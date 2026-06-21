#!/usr/bin/env python3
"""Cross-seed summary of predictive/retrospective/grid cell properties."""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np

from path_utils import analysis_dir_for_checkpoint, analysis_summary_dir, model_name_from_checkpoint
from predictive_retrospective_ablation import classify_from_scores


SEED_RE = re.compile(r"Seed\s+(\d+)")


def _seed_from_path(path: str) -> int:
    match = SEED_RE.search(path)
    return int(match.group(1)) if match else -1


def _compute_best_cm(scores_60: np.ndarray, lag_cm: np.ndarray) -> np.ndarray:
    best_idx = np.full(scores_60.shape[1], -1, dtype=int)
    for unit in range(scores_60.shape[1]):
        col = scores_60[:, unit]
        if np.isfinite(col).any():
            best_idx[unit] = int(np.nanargmax(col))
    best_cm = np.full(scores_60.shape[1], np.nan, dtype=float)
    valid = best_idx >= 0
    best_cm[valid] = lag_cm[best_idx[valid]]
    return best_cm


def _load_gridness(ckpt_path: str, analysis_subdir: str | None = None) -> Dict[str, np.ndarray]:
    analysis_dir = analysis_dir_for_checkpoint(Path(ckpt_path))
    if analysis_subdir:
        analysis_dir = analysis_dir / str(analysis_subdir)
    npz_path = analysis_dir / "gridness_data.npz"
    if not npz_path.exists():
        raise FileNotFoundError(f"Missing gridness_data.npz for checkpoint: {ckpt_path}\nExpected: {npz_path}")
    with np.load(npz_path) as data:
        return {k: data[k] for k in data.files}


def _record_from_checkpoint(
    ckpt_path: str,
    min_shift_cm: float,
    gridness_threshold: float,
    ng_use: int,
    analysis_subdir: str | None = None,
) -> Dict:
    data = _load_gridness(ckpt_path, analysis_subdir=analysis_subdir)
    scores_60 = np.asarray(data["scores_60"], dtype=float)
    lag_cm = np.asarray(data["lag_cm"], dtype=float)
    best_cm = np.asarray(data["best_cm"], dtype=float) if "best_cm" in data else _compute_best_cm(scores_60, lag_cm)

    ng_data = int(scores_60.shape[1])
    ng_eval = min(int(ng_use), ng_data) if int(ng_use) > 0 else ng_data
    scores_eval = scores_60[:, :ng_eval]
    best_cm_eval = best_cm[:ng_eval]

    cls_data = {
        "lag_cm": lag_cm,
        "scores_60": scores_eval,
    }
    classes = classify_from_scores(cls_data, min_shift_cm=min_shift_cm, gridness_threshold=gridness_threshold)
    predictive = np.asarray(classes["predictive"], dtype=int)
    retrospective = np.asarray(classes["retrospective"], dtype=int)
    normal = np.asarray(classes["normal"], dtype=int)
    grid_all = np.unique(np.concatenate([predictive, retrospective, normal])) if (predictive.size or retrospective.size or normal.size) else np.array([], dtype=int)
    non_grid = np.setdiff1d(np.arange(ng_eval, dtype=int), grid_all)

    def _fraction(count: int) -> float:
        return float(count / ng_eval) if ng_eval > 0 else float("nan")

    return {
        "checkpoint": ckpt_path,
        "analysis_subdir": analysis_subdir,
        "shift_mode": str(np.asarray(data.get("shift_mode", "unknown")).reshape(())),
        "seed": _seed_from_path(ckpt_path),
        "Ng_eval": ng_eval,
        "counts": {
            "predictive": int(predictive.size),
            "retrospective": int(retrospective.size),
            "normal": int(normal.size),
            "grid": int(grid_all.size),
            "non_grid": int(non_grid.size),
        },
        "fractions": {
            "predictive": _fraction(int(predictive.size)),
            "retrospective": _fraction(int(retrospective.size)),
            "normal": _fraction(int(normal.size)),
            "grid": _fraction(int(grid_all.size)),
            "non_grid": _fraction(int(non_grid.size)),
        },
        "predictive_shifts_cm": best_cm_eval[predictive].tolist(),
        "retrospective_shifts_cm": best_cm_eval[retrospective].tolist(),
    }


def _mean_sem(values: List[float]) -> Dict[str, float]:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"mean": float("nan"), "sem": float("nan")}
    sem = float(np.nanstd(arr, ddof=1) / np.sqrt(arr.size)) if arr.size > 1 else 0.0
    return {"mean": float(np.nanmean(arr)), "sem": sem}


def plot_summary(records: List[Dict], save_path: Path) -> None:
    records = sorted(records, key=lambda r: (r["seed"], r["checkpoint"]))
    seeds = [f"Seed {r['seed']}" if r["seed"] >= 0 else Path(r["checkpoint"]).name for r in records]
    x = np.arange(len(records))

    class_order = ["predictive", "retrospective", "normal", "non_grid"]
    class_colors = {
        "predictive": "#1f77b4",
        "retrospective": "#d62728",
        "normal": "#2ca02c",
        "non_grid": "#bdbdbd",
    }

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))

    # Panel A: stacked fractions per seed
    ax = axes[0, 0]
    bottom = np.zeros(len(records), dtype=float)
    for key in class_order:
        vals = np.array([r["fractions"][key] for r in records], dtype=float)
        ax.bar(x, vals, bottom=bottom, color=class_colors[key], label=key.replace("_", " ").title(), width=0.72)
        bottom += vals
    ax.set_xticks(x)
    ax.set_xticklabels(seeds, rotation=0)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Fraction of analyzed units")
    ax.set_title("A. Functional-class composition per seed")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, fontsize=9, ncol=2)

    # Panel B: class counts per seed
    ax = axes[0, 1]
    for key in ["predictive", "retrospective", "normal", "grid"]:
        vals = np.array([r["counts"][key] for r in records], dtype=float)
        ax.plot(x, vals, marker="o", lw=1.8, label=key.replace("_", " ").title())
    ax.set_xticks(x)
    ax.set_xticklabels(seeds, rotation=0)
    ax.set_ylabel("Unit count")
    ax.set_title("B. Counts across seeds")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=9)

    # Panel C: preferred shift distributions (pooled)
    ax = axes[1, 0]
    pred = np.concatenate([np.asarray(r["predictive_shifts_cm"], dtype=float) for r in records if len(r["predictive_shifts_cm"]) > 0]) if any(len(r["predictive_shifts_cm"]) > 0 for r in records) else np.array([], dtype=float)
    retro = np.concatenate([np.asarray(r["retrospective_shifts_cm"], dtype=float) for r in records if len(r["retrospective_shifts_cm"]) > 0]) if any(len(r["retrospective_shifts_cm"]) > 0 for r in records) else np.array([], dtype=float)
    combined = np.concatenate([pred, retro]) if (pred.size or retro.size) else np.array([], dtype=float)
    if combined.size:
        limit = max(5.0, float(np.nanmax(np.abs(combined))))
        bins = np.linspace(-limit, limit, 40)
        if retro.size:
            ax.hist(retro, bins=bins, alpha=0.65, color=class_colors["retrospective"], label=f"Retrospective (n={retro.size})")
        if pred.size:
            ax.hist(pred, bins=bins, alpha=0.65, color=class_colors["predictive"], label=f"Predictive (n={pred.size})")
        ax.axvline(0, color="k", lw=1.0)
        ax.legend(frameon=False, fontsize=9)
    else:
        ax.text(0.5, 0.5, "No preferred-shift data", ha="center", va="center", transform=ax.transAxes)
    ax.set_xlabel("Preferred shift (cm)")
    ax.set_ylabel("Unit count")
    ax.set_title("C. Preferred-shift distributions (pooled)")
    ax.grid(alpha=0.25)

    # Panel D: mean +/- SEM fractions
    ax = axes[1, 1]
    labels = ["predictive", "retrospective", "normal", "grid"]
    means = []
    sems = []
    for key in labels:
        stats = _mean_sem([r["fractions"][key] for r in records])
        means.append(stats["mean"])
        sems.append(stats["sem"] if math.isfinite(stats["sem"]) else 0.0)
    xpos = np.arange(len(labels))
    ax.bar(
        xpos,
        means,
        yerr=sems,
        capsize=4,
        color=[class_colors.get(k, "#777777") for k in labels],
        alpha=0.85,
        edgecolor="black",
        linewidth=0.6,
    )
    ax.set_xticks(xpos)
    ax.set_xticklabels([k.replace("_", " ").title() for k in labels], rotation=15)
    ax.set_ylabel("Fraction of analyzed units")
    ax.set_title("D. Mean +/- SEM across seeds")
    ax.grid(axis="y", alpha=0.25)

    fig.suptitle("Predictive/Retrospective/Grid Summary Across Single-Agent Seeds", y=1.01)
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint_paths", nargs="+", required=True, help="Model checkpoints to summarize.")
    parser.add_argument("--gridness_threshold", type=float, default=0.2, help="Minimum peak gridness to call a unit grid-like.")
    parser.add_argument("--min_shift_cm", type=float, default=5.0, help="Minimum shift magnitude for predictive/retrospective class.")
    parser.add_argument("--Ng_use", type=int, default=512, help="Number of units to include from each checkpoint (<=0 uses all cached).")
    parser.add_argument("--output_dir", default=None, help="Optional output folder override.")
    parser.add_argument(
        "--analysis_subdir",
        default=None,
        help="Optional subdirectory under each checkpoint analysis folder from which to load gridness_data.npz.",
    )
    args = parser.parse_args()

    if args.output_dir is None:
        model_name = model_name_from_checkpoint(Path(args.checkpoint_paths[0]))
        summary_name = "cross_seed_cell_class_summary"
        if args.analysis_subdir:
            safe_label = re.sub(r"[^A-Za-z0-9._-]+", "_", str(args.analysis_subdir)).strip("_")
            if safe_label:
                summary_name = f"{summary_name}_{safe_label}"
        output_dir = analysis_summary_dir(model_name, summary_name)
    else:
        output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    records = [
        _record_from_checkpoint(
            ckpt_path=ckpt,
            min_shift_cm=float(args.min_shift_cm),
            gridness_threshold=float(args.gridness_threshold),
            ng_use=int(args.Ng_use),
            analysis_subdir=args.analysis_subdir,
        )
        for ckpt in args.checkpoint_paths
    ]
    records = sorted(records, key=lambda r: (r["seed"], r["checkpoint"]))

    plot_path = output_dir / "cell_class_summary_across_seeds.png"
    plot_summary(records, plot_path)

    pooled = {
        "predictive_fraction": _mean_sem([r["fractions"]["predictive"] for r in records]),
        "retrospective_fraction": _mean_sem([r["fractions"]["retrospective"] for r in records]),
        "normal_fraction": _mean_sem([r["fractions"]["normal"] for r in records]),
        "grid_fraction": _mean_sem([r["fractions"]["grid"] for r in records]),
    }
    out_json = {
        "checkpoints": args.checkpoint_paths,
        "Ng_use": int(args.Ng_use),
        "gridness_threshold": float(args.gridness_threshold),
        "min_shift_cm": float(args.min_shift_cm),
        "analysis_subdir": args.analysis_subdir,
        "records": records,
        "pooled": pooled,
        "figure_path": str(plot_path),
    }
    with open(output_dir / "cell_class_summary_across_seeds.json", "w") as f:
        json.dump(out_json, f, indent=2)

    print(json.dumps({"figure": str(plot_path), "json": str(output_dir / "cell_class_summary_across_seeds.json")}, indent=2))


if __name__ == "__main__":
    main()
