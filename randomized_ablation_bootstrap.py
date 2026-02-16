#!/usr/bin/env python3
"""Randomized percentile ablations to compare predictive-grid removal against random controls.

For each checkpoint:
  - classify cells with the original predictive/retrospective/normal definitions,
  - ablate top predictive units at several percentiles,
  - compare against matched-size random ablations (within predictive class and across all units),
  - sample a small fraction (default 5%) of the top set vs an equal-sized global random control.

Results are saved as JSON plus summary plots in analysis_outputs/randomized_ablation_bootstrap.
"""

from __future__ import annotations

import argparse
import copy
import glob
import json
import math
import os
from typing import Dict, Iterable, List, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from model import RNN
from place_cells import PlaceCells
from trajectory_generator import TrajectoryGenerator
from predictive_retrospective_ablation import (
    default_checkpoints,
    extract_state,
    load_gridness_data,
    seed_label_from_path,
    classify_from_scores,
    class_score_vectors,
    select_top_percentile,
)
from multi_seed_predictive_analysis import (
    build_options,
    collect_eval_batches,
    infer_dims_from_state,
    mean_decoding_error_cm,
    zero_unit_weights_in_place,
)


def dedupe_paths(paths: Iterable[str]) -> List[str]:
    seen = set()
    ordered = []
    for p in paths:
        if p not in seen:
            seen.add(p)
            ordered.append(p)
    return ordered


def evaluate_ablation(base_model: RNN, units: Sequence[int], eval_batches) -> float:
    """Return decoding error (cm) after ablating the provided units."""
    if units is None or len(units) == 0:
        return mean_decoding_error_cm(base_model, eval_batches)
    model = copy.deepcopy(base_model)
    zero_unit_weights_in_place(model, units)
    return mean_decoding_error_cm(model, eval_batches)


def aggregate_results(results: List[Dict], percentiles: Sequence[float]) -> Dict:
    """Average metrics across seeds for plotting."""
    percentiles = [float(p) for p in percentiles]
    baseline_vals = [r.get("baseline_error_cm") for r in results if r.get("baseline_error_cm") is not None]

    conds = ["top_predictive", "random_predictive", "random_global", "subset_top", "subset_global"]
    series = {c: {p: [] for p in percentiles} for c in conds}
    subset_size = {p: [] for p in percentiles}
    target_sizes = {p: [] for p in percentiles}

    for res in results:
        per_pct = res.get("percentile_results", [])
        for row in per_pct:
            pct = float(row.get("percentile"))
            if pct not in series["top_predictive"]:
                continue
            target_sizes[pct].append(int(row.get("target_size", 0)))
            subset_size[pct].append(int(row.get("subset_size", 0)))
            series["top_predictive"][pct].append(row.get("top_error_cm"))
            if row.get("random_predictive_errors"):
                series["random_predictive"][pct].append(float(np.nanmean(row["random_predictive_errors"])))
            if row.get("random_global_errors"):
                series["random_global"][pct].append(float(np.nanmean(row["random_global_errors"])))
            if row.get("subset_top_errors"):
                series["subset_top"][pct].append(float(np.nanmean(row["subset_top_errors"])))
            if row.get("subset_global_errors"):
                series["subset_global"][pct].append(float(np.nanmean(row["subset_global_errors"])))

    stats = {
        "baseline_mean": float(np.nanmean(baseline_vals)) if baseline_vals else math.nan,
        "baseline_std": float(np.nanstd(baseline_vals)) if baseline_vals else math.nan,
        "per_percentile": {},
    }
    for pct in percentiles:
        entry = {"target_size_mean": float(np.nanmean(target_sizes[pct])) if target_sizes[pct] else 0,
                 "target_size_std": float(np.nanstd(target_sizes[pct])) if target_sizes[pct] else 0,
                 "subset_size_mean": float(np.nanmean(subset_size[pct])) if subset_size[pct] else 0,
                 "subset_size_std": float(np.nanstd(subset_size[pct])) if subset_size[pct] else 0}
        for cond in conds:
            vals = [v for v in series[cond][pct] if v is not None and math.isfinite(v)]
            entry[cond + "_mean"] = float(np.nanmean(vals)) if vals else math.nan
            entry[cond + "_std"] = float(np.nanstd(vals)) if vals else math.nan
        stats["per_percentile"][str(pct)] = entry
    return stats


def plot_percentile_summary(agg: Dict, percentiles: Sequence[float], save_path: str) -> None:
    """Plot mean decoding error vs percentile for top vs random ablations."""
    percentiles = [float(p) for p in percentiles]
    fig, ax = plt.subplots(figsize=(7.2, 4.6))

    baseline_mean = agg.get("baseline_mean")
    baseline_std = agg.get("baseline_std")
    if baseline_mean is not None and math.isfinite(baseline_mean):
        ax.axhline(baseline_mean, color="#aaaaaa", ls="--", lw=1.2, label="Baseline mean")
        if baseline_std is not None and math.isfinite(baseline_std):
            ax.axhspan(baseline_mean - baseline_std, baseline_mean + baseline_std, color="#dddddd", alpha=0.35)

    labels = [
        ("Top predictive", "top_predictive", "#2166ac"),
        ("Random predictive", "random_predictive", "#1a9850"),
        ("Random global", "random_global", "#b2182b"),
    ]
    for name, key, color in labels:
        means = []
        stds = []
        for pct in percentiles:
            per = agg["per_percentile"].get(str(pct), {})
            means.append(per.get(key + "_mean", math.nan))
            stds.append(per.get(key + "_std", 0.0))
        means = np.array(means, dtype=float)
        stds = np.array(stds, dtype=float)
        ax.plot(percentiles, means, marker="o", color=color, label=name)
        ax.fill_between(percentiles, means - stds, means + stds, color=color, alpha=0.18)

    ax.set_xlabel("Percentile of predictive cells ablated")
    ax.set_ylabel("Mean decoding error (cm)")
    ax.set_title("Predictive ablations vs random controls (avg across seeds)")
    ax.grid(alpha=0.3)
    ax.legend(frameon=False)
    fig.tight_layout()
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    fig.savefig(save_path, dpi=220)
    plt.close(fig)


def plot_subset_control(results: List[Dict], target_percentile: float, save_path: str) -> None:
    """Boxplot for the '5% of top predictive vs global random' control."""
    vals_top = []
    vals_global = []
    for res in results:
        for row in res.get("percentile_results", []):
            if float(row.get("percentile")) != float(target_percentile):
                continue
            vals_top.extend([v for v in row.get("subset_top_errors", []) if v is not None and math.isfinite(v)])
            vals_global.extend([v for v in row.get("subset_global_errors", []) if v is not None and math.isfinite(v)])
    if not vals_top and not vals_global:
        return
    fig, ax = plt.subplots(figsize=(4.8, 4.2))
    data = [vals_top, vals_global]
    labels = [f"Top {target_percentile}% sample", "Global random\nsame count"]
    bp = ax.boxplot(data, tick_labels=labels, patch_artist=True, showmeans=True)
    colors = ["#377eb8", "#e41a1c"]
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.4)
    ax.set_ylabel("Decoding error (cm)")
    ax.set_title(f"Randomly ablating {target_percentile}% of predictive units")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    fig.savefig(save_path, dpi=220)
    plt.close(fig)


def run_randomized_ablation(
    ckpt_path: str,
    args,
    percentiles: Sequence[float],
    rng: np.random.Generator,
) -> Dict:
    raw = torch.load(ckpt_path, map_location="cpu")
    state = extract_state(raw)
    Ng, Np, velocity_dim = infer_dims_from_state(state)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    options = build_options(args, (Ng, Np, velocity_dim), device, ckpt_path)

    grid_data = load_gridness_data(ckpt_path)
    classes = classify_from_scores(grid_data, args.min_shift_cm, args.gridness_threshold)
    score_vectors = class_score_vectors(grid_data, args.min_shift_cm)
    predictive_pool = np.asarray(classes["predictive"], dtype=int)
    all_units = np.arange(Ng, dtype=int)

    place_cells = PlaceCells(options)
    traj_gen = TrajectoryGenerator(options, place_cells)
    base_model = RNN(options, place_cells).to(options.device)
    base_model.load_state_dict(state)
    base_model.eval()

    eval_batches = collect_eval_batches(traj_gen, args.ablation_batches)
    baseline_err = mean_decoding_error_cm(base_model, eval_batches)

    per_percentile: List[Dict] = []
    for pct in percentiles:
        pct = float(pct)
        top_units = select_top_percentile(predictive_pool, score_vectors["predictive"], pct)
        n_target = int(top_units.size)
        subset_size = int(math.ceil(n_target * args.top_fraction)) if n_target > 0 else 0

        row = {
            "percentile": pct,
            "target_size": n_target,
            "subset_size": subset_size,
            "top_error_cm": float(baseline_err) if n_target == 0 else float(evaluate_ablation(base_model, top_units.tolist(), eval_batches)),
            "random_predictive_errors": [],
            "random_global_errors": [],
            "subset_top_errors": [],
            "subset_global_errors": [],
        }

        # Random ablations matching the full target size
        if n_target > 0 and predictive_pool.size >= n_target:
            for _ in range(args.repeats):
                sel = rng.choice(predictive_pool, size=n_target, replace=False)
                row["random_predictive_errors"].append(float(evaluate_ablation(base_model, sel.tolist(), eval_batches)))
        if n_target > 0 and all_units.size >= n_target:
            for _ in range(args.repeats):
                sel = rng.choice(all_units, size=n_target, replace=False)
                row["random_global_errors"].append(float(evaluate_ablation(base_model, sel.tolist(), eval_batches)))

        # Randomly ablating a small fraction of the top set vs anywhere
        if subset_size > 0 and top_units.size > 0:
            subset_size = min(subset_size, top_units.size)
            for _ in range(args.repeats):
                sel_top = rng.choice(top_units, size=subset_size, replace=False)
                row["subset_top_errors"].append(float(evaluate_ablation(base_model, sel_top.tolist(), eval_batches)))
                sel_global = rng.choice(all_units, size=subset_size, replace=False)
                row["subset_global_errors"].append(float(evaluate_ablation(base_model, sel_global.tolist(), eval_batches)))

        per_percentile.append(row)

    return {
        "checkpoint": ckpt_path,
        "seed": seed_label_from_path(ckpt_path),
        "baseline_error_cm": baseline_err,
        "class_counts": {k: int(len(v)) for k, v in classes.items()},
        "percentile_results": per_percentile,
    }


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint_paths", nargs="+", default=default_checkpoints(),
                        help="Checkpoints to evaluate (paths, dirs, or globs).")
    parser.add_argument("--ablate_percentiles", nargs="+", type=float, default=[5, 15, 30, 50],
                        help="Percentiles (within predictive class) to ablate.")
    parser.add_argument("--repeats", type=int, default=20, help="Bootstrap repetitions per percentile.")
    parser.add_argument("--top_fraction", type=float, default=0.05,
                        help="Fraction of the top set to randomly ablate for the small-fraction control.")
    parser.add_argument("--ablation_batches", type=int, default=4, help="Cached eval batches per checkpoint.")
    parser.add_argument("--batch_size", type=int, default=80)
    parser.add_argument("--sequence_length", type=int, default=20)
    parser.add_argument("--place_cell_rf", type=float, default=0.12)
    parser.add_argument("--surround_scale", type=float, default=2.0)
    parser.add_argument("--activation", default="relu")
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--box_width", type=float, default=2.2)
    parser.add_argument("--box_height", type=float, default=2.2)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--min_shift_cm", type=float, default=5.0)
    parser.add_argument("--gridness_threshold", type=float, default=0.3)
    parser.add_argument("--device", default=None, help="Override device (cpu/cuda).")
    parser.add_argument("--output_dir", default="analysis_outputs/randomized_ablation_bootstrap",
                        help="Directory to store figures/results.")
    parser.add_argument("--seed", type=int, default=0, help="Base seed for randomized ablations.")
    parser.add_argument("--traj_speed_scale", default=1.0, type=float)
    parser.add_argument("--traj_speed_max", default=None, type=float)
    parser.add_argument("--traj_velocity_smoothing", default=0.0, type=float)
    parser.add_argument("--traj_turn_sigma_scale", default=1.0, type=float)
    parser.add_argument("--traj_border_region", default=0.03, type=float)
    parser.add_argument("--traj_wall_slowdown", default=0.25, type=float)
    parser.add_argument("--traj_wall_turn_scale", default=1.0, type=float)
    return parser.parse_args()


def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    checkpoints: List[str] = []
    for path in args.checkpoint_paths:
        if any(ch in path for ch in "*?[]"):
            checkpoints.extend(sorted(glob.glob(path)))  # type: ignore[name-defined]
        elif os.path.isdir(path):
            for root, _, files in os.walk(path):
                for f in files:
                    if f.endswith(".pth"):
                        checkpoints.append(os.path.join(root, f))
        else:
            checkpoints.append(path)
    checkpoints = dedupe_paths(checkpoints)
    if not checkpoints:
        raise FileNotFoundError("No checkpoints matched the provided paths.")

    percentiles = [float(p) for p in args.ablate_percentiles]

    results: List[Dict] = []
    for ckpt in checkpoints:
        res = run_randomized_ablation(ckpt, args, percentiles, rng)
        results.append(res)

    os.makedirs(args.output_dir, exist_ok=True)
    agg = aggregate_results(results, percentiles)

    with open(os.path.join(args.output_dir, "randomized_ablation_results.json"), "w") as f:
        json.dump({"percentiles": percentiles, "seeds": results, "aggregate": agg}, f, indent=2)

    plot_percentile_summary(agg, percentiles, os.path.join(args.output_dir, "percentile_summary.png"))
    if 5.0 in percentiles:
        plot_subset_control(results, 5.0, os.path.join(args.output_dir, "top5pct_subset_boxplot.png"))


if __name__ == "__main__":
    main()
