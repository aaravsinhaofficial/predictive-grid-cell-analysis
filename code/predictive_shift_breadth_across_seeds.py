#!/usr/bin/env python3
"""Quantify breadth of predictive cells' preferred-shift distribution across seeds.

Uses per-seed `gridness_data.npz` files and reads:
  - `best_cm`: preferred shift (cm) at max 60-degree gridness per unit
  - `classes_predictive`: predictive unit indices

Outputs:
  - <stem>.png  : cross-seed distribution + breadth metrics figure
  - <stem>.csv  : per-seed and pooled breadth statistics
  - <stem>.json : metadata and metrics
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


EPS = 1e-9
SEED_RE = re.compile(r"Seed\s+(\d+)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--analysis-root",
        default="analysis_outputs",
        help="Root analysis folder. Prefix matching is used if exact path is missing (e.g., analysis_outputs_).",
    )
    parser.add_argument(
        "--model-subdir",
        default="Single agent path integration",
        help="Subdirectory under analysis root where per-seed outputs live.",
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Output directory. Defaults to <analysis-root>/<model-subdir>/summary/predictive_shift_breadth.",
    )
    parser.add_argument(
        "--save-stem",
        default="predictive_shift_breadth_across_seeds",
        help="Output filename stem.",
    )
    return parser.parse_args()


def resolve_analysis_root(path_like: str) -> Path:
    raw = Path(path_like).expanduser()
    if raw.exists():
        return raw.resolve()

    parent = raw.parent if str(raw.parent) else Path(".")
    candidates = sorted(p for p in parent.glob(f"{raw.name}*") if p.is_dir())
    if not candidates and raw.name.endswith("_"):
        trimmed = raw.name.rstrip("_")
        if trimmed:
            candidates = sorted(p for p in parent.glob(f"{trimmed}*") if p.is_dir())
    if not candidates:
        raise FileNotFoundError(f"Could not resolve analysis root from: {path_like}")

    for cand in candidates:
        if cand.name == "analysis_outputs":
            return cand.resolve()
    return candidates[0].resolve()


def extract_seed(path: Path) -> int:
    m = SEED_RE.search(str(path))
    return int(m.group(1)) if m else -1


def _compute_best_cm_if_missing(scores_60: np.ndarray, lag_cm: np.ndarray) -> np.ndarray:
    best_idx = np.full(scores_60.shape[1], -1, dtype=int)
    for u in range(scores_60.shape[1]):
        col = scores_60[:, u]
        if np.isfinite(col).any():
            best_idx[u] = int(np.nanargmax(col))
    best_cm = np.full(scores_60.shape[1], np.nan, dtype=float)
    valid = best_idx >= 0
    best_cm[valid] = lag_cm[best_idx[valid]]
    return best_cm


def summarize(values_cm: np.ndarray) -> Dict[str, float]:
    arr = np.asarray(values_cm, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {
            "count": 0,
            "mean_cm": float("nan"),
            "median_cm": float("nan"),
            "std_cm": float("nan"),
            "iqr_cm": float("nan"),
            "p10_cm": float("nan"),
            "p90_cm": float("nan"),
            "span_p10_p90_cm": float("nan"),
            "cv": float("nan"),
        }
    q10, q25, q50, q75, q90 = np.percentile(arr, [10, 25, 50, 75, 90])
    mean = float(np.mean(arr))
    std = float(np.std(arr))
    iqr = float(q75 - q25)
    return {
        "count": int(arr.size),
        "mean_cm": mean,
        "median_cm": float(q50),
        "std_cm": std,
        "iqr_cm": iqr,
        "p10_cm": float(q10),
        "p90_cm": float(q90),
        "span_p10_p90_cm": float(q90 - q10),
        "cv": float(std / (abs(mean) + EPS)),
    }


def collect_seed_data(search_root: Path) -> List[Dict[str, object]]:
    records: List[Dict[str, object]] = []
    for npz_path in sorted(search_root.rglob("gridness_data.npz")):
        seed = extract_seed(npz_path)
        with np.load(npz_path) as d:
            if "classes_predictive" not in d.files:
                continue
            pred_idxs = np.asarray(d["classes_predictive"], dtype=int).reshape(-1)
            if pred_idxs.size == 0:
                shifts = np.array([], dtype=float)
            else:
                if "best_cm" in d.files:
                    best_cm = np.asarray(d["best_cm"], dtype=float).reshape(-1)
                elif "scores_60" in d.files and "lag_cm" in d.files:
                    best_cm = _compute_best_cm_if_missing(np.asarray(d["scores_60"], dtype=float), np.asarray(d["lag_cm"], dtype=float))
                else:
                    continue

                pred_idxs = pred_idxs[(pred_idxs >= 0) & (pred_idxs < best_cm.shape[0])]
                shifts = best_cm[pred_idxs]
                shifts = shifts[np.isfinite(shifts)]

        stats = summarize(shifts)
        records.append(
            {
                "seed": int(seed),
                "gridness_path": str(npz_path),
                "shifts_cm": shifts,
                **stats,
            }
        )

    # Prefer one run per seed (largest predictive count).
    by_seed: Dict[int, Dict[str, object]] = {}
    unseeded: List[Dict[str, object]] = []
    for rec in records:
        seed = int(rec["seed"])
        if seed < 0:
            unseeded.append(rec)
            continue
        prev = by_seed.get(seed)
        if prev is None or int(rec["count"]) > int(prev["count"]):
            by_seed[seed] = rec
    return sorted(by_seed.values(), key=lambda r: int(r["seed"])) + unseeded


def plot_figure(records: List[Dict[str, object]], out_png: Path) -> None:
    seeds = [int(r["seed"]) for r in records]
    labels = [f"Seed {s}" if s >= 0 else f"Run {i}" for i, s in enumerate(seeds)]
    shift_arrays = [np.asarray(r["shifts_cm"], dtype=float) for r in records]
    stds = np.array([float(r["std_cm"]) for r in records], dtype=float)
    iqrs = np.array([float(r["iqr_cm"]) for r in records], dtype=float)
    counts = np.array([int(r["count"]) for r in records], dtype=int)

    pooled = np.concatenate([a[np.isfinite(a)] for a in shift_arrays if a.size > 0], axis=0)
    pooled_std = float(np.std(pooled)) if pooled.size else float("nan")
    pooled_iqr = float(np.percentile(pooled, 75) - np.percentile(pooled, 25)) if pooled.size else float("nan")

    fig = plt.figure(figsize=(12.6, 5.0), constrained_layout=True)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.4, 1.0], wspace=0.28)

    ax0 = fig.add_subplot(gs[0, 0])
    rng = np.random.default_rng(0)
    box_data = []
    box_pos = []
    for i, arr in enumerate(shift_arrays, start=1):
        if arr.size == 0:
            continue
        box_data.append(arr)
        box_pos.append(i)
    if box_data:
        bp = ax0.boxplot(
            box_data,
            positions=box_pos,
            widths=0.55,
            patch_artist=True,
            showfliers=False,
            medianprops={"color": "#111111", "linewidth": 1.6},
            boxprops={"facecolor": "#9fc2e6", "edgecolor": "#1d3f66", "alpha": 0.65},
            whiskerprops={"color": "#1d3f66", "linewidth": 1.1},
            capprops={"color": "#1d3f66", "linewidth": 1.1},
        )
        _ = bp

    max_points = 250
    for i, arr in enumerate(shift_arrays, start=1):
        if arr.size == 0:
            continue
        pts = arr
        if pts.size > max_points:
            idx = rng.choice(pts.size, size=max_points, replace=False)
            pts = pts[idx]
        jitter = rng.uniform(-0.16, 0.16, size=pts.size)
        ax0.scatter(
            np.full(pts.size, i, dtype=float) + jitter,
            pts,
            s=8,
            c="#2a6fbb",
            alpha=0.22,
            linewidths=0,
        )
        ax0.scatter([i], [np.mean(arr)], s=26, c="#d1495b", marker="D", zorder=4)

    ax0.set_xticks(np.arange(1, len(labels) + 1))
    ax0.set_xticklabels(labels, rotation=0)
    ax0.set_ylabel("Preferred shift of predictive units (cm)")
    ax0.set_title("Distribution Per Seed (boxplot + sampled points)")
    ax0.grid(axis="y", alpha=0.22)

    ax1 = fig.add_subplot(gs[0, 1])
    x = np.arange(len(records))
    w = 0.36
    ax1.bar(x - w / 2, stds, width=w, color="#4C72B0", label="SD (cm)")
    ax1.bar(x + w / 2, iqrs, width=w, color="#55A868", label="IQR (cm)")
    if math.isfinite(pooled_std):
        ax1.axhline(pooled_std, color="#4C72B0", linestyle="--", linewidth=1.3, alpha=0.8)
    if math.isfinite(pooled_iqr):
        ax1.axhline(pooled_iqr, color="#55A868", linestyle="--", linewidth=1.3, alpha=0.8)
    for i, c in enumerate(counts):
        ax1.text(i, max(stds[i], iqrs[i]) + 0.08, f"n={c}", ha="center", va="bottom", fontsize=8, color="#222222")

    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    ax1.set_ylabel("Breadth (cm)")
    ax1.set_title("Breadth Metrics Across Seeds")
    ax1.grid(axis="y", alpha=0.22)
    ax1.legend(frameon=False, loc="upper right")

    fig.suptitle("Predictive Preferred-Shift Breadth Across Seeds", fontsize=13)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=220)
    plt.close(fig)


def write_csv(records: List[Dict[str, object]], pooled_stats: Dict[str, float], out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "seed",
        "gridness_path",
        "count",
        "mean_cm",
        "median_cm",
        "std_cm",
        "iqr_cm",
        "p10_cm",
        "p90_cm",
        "span_p10_p90_cm",
        "cv",
    ]
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for rec in records:
            row = {k: rec.get(k, "") for k in fields}
            writer.writerow(row)
        pooled_row = {"seed": "pooled", "gridness_path": "all_seeds", **pooled_stats}
        writer.writerow(pooled_row)


def write_json(records: List[Dict[str, object]], pooled_stats: Dict[str, float], out_json: Path, search_root: Path) -> None:
    payload = {
        "metric_definition": {
            "std_cm": "Standard deviation of preferred shift (cm) among predictive units.",
            "iqr_cm": "Interquartile range Q75-Q25 (cm) among predictive units.",
            "span_p10_p90_cm": "P90-P10 span (cm) among predictive units.",
            "cv": "std_cm / |mean_cm|.",
        },
        "search_root": str(search_root),
        "seed_count": len(records),
        "pooled": pooled_stats,
        "records": [
            {k: v for k, v in rec.items() if k != "shifts_cm"} for rec in records
        ],
    }
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(payload, f, indent=2)


def main() -> None:
    args = parse_args()
    analysis_root = resolve_analysis_root(args.analysis_root)
    search_root = analysis_root / args.model_subdir if (analysis_root / args.model_subdir).exists() else analysis_root

    records = collect_seed_data(search_root)
    if not records:
        raise SystemExit(f"No gridness_data.npz with predictive classes found under: {search_root}")

    pooled = np.concatenate(
        [np.asarray(r["shifts_cm"], dtype=float) for r in records if int(r["count"]) > 0],
        axis=0,
    )
    pooled = pooled[np.isfinite(pooled)]
    pooled_stats = summarize(pooled)

    if args.out_dir is None:
        if (analysis_root / args.model_subdir).exists():
            out_dir = analysis_root / args.model_subdir / "summary" / "predictive_shift_breadth"
        else:
            out_dir = analysis_root / "summary" / "predictive_shift_breadth"
    else:
        out_dir = Path(args.out_dir).expanduser()

    stem = args.save_stem
    out_png = out_dir / f"{stem}.png"
    out_csv = out_dir / f"{stem}.csv"
    out_json = out_dir / f"{stem}.json"

    plot_figure(records, out_png)
    write_csv(records, pooled_stats, out_csv)
    write_json(records, pooled_stats, out_json, search_root)

    print(f"[predictive_shift_breadth] search_root: {search_root}")
    print(f"[predictive_shift_breadth] seeds processed: {len(records)}")
    print(f"[predictive_shift_breadth] wrote: {out_png}")
    print(f"[predictive_shift_breadth] wrote: {out_csv}")
    print(f"[predictive_shift_breadth] wrote: {out_json}")


if __name__ == "__main__":
    main()
