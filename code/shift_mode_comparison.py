#!/usr/bin/env python3
"""Compare temporal and spatial shift analyses plus their ablation summaries."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np

from multi_seed_predictive_analysis import checkpoint_analysis_dir
from path_utils import analysis_summary_dir, model_name_from_checkpoint
from predictive_retrospective_ablation import classify_from_scores, default_checkpoints, seed_label_from_path


CLASS_ORDER = ("predictive", "retrospective", "normal")
CLASS_COLORS = {
    "predictive": "#1f77b4",
    "retrospective": "#d62728",
    "normal": "#222222",
}
MODE_COLORS = {
    "temporal": "#4c78a8",
    "spatial": "#f58518",
}


def load_gridness_data(ckpt_path: str, subdir: str | None) -> Dict[str, np.ndarray]:
    grid_path = checkpoint_analysis_dir(ckpt_path, subdir) / "gridness_data.npz"
    if not grid_path.exists():
        raise FileNotFoundError(f"Missing gridness data: {grid_path}")
    with np.load(grid_path) as data:
        return {k: data[k] for k in data.files}


def best_shift_scores(scores_60: np.ndarray, lag_cm: np.ndarray, min_shift_cm: float) -> np.ndarray:
    mask = np.abs(lag_cm) >= float(min_shift_cm)
    if not np.any(mask):
        return np.full(scores_60.shape[1], np.nan, dtype=float)
    vals = np.nanmax(scores_60[mask], axis=0)
    vals[~np.isfinite(vals)] = np.nan
    return vals


def _safe_mean_sem(values: Sequence[float]) -> Tuple[float, float]:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return math.nan, 0.0
    mean = float(np.nanmean(arr))
    if arr.size == 1:
        return mean, 0.0
    sem = float(np.nanstd(arr, ddof=1) / np.sqrt(arr.size))
    return mean, sem


def plot_best_shift_scatter(
    x_vals: np.ndarray,
    y_vals: np.ndarray,
    seed_ids: np.ndarray,
    save_path: Path,
    max_points: int,
) -> None:
    valid = np.isfinite(x_vals) & np.isfinite(y_vals)
    if not np.any(valid):
        return
    x = x_vals[valid]
    y = y_vals[valid]
    seeds = seed_ids[valid]

    if x.size > max_points:
        rng = np.random.default_rng(0)
        keep = np.sort(rng.choice(x.size, size=max_points, replace=False))
        x = x[keep]
        y = y[keep]
        seeds = seeds[keep]

    unique_seeds = sorted(set(seeds.tolist()))
    cmap = plt.get_cmap("tab10", max(1, len(unique_seeds)))
    color_map = {seed: cmap(i) for i, seed in enumerate(unique_seeds)}

    fig, ax = plt.subplots(figsize=(6.6, 5.8))
    for seed in unique_seeds:
        mask = seeds == seed
        ax.scatter(
            x[mask],
            y[mask],
            s=10,
            alpha=0.22,
            color=color_map[seed],
            edgecolors="none",
            label=f"Seed {seed}",
        )

    lower = float(min(np.nanmin(x), np.nanmin(y)))
    upper = float(max(np.nanmax(x), np.nanmax(y)))
    lower = min(lower, 0.0)
    upper = max(upper, 0.0)
    ax.plot([lower, upper], [lower, upper], color="black", lw=1.0, ls="--")
    ax.set_xlabel("Best shifted gridness, temporal mode")
    ax.set_ylabel("Best shifted gridness, spatial mode")
    ax.set_title("Unit-wise comparison of best shifted gridness")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=8, ncol=2, loc="lower right")
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=220)
    plt.close(fig)


def plot_class_count_comparison(records: List[Dict[str, object]], save_path: Path) -> None:
    if not records:
        return

    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.8), sharex=True)
    metrics = ("count", "fraction")
    titles = ("Class counts", "Class fractions")
    ylabel = ("Units", "Fraction of analysed units")

    for ax, metric, title, ylab in zip(axes, metrics, titles, ylabel):
        x = np.arange(len(CLASS_ORDER))
        width = 0.34
        temporal_means = []
        temporal_sems = []
        spatial_means = []
        spatial_sems = []

        for cls_name in CLASS_ORDER:
            temporal_vals = [float(r["temporal"][metric][cls_name]) for r in records]
            spatial_vals = [float(r["spatial"][metric][cls_name]) for r in records]
            t_mean, t_sem = _safe_mean_sem(temporal_vals)
            s_mean, s_sem = _safe_mean_sem(spatial_vals)
            temporal_means.append(t_mean)
            temporal_sems.append(t_sem)
            spatial_means.append(s_mean)
            spatial_sems.append(s_sem)

            x_center = x[len(temporal_means) - 1]
            for rec in records:
                ax.plot(
                    [x_center - width / 2, x_center + width / 2],
                    [float(rec["temporal"][metric][cls_name]), float(rec["spatial"][metric][cls_name])],
                    color="#bbbbbb",
                    alpha=0.45,
                    lw=1.0,
                    zorder=1,
                )

        ax.bar(
            x - width / 2,
            temporal_means,
            width,
            yerr=temporal_sems,
            color=MODE_COLORS["temporal"],
            alpha=0.88,
            capsize=4,
            edgecolor="black",
            linewidth=0.6,
            label="Temporal",
            zorder=2,
        )
        ax.bar(
            x + width / 2,
            spatial_means,
            width,
            yerr=spatial_sems,
            color=MODE_COLORS["spatial"],
            alpha=0.88,
            capsize=4,
            edgecolor="black",
            linewidth=0.6,
            label="Spatial",
            zorder=2,
        )
        ax.set_xticks(x)
        ax.set_xticklabels([c.title() for c in CLASS_ORDER], rotation=15)
        ax.set_ylabel(ylab)
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.25)

    axes[0].legend(frameon=False)
    fig.suptitle("Temporal vs spatial class membership across seeds", y=1.02)
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _extract_ablation_delta(seed_entry: Dict[str, object], cls_name: str, percentile: float) -> Tuple[float, float]:
    percentiles = [float(p) for p in seed_entry.get("percentiles", [])]
    if percentile not in percentiles:
        return math.nan, math.nan
    idx = percentiles.index(percentile)
    baseline = seed_entry.get("baseline_error_cm")
    if baseline is None or not math.isfinite(float(baseline)):
        return math.nan, math.nan
    baseline_f = float(baseline)
    vals = seed_entry.get("errors", {}).get(cls_name, [])
    rand_vals = seed_entry.get("random_errors", {}).get(cls_name, [])
    targeted = math.nan
    matched = math.nan
    if len(vals) > idx and vals[idx] is not None and math.isfinite(float(vals[idx])):
        targeted = float(vals[idx]) - baseline_f
    if len(rand_vals) > idx and rand_vals[idx] is not None and math.isfinite(float(rand_vals[idx])):
        matched = float(vals[idx]) - float(rand_vals[idx]) if len(vals) > idx and vals[idx] is not None and math.isfinite(float(vals[idx])) else math.nan
    return targeted, matched


def plot_ablation_mode_comparison(
    temporal_summary: Dict[str, object],
    spatial_summary: Dict[str, object],
    percentile: float,
    save_path: Path,
) -> None:
    temporal_by_seed = {str(seed["seed"]): seed for seed in temporal_summary.get("seeds", [])}
    spatial_by_seed = {str(seed["seed"]): seed for seed in spatial_summary.get("seeds", [])}
    shared_seeds = sorted(set(temporal_by_seed) & set(spatial_by_seed), key=lambda x: int(x) if str(x).isdigit() else x)
    if not shared_seeds:
        return

    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.8), sharex=True)
    panels = (
        ("Targeted ablation minus baseline", 0),
        ("Targeted ablation minus matched random", 1),
    )

    for ax, (title, panel_idx) in zip(axes, panels):
        x = np.arange(len(CLASS_ORDER))
        width = 0.34
        temporal_means = []
        temporal_sems = []
        spatial_means = []
        spatial_sems = []
        for cls_name in CLASS_ORDER:
            temporal_vals = []
            spatial_vals = []
            for seed in shared_seeds:
                t_targeted, t_matched = _extract_ablation_delta(temporal_by_seed[seed], cls_name, percentile)
                s_targeted, s_matched = _extract_ablation_delta(spatial_by_seed[seed], cls_name, percentile)
                temporal_vals.append(t_targeted if panel_idx == 0 else t_matched)
                spatial_vals.append(s_targeted if panel_idx == 0 else s_matched)

            t_mean, t_sem = _safe_mean_sem(temporal_vals)
            s_mean, s_sem = _safe_mean_sem(spatial_vals)
            temporal_means.append(t_mean)
            temporal_sems.append(t_sem)
            spatial_means.append(s_mean)
            spatial_sems.append(s_sem)

            x_center = x[len(temporal_means) - 1]
            for t_val, s_val in zip(temporal_vals, spatial_vals):
                if math.isfinite(t_val) and math.isfinite(s_val):
                    ax.plot(
                        [x_center - width / 2, x_center + width / 2],
                        [t_val, s_val],
                        color="#bbbbbb",
                        alpha=0.45,
                        lw=1.0,
                        zorder=1,
                    )

        ax.bar(
            x - width / 2,
            temporal_means,
            width,
            yerr=temporal_sems,
            color=MODE_COLORS["temporal"],
            alpha=0.88,
            capsize=4,
            edgecolor="black",
            linewidth=0.6,
            label="Temporal",
            zorder=2,
        )
        ax.bar(
            x + width / 2,
            spatial_means,
            width,
            yerr=spatial_sems,
            color=MODE_COLORS["spatial"],
            alpha=0.88,
            capsize=4,
            edgecolor="black",
            linewidth=0.6,
            label="Spatial",
            zorder=2,
        )
        ax.axhline(0.0, color="#555555", lw=1.0, ls="--")
        ax.set_xticks(x)
        ax.set_xticklabels([c.title() for c in CLASS_ORDER], rotation=15)
        ax.set_ylabel("Delta decoding error (cm)")
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.25)

    axes[0].legend(frameon=False)
    fig.suptitle(f"Temporal vs spatial ablation effects at {percentile:.0f}% removal", y=1.02)
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _extract_all_percentile_deltas(
    seed_entry: Dict[str, object],
    cls_name: str,
) -> Tuple[List[float], List[float], List[float], List[float]]:
    """Return (percentiles, targeted_deltas, random_deltas, matched_deltas) for one seed+class."""
    percentiles = [float(p) for p in seed_entry.get("percentiles", [])]
    baseline = seed_entry.get("baseline_error_cm")
    if baseline is None or not math.isfinite(float(baseline)):
        return percentiles, [math.nan] * len(percentiles), [math.nan] * len(percentiles), [math.nan] * len(percentiles)
    baseline_f = float(baseline)
    vals = seed_entry.get("errors", {}).get(cls_name, [])
    rand_vals = seed_entry.get("random_errors", {}).get(cls_name, [])
    targeted = []
    random_delta = []
    matched = []
    for i in range(len(percentiles)):
        t = math.nan
        r = math.nan
        m = math.nan
        if i < len(vals) and vals[i] is not None and math.isfinite(float(vals[i])):
            t = float(vals[i]) - baseline_f
        if i < len(rand_vals) and rand_vals[i] is not None and math.isfinite(float(rand_vals[i])):
            r = float(rand_vals[i]) - baseline_f
        if math.isfinite(t) and math.isfinite(r):
            m = t - r  # targeted minus random (both already relative to baseline, so this is targeted_error - random_error)
            # Actually: t = err - baseline, r = rand_err - baseline, so t - r = err - rand_err
            # But we want targeted_err - rand_err = t - r + 0  ... wait
            # t = err - baseline, r = rand - baseline => t - r = err - rand.  Correct.
        targeted.append(t)
        random_delta.append(r)
        matched.append(m)
    return percentiles, targeted, random_delta, matched


def plot_ablation_percentile_curves(
    temporal_summary: Dict[str, object],
    spatial_summary: Dict[str, object],
    save_path: Path,
) -> None:
    """Line plots of ablation delta vs percentile for each class, comparing modes.

    Layout: 3 columns (predictive, retrospective, normal) x 2 rows
      Row 1: targeted ablation minus baseline
      Row 2: targeted minus matched-random
    Each panel has 4 lines: temporal targeted, temporal random, spatial targeted, spatial random
    (row 2 omits random lines since they are the baseline by definition).
    """
    temporal_seeds = temporal_summary.get("seeds", [])
    spatial_seeds = spatial_summary.get("seeds", [])
    if not temporal_seeds or not spatial_seeds:
        return

    temporal_by_seed = {str(s["seed"]): s for s in temporal_seeds}
    spatial_by_seed = {str(s["seed"]): s for s in spatial_seeds}
    shared_seeds = sorted(
        set(temporal_by_seed) & set(spatial_by_seed),
        key=lambda x: int(x) if str(x).isdigit() else x,
    )
    if not shared_seeds:
        return

    # Use percentiles from the first seed (assumed consistent)
    percentiles = [float(p) for p in temporal_seeds[0].get("percentiles", [])]
    if not percentiles:
        return

    fig, axes = plt.subplots(2, 3, figsize=(14.4, 8.0), sharex=True)

    cls_colors = {
        "predictive": ("#1f77b4", "#aec7e8"),
        "retrospective": ("#d62728", "#f4a6a6"),
        "normal": ("#222222", "#aaaaaa"),
    }

    for col_idx, cls_name in enumerate(CLASS_ORDER):
        # Gather per-seed curves
        t_targeted_rows = []
        t_random_rows = []
        s_targeted_rows = []
        s_random_rows = []
        t_matched_rows = []
        s_matched_rows = []

        for seed in shared_seeds:
            _, t_targ, t_rand, t_match = _extract_all_percentile_deltas(temporal_by_seed[seed], cls_name)
            _, s_targ, s_rand, s_match = _extract_all_percentile_deltas(spatial_by_seed[seed], cls_name)
            t_targeted_rows.append(t_targ)
            t_random_rows.append(t_rand)
            s_targeted_rows.append(s_targ)
            s_random_rows.append(s_rand)
            t_matched_rows.append(t_match)
            s_matched_rows.append(s_match)

        pcts = np.array(percentiles)

        def _mean_sem(rows):
            mat = np.array(rows, dtype=float)
            m = np.nanmean(mat, axis=0)
            if mat.shape[0] > 1:
                se = np.nanstd(mat, axis=0, ddof=1) / np.sqrt(mat.shape[0])
            else:
                se = np.zeros_like(m)
            return m, se

        t_targ_m, t_targ_se = _mean_sem(t_targeted_rows)
        t_rand_m, t_rand_se = _mean_sem(t_random_rows)
        s_targ_m, s_targ_se = _mean_sem(s_targeted_rows)
        s_rand_m, s_rand_se = _mean_sem(s_random_rows)
        t_match_m, t_match_se = _mean_sem(t_matched_rows)
        s_match_m, s_match_se = _mean_sem(s_matched_rows)

        # --- Row 0: targeted & random vs baseline ---
        ax0 = axes[0, col_idx]
        ax0.plot(pcts, t_targ_m, "o-", color=MODE_COLORS["temporal"], lw=2.0, label="Temporal targeted", zorder=3)
        ax0.fill_between(pcts, t_targ_m - t_targ_se, t_targ_m + t_targ_se, color=MODE_COLORS["temporal"], alpha=0.15)
        ax0.plot(pcts, t_rand_m, "x--", color=MODE_COLORS["temporal"], lw=1.3, alpha=0.7, label="Temporal random", zorder=2)

        ax0.plot(pcts, s_targ_m, "o-", color=MODE_COLORS["spatial"], lw=2.0, label="Spatial targeted", zorder=3)
        ax0.fill_between(pcts, s_targ_m - s_targ_se, s_targ_m + s_targ_se, color=MODE_COLORS["spatial"], alpha=0.15)
        ax0.plot(pcts, s_rand_m, "x--", color=MODE_COLORS["spatial"], lw=1.3, alpha=0.7, label="Spatial random", zorder=2)

        ax0.axhline(0.0, color="#555555", lw=0.8, ls="--")
        ax0.set_title(f"{cls_name.title()}", fontsize=12, fontweight="bold")
        ax0.set_ylabel("Δ error vs baseline (cm)" if col_idx == 0 else "")
        ax0.grid(alpha=0.25)
        if col_idx == 0:
            ax0.legend(frameon=False, fontsize=7, loc="upper left")

        # --- Row 1: targeted minus matched-random ---
        ax1 = axes[1, col_idx]
        ax1.plot(pcts, t_match_m, "o-", color=MODE_COLORS["temporal"], lw=2.0, label="Temporal", zorder=3)
        ax1.fill_between(pcts, t_match_m - t_match_se, t_match_m + t_match_se, color=MODE_COLORS["temporal"], alpha=0.15)

        ax1.plot(pcts, s_match_m, "o-", color=MODE_COLORS["spatial"], lw=2.0, label="Spatial", zorder=3)
        ax1.fill_between(pcts, s_match_m - s_match_se, s_match_m + s_match_se, color=MODE_COLORS["spatial"], alpha=0.15)

        ax1.axhline(0.0, color="#555555", lw=0.8, ls="--")
        ax1.set_xlabel("Top percentile of class removed")
        ax1.set_ylabel("Targeted − random Δ (cm)" if col_idx == 0 else "")
        ax1.grid(alpha=0.25)
        if col_idx == 0:
            ax1.legend(frameon=False, fontsize=7, loc="upper left")

    fig.suptitle("Ablation effects across removal percentiles: temporal vs spatial", fontsize=13, y=1.01)
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def load_ablation_summary(path: str) -> Dict[str, object]:
    with open(path, "r") as f:
        return json.load(f)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint_paths", nargs="+", default=default_checkpoints())
    parser.add_argument("--spatial_subdir", default="spatial_path")
    parser.add_argument("--temporal_subdir", default=None)
    parser.add_argument("--gridness_threshold", type=float, default=0.2)
    parser.add_argument("--min_shift_cm", type=float, default=5.0)
    parser.add_argument("--ablation_percentile", type=float, default=100.0)
    parser.add_argument("--temporal_ablation_json", default=None)
    parser.add_argument("--spatial_ablation_json", default=None)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--max_scatter_points", type=int, default=8000)
    args = parser.parse_args()

    checkpoints = [str(p) for p in args.checkpoint_paths if str(p)]
    if not checkpoints:
        raise FileNotFoundError("No checkpoints provided.")

    model_name = model_name_from_checkpoint(Path(checkpoints[0]))
    if args.output_dir is None:
        args.output_dir = str(analysis_summary_dir(model_name, "shift_mode_comparison"))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    records: List[Dict[str, object]] = []
    temporal_best_all: List[np.ndarray] = []
    spatial_best_all: List[np.ndarray] = []
    seed_ids_all: List[np.ndarray] = []

    for ckpt_path in checkpoints:
        temporal = load_gridness_data(ckpt_path, args.temporal_subdir)
        spatial = load_gridness_data(ckpt_path, args.spatial_subdir)

        temporal_scores = np.asarray(temporal["scores_60"], dtype=float)
        spatial_scores = np.asarray(spatial["scores_60"], dtype=float)
        temporal_lag_cm = np.asarray(temporal["lag_cm"], dtype=float)
        spatial_lag_cm = np.asarray(spatial["lag_cm"], dtype=float)

        temporal_classes = classify_from_scores(temporal, args.min_shift_cm, args.gridness_threshold)
        spatial_classes = classify_from_scores(spatial, args.min_shift_cm, args.gridness_threshold)

        temporal_units = temporal_scores.shape[1]
        spatial_units = spatial_scores.shape[1]
        seed = str(seed_label_from_path(ckpt_path))

        temporal_best = best_shift_scores(temporal_scores, temporal_lag_cm, args.min_shift_cm)
        spatial_best = best_shift_scores(spatial_scores, spatial_lag_cm, args.min_shift_cm)
        max_units = min(temporal_best.shape[0], spatial_best.shape[0])
        temporal_best_all.append(np.asarray(temporal_best[:max_units], dtype=float))
        spatial_best_all.append(np.asarray(spatial_best[:max_units], dtype=float))
        seed_ids_all.append(np.full(max_units, seed, dtype=object))

        records.append(
            {
                "seed": seed,
                "checkpoint": ckpt_path,
                "temporal": {
                    "count": {cls: int(len(temporal_classes[cls])) for cls in CLASS_ORDER},
                    "fraction": {cls: float(len(temporal_classes[cls]) / max(temporal_units, 1)) for cls in CLASS_ORDER},
                },
                "spatial": {
                    "count": {cls: int(len(spatial_classes[cls])) for cls in CLASS_ORDER},
                    "fraction": {cls: float(len(spatial_classes[cls]) / max(spatial_units, 1)) for cls in CLASS_ORDER},
                },
            }
        )

    if temporal_best_all and spatial_best_all and seed_ids_all:
        plot_best_shift_scatter(
            np.concatenate(temporal_best_all),
            np.concatenate(spatial_best_all),
            np.concatenate(seed_ids_all),
            output_dir / "best_shifted_gridness_temporal_vs_spatial.png",
            max_points=max(1, int(args.max_scatter_points)),
        )

    plot_class_count_comparison(records, output_dir / "class_counts_temporal_vs_spatial.png")

    temporal_ablation_json = args.temporal_ablation_json
    spatial_ablation_json = args.spatial_ablation_json
    if temporal_ablation_json is None:
        temporal_ablation_json = str(analysis_summary_dir(model_name, "predictive_retrospective_ablation") / "results.json")
    if spatial_ablation_json is None:
        spatial_ablation_json = str(analysis_summary_dir(model_name, f"predictive_retrospective_ablation_{args.spatial_subdir}") / "results.json")

    temporal_summary = load_ablation_summary(temporal_ablation_json)
    spatial_summary = load_ablation_summary(spatial_ablation_json)
    plot_ablation_mode_comparison(
        temporal_summary,
        spatial_summary,
        args.ablation_percentile,
        output_dir / f"ablation_deltas_temporal_vs_spatial_p{int(round(args.ablation_percentile))}.png",
    )
    plot_ablation_percentile_curves(
        temporal_summary,
        spatial_summary,
        output_dir / "ablation_percentile_curves_temporal_vs_spatial.png",
    )

    summary = {
        "checkpoints": checkpoints,
        "temporal_subdir": args.temporal_subdir,
        "spatial_subdir": args.spatial_subdir,
        "gridness_threshold": float(args.gridness_threshold),
        "min_shift_cm": float(args.min_shift_cm),
        "ablation_percentile": float(args.ablation_percentile),
        "records": records,
        "temporal_ablation_json": temporal_ablation_json,
        "spatial_ablation_json": spatial_ablation_json,
    }
    with open(output_dir / "shift_mode_comparison_summary.json", "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
