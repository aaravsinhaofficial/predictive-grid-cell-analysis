#!/usr/bin/env python3
"""Summarize predictive vs retrospective grid coding with consolidated figures."""

import argparse
import json
import os
from typing import Dict, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import torch

from model import RNN
from place_cells import PlaceCells
from trajectory_generator import TrajectoryGenerator
from visualize import predictive_gridness_analysis
from multi_seed_predictive_analysis import build_options, infer_dims_from_state
from replicate_predictive_grid_figure import cm_per_step_from_positions, safe_nanargmax


def _extract_state(raw: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Normalize checkpoint formats used across scripts."""
    if isinstance(raw, dict):
        if "state_dict" in raw:
            return raw["state_dict"]
        if "model_state_dict" in raw:
            return raw["model_state_dict"]
        if all(hasattr(v, "shape") for v in raw.values()):
            return raw
    raise TypeError(f"Unsupported checkpoint format: {type(raw)}")


def compute_shuffle_thresholds(xs: np.ndarray,
                               ys: np.ndarray,
                               activations: np.ndarray,
                               lags: Sequence[int],
                               scorer,
                               trials: int,
                               alpha: float,
                               seed: int):
    """Estimate chance-level gridness per (lag, unit) via shuffled activations."""
    rng = np.random.default_rng(seed)
    flat = activations.reshape(-1, activations.shape[-1])
    shuffle_scores = np.empty((trials, len(lags), activations.shape[-1]), dtype=np.float32)
    for i in range(trials):
        perm = rng.permutation(flat.shape[0])
        shuffled = flat[perm].reshape(activations.shape)
        s60, _ = scorer.predictive_grid_scores(xs, ys, shuffled, lags)
        shuffle_scores[i] = s60.astype(np.float32)
    percentile = 100.0 * (1.0 - alpha)
    thresholds = np.percentile(shuffle_scores, percentile, axis=0)
    mean = np.mean(shuffle_scores, axis=0)
    std = np.std(shuffle_scores, axis=0)
    return thresholds, mean, std


def classify_units(zero_scores: np.ndarray,
                   best_scores: np.ndarray,
                   best_shift_cm: np.ndarray,
                   min_shift_cm: float,
                   zero_threshold: float,
                   shift_threshold: float,
                   zero_significant: Optional[np.ndarray],
                   best_significant: Optional[np.ndarray]) -> Dict[str, np.ndarray]:
    """Assign units to predictive, retrospective, zero-lag, or low-grid categories."""
    zero_mask = np.isfinite(zero_scores) & (zero_scores >= zero_threshold)
    peak_mask = np.isfinite(best_scores) & (best_scores >= shift_threshold)
    shift_mask = np.isfinite(best_shift_cm)
    if zero_significant is not None:
        zero_mask &= zero_significant
    if best_significant is not None:
        peak_mask &= best_significant

    predictive = zero_mask & peak_mask & shift_mask & (best_shift_cm >= min_shift_cm)
    retrospective = zero_mask & peak_mask & shift_mask & (best_shift_cm <= -min_shift_cm)
    zero_lag = zero_mask & peak_mask & shift_mask & (np.abs(best_shift_cm) < min_shift_cm)
    low_grid = ~(zero_mask & peak_mask)
    other = ~(predictive | retrospective | zero_lag | low_grid)

    return {
        "predictive": np.where(predictive)[0],
        "retrospective": np.where(retrospective)[0],
        "zero_lag": np.where(zero_lag)[0],
        "low_grid": np.where(low_grid)[0],
        "other": np.where(other)[0],
    }


def _nanmean(x: np.ndarray) -> float:
    return float(np.nanmean(x)) if x.size else float("nan")


def plot_summary(fig_path: str,
                 ckpt_label: str,
                 args,
                 class_idxs: Dict[str, np.ndarray],
                 best_shift_cm: np.ndarray,
                 best_scores: np.ndarray,
                 zero_scores: np.ndarray) -> None:
    """Create the 3-panel summary figure."""
    colors = {
        "predictive": "#1f77b4",
        "retrospective": "#d62728",
        "zero": "#7f7f7f",
    }
    fig, axes = plt.subplots(1, 3, figsize=(14.0, 4.8))

    # Panel A: counts
    categories = [
        ("Predictive (+)", class_idxs["predictive"], colors["predictive"]),
        ("Retrospective (-)", class_idxs["retrospective"], colors["retrospective"]),
        ("Zero-lag", class_idxs["zero_lag"], colors["zero"]),
    ]
    counts = [len(idx) for _, idx, _ in categories]
    cat_colors = [color for _, _, color in categories]
    labels = [label for label, _, _ in categories]
    y = np.arange(len(categories))
    axes[0].barh(y, counts, color=cat_colors)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(labels)
    axes[0].invert_yaxis()
    axes[0].set_xlabel("Units")
    axes[0].set_title("Class counts (gridness ≥ {:.2f})".format(args.gridness_threshold))
    for val, yy in zip(counts, y):
        axes[0].text(val + 0.2, yy, f"{val}", va="center", fontsize=10)

    # Panel B: preferred shift distribution
    ax = axes[1]
    predictive_shifts = best_shift_cm[class_idxs["predictive"]]
    retrospective_shifts = best_shift_cm[class_idxs["retrospective"]]
    combined = np.concatenate([predictive_shifts, retrospective_shifts]) if (predictive_shifts.size or retrospective_shifts.size) else np.array([])
    if combined.size:
        max_abs = max(args.min_shift_cm, float(np.nanmax(np.abs(combined))))
        bins = np.linspace(-max_abs, max_abs, 31)
        if predictive_shifts.size:
            ax.hist(predictive_shifts, bins=bins, color=colors["predictive"], alpha=0.55, label="Predictive")
        if retrospective_shifts.size:
            ax.hist(retrospective_shifts, bins=bins, color=colors["retrospective"], alpha=0.55, label="Retrospective")
    else:
        bins = None
    ax.axvline(0, color="k", lw=1.0)
    ax.axvspan(-args.min_shift_cm, args.min_shift_cm, color="#cccccc", alpha=0.3, label="±min shift")
    ax.set_xlabel("Preferred shift (cm)")
    ax.set_ylabel("Units")
    ax.set_title("Preferred shift distribution")
    if predictive_shifts.size or retrospective_shifts.size:
        ax.legend(frameon=False)

    # Panel C: zero vs shifted gridness
    axc = axes[2]
    data = []
    labels = []
    colors_box = []
    if predictive_shifts.size:
        data.extend([zero_scores[class_idxs["predictive"]], best_scores[class_idxs["predictive"]]])
        labels.extend(["PGC zero", "PGC shift"])
        colors_box.extend([colors["predictive"], colors["predictive"]])
    if retrospective_shifts.size:
        data.extend([zero_scores[class_idxs["retrospective"]], best_scores[class_idxs["retrospective"]]])
        labels.extend(["RGC zero", "RGC shift"])
        colors_box.extend([colors["retrospective"], colors["retrospective"]])
    if data:
        positions = np.arange(len(data))
        box = axc.boxplot(data, positions=positions, widths=0.6, patch_artist=True, showmeans=True)
        for patch, color in zip(box["boxes"], colors_box):
            patch.set_facecolor(color)
            patch.set_alpha(0.35)
        axc.set_xticks(positions)
        axc.set_xticklabels(labels, rotation=20)
        axc.axhline(args.gridness_threshold, color="grey", lw=1.0, ls="--", label="Shift threshold")
    else:
        axc.text(0.5, 0.5, "No predictive/retrospective units", ha="center", va="center", transform=axc.transAxes)
        axc.set_xticks([])
    axc.set_ylabel("Gridness (60°)")
    axc.set_title("Gridness comparison")
    axc.grid(alpha=0.3)

    fig.suptitle(f"Predictive vs retrospective summary\n{ckpt_label}")
    fig.tight_layout(rect=[0, 0.02, 1, 0.95])
    fig.savefig(fig_path, dpi=200)
    plt.close(fig)


def summarize_checkpoint(args) -> Dict[str, float]:
    raw = torch.load(args.checkpoint_path, map_location="cpu")
    state = _extract_state(raw)
    Ng, Np, velocity_dim = infer_dims_from_state(state)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    options = build_options(args, (Ng, Np, velocity_dim), device, args.checkpoint_path)

    place_cells = PlaceCells(options)
    model = RNN(options, place_cells).to(options.device)
    model.load_state_dict(state)
    model.eval()
    traj_gen = TrajectoryGenerator(options, place_cells)

    lags = list(range(-args.max_lag, args.max_lag + 1))
    Ng_eval = min(args.Ng_use, Ng)
    scores_60, scores_90, xs, ys, activations, scorer = predictive_gridness_analysis(
        model,
        traj_gen,
        options,
        lags=lags,
        res=args.res,
        n_batches=args.n_batches,
        Ng=Ng_eval,
    )

    cm_step = cm_per_step_from_positions(xs, ys)
    lag_cm = np.array(lags, dtype=float) * cm_step
    zero_idx = lags.index(0)
    zero_scores = scores_60[zero_idx]
    best_idx, best_scores = safe_nanargmax(scores_60)
    best_shift_cm = np.full(best_idx.shape, np.nan)
    valid = best_idx >= 0
    best_shift_cm[valid] = lag_cm[best_idx[valid]]

    shuffle_thresholds = None
    shuffle_mean = None
    shuffle_std = None
    shuffle_significance = None
    if args.shuffle_trials > 0:
        shuffle_thresholds, shuffle_mean, shuffle_std = compute_shuffle_thresholds(
            xs,
            ys,
            activations,
            lags,
            scorer,
            args.shuffle_trials,
            args.shuffle_alpha,
            args.shuffle_seed,
        )
        shuffle_significance = scores_60 >= shuffle_thresholds

    zero_significant = shuffle_significance[zero_idx] if shuffle_significance is not None else None
    best_significant = None
    if shuffle_significance is not None:
        best_significant = np.zeros(best_idx.shape, dtype=bool)
        valid = best_idx >= 0
        best_significant[valid] = shuffle_significance[best_idx[valid], np.where(valid)[0]]

    class_idxs = classify_units(
        zero_scores,
        best_scores,
        best_shift_cm,
        min_shift_cm=args.min_shift_cm,
        zero_threshold=args.zero_shift_threshold,
        shift_threshold=args.gridness_threshold,
        zero_significant=zero_significant,
        best_significant=best_significant,
    )

    ckpt_dir = os.path.dirname(args.checkpoint_path) or "."
    out_dir = args.out_dir or os.path.join(ckpt_dir, "analysis_outputs", "predictive_retrospective")
    os.makedirs(out_dir, exist_ok=True)
    ckpt_label = os.path.basename(args.checkpoint_path)
    fig_path = os.path.join(out_dir, f"{ckpt_label}_summary.png")
    plot_summary(fig_path, ckpt_label, args, class_idxs, best_shift_cm, best_scores, zero_scores)

    summary = {
        "checkpoint": args.checkpoint_path,
        "cm_per_step": float(cm_step),
        "gridness_threshold": args.gridness_threshold,
        "zero_shift_threshold": args.zero_shift_threshold,
        "min_shift_cm": args.min_shift_cm,
        "shuffle_trials": args.shuffle_trials,
        "shuffle_alpha": args.shuffle_alpha if args.shuffle_trials > 0 else None,
        "counts": {k: int(len(v)) for k, v in class_idxs.items()},
        "preferred_shift_mean_cm": {
            "predictive": _nanmean(best_shift_cm[class_idxs["predictive"]]),
            "retrospective": _nanmean(best_shift_cm[class_idxs["retrospective"]]),
        },
        "preferred_shift_median_cm": {
            "predictive": float(np.nanmedian(best_shift_cm[class_idxs["predictive"]])) if class_idxs["predictive"].size else float("nan"),
            "retrospective": float(np.nanmedian(best_shift_cm[class_idxs["retrospective"]])) if class_idxs["retrospective"].size else float("nan"),
        },
        "gridness_means": {
            "predictive_zero": _nanmean(zero_scores[class_idxs["predictive"]]),
            "predictive_shift": _nanmean(best_scores[class_idxs["predictive"]]),
            "retrospective_zero": _nanmean(zero_scores[class_idxs["retrospective"]]),
            "retrospective_shift": _nanmean(best_scores[class_idxs["retrospective"]]),
        },
        "figure_path": fig_path,
    }

    summary_path = os.path.join(out_dir, f"{ckpt_label}_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    npz_path = os.path.join(out_dir, f"{ckpt_label}_summary_data.npz")
    np.savez(
        npz_path,
        lag_cm=lag_cm,
        scores_60=scores_60,
        scores_90=scores_90,
        zero_scores=zero_scores,
        best_idx=best_idx,
        best_scores=best_scores,
        best_shift_cm=best_shift_cm,
        classes_predictive=class_idxs["predictive"],
        classes_retrospective=class_idxs["retrospective"],
        classes_zero=class_idxs["zero_lag"],
        classes_low_grid=class_idxs["low_grid"],
        classes_other=class_idxs["other"],
        shuffle_thresholds=shuffle_thresholds if shuffle_thresholds is not None else np.array([]),
        shuffle_mean=shuffle_mean if shuffle_mean is not None else np.array([]),
        shuffle_std=shuffle_std if shuffle_std is not None else np.array([]),
    )

    print(json.dumps(summary, indent=2))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint_path", required=True, help="Path to a trained model (.pth).")
    parser.add_argument("--out_dir", default=None, help="Optional directory override for outputs.")
    parser.add_argument("--batch_size", default=100, type=int)
    parser.add_argument("--sequence_length", default=20, type=int)
    parser.add_argument("--place_cell_rf", default=0.12, type=float)
    parser.add_argument("--surround_scale", default=2.0, type=float)
    parser.add_argument("--activation", default="relu")
    parser.add_argument("--weight_decay", default=1e-4, type=float)
    parser.add_argument("--box_width", default=2.2, type=float)
    parser.add_argument("--box_height", default=2.2, type=float)
    parser.add_argument("--learning_rate", default=1e-4, type=float)
    parser.add_argument("--res", default=20, type=int)
    parser.add_argument("--n_batches", default=25, type=int, help="How many trajectory batches to aggregate.")
    parser.add_argument("--Ng_use", default=512, type=int, help="How many units to score.")
    parser.add_argument("--traj_speed_scale", default=1.0, type=float)
    parser.add_argument("--traj_speed_max", default=None, type=float)
    parser.add_argument("--traj_velocity_smoothing", default=0.0, type=float)
    parser.add_argument("--traj_turn_sigma_scale", default=1.0, type=float)
    parser.add_argument("--traj_border_region", default=0.03, type=float)
    parser.add_argument("--traj_wall_slowdown", default=0.25, type=float)
    parser.add_argument("--traj_wall_turn_scale", default=1.0, type=float)
    parser.add_argument("--gridness_threshold", default=0.5, type=float, help="Minimum gridness at the preferred shift.")
    parser.add_argument("--zero_shift_threshold", default=0.5, type=float, help="Minimum gridness at zero shift.")
    parser.add_argument("--min_shift_cm", default=5.0, type=float, help="Minimum |shift| (cm) to call a unit predictive/retrospective.")
    parser.add_argument("--max_lag", default=20, type=int, help="Evaluate lags from -max_lag to +max_lag.")
    parser.add_argument("--shuffle_trials", default=0, type=int, help="Number of shuffle permutations (0 disables significance testing).")
    parser.add_argument("--shuffle_alpha", default=0.05, type=float, help="Tail probability for shuffle thresholds.")
    parser.add_argument("--shuffle_seed", default=0, type=int)
    parser.add_argument("--device", default=None, help="Set to 'cpu' or 'cuda'. Defaults to best available.")
    args = parser.parse_args()

    summarize_checkpoint(args)


if __name__ == "__main__":
    main()
