#!/usr/bin/env python3
"""Band-cell analysis based on NeurIPS 2024 band-score definition.

This script computes band scores from rate maps, compares them with
predictive/retrospective/normal grid classes, and generates summary figures.

Outputs are written under analysis_outputs/<model>/<seed>/<run>/band_cells/ (or
analysis_outputs/<seed>/band_cells/ if checkpoints are outside Models/).
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch

from model import RNN
from place_cells import PlaceCells
from trajectory_generator import TrajectoryGenerator
from path_utils import analysis_dir_for_checkpoint
from visualize import compute_ratemaps
from scores import band_scores, border_score
from predictive_retrospective_ablation import classify_from_scores
from multi_seed_predictive_analysis import build_options, infer_dims_from_state, expand_checkpoints


COLORS = {
    "predictive": "#1f77b4",
    "retrospective": "#d62728",
    "normal": "#7f7f7f",
    "low_grid": "#bdbdbd",
    "band": "#2ca02c",
    "border": "#ff7f0e",
}


def extract_state(raw: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    if isinstance(raw, dict):
        if "state_dict" in raw:
            return raw["state_dict"]
        if "model_state_dict" in raw:
            return raw["model_state_dict"]
        if all(hasattr(v, "shape") for v in raw.values()):
            return raw
    raise TypeError(f"Unsupported checkpoint format: {type(raw)}")


def load_gridness_data(ckpt_path: str, analysis_subdir: Optional[str] = None) -> Dict[str, np.ndarray]:
    out_dir = analysis_dir_for_checkpoint(Path(ckpt_path))
    if analysis_subdir:
        out_dir = out_dir / str(analysis_subdir)
    grid_path = out_dir / "gridness_data.npz"
    if not os.path.exists(grid_path):
        raise FileNotFoundError(
            f"Missing gridness_data.npz beside checkpoint: {grid_path}. "
            "Run multi_seed_predictive_analysis.py first."
        )
    with np.load(grid_path) as data:
        return {k: data[k] for k in data.files}


def safe_nanargmax(arr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    idxs = np.full(arr.shape[1], -1, dtype=int)
    vals = np.full(arr.shape[1], np.nan, dtype=float)
    for u in range(arr.shape[1]):
        col = arr[:, u]
        if not np.isfinite(col).any():
            continue
        i = int(np.nanargmax(col))
        idxs[u] = i
        vals[u] = col[i]
    return idxs, vals


def select_band_units(scores: np.ndarray, percentile: float, threshold: Optional[float]) -> Tuple[np.ndarray, float]:
    finite = scores[np.isfinite(scores)]
    if finite.size == 0:
        return np.array([], dtype=int), float("nan")
    if threshold is not None:
        cutoff = float(threshold)
    else:
        cutoff = float(np.nanpercentile(finite, percentile))
    return np.where(scores >= cutoff)[0], cutoff


def _style_boxplot(ax, box, colors: Sequence[str]):
    for patch, color in zip(box["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.35)
    for median in box.get("medians", []):
        median.set_color("#222222")


def plot_band_hist(scores: np.ndarray, cutoff: float, save_path: str):
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    ax.hist(scores[np.isfinite(scores)], bins=30, color=COLORS["band"], alpha=0.75)
    if math.isfinite(cutoff):
        ax.axvline(cutoff, color="#222222", lw=1.5, ls="--", label=f"Cutoff {cutoff:.2f}")
        ax.legend(frameon=False)
    ax.set_title("Band score distribution")
    ax.set_xlabel("Band score")
    ax.set_ylabel("Units")
    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)


def plot_band_by_class(scores: np.ndarray, class_idxs: Dict[str, np.ndarray], save_path: str):
    labels = []
    data = []
    colors = []
    for key in ("predictive", "retrospective", "normal", "low_grid"):
        idxs = class_idxs.get(key, np.array([], dtype=int))
        if idxs.size == 0:
            continue
        labels.append(key.replace("_", " ").title())
        data.append(scores[idxs])
        colors.append(COLORS.get(key, "#aaaaaa"))
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    if data:
        box = ax.boxplot(data, patch_artist=True, showmeans=True)
        _style_boxplot(ax, box, colors)
        ax.set_xticklabels(labels, rotation=20)
    else:
        ax.text(0.5, 0.5, "No class data available", ha="center", va="center", transform=ax.transAxes)
        ax.set_xticks([])
    ax.set_ylabel("Band score")
    ax.set_title("Band scores by predictive class")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)


def plot_band_vs_gridness(best_scores: np.ndarray, band_scores_vec: np.ndarray, class_idxs: Dict[str, np.ndarray], save_path: str):
    fig, ax = plt.subplots(figsize=(6.5, 4.6))
    plotted = False
    for key in ("predictive", "retrospective", "normal", "low_grid"):
        idxs = class_idxs.get(key, np.array([], dtype=int))
        if idxs.size == 0:
            continue
        ax.scatter(best_scores[idxs], band_scores_vec[idxs], s=24, alpha=0.7, color=COLORS.get(key, "#aaaaaa"), label=key)
        plotted = True
    if not plotted:
        ax.text(0.5, 0.5, "No class data available", ha="center", va="center", transform=ax.transAxes)
    ax.set_xlabel("Best gridness (60°)")
    ax.set_ylabel("Band score")
    ax.set_title("Band vs gridness")
    if plotted:
        ax.legend(frameon=False, fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)


def plot_overlap_counts(counts: Dict[str, int], save_path: str, title: str):
    labels = list(counts.keys())
    vals = [counts[k] for k in labels]
    fig, ax = plt.subplots(figsize=(6.8, 4.0))
    x = np.arange(len(labels))
    ax.bar(x, vals, color=COLORS["band"], alpha=0.7)
    ax.set_ylabel("Units")
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20)
    for i, v in enumerate(vals):
        ax.text(i, v + 0.3, str(v), ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)


def plot_top_ratemaps(rate_maps: np.ndarray, band_scores_vec: np.ndarray, idxs: np.ndarray, save_path: str, n_show: int = 16):
    if rate_maps.size == 0:
        return
    order = np.argsort(np.nan_to_num(band_scores_vec, nan=-np.inf))[::-1]
    order = order[:min(n_show, order.size)]
    if order.size == 0:
        return
    cols = 4
    rows = int(np.ceil(order.size / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3.2 * cols, 3.0 * rows))
    axes = np.array(axes).reshape(rows, cols)
    for i in range(rows * cols):
        ax = axes.flat[i]
        if i >= order.size:
            ax.axis("off")
            continue
        u = order[i]
        ax.imshow(rate_maps[u], cmap="jet", interpolation="nearest")
        ax.set_title(f"u{int(idxs[u])} bs={band_scores_vec[u]:.2f}", fontsize=9)
        ax.axis("off")
    fig.suptitle("Top band-score rate maps")
    fig.tight_layout(rect=[0, 0.02, 1, 0.96])
    fig.savefig(save_path, dpi=200)
    plt.close(fig)


def analyse_checkpoint(ckpt_path: str, args) -> Dict[str, float]:
    raw = torch.load(ckpt_path, map_location="cpu")
    state = extract_state(raw)
    Ng, Np, velocity_dim = infer_dims_from_state(state)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    options = build_options(args, (Ng, Np, velocity_dim), device, ckpt_path)

    base_out_dir = analysis_dir_for_checkpoint(Path(ckpt_path))
    if getattr(args, "analysis_subdir", None):
        base_out_dir = base_out_dir / str(args.analysis_subdir)
    out_dir = base_out_dir / "band_cells"
    os.makedirs(out_dir, exist_ok=True)

    place_cells = PlaceCells(options)
    model = RNN(options, place_cells).to(options.device)
    model.load_state_dict(state)
    model.eval()
    traj_gen = TrajectoryGenerator(options, place_cells)

    Ng_eval = min(args.Ng_use, Ng)
    idxs = np.arange(Ng_eval, dtype=int)
    activations, _, _, _ = compute_ratemaps(
        model,
        traj_gen,
        options,
        res=args.res,
        n_avg=args.n_avg,
        Ng=Ng_eval,
        idxs=idxs,
    )
    band_vals, band_kx, band_ky = band_scores(activations, args.res, options.box_width,
                                              k_values=np.arange(0.0, args.band_k_max + 1e-6, args.band_k_step))

    border_vals = np.zeros((Ng_eval,), dtype=np.float32)
    for i in range(Ng_eval):
        bscore, _, _ = border_score(activations[i], args.res, options.box_width)
        border_vals[i] = bscore

    # Load predictive classes
    grid_data = load_gridness_data(ckpt_path, getattr(args, "analysis_subdir", None))
    scores_60 = np.asarray(grid_data["scores_60"], dtype=float)
    if scores_60.shape[1] < Ng_eval:
        Ng_eval = scores_60.shape[1]
        band_vals = band_vals[:Ng_eval]
        band_kx = band_kx[:Ng_eval]
        band_ky = band_ky[:Ng_eval]
        border_vals = border_vals[:Ng_eval]
        activations = activations[:Ng_eval]
        idxs = idxs[:Ng_eval]

    class_idxs = classify_from_scores(grid_data, args.min_shift_cm, args.gridness_threshold)
    class_idxs = {k: np.asarray(v, dtype=int) for k, v in class_idxs.items()}
    class_idxs = {k: v[v < Ng_eval] for k, v in class_idxs.items()}
    assigned = np.zeros((Ng_eval,), dtype=bool)
    for v in class_idxs.values():
        assigned[v] = True
    class_idxs["low_grid"] = np.where(~assigned)[0]

    best_idx, best_scores = safe_nanargmax(scores_60[:, :Ng_eval])

    band_units, band_cutoff = select_band_units(band_vals, args.band_percentile, args.band_threshold)
    band_mask = np.zeros((Ng_eval,), dtype=bool)
    band_mask[band_units] = True

    border_mask = border_vals >= args.border_threshold

    overlap_counts = {
        "predictive": int(np.intersect1d(band_units, class_idxs["predictive"]).size),
        "retrospective": int(np.intersect1d(band_units, class_idxs["retrospective"]).size),
        "normal": int(np.intersect1d(band_units, class_idxs["normal"]).size),
        "low_grid": int(np.intersect1d(band_units, class_idxs["low_grid"]).size),
    }

    border_overlap = {
        "border": int(np.sum(border_mask)),
        "band": int(band_units.size),
        "band_and_border": int(np.sum(band_mask & border_mask)),
    }

    predictive_border = {
        "predictive": int(np.sum(border_mask & np.isin(np.arange(Ng_eval), class_idxs["predictive"]))),
        "retrospective": int(np.sum(border_mask & np.isin(np.arange(Ng_eval), class_idxs["retrospective"]))),
        "normal": int(np.sum(border_mask & np.isin(np.arange(Ng_eval), class_idxs["normal"]))),
    }

    summary = {
        "checkpoint": ckpt_path,
        "analysis_subdir": None if getattr(args, "analysis_subdir", None) is None else str(args.analysis_subdir),
        "gridness_shift_mode": str(np.asarray(grid_data.get("shift_mode", "unknown")).reshape(())),
        "gridness_space_projection": str(np.asarray(grid_data.get("space_projection", "")).reshape(())) if "space_projection" in grid_data else None,
        "num_units_scored": int(Ng_eval),
        "band": {
            "percentile": float(args.band_percentile),
            "threshold": None if args.band_threshold is None else float(args.band_threshold),
            "cutoff": float(band_cutoff),
            "count": int(band_units.size),
            "fraction": float(band_units.size / max(Ng_eval, 1)),
        },
        "border": {
            "threshold": float(args.border_threshold),
            "count": int(np.sum(border_mask)),
        },
        "classes": {
            "predictive": int(class_idxs["predictive"].size),
            "retrospective": int(class_idxs["retrospective"].size),
            "normal": int(class_idxs["normal"].size),
            "low_grid": int(class_idxs["low_grid"].size),
        },
        "band_overlap": overlap_counts,
        "border_overlap": border_overlap,
        "predictive_border_overlap": predictive_border,
    }

    with open(out_dir / "band_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    np.savez(
        out_dir / "band_scores.npz",
        band_scores=band_vals,
        band_kx=band_kx,
        band_ky=band_ky,
        border_scores=border_vals,
        best_grid_scores=best_scores[:Ng_eval],
        band_units=band_units,
    )

    plot_band_hist(band_vals, band_cutoff, str(out_dir / "band_score_hist.png"))
    plot_band_by_class(band_vals, class_idxs, str(out_dir / "band_score_by_class.png"))
    plot_band_vs_gridness(best_scores[:Ng_eval], band_vals, class_idxs, str(out_dir / "band_vs_gridness_scatter.png"))
    plot_overlap_counts(overlap_counts, str(out_dir / "band_overlap_counts.png"), "Band overlap with predictive classes")
    plot_overlap_counts(border_overlap, str(out_dir / "band_border_overlap.png"), "Band/border overlap")
    plot_top_ratemaps(activations, band_vals, idxs, str(out_dir / "band_top_ratemaps.png"), n_show=args.top_ratemaps)

    return {
        "band_count": int(band_units.size),
        "border_count": int(np.sum(border_mask)),
        "predictive_count": int(class_idxs["predictive"].size),
        "retrospective_count": int(class_idxs["retrospective"].size),
        "normal_count": int(class_idxs["normal"].size),
    }


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint_paths", nargs="+", required=True)
    parser.add_argument("--batch_size", default=100, type=int)
    parser.add_argument("--sequence_length", default=20, type=int)
    parser.add_argument("--place_cell_rf", type=float, default=0.12)
    parser.add_argument("--surround_scale", type=float, default=2.0)
    parser.add_argument("--activation", default="relu")
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--box_width", type=float, default=2.2)
    parser.add_argument("--box_height", type=float, default=2.2)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--res", type=int, default=40)
    parser.add_argument("--n_avg", type=int, default=None)
    parser.add_argument("--Ng_use", type=int, default=512)
    parser.add_argument("--gridness_threshold", type=float, default=0.2)
    parser.add_argument("--min_shift_cm", type=float, default=5.0)
    parser.add_argument("--band_percentile", type=float, default=90.0)
    parser.add_argument("--band_threshold", type=float, default=None)
    parser.add_argument("--band_k_max", type=float, default=2.0)
    parser.add_argument("--band_k_step", type=float, default=0.1)
    parser.add_argument("--border_threshold", type=float, default=0.5)
    parser.add_argument("--top_ratemaps", type=int, default=16)
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--analysis_subdir",
        default=None,
        help="Optional subdirectory under each checkpoint analysis folder from which to load gridness_data.npz and write band outputs.",
    )
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
    checkpoints = expand_checkpoints(args.checkpoint_paths)
    if not checkpoints:
        raise FileNotFoundError("No checkpoints matched the provided paths.")
    for ckpt in checkpoints:
        print(f"[band_cell_analysis] Processing {ckpt}")
        analyse_checkpoint(ckpt, args)


if __name__ == "__main__":
    main()
