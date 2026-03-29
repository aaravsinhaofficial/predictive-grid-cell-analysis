#!/usr/bin/env python3
"""Predictive/retrospective border-cell analysis across temporal lags.

For each checkpoint, this script:
1) computes border scores across lags for each hidden unit,
2) classifies predictive/retrospective/near-zero border cells,
3) compares overlap with predictive/retrospective/normal grid classes,
4) saves counts and summary plots.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch

from model import RNN
from place_cells import PlaceCells
from trajectory_generator import TrajectoryGenerator
from path_utils import analysis_dir_for_checkpoint
from scores import GridScorer, border_score
from visualize import collect_sequences
from multi_seed_predictive_analysis import build_options, infer_dims_from_state, expand_checkpoints
from predictive_retrospective_ablation import classify_from_scores
from replicate_predictive_grid_figure import cm_per_step_from_positions
from shift_utils import build_shift_values, shift_config_label, shift_values_to_cm


def extract_state(raw):
    if isinstance(raw, dict):
        if "state_dict" in raw:
            return raw["state_dict"]
        if "model_state_dict" in raw:
            return raw["model_state_dict"]
        if all(hasattr(v, "shape") for v in raw.values()):
            return raw
    raise TypeError(f"Unsupported checkpoint format: {type(raw)}")


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


def compute_border_scores_by_lag(
    xs: np.ndarray,
    ys: np.ndarray,
    activations: np.ndarray,
    lags: Sequence[int],
    res: int,
    box_width: float,
    box_height: float,
    shift_mode: str,
    periodic: bool,
    space_projection: str,
) -> np.ndarray:
    coord_range = ((-box_width / 2, box_width / 2), (-box_height / 2, box_height / 2))
    scorer = GridScorer(res, coord_range, [(0.2, 0.4)])
    _, _, Ng = activations.shape
    out = np.full((len(lags), Ng), np.nan, dtype=np.float32)
    for li, lag in enumerate(lags):
        flat_x, flat_y, flat_a = scorer.aligned_samples(
            xs,
            ys,
            activations,
            lag,
            shift_mode=shift_mode,
            periodic=periodic,
            space_projection=space_projection,
        )
        if flat_x.size == 0:
            continue
        for u in range(Ng):
            rm = scorer.calculate_ratemap(flat_x, flat_y, flat_a[:, u], statistic="mean")
            bscore, _, _ = border_score(rm, res, box_width)
            out[li, u] = bscore
    return out


def plot_counts(counts: Dict[str, int], save_path: Path):
    labels = list(counts.keys())
    vals = [counts[k] for k in labels]
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(6.8, 4.0))
    ax.bar(x, vals, color=["#1f77b4", "#d62728", "#7f7f7f", "#222222"], alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20)
    ax.set_ylabel("Units")
    ax.set_title("Predictive border-cell counts")
    for i, v in enumerate(vals):
        ax.text(i, v + 0.3, str(v), ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)


def plot_shift_hist(best_shift_cm: np.ndarray, pred_idx: np.ndarray, retro_idx: np.ndarray, min_shift_cm: float, save_path: Path):
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    vals = np.concatenate([best_shift_cm[pred_idx], best_shift_cm[retro_idx]]) if (pred_idx.size or retro_idx.size) else np.array([])
    if vals.size:
        m = max(min_shift_cm, float(np.nanmax(np.abs(vals))))
        bins = np.linspace(-m, m, 31)
        if pred_idx.size:
            ax.hist(best_shift_cm[pred_idx], bins=bins, color="#1f77b4", alpha=0.55, label="Predictive border")
        if retro_idx.size:
            ax.hist(best_shift_cm[retro_idx], bins=bins, color="#d62728", alpha=0.55, label="Retrospective border")
        ax.legend(frameon=False)
    ax.axvline(0, color="k", lw=1.0)
    ax.axvspan(-min_shift_cm, min_shift_cm, color="#cccccc", alpha=0.3)
    ax.set_xlabel("Preferred shift (cm)")
    ax.set_ylabel("Units")
    ax.set_title("Preferred shift for border cells")
    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)


def plot_zero_vs_best(zero_scores: np.ndarray, best_scores: np.ndarray, best_shift_cm: np.ndarray, threshold: float, save_path: Path):
    fig, ax = plt.subplots(figsize=(6.8, 4.6))
    c = np.nan_to_num(best_shift_cm, nan=0.0)
    sc = ax.scatter(zero_scores, best_scores, c=c, cmap="coolwarm", s=20, alpha=0.7, linewidths=0)
    ax.axhline(threshold, color="#666666", ls="--", lw=1.0)
    ax.axvline(threshold, color="#666666", ls="--", lw=1.0)
    ax.set_xlabel("Border score at lag 0")
    ax.set_ylabel("Best border score across lags")
    ax.set_title("Predictive border scatter")
    cb = fig.colorbar(sc, ax=ax)
    cb.set_label("Preferred shift (cm)")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)


def analyse_checkpoint(args, ckpt_path: str):
    raw = torch.load(ckpt_path, map_location="cpu")
    state = extract_state(raw)
    Ng, Np, velocity_dim = infer_dims_from_state(state)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    options = build_options(args, (Ng, Np, velocity_dim), device, ckpt_path)

    place_cells = PlaceCells(options)
    model = RNN(options, place_cells).to(options.device)
    model.load_state_dict(state)
    model.eval()
    traj_gen = TrajectoryGenerator(options, place_cells)

    Ng_eval = min(args.Ng_use, Ng)
    lags = build_shift_values(
        args.shift_mode,
        max_lag=args.max_lag,
        max_shift_cm=args.max_shift_cm,
        shift_step_cm=args.shift_step_cm,
    )
    xs, ys, activations = collect_sequences(
        model,
        traj_gen,
        options,
        n_batches=args.n_batches,
        Ng=Ng_eval,
        idxs=np.arange(Ng_eval, dtype=int),
    )

    border_lag = compute_border_scores_by_lag(
        xs,
        ys,
        activations,
        lags,
        args.res,
        options.box_width,
        options.box_height,
        args.shift_mode,
        bool(getattr(options, "periodic", False)),
        args.space_projection,
    )

    cm_step = cm_per_step_from_positions(xs, ys)
    lag_cm = shift_values_to_cm(lags, args.shift_mode, cm_step)
    print(f"[predictive_border_analysis] shift_mode={shift_config_label(args.shift_mode, args.space_projection)}")
    print(f"[predictive_border_analysis] cm_per_step={cm_step:.3f}")
    zero_idx = lags.index(0)
    zero_scores = border_lag[zero_idx]
    best_idx, best_scores = safe_nanargmax(border_lag)
    best_shift_cm = np.full(best_idx.shape, np.nan, dtype=float)
    valid = best_idx >= 0
    best_shift_cm[valid] = lag_cm[best_idx[valid]]

    qual = np.isfinite(zero_scores) & np.isfinite(best_scores) & (zero_scores >= args.border_threshold) & (best_scores >= args.border_threshold)
    predictive = np.where(qual & (best_shift_cm >= args.min_shift_cm))[0]
    retrospective = np.where(qual & (best_shift_cm <= -args.min_shift_cm))[0]
    normal = np.where(qual & (np.abs(best_shift_cm) < args.min_shift_cm))[0]

    # Compare with existing predictive-grid classes if present.
    out_root = analysis_dir_for_checkpoint(Path(ckpt_path))
    grid_data_path = out_root / "gridness_data.npz"
    overlap = {}
    if grid_data_path.exists():
        with np.load(grid_data_path) as data:
            grid_data = {k: data[k] for k in data.files}
        grid_classes = classify_from_scores(grid_data, args.min_shift_cm, args.gridness_threshold)
        for key in ("predictive", "retrospective", "normal"):
            grid_idx = np.asarray(grid_classes.get(key, []), dtype=int)
            grid_idx = grid_idx[grid_idx < Ng_eval]
            if key == "predictive":
                overlap[key] = int(np.intersect1d(grid_idx, predictive).size)
            elif key == "retrospective":
                overlap[key] = int(np.intersect1d(grid_idx, retrospective).size)
            else:
                overlap[key] = int(np.intersect1d(grid_idx, normal).size)

    out_dir = out_root / "predictive_border"
    os.makedirs(out_dir, exist_ok=True)

    counts = {
        "Predictive": int(predictive.size),
        "Retrospective": int(retrospective.size),
        "Near-zero": int(normal.size),
        "Any border": int(qual.sum()),
    }

    summary = {
        "checkpoint": ckpt_path,
        "num_units_scored": int(Ng_eval),
        "lags": lags,
        "shift_mode": args.shift_mode,
        "space_projection": args.space_projection,
        "shift_values": np.asarray(lags, dtype=float).tolist(),
        "lag_cm": lag_cm.tolist(),
        "cm_per_step": float(cm_step),
        "border_threshold": float(args.border_threshold),
        "min_shift_cm": float(args.min_shift_cm),
        "counts": counts,
        "overlap_with_grid_classes": overlap,
    }

    with open(out_dir / "predictive_border_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    np.savez(
        out_dir / "predictive_border_scores.npz",
        border_scores_lag=border_lag,
        zero_scores=zero_scores,
        best_scores=best_scores,
        best_shift_cm=best_shift_cm,
        shift_mode=np.array(args.shift_mode),
        space_projection=np.array(args.space_projection),
        shift_values=np.asarray(lags, dtype=float),
        lag_cm=lag_cm,
        predictive=predictive,
        retrospective=retrospective,
        normal=normal,
    )

    plot_counts(counts, out_dir / "predictive_border_counts.png")
    plot_shift_hist(best_shift_cm, predictive, retrospective, args.min_shift_cm, out_dir / "predictive_border_shift_hist.png")
    plot_zero_vs_best(zero_scores, best_scores, best_shift_cm, args.border_threshold, out_dir / "predictive_border_scatter.png")
    print(f"[predictive_border_analysis] Saved results to {out_dir}")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint_paths", nargs="+", required=True)
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
    parser.add_argument("--n_batches", default=10, type=int)
    parser.add_argument("--Ng_use", default=512, type=int)
    parser.add_argument("--shift_mode", default="time", choices=["time", "space"])
    parser.add_argument("--space_projection", default="path", choices=["path", "heading"])
    parser.add_argument("--max_lag", default=20, type=int)
    parser.add_argument("--max_shift_cm", default=20.0, type=float)
    parser.add_argument("--shift_step_cm", default=1.0, type=float)
    parser.add_argument("--min_shift_cm", default=5.0, type=float)
    parser.add_argument("--gridness_threshold", default=0.2, type=float)
    parser.add_argument("--border_threshold", default=0.5, type=float)
    parser.add_argument("--device", default=None)
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
        analyse_checkpoint(args, ckpt)


if __name__ == "__main__":
    main()
