#!/usr/bin/env python3
"""Ablate saved predictive/retrospective/grid classes from an analysis-output folder.

This script is intended for checkpoints that already have class memberships saved as
`predictive_ids_<suffix>.npy`, `retrospective_ids_<suffix>.npy`, and `grid_ids_<suffix>.npy`
under an analysis-output directory. It ranks units within each class using freshly
computed heading- or path-based spatial shift scores, then evaluates path-integration
performance after ablating 0-100% of each class.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import matplotlib.pyplot as plt
import numpy as np
import torch

from model import RNN
from place_cells import PlaceCells
from trajectory_generator import TrajectoryGenerator
from multi_seed_predictive_analysis import (
    build_options,
    collect_eval_batches,
    infer_dims_from_state,
    mean_decoding_error_cm,
    zero_unit_weights_in_place,
)
from shift_utils import build_shift_values
from visualize import predictive_gridness_analysis


def extract_state(raw):
    if isinstance(raw, dict):
        if "state_dict" in raw:
            return raw["state_dict"]
        if "model_state_dict" in raw:
            return raw["model_state_dict"]
        if all(hasattr(v, "shape") for v in raw.values()):
            return raw
    raise TypeError(f"Unsupported checkpoint format: {type(raw)}")


def load_class_ids(class_dir: Path, suffix: str) -> Dict[str, np.ndarray]:
    """Load saved predictive/retrospective/grid IDs and derive normal grid IDs."""
    def _load(name: str) -> np.ndarray:
        arr = np.load(class_dir / f"{name}_{suffix}.npy")
        return np.asarray(arr, dtype=int).reshape(-1)

    predictive = _load("predictive_ids")
    retrospective = _load("retrospective_ids")
    grid = _load("grid_ids")
    dead_path = class_dir / f"dead_unit_ids_{suffix}.npy"
    dead_mask = np.load(dead_path).astype(bool).reshape(-1) if dead_path.exists() else None

    normal = np.setdiff1d(grid, np.union1d(predictive, retrospective), assume_unique=False)
    if dead_mask is not None and dead_mask.size:
        alive = np.where(~dead_mask)[0]
        predictive = np.intersect1d(predictive, alive, assume_unique=False)
        retrospective = np.intersect1d(retrospective, alive, assume_unique=False)
        grid = np.intersect1d(grid, alive, assume_unique=False)
        normal = np.intersect1d(normal, alive, assume_unique=False)

    return {
        "predictive": predictive,
        "retrospective": retrospective,
        "grid": grid,
        "normal": normal,
    }


def select_top_percent(indices: Iterable[int], score_vec: np.ndarray, percent: float) -> np.ndarray:
    idxs = np.asarray(list(indices), dtype=int).reshape(-1)
    if percent <= 0 or idxs.size == 0:
        return np.array([], dtype=int)

    valid = (idxs >= 0) & (idxs < score_vec.shape[0])
    idxs = idxs[valid]
    if idxs.size == 0:
        return np.array([], dtype=int)

    scores = score_vec[idxs]
    finite = np.isfinite(scores)
    idxs = idxs[finite]
    scores = scores[finite]
    if idxs.size == 0:
        return np.array([], dtype=int)

    k = max(1, int(math.ceil(idxs.size * (percent / 100.0))))
    order = np.argsort(scores)[::-1]
    return idxs[order][:k]


def class_score_vectors(scores_60: np.ndarray, lag_cm: np.ndarray, min_shift_cm: float) -> Dict[str, np.ndarray]:
    zero_idx = int(np.nanargmin(np.abs(lag_cm)))
    zero_scores = scores_60[zero_idx]

    pos_mask = lag_cm >= min_shift_cm
    neg_mask = lag_cm <= -min_shift_cm
    pos_scores = np.nanmax(scores_60[pos_mask], axis=0) if np.any(pos_mask) else np.full(scores_60.shape[1], np.nan)
    neg_scores = np.nanmax(scores_60[neg_mask], axis=0) if np.any(neg_mask) else np.full(scores_60.shape[1], np.nan)

    return {
        "predictive": pos_scores,
        "retrospective": neg_scores,
        "normal": zero_scores,
    }


def plot_ablation_lines(percentiles: Sequence[float], baseline: float, errors: Dict[str, List[float]], save_path: Path, title: str) -> None:
    colors = {
        "predictive": "#2086cf",
        "retrospective": "#d62728",
        "normal": "#111111",
    }
    fig, ax = plt.subplots(figsize=(7.0, 4.6))
    if baseline is not None and math.isfinite(baseline):
        ax.axhline(float(baseline), color="#999999", lw=1.2, ls="--", label="Baseline")

    for label in ("predictive", "retrospective", "normal"):
        vals = np.asarray(errors.get(label, []), dtype=float)
        if vals.size != len(percentiles):
            continue
        ax.plot(percentiles, vals, marker="o", lw=2.0, color=colors[label], label=label.title())

    ax.set_xlabel("Percent of class ablated")
    ax.set_ylabel("Mean decoding error (cm)")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend(frameon=False)
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=200)
    plt.close(fig)


def evaluate_for_suffix(args, model_state, ckpt_path: Path, suffix: str, output_dir: Path) -> Dict[str, object]:
    class_dir = Path(args.class_dir)
    classes = load_class_ids(class_dir, suffix)
    print(f"[analysis_output_class_ablation] {suffix}: class counts "
          f"pred={classes['predictive'].size}, retro={classes['retrospective'].size}, "
          f"normal={classes['normal'].size}, grid={classes['grid'].size}", flush=True)

    Ng, Np, velocity_dim = infer_dims_from_state(model_state)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    options = build_options(args, (Ng, Np, velocity_dim), device, str(ckpt_path))
    options.trajectory_style = suffix
    options.trajectory_fixed_speed = args.trajectory_fixed_speed if suffix == "straight" else None

    place_cells = PlaceCells(options)
    model = RNN(options, place_cells).to(options.device)
    model.load_state_dict(model_state)
    model.eval()
    traj_gen = TrajectoryGenerator(options, place_cells)

    ranking_units = np.unique(np.concatenate([classes["grid"], classes["predictive"], classes["retrospective"]]))
    ranking_units = ranking_units[(ranking_units >= 0) & (ranking_units < Ng)]
    shift_values = build_shift_values(
        "space",
        max_shift_cm=args.max_shift_cm,
        shift_step_cm=args.shift_step_cm,
    )
    print(f"[analysis_output_class_ablation] {suffix}: ranking with {len(shift_values)} heading-space shifts "
          f"over {ranking_units.size} saved class units", flush=True)
    scores_60, _, xs, ys, _, _ = predictive_gridness_analysis(
        model,
        traj_gen,
        options,
        lags=shift_values,
        res=args.res,
        n_batches=args.score_batches,
        Ng=len(ranking_units),
        idxs=ranking_units,
        shift_mode="space",
        space_projection="heading",
    )
    lag_cm = np.asarray(shift_values, dtype=float)
    score_subset = class_score_vectors(scores_60, lag_cm, args.min_shift_cm)
    score_vectors = {
        key: np.full(Ng, np.nan, dtype=float)
        for key in ("predictive", "retrospective", "normal")
    }
    for key, vals in score_subset.items():
        score_vectors[key][ranking_units] = vals

    eval_batches = collect_eval_batches(traj_gen, args.ablation_batches)
    baseline = mean_decoding_error_cm(model, eval_batches)
    baseline = float(baseline) if math.isfinite(baseline) else None
    if baseline is None:
        print(f"[analysis_output_class_ablation] {suffix}: baseline error is NaN", flush=True)
    else:
        print(f"[analysis_output_class_ablation] {suffix}: baseline error = {baseline:.3f} cm", flush=True)

    errors: Dict[str, List[float]] = {k: [] for k in ("predictive", "retrospective", "normal")}
    removed_counts: Dict[str, List[int]] = {k: [] for k in ("predictive", "retrospective", "normal")}
    for pct in args.ablate_percentiles:
        print(f"[analysis_output_class_ablation] {suffix}: evaluating {pct:.0f}% ablations", flush=True)
        for label in ("predictive", "retrospective", "normal"):
            selected = select_top_percent(classes[label], score_vectors[label], pct)
            removed_counts[label].append(int(selected.size))
            if selected.size == 0:
                errors[label].append(baseline)
                continue
            ablated_model = copy.deepcopy(model)
            zero_unit_weights_in_place(ablated_model, selected.tolist())
            err_val = mean_decoding_error_cm(ablated_model, eval_batches)
            errors[label].append(float(err_val) if math.isfinite(err_val) else math.nan)
            print(f"[analysis_output_class_ablation] {suffix}: {label} {pct:.0f}% -> "
                  f"{selected.size} units, error {errors[label][-1]:.3f} cm", flush=True)

    title = f"Seed 0 saved {suffix} classes | heading-space ranking"
    plot_path = output_dir / f"{suffix}_heading_space_ablation.png"
    plot_ablation_lines(args.ablate_percentiles, baseline, errors, plot_path, title)

    result = {
        "suffix": suffix,
        "checkpoint": str(ckpt_path),
        "class_dir": str(class_dir),
        "shift_mode": "space",
        "space_projection": "heading",
        "max_shift_cm": float(args.max_shift_cm),
        "shift_step_cm": float(args.shift_step_cm),
        "min_shift_cm": float(args.min_shift_cm),
        "score_batches": int(args.score_batches),
        "ablation_batches": int(args.ablation_batches),
        "percentiles": [float(x) for x in args.ablate_percentiles],
        "baseline_error_cm": baseline,
        "class_counts": {k: int(classes[k].size) for k in ("predictive", "retrospective", "normal", "grid")},
        "removed_counts": removed_counts,
        "errors_cm": errors,
        "plot_path": str(plot_path),
    }
    with open(output_dir / f"{suffix}_heading_space_ablation.json", "w") as f:
        json.dump(result, f, indent=2)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint_path", required=True)
    parser.add_argument("--class_dir", required=True)
    parser.add_argument("--suffixes", nargs="+", default=["random_walk", "straight"])
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--ablate_percentiles", nargs="+", type=float, default=[0, 25, 50, 75, 100])
    parser.add_argument("--ablation_batches", type=int, default=8)
    parser.add_argument("--score_batches", type=int, default=10)
    parser.add_argument("--res", type=int, default=20)
    parser.add_argument("--max_shift_cm", type=float, default=20.0)
    parser.add_argument("--shift_step_cm", type=float, default=1.0)
    parser.add_argument("--min_shift_cm", type=float, default=5.0)
    parser.add_argument("--batch_size", type=int, default=200)
    parser.add_argument("--sequence_length", type=int, default=40)
    parser.add_argument("--place_cell_rf", type=float, default=0.12)
    parser.add_argument("--surround_scale", type=float, default=2.0)
    parser.add_argument("--activation", default="relu")
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--box_width", type=float, default=2.2)
    parser.add_argument("--box_height", type=float, default=2.2)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--trajectory_fixed_speed", type=float, default=None)
    parser.add_argument("--traj_speed_scale", type=float, default=1.0)
    parser.add_argument("--traj_speed_max", type=float, default=None)
    parser.add_argument("--traj_velocity_smoothing", type=float, default=0.0)
    parser.add_argument("--traj_turn_sigma_scale", type=float, default=1.0)
    parser.add_argument("--traj_border_region", type=float, default=0.03)
    parser.add_argument("--traj_wall_slowdown", type=float, default=0.25)
    parser.add_argument("--traj_wall_turn_scale", type=float, default=1.0)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    ckpt_path = Path(args.checkpoint_path)
    raw = torch.load(ckpt_path, map_location="cpu")
    state = extract_state(raw)
    out_dir = Path(args.output_dir) if args.output_dir else Path(args.class_dir) / "heading_space_ablation"
    out_dir.mkdir(parents=True, exist_ok=True)

    all_results: List[Dict[str, object]] = []
    for suffix in args.suffixes:
        print(f"[analysis_output_class_ablation] Running {suffix} classes")
        all_results.append(evaluate_for_suffix(args, state, ckpt_path, suffix, out_dir))

    with open(out_dir / "results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"[analysis_output_class_ablation] Wrote outputs to {out_dir}")


if __name__ == "__main__":
    main()
