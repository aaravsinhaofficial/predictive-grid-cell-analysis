#!/usr/bin/env python3
"""Run predictive/retrospective/zero-lag grid-cell ablations across checkpoints.

This script sweeps ablation percentiles for three classes (predictive, retrospective,
and near-zero/"normal" grid cells) and evaluates every combination of class ablations
per checkpoint. Results are averaged across seeds and plotted as summary figures.
"""

import argparse
import copy
import glob
import json
import math
import os
import re
from typing import Dict, Iterable, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch

from model import RNN
from place_cells import PlaceCells
from trajectory_generator import TrajectoryGenerator
from multi_seed_predictive_analysis import (
    build_options,
    infer_dims_from_state,
    zero_unit_weights_in_place,
    collect_eval_batches,
    mean_decoding_error_cm,
)


COMBOS: Dict[str, Sequence[str]] = {
    "baseline": [],
    "predictive": ["predictive"],
    "retrospective": ["retrospective"],
    "normal": ["normal"],
    "predictive+retrospective": ["predictive", "retrospective"],
    "predictive+normal": ["predictive", "normal"],
    "retrospective+normal": ["retrospective", "normal"],
    "all": ["predictive", "retrospective", "normal"],
}


def default_checkpoints() -> List[str]:
    """Return the five default single-agent checkpoints if they exist."""
    base = "Models/Single agent path integration"
    suffix = (
        "steps_20_batch_200_RNN_4096_relu_rf_012_DoG_True_"
        "periodic_False_lr_00001_weight_decay_1e-06/final_model.pth"
    )
    paths = []
    for seed in range(5):
        ckpt = os.path.join(base, f"Seed {seed} weight decay 1e-06", suffix)
        if os.path.exists(ckpt):
            paths.append(ckpt)
    return paths


def extract_state(raw: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Normalize checkpoint formats used across scripts."""
    if isinstance(raw, dict):
        if "state_dict" in raw:
            return raw["state_dict"]
        if "model_state_dict" in raw:
            return raw["model_state_dict"]
        if all(hasattr(v, "shape") for v in raw.values()):
            return raw
    raise TypeError(f"Unsupported checkpoint format: {type(raw)}")


def load_gridness_data(ckpt_path: str) -> Dict[str, np.ndarray]:
    """Load cached gridness data; raises if missing."""
    out_dir = os.path.join(os.path.dirname(ckpt_path), "analysis_outputs")
    grid_path = os.path.join(out_dir, "gridness_data.npz")
    if not os.path.exists(grid_path):
        raise FileNotFoundError(
            f"Missing gridness_data.npz beside checkpoint: {grid_path}. "
            "Run multi_seed_predictive_analysis.py first."
        )
    with np.load(grid_path) as data:
        return {k: data[k] for k in data.files}


def _safe_nanargmax(arr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Nan-safe argmax per column, returning (idx, value)."""
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


def classify_from_scores(data: Dict[str, np.ndarray], min_shift_cm: float, gridness_threshold: float) -> Dict[str, np.ndarray]:
    """Classify units using tighter definitions based on lag_cm + scores_60."""
    lag_cm = np.asarray(data["lag_cm"], dtype=float)
    scores = np.asarray(data["scores_60"], dtype=float)
    best_idx, best_vals = _safe_nanargmax(scores)
    best_cm = np.full(best_idx.shape, np.nan, dtype=float)
    valid = best_idx >= 0
    best_cm[valid] = lag_cm[best_idx[valid]]

    qual = valid & np.isfinite(best_vals) & (best_vals >= gridness_threshold)
    predictive = qual & (best_cm >= min_shift_cm)
    retrospective = qual & (best_cm <= -min_shift_cm)
    normal = qual & ~(predictive | retrospective)

    return {
        "predictive": np.where(predictive)[0],
        "retrospective": np.where(retrospective)[0],
        "normal": np.where(normal)[0],
    }


def class_score_vectors(data: Dict[str, np.ndarray], min_shift_cm: float) -> Dict[str, np.ndarray]:
    """Return score vectors per class for ranking units within each group."""
    lag_cm = np.asarray(data["lag_cm"], dtype=float)
    scores = np.asarray(data["scores_60"], dtype=float)

    zero_idx = int(np.argmin(np.abs(lag_cm)))
    zero_scores = scores[zero_idx]

    pos_mask = lag_cm >= min_shift_cm
    neg_mask = lag_cm <= -min_shift_cm
    pos_scores = np.nanmax(scores[pos_mask], axis=0) if np.any(pos_mask) else np.full(scores.shape[1], np.nan)
    neg_scores = np.nanmax(scores[neg_mask], axis=0) if np.any(neg_mask) else np.full(scores.shape[1], np.nan)

    return {
        "predictive": pos_scores,
        "retrospective": neg_scores,
        "normal": zero_scores,
    }


def select_top_percentile(indices: Iterable[int], score_vec: np.ndarray, percentile: float) -> np.ndarray:
    """Return the top-k unit indices within a class for the given percentile."""
    idxs = np.asarray(list(indices), dtype=int)
    idxs = idxs[(idxs >= 0) & (idxs < score_vec.shape[-1])]
    if percentile <= 0 or idxs.size == 0:
        return np.array([], dtype=int)

    scores = score_vec[idxs]
    finite_mask = np.isfinite(scores)
    idxs = idxs[finite_mask]
    scores = scores[finite_mask]
    if idxs.size == 0:
        return np.array([], dtype=int)

    k = max(1, int(math.ceil(idxs.size * (percentile / 100.0))))
    order = np.argsort(scores)[::-1]
    return idxs[order][:k]


def select_top_count(indices: Iterable[int], score_vec: np.ndarray, count: int) -> np.ndarray:
    """Return the top-N unit indices within a class."""
    idxs = np.asarray(list(indices), dtype=int)
    idxs = idxs[(idxs >= 0) & (idxs < score_vec.shape[-1])]
    if count <= 0 or idxs.size == 0:
        return np.array([], dtype=int)

    scores = score_vec[idxs]
    finite_mask = np.isfinite(scores)
    idxs = idxs[finite_mask]
    scores = scores[finite_mask]
    if idxs.size == 0:
        return np.array([], dtype=int)

    order = np.argsort(scores)[::-1]
    k = min(int(count), order.size)
    return idxs[order][:k]


def seed_label_from_path(path: str) -> str:
    match = re.search(r"Seed\s+(\d+)", path)
    return match.group(1) if match else os.path.basename(path)


def run_ablation_for_checkpoint(
    ckpt_path: str,
    args,
    percentiles: Sequence[float],
    combos: Dict[str, Sequence[str]],
    count_targets: Sequence[int],
):
    raw = torch.load(ckpt_path, map_location="cpu")
    state = extract_state(raw)
    Ng, Np, velocity_dim = infer_dims_from_state(state)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    options = build_options(args, (Ng, Np, velocity_dim), device, ckpt_path)

    grid_data = load_gridness_data(ckpt_path)
    class_indices = classify_from_scores(grid_data, args.min_shift_cm, args.gridness_threshold)
    score_vectors = class_score_vectors(grid_data, args.min_shift_cm)

    place_cells = PlaceCells(options)
    traj_gen = TrajectoryGenerator(options, place_cells)
    base_model = RNN(options, place_cells).to(options.device)
    base_model.load_state_dict(state)
    base_model.eval()

    eval_batches = collect_eval_batches(traj_gen, args.ablation_batches)
    if not eval_batches:
        raise RuntimeError("No evaluation batches collected; increase --ablation_batches.")

    baseline_raw = mean_decoding_error_cm(base_model, eval_batches)
    baseline_err = float(baseline_raw) if math.isfinite(baseline_raw) else None

    top_units: Dict[str, Dict[float, np.ndarray]] = {k: {} for k in score_vectors.keys()}
    for cls_name, scores in score_vectors.items():
        for pct in percentiles:
            top_units[cls_name][pct] = select_top_percentile(class_indices[cls_name], scores, pct)

    errors: Dict[str, List[float]] = {name: [] for name in combos}
    removed_counts: Dict[str, List[int]] = {name: [] for name in combos}
    for pct in percentiles:
        for combo_name, combo_classes in combos.items():
            if not combo_classes:
                errors[combo_name].append(baseline_err)
                removed_counts[combo_name].append(0)
                continue

            combo_unit_lists = [top_units[c].get(pct, np.array([], dtype=int)) for c in combo_classes]
            units = np.unique(np.concatenate(combo_unit_lists)) if combo_unit_lists else np.array([], dtype=int)
            removed_counts[combo_name].append(int(units.size))

            if units.size == 0:
                errors[combo_name].append(baseline_err)
                continue

            ablated_model = copy.deepcopy(base_model)
            zero_unit_weights_in_place(ablated_model, units.tolist())
            err_val = mean_decoding_error_cm(ablated_model, eval_batches)
            errors[combo_name].append(float(err_val) if math.isfinite(err_val) else None)

    count_errors: Dict[str, List[float]] = {"predictive": [], "retrospective": []}
    count_removed: Dict[str, List[int]] = {"predictive": [], "retrospective": []}
    for cnt in count_targets:
        count_int = int(cnt)
        for cls_name in ("predictive", "retrospective"):
            selected = select_top_count(class_indices.get(cls_name, []), score_vectors.get(cls_name, np.array([])), count_int)
            count_removed[cls_name].append(int(selected.size))
            if selected.size < count_int or selected.size == 0:
                count_errors[cls_name].append(math.nan)
                continue
            ablated_model = copy.deepcopy(base_model)
            zero_unit_weights_in_place(ablated_model, selected.tolist())
            err_val = mean_decoding_error_cm(ablated_model, eval_batches)
            count_errors[cls_name].append(float(err_val) if math.isfinite(err_val) else None)

    return {
        "checkpoint": ckpt_path,
        "seed": seed_label_from_path(ckpt_path),
        "baseline_error_cm": baseline_err,
        "class_counts": {k: int(v.size) for k, v in class_indices.items()},
        "errors": errors,
        "removed_counts": removed_counts,
        "percentiles": list(percentiles),
        "count_targets": list(count_targets),
        "count_errors": count_errors,
        "count_removed": count_removed,
    }


def plot_single_class_lines(results: List[Dict], percentiles: Sequence[float], save_path: str) -> None:
    labels = ["predictive", "retrospective", "normal"]
    colors = {
        "predictive": "#2086cf",
        "retrospective": "#d72020",
        "normal": "#0A0A0A",
    }

    fig, ax = plt.subplots(figsize=(6.4, 4.1))
    baseline_vals = [r.get("baseline_error_cm") for r in results if r.get("baseline_error_cm") is not None]
    baseline_mean = float(np.nanmean(baseline_vals)) if baseline_vals else math.nan
    if math.isfinite(baseline_mean):
        ax.axhline(baseline_mean, color="#bbbbbb", ls="--", lw=1.2, label="Baseline mean")

    for label in labels:
        rows = []
        for r in results:
            vals = r.get("errors", {}).get(label, [])
            if len(vals) == len(percentiles):
                rows.append(np.array(vals, dtype=float))
        if not rows:
            continue
        mat = np.vstack(rows)
        mean = np.nanmean(mat, axis=0)
        std = np.nanstd(mat, axis=0)
        ax.plot(percentiles, mean, marker="o", color=colors[label], label=label.title())
        ax.fill_between(percentiles, mean - std, mean + std, color=colors[label], alpha=0.2)

    ax.set_xlabel("Top percentile within class removed")
    ax.set_ylabel("Mean decoding error (cm)")
    ax.set_title("Single-class ablations (avg. across seeds)")
    ax.grid(alpha=0.3)
    ax.legend(frameon=False)
    fig.tight_layout()
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    fig.savefig(save_path, dpi=200)
    plt.close(fig)


def plot_combo_heatmap(results: List[Dict], percentiles: Sequence[float], save_path: str) -> None:
    combo_labels = [c for c in COMBOS.keys() if c != "baseline"]
    mat = np.full((len(combo_labels), len(percentiles)), np.nan, dtype=float)
    for i, combo in enumerate(combo_labels):
        rows = []
        for r in results:
            vals = r.get("errors", {}).get(combo, [])
            if len(vals) == len(percentiles):
                rows.append(np.array(vals, dtype=float))
        if rows:
            mat[i] = np.nanmean(np.vstack(rows), axis=0)

    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    im = ax.imshow(mat, aspect="auto", cmap="viridis")
    ax.set_xticks(np.arange(len(percentiles)))
    ax.set_xticklabels([str(p) for p in percentiles])
    ax.set_yticks(np.arange(len(combo_labels)))
    ax.set_yticklabels(combo_labels)
    ax.set_xlabel("Top percentile removed")
    ax.set_ylabel("Ablated class combination")
    ax.set_title("Decoding error (cm) across ablation combinations")

    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            val = mat[i, j]
            if math.isfinite(val):
                ax.text(j, i, f"{val:.1f}", ha="center", va="center", color="white", fontsize=8)

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02, label="Mean error (cm)")
    fig.tight_layout()
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    fig.savefig(save_path, dpi=200)
    plt.close(fig)


def plot_fixed_count_lines(results: List[Dict], counts: Sequence[int], save_path: str) -> None:
    if not counts:
        return
    labels = ["predictive", "retrospective"]
    colors = {
        "predictive": "#1f77b4",
        "retrospective": "#d62728",
    }
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    for label in labels:
        rows = []
        for r in results:
            vals = r.get("count_errors", {}).get(label, [])
            if len(vals) == len(counts):
                rows.append(np.array(vals, dtype=float))
        if not rows:
            continue
        mat = np.vstack(rows)
        mean = np.nanmean(mat, axis=0)
        std = np.nanstd(mat, axis=0)
        ax.plot(counts, mean, marker="o", color=colors[label], label=f"{label.title()} ablation")
        ax.fill_between(counts, mean - std, mean + std, color=colors[label], alpha=0.2)
    ax.set_xlabel("Units removed from class")
    ax.set_ylabel("Mean decoding error (cm)")
    ax.set_title("Fixed-count ablations")
    ax.grid(alpha=0.3)
    ax.legend(frameon=False)
    fig.tight_layout()
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    fig.savefig(save_path, dpi=200)
    plt.close(fig)


def save_results_json(results: List[Dict], percentiles: Sequence[float], counts: Sequence[int], save_path: str) -> None:
    payload = {
        "percentiles": list(percentiles),
        "counts": list(counts),
        "combos": list(COMBOS.keys()),
        "seeds": results,
    }
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    with open(save_path, "w") as f:
        json.dump(payload, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint_paths", nargs="+", default=default_checkpoints(),
                        help="Checkpoints to evaluate (paths, dirs, or globs).")
    parser.add_argument("--ablate_percentiles", nargs="+", type=float,
                        default=[0, 5, 10, 15, 20, 25, 50, 75, 100],
                        help="Percentiles (within each class) to ablate.")
    parser.add_argument("--ablate_counts", nargs="+", type=int, default=[],
                        help="Absolute number of predictive/retrospective units to ablate (per class).")
    parser.add_argument("--ablation_batches", default=8, type=int,
                        help="Cached eval batches per checkpoint.")
    parser.add_argument("--batch_size", default=100, type=int)
    parser.add_argument("--sequence_length", default=20, type=int)
    parser.add_argument("--place_cell_rf", default=0.12, type=float)
    parser.add_argument("--surround_scale", default=2.0, type=float)
    parser.add_argument("--activation", default="relu")
    parser.add_argument("--weight_decay", default=1e-4, type=float)
    parser.add_argument("--box_width", default=2.2, type=float)
    parser.add_argument("--box_height", default=2.2, type=float)
    parser.add_argument("--learning_rate", default=1e-4, type=float)
    parser.add_argument("--min_shift_cm", default=5.0, type=float,
                        help="Minimum shift used to separate predictive vs retrospective units.")
    parser.add_argument("--gridness_threshold", default=0.2, type=float,
                        help="Minimum gridness for class membership (tighter = fewer units).")
    parser.add_argument("--device", default=None, help="Override device (cpu/cuda).")
    parser.add_argument("--output_dir", default="analysis_outputs/predictive_retrospective_ablation",
                        help="Directory to store figures/results.")
    parser.add_argument("--traj_speed_scale", default=1.0, type=float,
                        help="Multiplier on the Rayleigh speed scale (1.0 = default).")
    parser.add_argument("--traj_speed_max", default=None, type=float,
                        help="Optional cap on the forward speed in m/s.")
    parser.add_argument("--traj_velocity_smoothing", default=0.0, type=float,
                        help="EMA factor (0-0.99) applied to speeds for smoother motion.")
    parser.add_argument("--traj_turn_sigma_scale", default=1.0, type=float,
                        help="Multiplier applied to the turning noise σ (lower = smoother).")
    parser.add_argument("--traj_border_region", default=0.03, type=float,
                        help="Wall avoidance triggers within this distance in meters.")
    parser.add_argument("--traj_wall_slowdown", default=0.25, type=float,
                        help="Speed multiplier applied near walls (default 0.25).")
    parser.add_argument("--traj_wall_turn_scale", default=1.0, type=float,
                        help="Scales the corrective turn applied near walls.")
    args = parser.parse_args()

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

    if not checkpoints:
        raise FileNotFoundError("No checkpoints matched the provided paths.")

    # Deduplicate while preserving order
    seen = set()
    ordered = []
    for p in checkpoints:
        if p not in seen:
            seen.add(p)
            ordered.append(p)
    checkpoints = ordered

    percentiles = [float(p) for p in args.ablate_percentiles]
    count_targets = []
    seen_counts = set()
    for c in args.ablate_counts:
        val = int(c)
        if val <= 0:
            continue
        if val not in seen_counts:
            seen_counts.add(val)
            count_targets.append(val)

    results: List[Dict] = []
    for ckpt in checkpoints:
        res = run_ablation_for_checkpoint(ckpt, args, percentiles, COMBOS, count_targets)
        results.append(res)

    os.makedirs(args.output_dir, exist_ok=True)
    save_results_json(results, percentiles, count_targets, os.path.join(args.output_dir, "results.json"))
    plot_single_class_lines(results, percentiles, os.path.join(args.output_dir, "single_class_ablation.png"))
    plot_combo_heatmap(results, percentiles, os.path.join(args.output_dir, "combo_heatmap.png"))
    if count_targets:
        plot_fixed_count_lines(results, count_targets, os.path.join(args.output_dir, "fixed_count_ablation.png"))


if __name__ == "__main__":
    main()
