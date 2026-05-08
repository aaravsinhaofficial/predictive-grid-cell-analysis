#!/usr/bin/env python3
"""Train low-rank RNNs and run spatial predictive-grid analyses.

This CLI is intentionally low-rank first: it trains ``model_low_rank.RNN`` on the
path-integration task, tracks spatial predictive-grid emergence during training,
then runs full-unit spatial scoring and class ablations on the final model.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import random
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from model_low_rank import RNN as LowRankRNN
from place_cells import PlaceCells
from replicate_predictive_grid_figure import prepare_group_matrix, plot_heatmaps_and_curves
from shift_utils import build_shift_values, shift_config_label, shift_values_to_cm
from trainer import Trainer
from trajectory_generator import TrajectoryGenerator
from utils import generate_run_ID
from visualize import predictive_gridness_analysis


CLASS_COLORS = {
    "predictive": "#1f77b4",
    "phase_precession": "#d62728",
    "phase_locked": "#2ca02c",
}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def json_ready(obj: Any) -> Any:
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        val = float(obj)
        return val if math.isfinite(val) else None
    if isinstance(obj, dict):
        return {str(k): json_ready(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_ready(v) for v in obj]
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    return obj


def save_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(json_ready(payload), f, indent=2)


def safe_nanargmax(arr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    arr = np.asarray(arr, dtype=float)
    idxs = np.full(arr.shape[1], -1, dtype=int)
    vals = np.full(arr.shape[1], np.nan, dtype=float)
    for u in range(arr.shape[1]):
        col = arr[:, u]
        if np.isfinite(col).any():
            i = int(np.nanargmax(col))
            idxs[u] = i
            vals[u] = col[i]
    return idxs, vals


def classify_spatial_scores(
    lag_cm: np.ndarray,
    scores_60: np.ndarray,
    gridness_threshold: float,
    min_shift_cm: float,
    strong_gridness_threshold: float,
) -> Tuple[Dict[str, np.ndarray], Dict[str, Any], np.ndarray, np.ndarray]:
    best_idx, best_scores = safe_nanargmax(scores_60)
    best_shift_cm = np.full(best_idx.shape, np.nan, dtype=float)
    valid = best_idx >= 0
    best_shift_cm[valid] = lag_cm[best_idx[valid]]

    valid_best = valid & np.isfinite(best_scores) & np.isfinite(best_shift_cm)
    grid = valid_best & (best_scores >= gridness_threshold)
    strong_grid = valid_best & (best_scores >= strong_gridness_threshold)
    predictive = grid & (best_shift_cm >= min_shift_cm)
    retrospective = grid & (best_shift_cm <= -min_shift_cm)
    zero_lag = grid & ~(predictive | retrospective)
    low_grid = ~grid

    classes = {
        "grid": np.where(grid)[0],
        "strong_grid": np.where(strong_grid)[0],
        "predictive": np.where(predictive)[0],
        "retrospective": np.where(retrospective)[0],
        "zero_lag": np.where(zero_lag)[0],
        "low_grid": np.where(low_grid)[0],
        # Aliases used by the existing heatmap helper.
        "phase_precession": np.where(retrospective)[0],
        "phase_locked": np.where(zero_lag)[0],
    }

    n_units = int(scores_60.shape[1])
    denom = float(max(1, n_units))
    summary = {
        "Ng_evaluated": n_units,
        "gridness_threshold": float(gridness_threshold),
        "strong_gridness_threshold": float(strong_gridness_threshold),
        "min_shift_cm": float(min_shift_cm),
        "class_counts": {
            "grid": int(np.sum(grid)),
            "strong_grid": int(np.sum(strong_grid)),
            "predictive": int(np.sum(predictive)),
            "retrospective": int(np.sum(retrospective)),
            "zero_lag": int(np.sum(zero_lag)),
            "low_grid": int(np.sum(low_grid)),
        },
        "class_percent": {
            "grid": 100.0 * float(np.sum(grid) / denom),
            "strong_grid": 100.0 * float(np.sum(strong_grid) / denom),
            "predictive": 100.0 * float(np.sum(predictive) / denom),
            "retrospective": 100.0 * float(np.sum(retrospective) / denom),
            "zero_lag": 100.0 * float(np.sum(zero_lag) / denom),
            "low_grid": 100.0 * float(np.sum(low_grid) / denom),
        },
        "mean_best_gridness": float(np.nanmean(best_scores)),
        "median_best_shift_cm": float(np.nanmedian(best_shift_cm)) if np.isfinite(best_shift_cm).any() else None,
    }
    return classes, summary, best_shift_cm, best_scores


def class_score_vectors(scores_60: np.ndarray, lag_cm: np.ndarray, min_shift_cm: float) -> Dict[str, np.ndarray]:
    zero_idx = int(np.argmin(np.abs(lag_cm)))
    pos_mask = lag_cm >= min_shift_cm
    neg_mask = lag_cm <= -min_shift_cm
    return {
        "predictive": np.nanmax(scores_60[pos_mask], axis=0) if np.any(pos_mask) else np.full(scores_60.shape[1], np.nan),
        "retrospective": np.nanmax(scores_60[neg_mask], axis=0) if np.any(neg_mask) else np.full(scores_60.shape[1], np.nan),
        "zero_lag": scores_60[zero_idx],
        "grid": np.nanmax(scores_60, axis=0),
    }


def select_top_percentile(unit_indices: Iterable[int], score_vec: np.ndarray, percentile: float) -> np.ndarray:
    idxs = np.asarray(list(unit_indices), dtype=int)
    idxs = idxs[(idxs >= 0) & (idxs < score_vec.shape[0])]
    if percentile <= 0 or idxs.size == 0:
        return np.array([], dtype=int)
    scores = np.asarray(score_vec, dtype=float)[idxs]
    finite = np.isfinite(scores)
    idxs = idxs[finite]
    scores = scores[finite]
    if idxs.size == 0:
        return np.array([], dtype=int)
    k = max(1, int(math.ceil(idxs.size * float(percentile) / 100.0)))
    order = np.argsort(scores)[::-1]
    return idxs[order[:k]]


def zero_unit_weights_in_place(model: torch.nn.Module, unit_indices: Sequence[int]) -> None:
    idx_arr = np.asarray(unit_indices, dtype=int).reshape(-1)
    idx_arr = idx_arr[(idx_arr >= 0) & (idx_arr < int(model.Ng))]
    if idx_arr.size == 0:
        return
    idx_arr = np.unique(idx_arr)
    device = next(model.parameters()).device
    idx = torch.as_tensor(idx_arr, dtype=torch.long, device=device)

    with torch.no_grad():
        if hasattr(model, "decoder"):
            model.decoder.weight[:, idx] = 0
        if hasattr(model, "encoder"):
            model.encoder.weight[idx, :] = 0

        rnn = getattr(model, "RNN", None)
        if rnn is None:
            return
        if hasattr(rnn, "weight_hh_l0"):
            rnn.weight_hh_l0[idx, :] = 0
            rnn.weight_hh_l0[:, idx] = 0
        if hasattr(rnn, "weight_ih_l0"):
            rnn.weight_ih_l0[idx, :] = 0
        if hasattr(rnn, "bias_ih_l0") and rnn.bias_ih_l0 is not None:
            rnn.bias_ih_l0[idx] = 0

        if hasattr(rnn, "M") and hasattr(rnn, "N"):
            rnn.M[idx, :] = 0
            rnn.N[:, idx] = 0
        if hasattr(rnn, "weight_ih"):
            rnn.weight_ih[idx, :] = 0
        if hasattr(rnn, "bias_ih") and rnn.bias_ih is not None:
            rnn.bias_ih[idx] = 0


def collect_eval_batches(traj_gen: TrajectoryGenerator, n_batches: int):
    batches = []
    for _ in range(max(0, int(n_batches))):
        inputs, pos_batch, _ = traj_gen.get_test_batch()
        v, init = inputs
        batches.append(((v.detach().clone(), init.detach().clone()), pos_batch.detach().clone()))
    return batches


def mean_decoding_error_cm(model: torch.nn.Module, eval_batches) -> float:
    if not eval_batches:
        return float("nan")
    model.eval()
    errors = []
    with torch.no_grad():
        for inputs, pos_batch in eval_batches:
            preds = model.predict(inputs)
            pred_pos = model.place_cells.get_nearest_cell_pos(preds)
            err = torch.sqrt(((pos_batch - pred_pos) ** 2).sum(-1)) * 100.0
            errors.append(err.detach().cpu().numpy())
    return float(np.mean(np.concatenate(errors, axis=None))) if errors else float("nan")


def plot_class_counts(summary: Dict[str, Any], save_path: Path) -> None:
    labels = ["grid", "strong_grid", "predictive", "retrospective", "zero_lag", "low_grid"]
    counts = [summary["class_counts"][key] for key in labels]
    perc = [summary["class_percent"][key] for key in labels]
    fig, ax = plt.subplots(figsize=(8.0, 4.4))
    colors = ["#1f77b4", "#111111", "#2ca02c", "#d62728", "#9467bd", "#7f7f7f"]
    x = np.arange(len(labels))
    ax.bar(x, counts, color=colors, alpha=0.86)
    ax.set_xticks(x)
    ax.set_xticklabels(["Grid", "Strong", "Predictive", "Retro", "Zero-lag", "Low"], rotation=20)
    ax.set_ylabel("units")
    ax.set_title("Spatial predictive-grid classes")
    ax.grid(axis="y", alpha=0.25)
    for i, (count, pct) in enumerate(zip(counts, perc)):
        ax.text(i, count, f"{pct:.1f}%", ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=200)
    plt.close(fig)


def plot_zero_vs_shift(
    lag_cm: np.ndarray,
    scores_60: np.ndarray,
    best_shift_cm: np.ndarray,
    classes: Dict[str, np.ndarray],
    min_shift_cm: float,
    gridness_threshold: float,
    save_path: Path,
) -> None:
    zero_idx = int(np.argmin(np.abs(lag_cm)))
    zero_scores = scores_60[zero_idx]
    shifted_mask = np.abs(lag_cm) >= min_shift_cm
    shifted_scores = np.nanmax(scores_60[shifted_mask], axis=0) if np.any(shifted_mask) else np.full(scores_60.shape[1], np.nan)
    valid = np.isfinite(zero_scores) & np.isfinite(shifted_scores)

    finite_shift = best_shift_cm[np.isfinite(best_shift_cm)]
    span = max(float(np.max(np.abs(finite_shift))) if finite_shift.size else 1.0, float(min_shift_cm), 1.0)
    cmap = plt.cm.get_cmap("coolwarm")
    norm = plt.Normalize(vmin=-span, vmax=span)

    fig, ax = plt.subplots(figsize=(6.5, 5.4))
    ax.scatter(
        zero_scores[valid],
        shifted_scores[valid],
        c=cmap(norm(np.nan_to_num(best_shift_cm[valid], nan=0.0))),
        s=22,
        alpha=0.68,
        edgecolors="none",
    )
    markers = {
        "predictive": ("o", "#1f77b4"),
        "retrospective": ("^", "#d62728"),
        "zero_lag": ("s", "#2ca02c"),
    }
    for key, (marker, color) in markers.items():
        idxs = classes.get(key, np.array([], dtype=int))
        idxs = idxs[(idxs >= 0) & (idxs < scores_60.shape[1])]
        idxs = idxs[valid[idxs]] if idxs.size else idxs
        if idxs.size:
            ax.scatter(zero_scores[idxs], shifted_scores[idxs], marker=marker, s=44, facecolors="none",
                       edgecolors=color, linewidths=1.0, label=f"{key.replace('_', '-')} (n={idxs.size})")
    ax.axhline(gridness_threshold, color="grey", ls="--", lw=1.0)
    ax.axvline(gridness_threshold, color="grey", ls="--", lw=1.0)
    ax.axline((0, 0), slope=1, color="black", lw=0.9, ls=":")
    ax.set_xlabel("Gridness at 0 cm shift")
    ax.set_ylabel(f"Best gridness for |shift| >= {min_shift_cm:g} cm")
    ax.set_title("Zero-lag vs spatially shifted gridness")
    ax.grid(alpha=0.25)
    if ax.get_legend_handles_labels()[0]:
        ax.legend(frameon=False, fontsize=8)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.02)
    cbar.set_label("Preferred shift (cm)")
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=200)
    plt.close(fig)


def plot_preferred_shift_hist(best_shift_cm: np.ndarray, classes: Dict[str, np.ndarray], min_shift_cm: float, save_path: Path) -> None:
    valid = np.isfinite(best_shift_cm)
    fig, ax = plt.subplots(figsize=(7.0, 4.4))
    if not np.any(valid):
        ax.text(0.5, 0.5, "No finite preferred shifts", ha="center", va="center", transform=ax.transAxes)
    else:
        span = max(float(np.nanmax(np.abs(best_shift_cm[valid]))), min_shift_cm)
        bins = np.linspace(-span, span, 41)
        ax.hist(best_shift_cm[valid], bins=bins, color="#9ecae1", alpha=0.55, label="All units")
        for key, color in (("predictive", "#1f77b4"), ("retrospective", "#d62728"), ("zero_lag", "#2ca02c")):
            idxs = classes.get(key, np.array([], dtype=int))
            idxs = idxs[(idxs >= 0) & (idxs < best_shift_cm.shape[0])]
            if idxs.size:
                ax.hist(best_shift_cm[idxs], bins=bins, color=color, alpha=0.55, label=key.replace("_", "-"))
        ax.axvline(0, color="black", lw=1.0)
        ax.axvspan(-min_shift_cm, min_shift_cm, color="#bbbbbb", alpha=0.24, label="+/- min shift")
    ax.set_xlabel("Preferred spatial shift (cm)")
    ax.set_ylabel("units")
    ax.set_title("Distribution of preferred spatial shifts")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=200)
    plt.close(fig)


def plot_top_ratemaps(
    xs: np.ndarray,
    ys: np.ndarray,
    activations: np.ndarray,
    scorer,
    unit_indices: Sequence[int],
    rank_scores: np.ndarray,
    save_path: Path,
    title: str,
    max_units: int = 16,
) -> None:
    idxs = np.asarray(unit_indices, dtype=int)
    idxs = idxs[(idxs >= 0) & (idxs < activations.shape[2])]
    if idxs.size:
        order = np.argsort(rank_scores[idxs])[::-1]
        idxs = idxs[order[:max_units]]

    n = max(1, int(idxs.size))
    cols = min(4, n)
    rows = int(math.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(2.5 * cols, 2.6 * rows), squeeze=False)
    flat_x = xs.reshape(-1)
    flat_y = ys.reshape(-1)
    if idxs.size == 0:
        axes[0, 0].text(0.5, 0.5, "No units", ha="center", va="center", transform=axes[0, 0].transAxes)
        axes[0, 0].axis("off")
    for i, u in enumerate(idxs):
        ax = axes[i // cols, i % cols]
        rm = scorer.calculate_ratemap(flat_x, flat_y, activations[:, :, u].reshape(-1), statistic="mean")
        ax.imshow(rm, cmap="jet", interpolation="nearest")
        ax.set_title(f"Unit {u}\nGS {rank_scores[u]:.2f}", fontsize=9)
        ax.axis("off")
    for j in range(idxs.size, rows * cols):
        axes[j // cols, j % cols].axis("off")
    fig.suptitle(title)
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=200)
    plt.close(fig)


def plot_ablation_results(results: Dict[str, Any], save_path: Path) -> None:
    percentiles = np.asarray(results["percentiles"], dtype=float)
    groups = list(results["groups"].keys())
    cols = 3
    rows = int(math.ceil(len(groups) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(5.0 * cols, 3.8 * rows), squeeze=False)
    baseline = results.get("baseline_error_cm")
    for i, group in enumerate(groups):
        ax = axes[i // cols, i % cols]
        vals = np.asarray(results["groups"][group]["target_error_cm"], dtype=float)
        rand = np.asarray(results["groups"][group]["random_error_cm_mean"], dtype=float)
        ax.plot(percentiles, vals, marker="o", lw=2.0, label="targeted")
        ax.plot(percentiles, rand, marker="x", lw=1.5, ls="--", label="matched random")
        if baseline is not None and math.isfinite(float(baseline)):
            ax.axhline(float(baseline), color="#777777", ls=":", lw=1.1, label="baseline")
        ax.set_title(group.replace("_", " "))
        ax.set_xlabel("class percentile removed")
        ax.set_ylabel("decoding error (cm)")
        ax.grid(alpha=0.25)
        ax.legend(frameon=False, fontsize=8)
    for j in range(len(groups), rows * cols):
        axes[j // cols, j % cols].axis("off")
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=200)
    plt.close(fig)


def plot_full_ablation_bars(results: Dict[str, Any], save_path: Path) -> None:
    percentiles = list(results["percentiles"])
    if 100.0 not in [float(p) for p in percentiles]:
        return
    idx = [float(p) for p in percentiles].index(100.0)
    labels = ["baseline"] + list(results["groups"].keys())
    vals = [results.get("baseline_error_cm")]
    for group in results["groups"].values():
        arr = group.get("target_error_cm", [])
        vals.append(arr[idx] if idx < len(arr) else math.nan)

    fig, ax = plt.subplots(figsize=(8.2, 4.4))
    x = np.arange(len(labels))
    ax.bar(x, vals, color=["#777777"] + ["#1f77b4"] * (len(labels) - 1), alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels([label.replace("_", " ") for label in labels], rotation=25, ha="right")
    ax.set_ylabel("decoding error (cm)")
    ax.set_title("Full-class ablation summary")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=200)
    plt.close(fig)


def run_ablation_study(
    model: torch.nn.Module,
    traj_gen: TrajectoryGenerator,
    classes: Dict[str, np.ndarray],
    score_vectors: Dict[str, np.ndarray],
    args: argparse.Namespace,
    out_dir: Path,
    rng: np.random.Generator,
    analysed_units: int,
) -> Dict[str, Any]:
    eval_batches = collect_eval_batches(traj_gen, args.ablation_batches)
    baseline = mean_decoding_error_cm(model, eval_batches)
    percentiles = [float(p) for p in args.ablation_percentiles]
    pool = np.arange(analysed_units, dtype=int)
    groups = {
        "predictive": ("predictive",),
        "retrospective": ("retrospective",),
        "zero_lag": ("zero_lag",),
        "all_grid": ("grid",),
        "predictive_plus_retrospective": ("predictive", "retrospective"),
    }

    results: Dict[str, Any] = {
        "baseline_error_cm": float(baseline) if math.isfinite(baseline) else None,
        "percentiles": percentiles,
        "ablation_batches": int(args.ablation_batches),
        "random_trials": int(args.ablation_random_trials),
        "analysed_units": int(analysed_units),
        "groups": {},
    }

    for group_name, class_names in groups.items():
        target_errors: List[float] = []
        random_means: List[float] = []
        random_sems: List[float] = []
        removed_counts: List[int] = []

        for pct in percentiles:
            unit_sets = []
            for class_name in class_names:
                class_units = classes.get(class_name, np.array([], dtype=int))
                scores = score_vectors["grid" if class_name == "grid" else class_name]
                unit_sets.append(select_top_percentile(class_units, scores, pct))
            units = np.unique(np.concatenate(unit_sets)) if unit_sets else np.array([], dtype=int)
            removed_counts.append(int(units.size))

            if units.size == 0:
                target_errors.append(float(baseline) if math.isfinite(baseline) else math.nan)
                random_means.append(float(baseline) if math.isfinite(baseline) else math.nan)
                random_sems.append(0.0)
                continue

            ablated = copy.deepcopy(model)
            zero_unit_weights_in_place(ablated, units)
            target_errors.append(mean_decoding_error_cm(ablated, eval_batches))

            rand_vals: List[float] = []
            for _ in range(max(0, int(args.ablation_random_trials))):
                rand_units = rng.choice(pool, size=int(units.size), replace=False)
                rand_model = copy.deepcopy(model)
                zero_unit_weights_in_place(rand_model, rand_units)
                err = mean_decoding_error_cm(rand_model, eval_batches)
                if math.isfinite(err):
                    rand_vals.append(float(err))
            if rand_vals:
                arr = np.asarray(rand_vals, dtype=float)
                random_means.append(float(np.nanmean(arr)))
                random_sems.append(float(np.nanstd(arr, ddof=1) / np.sqrt(arr.size)) if arr.size > 1 else 0.0)
            else:
                random_means.append(math.nan)
                random_sems.append(0.0)

        results["groups"][group_name] = {
            "classes": list(class_names),
            "removed_counts": removed_counts,
            "target_error_cm": target_errors,
            "random_error_cm_mean": random_means,
            "random_error_cm_sem": random_sems,
        }

    save_json(out_dir / "spatial_ablation_metrics.json", results)
    plot_ablation_results(results, out_dir / "spatial_ablation_curves.png")
    plot_full_ablation_bars(results, out_dir / "spatial_ablation_full_class_summary.png")
    return results


def build_options(args: argparse.Namespace, rank: int, seed: int) -> argparse.Namespace:
    options = argparse.Namespace()
    for key, value in vars(args).items():
        setattr(options, key, value)
    options.RNN_type = "low_rank"
    options.rank = int(rank)
    options.low_rank_rank = int(rank)
    options.device = args.device
    options.seed = int(seed)
    options.velocity_dim = 2
    options.grid_eval_shift_mode = "space"
    options.grid_eval_space_projection = args.space_projection
    options.grid_eval_lags = build_shift_values(
        "space",
        max_shift_cm=args.grid_eval_max_shift_cm,
        shift_step_cm=args.grid_eval_shift_step_cm,
    )
    options.grid_eval_min_shift_cm = args.min_shift_cm
    options.grid_eval_threshold = args.gridness_threshold
    options.grid_eval_strong_threshold = args.strong_gridness_threshold
    options.grid_eval_max_units = args.grid_eval_max_units
    options.grid_eval_res = args.grid_eval_res
    options.grid_eval_batches = args.grid_eval_batches
    options.grid_eval_interval = args.grid_eval_interval
    options.save_ratemaps_interval = args.save_ratemaps_interval

    base_run_id = generate_run_ID(options)
    label = f"seed_{seed}_spatial"
    if args.run_label:
        label = f"{label}_{args.run_label}"
    options.run_ID = f"{base_run_id}_{label}"
    return options


def save_params(options: argparse.Namespace, out_dir: Path) -> None:
    payload = {key: value for key, value in vars(options).items() if not callable(value)}
    save_json(out_dir / "params.json", payload)


def run_final_spatial_analysis(
    model: torch.nn.Module,
    traj_gen: TrajectoryGenerator,
    options: argparse.Namespace,
    args: argparse.Namespace,
    out_dir: Path,
    rng: np.random.Generator,
) -> Dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    shift_values = build_shift_values(
        "space",
        max_shift_cm=args.analysis_max_shift_cm,
        shift_step_cm=args.analysis_shift_step_cm,
    )
    Ng_eval = options.Ng if args.analysis_Ng_use <= 0 else min(int(args.analysis_Ng_use), int(options.Ng))
    idxs = np.arange(Ng_eval, dtype=int)

    print(f"Running final spatial analysis on {Ng_eval}/{options.Ng} units.")
    scores_60, scores_90, xs, ys, activations, scorer = predictive_gridness_analysis(
        model,
        traj_gen,
        options,
        lags=shift_values,
        res=args.analysis_res,
        n_batches=args.analysis_batches,
        Ng=Ng_eval,
        idxs=idxs,
        shift_mode="space",
        space_projection=args.space_projection,
    )

    cm_step = float(np.nanmean(np.sqrt(np.diff(xs, axis=0) ** 2 + np.diff(ys, axis=0) ** 2)) * 100.0)
    lag_cm = shift_values_to_cm(shift_values, "space", cm_step)
    classes, summary, best_shift_cm, best_scores = classify_spatial_scores(
        lag_cm,
        scores_60,
        gridness_threshold=args.gridness_threshold,
        min_shift_cm=args.min_shift_cm,
        strong_gridness_threshold=args.strong_gridness_threshold,
    )
    summary.update({
        "shift_mode": "space",
        "space_projection": args.space_projection,
        "shift_config": shift_config_label("space", args.space_projection),
        "shift_values_cm": lag_cm.tolist(),
        "cm_per_step_observed": cm_step,
        "analysis_batches": int(args.analysis_batches),
        "analysis_res": int(args.analysis_res),
    })

    save_json(out_dir / "spatial_summary.json", summary)
    np.savez(
        out_dir / "spatial_gridness_data.npz",
        shift_mode=np.array("space"),
        space_projection=np.array(args.space_projection),
        shift_values=np.asarray(shift_values, dtype=float),
        lag_cm=lag_cm,
        scores_60=scores_60,
        scores_90=scores_90,
        best_shift_cm=best_shift_cm,
        best_scores=best_scores,
        classes_grid=classes["grid"],
        classes_strong_grid=classes["strong_grid"],
        classes_predictive=classes["predictive"],
        classes_retrospective=classes["retrospective"],
        classes_zero_lag=classes["zero_lag"],
        classes_low_grid=classes["low_grid"],
    )

    plot_class_counts(summary, out_dir / "spatial_class_counts.png")
    plot_zero_vs_shift(lag_cm, scores_60, best_shift_cm, classes, args.min_shift_cm,
                       args.gridness_threshold, out_dir / "spatial_zero_vs_shift.png")
    plot_preferred_shift_hist(best_shift_cm, classes, args.min_shift_cm, out_dir / "spatial_preferred_shift_distribution.png")

    heatmap_groups = {
        "predictive": scores_60[:, classes["predictive"]],
        "phase_precession": scores_60[:, classes["retrospective"]],
        "phase_locked": scores_60[:, classes["zero_lag"]],
    }
    heatmap_mats = {
        "predictive": prepare_group_matrix(scores_60, classes["predictive"], best_shift_cm),
        "phase_precession": prepare_group_matrix(scores_60, classes["retrospective"], best_shift_cm),
        "phase_locked": prepare_group_matrix(scores_60, classes["zero_lag"], best_shift_cm),
    }
    plot_heatmaps_and_curves(lag_cm, heatmap_groups, heatmap_mats, CLASS_COLORS,
                             save_path=str(out_dir / "spatial_predictive_classes.png"))

    plot_top_ratemaps(xs, ys, activations, scorer, classes["strong_grid"], best_scores,
                      out_dir / "top_strong_grid_ratemaps.png", "Top strong-grid ratemaps")
    plot_top_ratemaps(xs, ys, activations, scorer, classes["predictive"], best_scores,
                      out_dir / "top_predictive_ratemaps.png", "Top predictive-grid ratemaps")

    score_vectors = class_score_vectors(scores_60, lag_cm, args.min_shift_cm)
    if not args.skip_ablation and args.ablation_batches > 0:
        ablation = run_ablation_study(model, traj_gen, classes, score_vectors, args, out_dir, rng, Ng_eval)
        summary["ablation_metrics_path"] = str(out_dir / "spatial_ablation_metrics.json")
        summary["ablation_baseline_error_cm"] = ablation.get("baseline_error_cm")
        save_json(out_dir / "spatial_summary.json", summary)

    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--save_dir", default="low_rank_models", help="Directory for checkpoints and analysis outputs.")
    parser.add_argument("--run_label", default=None, help="Optional suffix added to the run directory.")
    parser.add_argument("--device", default="cuda", help="Device for training/eval. Use cuda on the AWS L40 instance.")
    parser.add_argument("--require_cuda", action="store_true", help="Fail fast if CUDA is unavailable.")

    parser.add_argument("--seeds", nargs="+", default=[0], type=int)
    parser.add_argument("--ranks", nargs="+", default=[8], type=int, help="Low-rank K values to train.")
    parser.add_argument("--n_epochs", default=50, type=int)
    parser.add_argument("--n_steps", default=250, type=int, help="Training batches per epoch.")
    parser.add_argument("--batch_size", default=200, type=int)
    parser.add_argument("--sequence_length", default=20, type=int)
    parser.add_argument("--learning_rate", default=1e-4, type=float)
    parser.add_argument("--weight_decay", default=1e-6, type=float)
    parser.add_argument("--activation", default="relu", choices=["relu", "tanh"])
    parser.add_argument("--low_rank_factor_init", default="balanced", choices=["balanced", "legacy"],
                        help="Balanced preserves recurrent variance while giving M/N comparable scales.")
    parser.add_argument("--low_rank_recurrent_gain", default=1.0, type=float)
    parser.add_argument("--low_rank_input_init_scale", default=1.0, type=float)
    parser.add_argument("--Np", default=256, type=int)
    parser.add_argument("--Ng", default=4096, type=int)
    parser.add_argument("--place_cell_rf", default=0.12, type=float)
    parser.add_argument("--surround_scale", default=2.0, type=float)
    parser.add_argument("--DoG", default=True, type=lambda x: str(x).lower() not in {"false", "0", "no"})
    parser.add_argument("--periodic", default=False, type=lambda x: str(x).lower() in {"true", "1", "yes"})
    parser.add_argument("--box_width", default=2.2, type=float)
    parser.add_argument("--box_height", default=2.2, type=float)
    parser.add_argument("--no_restore", action="store_true", help="Start from scratch even if most_recent_model.pth exists.")
    parser.add_argument("--skip_train", action="store_true", help="Skip training and only analyze the initialized/restored model.")

    parser.add_argument("--trajectory_style", default="random_walk", choices=["random_walk", "straight", "per_step_random"])
    parser.add_argument("--trajectory_fixed_speed", default=None, type=float)
    parser.add_argument("--trajectory_dt", default=0.02, type=float)
    parser.add_argument("--trajectory_turn_sigma_scale", default=1.0, type=float)
    parser.add_argument("--trajectory_speed_scale", default=1.0, type=float)
    parser.add_argument("--trajectory_speed_max", default=None, type=float)
    parser.add_argument("--trajectory_velocity_smoothing", default=0.0, type=float)
    parser.add_argument("--trajectory_border_region", default=0.03, type=float)
    parser.add_argument("--trajectory_wall_slowdown", default=0.25, type=float)
    parser.add_argument("--trajectory_wall_turn_scale", default=1.0, type=float)

    parser.add_argument("--space_projection", default="path", choices=["path", "heading"],
                        help="Spatial predictive definition: arc length along the path or heading-projected cm.")
    parser.add_argument("--gridness_threshold", default=0.3, type=float)
    parser.add_argument("--strong_gridness_threshold", default=0.5, type=float)
    parser.add_argument("--min_shift_cm", default=5.0, type=float)

    parser.add_argument("--grid_eval_interval", default=1, type=int,
                        help="Run spatial emergence diagnostics every N epochs; 0 disables.")
    parser.add_argument("--grid_eval_batches", default=5, type=int)
    parser.add_argument("--grid_eval_res", default=20, type=int)
    parser.add_argument("--grid_eval_max_units", default=0, type=int,
                        help="Units for per-epoch emergence diagnostics; <=0 means all 4096.")
    parser.add_argument("--grid_eval_max_shift_cm", default=20.0, type=float)
    parser.add_argument("--grid_eval_shift_step_cm", default=2.0, type=float)
    parser.add_argument("--save_ratemaps_interval", default=0, type=int,
                        help="Save full ratemap mosaic every N epochs; 0 skips per-epoch mosaics.")

    parser.add_argument("--skip_final_analysis", action="store_true")
    parser.add_argument("--analysis_batches", default=25, type=int)
    parser.add_argument("--analysis_res", default=20, type=int)
    parser.add_argument("--analysis_Ng_use", default=0, type=int,
                        help="Units for final analysis; <=0 means all trained units.")
    parser.add_argument("--analysis_max_shift_cm", default=20.0, type=float)
    parser.add_argument("--analysis_shift_step_cm", default=1.0, type=float)

    parser.add_argument("--skip_ablation", action="store_true")
    parser.add_argument("--ablation_batches", default=8, type=int)
    parser.add_argument("--ablation_random_trials", default=5, type=int)
    parser.add_argument("--ablation_percentiles", nargs="+", default=[0, 25, 50, 75, 100], type=float)

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.require_cuda and not torch.cuda.is_available():
        raise RuntimeError("CUDA was required but torch.cuda.is_available() is False.")
    if str(args.device).startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("Requested CUDA device, but CUDA is not available. Pass --device cpu to override.")

    all_summaries = []
    for seed in args.seeds:
        for rank in args.ranks:
            set_seed(seed)
            options = build_options(args, rank=rank, seed=seed)
            print(f"\n=== Low-rank run: rank={rank}, seed={seed}, device={options.device} ===")
            print(f"Spatial predictive diagnostics: {shift_config_label('space', args.space_projection)}")

            place_cells = PlaceCells(options)
            model = LowRankRNN(options, place_cells).to(options.device)
            traj_gen = TrajectoryGenerator(options, place_cells)
            trainer = Trainer(options, model, traj_gen, restore=not args.no_restore)
            run_dir = Path(trainer.ckpt_dir)
            save_params(options, run_dir)

            if not args.skip_train:
                trainer.train(n_epochs=options.n_epochs, n_steps=options.n_steps, save=True)

            final_ckpt = run_dir / "final_model.pth"
            torch.save(model.state_dict(), final_ckpt)
            print(f"Saved final checkpoint to {final_ckpt}")

            if not args.skip_final_analysis:
                analysis_dir = run_dir / "spatial_analysis_all_units"
                rng = np.random.default_rng(seed + 1009 * int(rank))
                summary = run_final_spatial_analysis(model, traj_gen, options, args, analysis_dir, rng)
                summary.update({"rank": int(rank), "seed": int(seed), "run_dir": str(run_dir), "checkpoint": str(final_ckpt)})
                all_summaries.append(summary)

    if all_summaries:
        save_json(Path(args.save_dir) / "low_rank_spatial_experiment_summary.json", {"runs": all_summaries})


if __name__ == "__main__":
    main()
