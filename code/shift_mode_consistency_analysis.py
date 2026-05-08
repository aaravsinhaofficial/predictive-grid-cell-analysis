#!/usr/bin/env python3
"""Rigorous temporal-vs-spatial shift consistency analysis on shared trajectories.

This script evaluates whether temporal and spatial shift analyses recover the
same underlying predictive/retrospective signal when applied to the exact same
hidden activations and trajectory samples from a trained RNN.

Outputs:
  - consistency_summary.json: statistical summary of agreement tests
  - shared_shift_scores.npz: cached scores/labels for downstream analysis
  - profile_correlation_distribution.png: distribution of within-unit profile agreement
  - preferred_shift_scatter.png: temporal vs spatial preferred shift comparison
  - class_confusion_matrix.png: class-label agreement matrix
  - rnn_exemplar_ratemaps_shift_modes.png: same predictive/retrospective units under both shift definitions
  - rnn_exemplar_pairs/: one side-by-side PNG per exemplar unit
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats
import torch

from model import RNN
from multi_seed_predictive_analysis import build_options, infer_dims_from_state
from path_utils import analysis_dir_for_checkpoint
from place_cells import PlaceCells
from predictive_retrospective_ablation import classify_from_scores, extract_state
from scores import GridScorer
from shift_utils import build_shift_values, shift_values_to_cm
from trajectory_generator import TrajectoryGenerator
from visualize import collect_sequences


CLASS_NAMES = ("predictive", "retrospective", "normal", "non_grid")
CLASS_TO_INT = {name: idx for idx, name in enumerate(CLASS_NAMES)}
CLASS_COLORS = {
    "predictive": "#1f77b4",
    "retrospective": "#d62728",
    "normal": "#2c2c2c",
    "non_grid": "#b0b0b0",
}
MODE_COLORS = {
    "temporal": "#4c78a8",
    "spatial": "#f58518",
}


def format_duration(seconds: float) -> str:
    seconds = float(seconds)
    if not np.isfinite(seconds) or seconds < 0:
        return "?"
    total_seconds = int(round(seconds))
    minutes, secs = divmod(total_seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours > 0:
        return f"{hours}h {minutes:02d}m {secs:02d}s"
    if minutes > 0:
        return f"{minutes}m {secs:02d}s"
    return f"{secs}s"


def log(message: str) -> None:
    stamp = time.strftime("%H:%M:%S")
    print(f"[{stamp}] {message}", flush=True)


def build_grid_scorer(res: int, options) -> GridScorer:
    starts = [0.2] * 10
    ends = np.linspace(0.4, 1.0, num=10)
    coord_range = (
        (-options.box_width / 2, options.box_width / 2),
        (-options.box_height / 2, options.box_height / 2),
    )
    masks_parameters = zip(starts, ends.tolist())
    return GridScorer(res, coord_range, masks_parameters)


def safe_nanargmax(arr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    idxs = np.full(arr.shape[1], -1, dtype=int)
    vals = np.full(arr.shape[1], np.nan, dtype=float)
    for u in range(arr.shape[1]):
        col = np.asarray(arr[:, u], dtype=float)
        if not np.isfinite(col).any():
            continue
        idx = int(np.nanargmax(col))
        idxs[u] = idx
        vals[u] = float(col[idx])
    return idxs, vals


def best_shift_scores(scores: np.ndarray, lag_cm: np.ndarray, min_shift_cm: float) -> np.ndarray:
    mask = np.abs(np.asarray(lag_cm, dtype=float)) >= float(min_shift_cm)
    if not np.any(mask):
        return np.full(scores.shape[1], np.nan, dtype=float)
    vals = np.nanmax(scores[mask], axis=0)
    vals[~np.isfinite(vals)] = np.nan
    return vals


def directional_best(
    scores: np.ndarray,
    lag_cm: np.ndarray,
    min_shift_cm: float,
    direction: str,
) -> Tuple[np.ndarray, np.ndarray]:
    lag_cm = np.asarray(lag_cm, dtype=float)
    if direction == "positive":
        mask = lag_cm >= float(min_shift_cm)
    elif direction == "negative":
        mask = lag_cm <= -float(min_shift_cm)
    else:
        raise ValueError("direction must be 'positive' or 'negative'")
    if not np.any(mask):
        n_units = scores.shape[1]
        return np.full(n_units, np.nan, dtype=float), np.full(n_units, np.nan, dtype=float)
    sub = np.asarray(scores[mask], dtype=float)
    idxs, vals = safe_nanargmax(sub)
    best_cm = np.full(scores.shape[1], np.nan, dtype=float)
    valid = idxs >= 0
    grid = lag_cm[mask]
    best_cm[valid] = grid[idxs[valid]]
    return best_cm, vals


def labels_from_classes(classes: Dict[str, np.ndarray], n_units: int) -> np.ndarray:
    labels = np.full(n_units, CLASS_TO_INT["non_grid"], dtype=int)
    for name in ("predictive", "retrospective", "normal"):
        idxs = np.asarray(classes.get(name, []), dtype=int)
        idxs = idxs[(idxs >= 0) & (idxs < n_units)]
        labels[idxs] = CLASS_TO_INT[name]
    return labels


def interpolate_profile(source_x: np.ndarray, source_y: np.ndarray, target_x: np.ndarray) -> np.ndarray:
    source_x = np.asarray(source_x, dtype=float)
    source_y = np.asarray(source_y, dtype=float)
    target_x = np.asarray(target_x, dtype=float)
    valid = np.isfinite(source_x) & np.isfinite(source_y)
    if np.sum(valid) < 2:
        return np.full(target_x.shape, np.nan, dtype=float)
    xs = source_x[valid]
    ys = source_y[valid]
    order = np.argsort(xs)
    xs = xs[order]
    ys = ys[order]
    uniq_x, uniq_idx = np.unique(xs, return_index=True)
    ys = ys[uniq_idx]
    if uniq_x.size < 2:
        return np.full(target_x.shape, np.nan, dtype=float)
    out = np.interp(target_x, uniq_x, ys, left=np.nan, right=np.nan)
    outside = (target_x < uniq_x[0]) | (target_x > uniq_x[-1])
    out[outside] = np.nan
    return out


def unitwise_profile_correlations(
    temporal_scores: np.ndarray,
    temporal_cm: np.ndarray,
    spatial_scores: np.ndarray,
    spatial_cm: np.ndarray,
    min_points: int,
) -> Tuple[np.ndarray, np.ndarray]:
    lo = max(float(np.nanmin(temporal_cm)), float(np.nanmin(spatial_cm)))
    hi = min(float(np.nanmax(temporal_cm)), float(np.nanmax(spatial_cm)))
    common_grid = np.asarray(spatial_cm, dtype=float)
    common_grid = common_grid[(common_grid >= lo) & (common_grid <= hi)]
    corrs = np.full(temporal_scores.shape[1], np.nan, dtype=float)
    for u in range(temporal_scores.shape[1]):
        temp_interp = interpolate_profile(temporal_cm, temporal_scores[:, u], common_grid)
        spat_interp = interpolate_profile(spatial_cm, spatial_scores[:, u], common_grid)
        valid = np.isfinite(temp_interp) & np.isfinite(spat_interp)
        if np.sum(valid) < int(min_points):
            continue
        x = temp_interp[valid]
        y = spat_interp[valid]
        if np.nanstd(x) < 1e-10 or np.nanstd(y) < 1e-10:
            continue
        corrs[u] = float(np.corrcoef(x, y)[0, 1])
    return corrs, common_grid


def bootstrap_ci(
    values: Sequence[float],
    stat_fn=np.nanmedian,
    n_boot: int = 4000,
    seed: int = 0,
) -> Tuple[float, float]:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return math.nan, math.nan
    rng = np.random.default_rng(seed)
    boots = np.empty(n_boot, dtype=float)
    for i in range(int(n_boot)):
        sample = rng.choice(arr, size=arr.size, replace=True)
        boots[i] = float(stat_fn(sample))
    return float(np.nanpercentile(boots, 2.5)), float(np.nanpercentile(boots, 97.5))


def permutation_spearman(
    x: Sequence[float],
    y: Sequence[float],
    n_perm: int,
    seed: int,
) -> Tuple[float, float]:
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    valid = np.isfinite(x_arr) & np.isfinite(y_arr)
    x_arr = x_arr[valid]
    y_arr = y_arr[valid]
    if x_arr.size < 3:
        return math.nan, math.nan
    obs = float(scipy.stats.spearmanr(x_arr, y_arr).correlation)
    rng = np.random.default_rng(seed)
    count = 0
    for _ in range(int(n_perm)):
        perm = rng.permutation(y_arr)
        val = float(scipy.stats.spearmanr(x_arr, perm).correlation)
        if abs(val) >= abs(obs):
            count += 1
    p_val = float((count + 1) / (n_perm + 1))
    return obs, p_val


def cohen_kappa(labels_a: np.ndarray, labels_b: np.ndarray, n_classes: int) -> Tuple[np.ndarray, float, float]:
    labels_a = np.asarray(labels_a, dtype=int)
    labels_b = np.asarray(labels_b, dtype=int)
    valid = (labels_a >= 0) & (labels_a < n_classes) & (labels_b >= 0) & (labels_b < n_classes)
    labels_a = labels_a[valid]
    labels_b = labels_b[valid]
    conf = np.zeros((n_classes, n_classes), dtype=int)
    for a, b in zip(labels_a, labels_b):
        conf[a, b] += 1
    total = conf.sum()
    if total == 0:
        return conf, math.nan, math.nan
    po = float(np.trace(conf) / total)
    row = conf.sum(axis=1).astype(float)
    col = conf.sum(axis=0).astype(float)
    pe = float(np.sum(row * col) / (total ** 2))
    if 1.0 - pe <= 1e-12:
        kappa = math.nan
    else:
        kappa = float((po - pe) / (1.0 - pe))
    return conf, po, kappa


def permutation_label_agreement(labels_a: np.ndarray, labels_b: np.ndarray, n_perm: int, seed: int) -> Tuple[float, float]:
    labels_a = np.asarray(labels_a, dtype=int)
    labels_b = np.asarray(labels_b, dtype=int)
    valid = np.isfinite(labels_a) & np.isfinite(labels_b)
    labels_a = labels_a[valid]
    labels_b = labels_b[valid]
    if labels_a.size == 0:
        return math.nan, math.nan
    obs = float(np.mean(labels_a == labels_b))
    rng = np.random.default_rng(seed)
    count = 0
    for _ in range(int(n_perm)):
        perm = rng.permutation(labels_b)
        val = float(np.mean(labels_a == perm))
        if val >= obs:
            count += 1
    p_val = float((count + 1) / (n_perm + 1))
    return obs, p_val


def overlap_stats(total_units: int, a: np.ndarray, b: np.ndarray) -> Dict[str, float]:
    a = np.asarray(a, dtype=int)
    b = np.asarray(b, dtype=int)
    a = np.unique(a[(a >= 0) & (a < total_units)])
    b = np.unique(b[(b >= 0) & (b < total_units)])
    inter = np.intersect1d(a, b)
    union = np.union1d(a, b)
    p_val = float(
        scipy.stats.hypergeom.sf(
            max(int(inter.size) - 1, -1),
            int(total_units),
            int(a.size),
            int(b.size),
        )
    )
    return {
        "temporal_count": int(a.size),
        "spatial_count": int(b.size),
        "overlap_count": int(inter.size),
        "jaccard": float(inter.size / max(union.size, 1)),
        "hypergeom_p_value": p_val,
    }


def plot_profile_correlation_distribution(
    all_corrs: np.ndarray,
    directional_corrs: np.ndarray,
    save_path: Path,
    directional_median: float,
    directional_ci: Tuple[float, float],
    positive_fraction: float,
    binom_p: float,
) -> None:
    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    bins = np.linspace(-1.0, 1.0, 31)
    if np.isfinite(all_corrs).any():
        ax.hist(all_corrs[np.isfinite(all_corrs)], bins=bins, alpha=0.35, color="#b9c0c8", label="All analysed units")
    if np.isfinite(directional_corrs).any():
        ax.hist(directional_corrs[np.isfinite(directional_corrs)], bins=bins, alpha=0.65, color=MODE_COLORS["temporal"], label="Predictive/retrospective units")
    ax.axvline(0.0, color="black", lw=1.0, ls="--")
    if math.isfinite(directional_median):
        ax.axvline(directional_median, color=MODE_COLORS["spatial"], lw=2.0, label=f"Median directional r = {directional_median:.2f}")
    ax.set_xlabel("Within-unit correlation between temporal and spatial score profiles")
    ax.set_ylabel("Units")
    ax.set_title("Shared-trajectory temporal vs spatial profile agreement")
    ax.grid(alpha=0.25)
    subtitle = (
        f"Directional units: median r = {directional_median:.3f} "
        f"[95% bootstrap CI {directional_ci[0]:.3f}, {directional_ci[1]:.3f}] | "
        f"fraction r>0 = {positive_fraction:.3f}, binomial p = {binom_p:.3g}"
    )
    ax.text(0.02, 0.98, subtitle, transform=ax.transAxes, ha="left", va="top", fontsize=9)
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=220)
    plt.close(fig)


def plot_preferred_shift_scatter(
    temporal_best_cm: np.ndarray,
    spatial_best_cm: np.ndarray,
    temporal_labels: np.ndarray,
    spatial_labels: np.ndarray,
    save_path: Path,
    spearman_rho: float,
    spearman_p: float,
    sign_agreement: float,
) -> None:
    directional_mask = (
        np.isfinite(temporal_best_cm)
        & np.isfinite(spatial_best_cm)
        & (
            (temporal_labels != CLASS_TO_INT["non_grid"])
            | (spatial_labels != CLASS_TO_INT["non_grid"])
        )
    )
    if not np.any(directional_mask):
        return

    x = temporal_best_cm[directional_mask]
    y = spatial_best_cm[directional_mask]
    t_lab = temporal_labels[directional_mask]
    s_lab = spatial_labels[directional_mask]

    groups = {
        "predictive": (t_lab == CLASS_TO_INT["predictive"]) & (s_lab == CLASS_TO_INT["predictive"]),
        "retrospective": (t_lab == CLASS_TO_INT["retrospective"]) & (s_lab == CLASS_TO_INT["retrospective"]),
        "discordant": ~(
            ((t_lab == CLASS_TO_INT["predictive"]) & (s_lab == CLASS_TO_INT["predictive"]))
            | ((t_lab == CLASS_TO_INT["retrospective"]) & (s_lab == CLASS_TO_INT["retrospective"]))
        ),
    }
    colors = {
        "predictive": CLASS_COLORS["predictive"],
        "retrospective": CLASS_COLORS["retrospective"],
        "discordant": "#7f7f7f",
    }

    fig, ax = plt.subplots(figsize=(6.5, 6.0))
    for label, mask in groups.items():
        if not np.any(mask):
            continue
        ax.scatter(
            x[mask],
            y[mask],
            s=34,
            alpha=0.75,
            color=colors[label],
            edgecolors="none",
            label=f"{label.title()} (n={int(np.sum(mask))})",
        )
    lim = max(float(np.nanmax(np.abs(x))), float(np.nanmax(np.abs(y))), 5.0)
    ax.plot([-lim, lim], [-lim, lim], color="black", lw=1.0, ls="--")
    ax.axhline(0.0, color="#666666", lw=0.8)
    ax.axvline(0.0, color="#666666", lw=0.8)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_xlabel("Preferred shift from temporal analysis (cm)")
    ax.set_ylabel("Preferred shift from spatial analysis (cm)")
    ax.set_title("Preferred shift agreement for directional units")
    ax.text(
        0.02,
        0.98,
        f"Spearman rho = {spearman_rho:.3f}, permutation p = {spearman_p:.3g}\n"
        f"Sign agreement = {sign_agreement:.3f}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9,
    )
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=220)
    plt.close(fig)


def plot_class_confusion_matrix(
    confusion: np.ndarray,
    save_path: Path,
    agreement: float,
    kappa: float,
    p_value: float,
) -> None:
    fig, ax = plt.subplots(figsize=(6.2, 5.4))
    im = ax.imshow(confusion, cmap="Blues")
    ax.set_xticks(np.arange(len(CLASS_NAMES)))
    ax.set_yticks(np.arange(len(CLASS_NAMES)))
    ax.set_xticklabels([name.title() for name in CLASS_NAMES], rotation=20)
    ax.set_yticklabels([name.title() for name in CLASS_NAMES])
    ax.set_xlabel("Spatial label")
    ax.set_ylabel("Temporal label")
    ax.set_title("Class-label agreement on shared trajectories")
    total = confusion.sum()
    for i in range(confusion.shape[0]):
        for j in range(confusion.shape[1]):
            val = int(confusion[i, j])
            frac = (val / total) if total else 0.0
            ax.text(
                j,
                i,
                f"{val}\n({frac:.1%})",
                ha="center",
                va="center",
                fontsize=8,
                color="white" if confusion[i, j] > np.nanmax(confusion) * 0.45 else "black",
            )
    ax.text(
        0.02,
        -0.18,
        f"Accuracy = {agreement:.3f} | Cohen's kappa = {kappa:.3f} | permutation p = {p_value:.3g}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9,
    )
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Units")
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _class_specific_candidates(
    temporal_labels: np.ndarray,
    spatial_labels: np.ndarray,
    temporal_score: np.ndarray,
    spatial_score: np.ndarray,
    class_name: str,
) -> np.ndarray:
    class_idx = CLASS_TO_INT[class_name]
    score_combo = np.nan_to_num(temporal_score, nan=-np.inf) + np.nan_to_num(spatial_score, nan=-np.inf)
    joint = np.where((temporal_labels == class_idx) & (spatial_labels == class_idx))[0]
    union = np.where((temporal_labels == class_idx) | (spatial_labels == class_idx))[0]

    ranked_parts: List[np.ndarray] = []
    if joint.size:
        joint_rank = np.argsort(score_combo[joint])[::-1]
        ranked_parts.append(joint[joint_rank])

    union_only = np.setdiff1d(union, joint, assume_unique=False)
    if union_only.size:
        union_rank = np.argsort(score_combo[union_only])[::-1]
        ranked_parts.append(union_only[union_rank])

    remainder = np.setdiff1d(np.arange(score_combo.size, dtype=int), union, assume_unique=False)
    if remainder.size:
        remainder_rank = np.argsort(score_combo[remainder])[::-1]
        ranked_parts.append(remainder[remainder_rank])

    if not ranked_parts:
        return np.asarray([], dtype=int)
    return np.concatenate(ranked_parts)


def select_example_units(
    temporal_labels: np.ndarray,
    spatial_labels: np.ndarray,
    temporal_pos_score: np.ndarray,
    spatial_pos_score: np.ndarray,
    temporal_neg_score: np.ndarray,
    spatial_neg_score: np.ndarray,
    examples_per_class: int,
) -> Dict[str, List[int]]:
    result: Dict[str, List[int]] = {}
    pred_candidates = _class_specific_candidates(
        temporal_labels,
        spatial_labels,
        temporal_pos_score,
        spatial_pos_score,
        "predictive",
    )
    pred_rank = np.nan_to_num(temporal_pos_score[pred_candidates], nan=-np.inf) + np.nan_to_num(
        spatial_pos_score[pred_candidates], nan=-np.inf
    )
    result["predictive"] = pred_candidates[np.argsort(pred_rank)[::-1][:examples_per_class]].tolist()

    retro_candidates = _class_specific_candidates(
        temporal_labels,
        spatial_labels,
        temporal_neg_score,
        spatial_neg_score,
        "retrospective",
    )
    retro_rank = np.nan_to_num(temporal_neg_score[retro_candidates], nan=-np.inf) + np.nan_to_num(
        spatial_neg_score[retro_candidates], nan=-np.inf
    )
    result["retrospective"] = retro_candidates[np.argsort(retro_rank)[::-1][:examples_per_class]].tolist()
    return result


def rate_map_for_unit(
    scorer: GridScorer,
    xs: np.ndarray,
    ys: np.ndarray,
    activations: np.ndarray,
    unit_idx: int,
    shift_value: float,
    shift_mode: str,
    space_projection: str,
    periodic: bool,
) -> Tuple[np.ndarray, float]:
    s60, _, rm, _ = scorer.get_scores_with_shift(
        xs,
        ys,
        activations[:, :, unit_idx],
        shift_value,
        statistic="mean",
        return_maps=True,
        shift_mode=shift_mode,
        periodic=periodic,
        space_projection=space_projection,
    )
    return rm, float(s60) if s60 is not None and np.isfinite(s60) else math.nan


def draw_rate_map(ax: plt.Axes, ratemap: Optional[np.ndarray], title: str, vmin: float, vmax: float) -> None:
    if ratemap is None or not np.isfinite(np.asarray(ratemap, dtype=float)).any():
        ax.set_facecolor("#f0f0f0")
        ax.text(0.5, 0.5, "No valid\nsamples", ha="center", va="center", fontsize=10, color="#555555")
    else:
        ax.imshow(ratemap, cmap="jet", interpolation="nearest", vmin=vmin, vmax=vmax)
    ax.set_title(title, fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])


def draw_exemplar_comparison(
    ax_curve: plt.Axes,
    ax_zero: plt.Axes,
    ax_temp: plt.Axes,
    ax_space: plt.Axes,
    *,
    class_name: str,
    unit_idx: int,
    temporal_cm: np.ndarray,
    temporal_profile: np.ndarray,
    spatial_cm: np.ndarray,
    spatial_profile: np.ndarray,
    temp_best_cm: float,
    temp_best_score: float,
    space_best_cm: float,
    space_best_score: float,
    rm_zero: Optional[np.ndarray],
    rm_temp: Optional[np.ndarray],
    rm_space: Optional[np.ndarray],
    show_legend: bool,
) -> None:
    rm_stack = [
        np.asarray(rm, dtype=float)
        for rm in (rm_zero, rm_temp, rm_space)
        if rm is not None and np.isfinite(np.asarray(rm, dtype=float)).any()
    ]
    if rm_stack:
        vmin = float(min(np.nanmin(rm) for rm in rm_stack))
        vmax = float(max(np.nanmax(rm) for rm in rm_stack))
    else:
        vmin, vmax = 0.0, 1.0

    ax_curve.plot(temporal_cm, temporal_profile, "o-", color=MODE_COLORS["temporal"], lw=1.8, ms=4, label="Temporal")
    ax_curve.plot(spatial_cm, spatial_profile, "-", color=MODE_COLORS["spatial"], lw=2.0, label="Spatial")
    if np.isfinite(temp_best_cm) and np.isfinite(temp_best_score):
        ax_curve.scatter([temp_best_cm], [temp_best_score], color=MODE_COLORS["temporal"], s=48, zorder=4)
    if np.isfinite(space_best_cm) and np.isfinite(space_best_score):
        ax_curve.scatter([space_best_cm], [space_best_score], color=MODE_COLORS["spatial"], s=48, zorder=4)
    ax_curve.axhline(0.0, color="black", lw=0.8)
    ax_curve.axvline(0.0, color="black", lw=0.8, ls="--")
    ax_curve.set_xlabel("Shift (cm)")
    ax_curve.set_ylabel("Gridness (60°)")
    ax_curve.grid(alpha=0.25)
    ax_curve.set_title(
        f"{class_name.title()} unit {unit_idx}\n"
        f"T: {temp_best_cm:+.1f} cm ({temp_best_score:.2f}) | "
        f"S: {space_best_cm:+.1f} cm ({space_best_score:.2f})",
        fontsize=10,
    )
    if show_legend:
        ax_curve.legend(frameon=False, fontsize=8, loc="upper left")

    draw_rate_map(ax_zero, rm_zero, "Zero shift", vmin, vmax)
    draw_rate_map(ax_temp, rm_temp, f"Temporal best\n{temp_best_cm:+.1f} cm", vmin, vmax)
    draw_rate_map(ax_space, rm_space, f"Spatial best\n{space_best_cm:+.1f} cm", vmin, vmax)


def plot_exemplar_ratemaps(
    scorer: GridScorer,
    xs: np.ndarray,
    ys: np.ndarray,
    activations: np.ndarray,
    temporal_scores: np.ndarray,
    temporal_cm: np.ndarray,
    spatial_scores: np.ndarray,
    spatial_cm: np.ndarray,
    temporal_labels: np.ndarray,
    spatial_labels: np.ndarray,
    temporal_pos_cm: np.ndarray,
    temporal_pos_score: np.ndarray,
    temporal_neg_cm: np.ndarray,
    temporal_neg_score: np.ndarray,
    spatial_pos_cm: np.ndarray,
    spatial_pos_score: np.ndarray,
    spatial_neg_cm: np.ndarray,
    spatial_neg_score: np.ndarray,
    examples: Dict[str, List[int]],
    save_path: Path,
    space_projection: str,
    periodic: bool,
) -> Dict[str, List[Dict[str, object]]]:
    rows: List[Tuple[str, int]] = []
    for class_name in ("predictive", "retrospective"):
        for unit_idx in examples.get(class_name, []):
            rows.append((class_name, int(unit_idx)))
    if not rows:
        return {}

    fig, axes = plt.subplots(
        len(rows),
        4,
        figsize=(14.5, 3.8 * len(rows)),
        gridspec_kw={"width_ratios": [1.7, 1.0, 1.0, 1.0]},
        squeeze=False,
    )
    selected_payload: Dict[str, List[Dict[str, object]]] = {"predictive": [], "retrospective": []}

    for row_idx, (class_name, unit_idx) in enumerate(rows):
        ax_curve, ax_zero, ax_temp, ax_space = axes[row_idx]
        if class_name == "predictive":
            temp_best_cm = float(temporal_pos_cm[unit_idx])
            temp_best_score = float(temporal_pos_score[unit_idx])
            space_best_cm = float(spatial_pos_cm[unit_idx])
            space_best_score = float(spatial_pos_score[unit_idx])
        else:
            temp_best_cm = float(temporal_neg_cm[unit_idx])
            temp_best_score = float(temporal_neg_score[unit_idx])
            space_best_cm = float(spatial_neg_cm[unit_idx])
            space_best_score = float(spatial_neg_score[unit_idx])

        temp_shift_value = int(np.nanargmin(np.abs(temporal_cm - temp_best_cm))) if np.isfinite(temp_best_cm) else int(np.nanargmin(np.abs(temporal_cm)))
        temporal_shift = int(round(float(temporal_cm[temp_shift_value] / max(np.nanmean(np.diff(temporal_cm)), 1.0))))  # placeholder, replaced below
        # Use score grid directly so the plotted shift matches an actual evaluated temporal lag.
        temporal_shift = int(np.asarray(build_shift_values("time", max_lag=int((len(temporal_cm) - 1) // 2)))[temp_shift_value]) if np.isfinite(temp_best_cm) else 0

        # Reconstruct the actual temporal shift value from the evaluated cm grid.
        best_temporal_idx = int(np.nanargmin(np.abs(temporal_cm - temp_best_cm))) if np.isfinite(temp_best_cm) else int(np.nanargmin(np.abs(temporal_cm)))
        temporal_shift = best_temporal_idx  # replaced by caller metadata via closure below

        # The actual temporal shift values are not recoverable from cm alone if step size is non-integer.
        # The caller passes a temporal cm grid generated from known discrete lags, so we map back by nearest index.
        # This index is later overwritten by the true lag lookup before plotting.
        del temporal_shift

        selected_payload[class_name].append(
            {
                "unit_index": int(unit_idx),
                "temporal_label": CLASS_NAMES[int(temporal_labels[unit_idx])],
                "spatial_label": CLASS_NAMES[int(spatial_labels[unit_idx])],
                "temporal_best_cm": temp_best_cm,
                "temporal_best_score": temp_best_score,
                "spatial_best_cm": space_best_cm,
                "spatial_best_score": space_best_score,
            }
        )

    # Plot after metadata collection so we can use consistent indexing logic.
    plt.close(fig)
    return selected_payload


def plot_exemplar_ratemaps_with_lags(
    scorer: GridScorer,
    xs: np.ndarray,
    ys: np.ndarray,
    activations: np.ndarray,
    temporal_scores: np.ndarray,
    temporal_cm: np.ndarray,
    temporal_lags: Sequence[int],
    spatial_scores: np.ndarray,
    spatial_cm: np.ndarray,
    temporal_labels: np.ndarray,
    spatial_labels: np.ndarray,
    temporal_pos_cm: np.ndarray,
    temporal_pos_score: np.ndarray,
    temporal_neg_cm: np.ndarray,
    temporal_neg_score: np.ndarray,
    spatial_pos_cm: np.ndarray,
    spatial_pos_score: np.ndarray,
    spatial_neg_cm: np.ndarray,
    spatial_neg_score: np.ndarray,
    examples: Dict[str, List[int]],
    save_path: Path,
    individual_dir: Optional[Path],
    progress_label: Optional[str],
    space_projection: str,
    periodic: bool,
) -> Dict[str, List[Dict[str, object]]]:
    rows: List[Tuple[str, int]] = []
    for class_name in ("predictive", "retrospective"):
        for unit_idx in examples.get(class_name, []):
            rows.append((class_name, int(unit_idx)))
    if not rows:
        return {}

    fig, axes = plt.subplots(
        len(rows),
        4,
        figsize=(14.5, 3.8 * len(rows)),
        gridspec_kw={"width_ratios": [1.8, 1.0, 1.0, 1.0]},
        squeeze=False,
    )
    selected_payload: Dict[str, List[Dict[str, object]]] = {"predictive": [], "retrospective": []}
    if individual_dir is not None:
        individual_dir.mkdir(parents=True, exist_ok=True)

    for row_idx, (class_name, unit_idx) in enumerate(rows):
        if progress_label:
            print(
                f"[{progress_label}] Rendering {row_idx + 1}/{len(rows)}: {class_name} unit {unit_idx}",
                flush=True,
            )
        ax_curve, ax_zero, ax_temp, ax_space = axes[row_idx]
        if class_name == "predictive":
            temp_best_cm = float(temporal_pos_cm[unit_idx])
            temp_best_score = float(temporal_pos_score[unit_idx])
            space_best_cm = float(spatial_pos_cm[unit_idx])
            space_best_score = float(spatial_pos_score[unit_idx])
        else:
            temp_best_cm = float(temporal_neg_cm[unit_idx])
            temp_best_score = float(temporal_neg_score[unit_idx])
            space_best_cm = float(spatial_neg_cm[unit_idx])
            space_best_score = float(spatial_neg_score[unit_idx])

        temp_best_idx = int(np.nanargmin(np.abs(temporal_cm - temp_best_cm))) if np.isfinite(temp_best_cm) else int(np.nanargmin(np.abs(temporal_cm)))
        spatial_best_value = float(space_best_cm) if np.isfinite(space_best_cm) else 0.0
        temporal_shift_value = int(temporal_lags[temp_best_idx])

        rm_zero, _ = rate_map_for_unit(
            scorer,
            xs,
            ys,
            activations,
            unit_idx,
            0,
            shift_mode="time",
            space_projection=space_projection,
            periodic=periodic,
        )
        rm_temp, _ = rate_map_for_unit(
            scorer,
            xs,
            ys,
            activations,
            unit_idx,
            temporal_shift_value,
            shift_mode="time",
            space_projection=space_projection,
            periodic=periodic,
        )
        rm_space, _ = rate_map_for_unit(
            scorer,
            xs,
            ys,
            activations,
            unit_idx,
            spatial_best_value,
            shift_mode="space",
            space_projection=space_projection,
            periodic=periodic,
        )
        draw_exemplar_comparison(
            ax_curve,
            ax_zero,
            ax_temp,
            ax_space,
            class_name=class_name,
            unit_idx=unit_idx,
            temporal_cm=temporal_cm,
            temporal_profile=temporal_scores[:, unit_idx],
            spatial_cm=spatial_cm,
            spatial_profile=spatial_scores[:, unit_idx],
            temp_best_cm=temp_best_cm,
            temp_best_score=temp_best_score,
            space_best_cm=space_best_cm,
            space_best_score=space_best_score,
            rm_zero=rm_zero,
            rm_temp=rm_temp,
            rm_space=rm_space,
            show_legend=(row_idx == 0),
        )

        individual_path = None
        if individual_dir is not None:
            individual_path = individual_dir / f"{row_idx + 1:02d}_{class_name}_unit_{unit_idx}.png"
            fig_single, axes_single = plt.subplots(
                1,
                4,
                figsize=(14.5, 3.8),
                gridspec_kw={"width_ratios": [1.8, 1.0, 1.0, 1.0]},
                squeeze=False,
            )
            draw_exemplar_comparison(
                axes_single[0, 0],
                axes_single[0, 1],
                axes_single[0, 2],
                axes_single[0, 3],
                class_name=class_name,
                unit_idx=unit_idx,
                temporal_cm=temporal_cm,
                temporal_profile=temporal_scores[:, unit_idx],
                spatial_cm=spatial_cm,
                spatial_profile=spatial_scores[:, unit_idx],
                temp_best_cm=temp_best_cm,
                temp_best_score=temp_best_score,
                space_best_cm=space_best_cm,
                space_best_score=space_best_score,
                rm_zero=rm_zero,
                rm_temp=rm_temp,
                rm_space=rm_space,
                show_legend=True,
            )
            fig_single.tight_layout()
            fig_single.savefig(individual_path, dpi=220, bbox_inches="tight")
            plt.close(fig_single)

        selected_payload[class_name].append(
            {
                "unit_index": int(unit_idx),
                "temporal_label": CLASS_NAMES[int(temporal_labels[unit_idx])],
                "spatial_label": CLASS_NAMES[int(spatial_labels[unit_idx])],
                "temporal_best_cm": temp_best_cm,
                "temporal_best_score": temp_best_score,
                "temporal_best_lag": temporal_shift_value,
                "spatial_best_cm": space_best_cm,
                "spatial_best_score": space_best_score,
                "individual_png": None if individual_path is None else str(individual_path),
            }
        )

    fig.suptitle("Same RNN units under temporal and spatial shifting", fontsize=13, y=1.01)
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return selected_payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint_path", required=True)
    parser.add_argument("--batch_size", type=int, default=200)
    parser.add_argument("--sequence_length", type=int, default=40)
    parser.add_argument("--place_cell_rf", type=float, default=0.12)
    parser.add_argument("--surround_scale", type=float, default=2.0)
    parser.add_argument("--activation", default="relu")
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--box_width", type=float, default=2.2)
    parser.add_argument("--box_height", type=float, default=2.2)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--traj_speed_scale", type=float, default=1.0)
    parser.add_argument("--traj_speed_max", type=float, default=None)
    parser.add_argument("--traj_velocity_smoothing", type=float, default=0.0)
    parser.add_argument("--traj_turn_sigma_scale", type=float, default=1.0)
    parser.add_argument("--traj_border_region", type=float, default=0.03)
    parser.add_argument("--traj_wall_slowdown", type=float, default=0.25)
    parser.add_argument("--traj_wall_turn_scale", type=float, default=1.0)
    parser.add_argument("--trajectory_style", default="random_walk", choices=["random_walk", "straight", "per_step_random"])
    parser.add_argument("--trajectory_fixed_speed", type=float, default=None)
    parser.add_argument("--res", type=int, default=20)
    parser.add_argument("--n_batches", type=int, default=8)
    parser.add_argument("--Ng_use", type=int, default=256)
    parser.add_argument("--gridness_threshold", type=float, default=0.2)
    parser.add_argument("--min_shift_cm", type=float, default=5.0)
    parser.add_argument("--max_lag", type=int, default=10)
    parser.add_argument("--max_shift_cm", type=float, default=20.0)
    parser.add_argument("--shift_step_cm", type=float, default=1.0)
    parser.add_argument("--space_projection", default="path", choices=["path", "heading"])
    parser.add_argument("--min_corr_points", type=int, default=8)
    parser.add_argument("--n_perm", type=int, default=4000)
    parser.add_argument("--n_boot", type=int, default=4000)
    parser.add_argument("--examples_per_class", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default=None)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--individual_examples_dirname", default="rnn_exemplar_pairs")
    parser.add_argument("--progress_every_units", type=int, default=50)
    args = parser.parse_args()

    run_start = time.time()
    ckpt_path = Path(args.checkpoint_path)
    log(f"Loading checkpoint from {ckpt_path}")
    raw = torch.load(ckpt_path, map_location="cpu")
    state = extract_state(raw)
    dims = infer_dims_from_state(state)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    options = build_options(args, dims, device, str(ckpt_path))
    options.trajectory_style = args.trajectory_style
    options.trajectory_fixed_speed = args.trajectory_fixed_speed

    if args.output_dir is None:
        args.output_dir = str(analysis_dir_for_checkpoint(ckpt_path) / "shift_mode_consistency")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    log(f"Output directory: {output_dir}")

    place_cells = PlaceCells(options)
    np.random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))
    model = RNN(options, place_cells).to(options.device)
    model.load_state_dict(state)
    model.eval()
    traj_gen = TrajectoryGenerator(options, place_cells)

    ng_total = dims[0]
    if args.Ng_use is not None and int(args.Ng_use) > 0:
        ng_eval = min(int(args.Ng_use), int(ng_total))
    else:
        ng_eval = int(ng_total)
    unit_indices = np.arange(ng_eval, dtype=int)
    log(f"Evaluating {ng_eval} units with seed {args.seed}.")

    stage_start = time.time()
    log(f"Collecting shared trajectories and activations across {int(args.n_batches)} batches.")
    xs, ys, activations = collect_sequences(
        model,
        traj_gen,
        options,
        n_batches=int(args.n_batches),
        Ng=ng_eval,
        idxs=unit_indices,
        progress=True,
        progress_label="sequences",
    )
    log(f"Collected sequences in {format_duration(time.time() - stage_start)}.")
    cm_per_step = float(np.nanmean(np.sqrt(np.diff(xs, axis=0) ** 2 + np.diff(ys, axis=0) ** 2)) * 100.0)
    scorer = build_grid_scorer(int(args.res), options)
    periodic = bool(getattr(options, "periodic", False))
    log(f"Estimated step size: {cm_per_step:.2f} cm per time step.")

    temporal_lags = build_shift_values("time", max_lag=int(args.max_lag))
    spatial_shifts = build_shift_values("space", max_shift_cm=float(args.max_shift_cm), shift_step_cm=float(args.shift_step_cm))
    stage_start = time.time()
    log(f"Scoring temporal shifts for {ng_eval} units across {len(temporal_lags)} lags.")
    temporal_scores, _ = scorer.predictive_grid_scores(
        xs,
        ys,
        activations,
        temporal_lags,
        shift_mode="time",
        periodic=periodic,
        space_projection=args.space_projection,
        progress=True,
        progress_label="temporal",
        progress_every=int(args.progress_every_units),
    )
    log(f"Finished temporal scoring in {format_duration(time.time() - stage_start)}.")
    stage_start = time.time()
    log(f"Scoring spatial shifts for {ng_eval} units across {len(spatial_shifts)} shifts.")
    spatial_scores, _ = scorer.predictive_grid_scores(
        xs,
        ys,
        activations,
        spatial_shifts,
        shift_mode="space",
        periodic=periodic,
        space_projection=args.space_projection,
        progress=True,
        progress_label="spatial",
        progress_every=int(args.progress_every_units),
    )
    log(f"Finished spatial scoring in {format_duration(time.time() - stage_start)}.")
    temporal_cm = shift_values_to_cm(temporal_lags, "time", cm_per_step)
    spatial_cm = shift_values_to_cm(spatial_shifts, "space", cm_per_step)

    stage_start = time.time()
    log("Classifying units and computing agreement statistics.")
    temporal_data = {
        "scores_60": np.asarray(temporal_scores, dtype=float),
        "lag_cm": np.asarray(temporal_cm, dtype=float),
    }
    spatial_data = {
        "scores_60": np.asarray(spatial_scores, dtype=float),
        "lag_cm": np.asarray(spatial_cm, dtype=float),
    }
    temporal_classes = classify_from_scores(temporal_data, args.min_shift_cm, args.gridness_threshold)
    spatial_classes = classify_from_scores(spatial_data, args.min_shift_cm, args.gridness_threshold)
    temporal_labels = labels_from_classes(temporal_classes, ng_eval)
    spatial_labels = labels_from_classes(spatial_classes, ng_eval)

    temporal_best_cm, temporal_best_score = safe_nanargmax(np.asarray(temporal_scores, dtype=float))
    spatial_best_cm_idx, spatial_best_score = safe_nanargmax(np.asarray(spatial_scores, dtype=float))
    temporal_best_shift_cm = np.full(ng_eval, np.nan, dtype=float)
    spatial_best_shift_cm = np.full(ng_eval, np.nan, dtype=float)
    valid_t = temporal_best_cm >= 0
    valid_s = spatial_best_cm_idx >= 0
    temporal_best_shift_cm[valid_t] = temporal_cm[temporal_best_cm[valid_t]]
    spatial_best_shift_cm[valid_s] = spatial_cm[spatial_best_cm_idx[valid_s]]

    temporal_shifted_score = best_shift_scores(np.asarray(temporal_scores, dtype=float), temporal_cm, args.min_shift_cm)
    spatial_shifted_score = best_shift_scores(np.asarray(spatial_scores, dtype=float), spatial_cm, args.min_shift_cm)
    temporal_pos_cm, temporal_pos_score = directional_best(np.asarray(temporal_scores, dtype=float), temporal_cm, args.min_shift_cm, "positive")
    temporal_neg_cm, temporal_neg_score = directional_best(np.asarray(temporal_scores, dtype=float), temporal_cm, args.min_shift_cm, "negative")
    spatial_pos_cm, spatial_pos_score = directional_best(np.asarray(spatial_scores, dtype=float), spatial_cm, args.min_shift_cm, "positive")
    spatial_neg_cm, spatial_neg_score = directional_best(np.asarray(spatial_scores, dtype=float), spatial_cm, args.min_shift_cm, "negative")

    profile_corrs_all, common_grid_cm = unitwise_profile_correlations(
        np.asarray(temporal_scores, dtype=float),
        np.asarray(temporal_cm, dtype=float),
        np.asarray(spatial_scores, dtype=float),
        np.asarray(spatial_cm, dtype=float),
        min_points=int(args.min_corr_points),
    )
    directional_mask = (
        (temporal_labels == CLASS_TO_INT["predictive"])
        | (temporal_labels == CLASS_TO_INT["retrospective"])
        | (spatial_labels == CLASS_TO_INT["predictive"])
        | (spatial_labels == CLASS_TO_INT["retrospective"])
    )
    profile_corrs_directional = profile_corrs_all[directional_mask]
    directional_valid = profile_corrs_directional[np.isfinite(profile_corrs_directional)]
    directional_median = float(np.nanmedian(directional_valid)) if directional_valid.size else math.nan
    directional_ci = bootstrap_ci(directional_valid, stat_fn=np.nanmedian, n_boot=int(args.n_boot), seed=int(args.seed) + 1)
    positive_fraction = float(np.mean(directional_valid > 0)) if directional_valid.size else math.nan
    binom_p = (
        float(scipy.stats.binomtest(int(np.sum(directional_valid > 0)), int(directional_valid.size), p=0.5, alternative="greater").pvalue)
        if directional_valid.size
        else math.nan
    )

    score_mask = (
        np.isfinite(temporal_shifted_score)
        & np.isfinite(spatial_shifted_score)
        & (
            (temporal_shifted_score >= args.gridness_threshold)
            | (spatial_shifted_score >= args.gridness_threshold)
        )
    )
    score_rho, score_p = permutation_spearman(
        temporal_shifted_score[score_mask],
        spatial_shifted_score[score_mask],
        n_perm=int(args.n_perm),
        seed=int(args.seed) + 2,
    )

    preferred_mask = (
        np.isfinite(temporal_best_shift_cm)
        & np.isfinite(spatial_best_shift_cm)
        & directional_mask
    )
    preferred_rho, preferred_p = permutation_spearman(
        temporal_best_shift_cm[preferred_mask],
        spatial_best_shift_cm[preferred_mask],
        n_perm=int(args.n_perm),
        seed=int(args.seed) + 3,
    )
    sign_agreement = (
        float(np.mean(np.sign(temporal_best_shift_cm[preferred_mask]) == np.sign(spatial_best_shift_cm[preferred_mask])))
        if np.any(preferred_mask)
        else math.nan
    )

    confusion, agreement, kappa = cohen_kappa(temporal_labels, spatial_labels, len(CLASS_NAMES))
    _, agreement_p = permutation_label_agreement(
        temporal_labels,
        spatial_labels,
        n_perm=int(args.n_perm),
        seed=int(args.seed) + 4,
    )
    predictive_overlap = overlap_stats(ng_eval, temporal_classes["predictive"], spatial_classes["predictive"])
    retrospective_overlap = overlap_stats(ng_eval, temporal_classes["retrospective"], spatial_classes["retrospective"])
    log(f"Computed statistics in {format_duration(time.time() - stage_start)}.")

    stage_start = time.time()
    log("Writing summary comparison figures.")
    plot_profile_correlation_distribution(
        profile_corrs_all,
        profile_corrs_directional,
        output_dir / "profile_correlation_distribution.png",
        directional_median,
        directional_ci,
        positive_fraction,
        binom_p,
    )
    plot_preferred_shift_scatter(
        temporal_best_shift_cm,
        spatial_best_shift_cm,
        temporal_labels,
        spatial_labels,
        output_dir / "preferred_shift_scatter.png",
        preferred_rho,
        preferred_p,
        sign_agreement,
    )
    plot_class_confusion_matrix(
        confusion,
        output_dir / "class_confusion_matrix.png",
        agreement,
        kappa,
        agreement_p,
    )
    log(f"Saved summary figures in {format_duration(time.time() - stage_start)}.")

    stage_start = time.time()
    log(
        f"Selecting up to {int(args.examples_per_class)} predictive and "
        f"{int(args.examples_per_class)} retrospective exemplar units."
    )
    examples = select_example_units(
        temporal_labels,
        spatial_labels,
        temporal_pos_score,
        spatial_pos_score,
        temporal_neg_score,
        spatial_neg_score,
        examples_per_class=int(args.examples_per_class),
    )
    exemplar_units = plot_exemplar_ratemaps_with_lags(
        scorer,
        xs,
        ys,
        activations,
        np.asarray(temporal_scores, dtype=float),
        np.asarray(temporal_cm, dtype=float),
        temporal_lags,
        np.asarray(spatial_scores, dtype=float),
        np.asarray(spatial_cm, dtype=float),
        temporal_labels,
        spatial_labels,
        temporal_pos_cm,
        temporal_pos_score,
        temporal_neg_cm,
        temporal_neg_score,
        spatial_pos_cm,
        spatial_pos_score,
        spatial_neg_cm,
        spatial_neg_score,
        examples,
        output_dir / "rnn_exemplar_ratemaps_shift_modes.png",
        output_dir / args.individual_examples_dirname,
        "exemplars",
        space_projection=args.space_projection,
        periodic=periodic,
    )
    exemplar_count = int(sum(len(exemplar_units.get(name, [])) for name in ("predictive", "retrospective")))
    log(f"Rendered {exemplar_count} exemplar comparisons in {format_duration(time.time() - stage_start)}.")

    np.savez(
        output_dir / "shared_shift_scores.npz",
        temporal_lags=np.asarray(temporal_lags, dtype=int),
        temporal_cm=np.asarray(temporal_cm, dtype=float),
        spatial_shifts=np.asarray(spatial_shifts, dtype=float),
        spatial_cm=np.asarray(spatial_cm, dtype=float),
        temporal_scores=np.asarray(temporal_scores, dtype=np.float32),
        spatial_scores=np.asarray(spatial_scores, dtype=np.float32),
        temporal_labels=temporal_labels.astype(np.int16),
        spatial_labels=spatial_labels.astype(np.int16),
        profile_correlations=np.asarray(profile_corrs_all, dtype=np.float32),
        common_grid_cm=np.asarray(common_grid_cm, dtype=np.float32),
        temporal_best_shift_cm=np.asarray(temporal_best_shift_cm, dtype=np.float32),
        spatial_best_shift_cm=np.asarray(spatial_best_shift_cm, dtype=np.float32),
        temporal_shifted_score=np.asarray(temporal_shifted_score, dtype=np.float32),
        spatial_shifted_score=np.asarray(spatial_shifted_score, dtype=np.float32),
    )

    summary = {
        "checkpoint_path": str(ckpt_path),
        "output_dir": str(output_dir),
        "settings": {
            "batch_size": int(args.batch_size),
            "sequence_length": int(args.sequence_length),
            "n_batches": int(args.n_batches),
            "Ng_use": int(ng_eval),
            "gridness_threshold": float(args.gridness_threshold),
            "min_shift_cm": float(args.min_shift_cm),
            "max_lag": int(args.max_lag),
            "max_shift_cm": float(args.max_shift_cm),
            "shift_step_cm": float(args.shift_step_cm),
            "space_projection": args.space_projection,
            "trajectory_style": args.trajectory_style,
            "trajectory_fixed_speed": None if args.trajectory_fixed_speed is None else float(args.trajectory_fixed_speed),
            "seed": int(args.seed),
        },
        "cm_per_step": cm_per_step,
        "tests": {
            "profile_correlation_all_units": {
                "n_valid_units": int(np.sum(np.isfinite(profile_corrs_all))),
                "median": float(np.nanmedian(profile_corrs_all)) if np.isfinite(profile_corrs_all).any() else math.nan,
                "mean": float(np.nanmean(profile_corrs_all)) if np.isfinite(profile_corrs_all).any() else math.nan,
            },
            "profile_correlation_directional_units": {
                "n_valid_units": int(directional_valid.size),
                "median": directional_median,
                "mean": float(np.nanmean(directional_valid)) if directional_valid.size else math.nan,
                "bootstrap_ci_95": [directional_ci[0], directional_ci[1]],
                "fraction_positive": positive_fraction,
                "binomial_p_value_gt_zero": binom_p,
            },
            "best_shifted_score_spearman": {
                "n_units": int(np.sum(score_mask)),
                "rho": score_rho,
                "permutation_p_value": score_p,
            },
            "preferred_shift_spearman": {
                "n_units": int(np.sum(preferred_mask)),
                "rho": preferred_rho,
                "permutation_p_value": preferred_p,
                "sign_agreement": sign_agreement,
            },
            "class_agreement": {
                "accuracy": agreement,
                "cohen_kappa": kappa,
                "permutation_p_value": agreement_p,
                "confusion_matrix": confusion.tolist(),
                "labels": list(CLASS_NAMES),
            },
            "predictive_overlap": predictive_overlap,
            "retrospective_overlap": retrospective_overlap,
        },
        "class_counts": {
            "temporal": {name: int(len(temporal_classes[name])) for name in ("predictive", "retrospective", "normal")},
            "spatial": {name: int(len(spatial_classes[name])) for name in ("predictive", "retrospective", "normal")},
        },
        "selected_rnn_units": exemplar_units,
        "individual_rnn_example_dir": str(output_dir / args.individual_examples_dirname),
        "neural_data_status": "No neural recording dataset was found in this workspace, so neural-vs-RNN side-by-side plots were not generated.",
    }
    with open(output_dir / "consistency_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    log(f"Saved summary JSON to {output_dir / 'consistency_summary.json'}.")
    log(f"Run completed in {format_duration(time.time() - run_start)}.")


if __name__ == "__main__":
    main()
