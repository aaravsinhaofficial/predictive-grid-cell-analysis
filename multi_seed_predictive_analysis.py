#!/usr/bin/env python3
"""Batch predictive-grid analysis across multiple checkpoints.

Outputs per checkpoint:
  1. Predictive/phase-precession/phase-locked heatmaps (reuses figure from replicate script).
  2. Scatter of gridness at zero shift vs best gridness for |shift| >= threshold_cm.
  3. Rate maps for units whose gridness never exceeds a low threshold.

Also prints diagnostics (cm-per-step, lag range) to help verify the shift calculation.
"""

import argparse
import copy
import glob
import json
import math
import os
from typing import Any, Dict, Iterable, List, Sequence, Tuple, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import gridspec

from model import RNN
from place_cells import PlaceCells
from trajectory_generator import TrajectoryGenerator
from visualize import predictive_gridness_analysis
from replicate_predictive_grid_figure import (
    classify_units,
    cm_per_step_from_positions,
    plot_heatmaps_and_curves,
    prepare_group_matrix,
    safe_nanargmax,
)

COLORS = {
    "predictive": "#1f77b4",
    "phase_precession": "#d62728",
    "phase_locked": "#2ca02c",
}


def get_class_indices(classes: Optional[Dict[str, np.ndarray]], *keys: str) -> np.ndarray:
    """Return the first matching class array (or empty)."""
    if not classes:
        return np.array([], dtype=int)
    for key in keys:
        arr = classes.get(key)
        if arr is not None and np.size(arr) > 0:
            return np.asarray(arr, dtype=int)
    return np.array([], dtype=int)


def infer_dims_from_state(state: Dict[str, torch.Tensor]) -> Tuple[int, int, int]:
    """Infer (Ng, Np, velocity_dim) from a state dict."""
    missing = [k for k in ("encoder.weight", "decoder.weight", "RNN.weight_ih_l0") if k not in state]
    if missing:
        raise KeyError(f"Checkpoint missing required tensors: {missing}")
    enc = state["encoder.weight"]
    dec = state["decoder.weight"]
    rnn = state["RNN.weight_ih_l0"]
    Ng_enc, Np = enc.shape
    Np_dec, Ng_dec = dec.shape
    if Ng_enc != Ng_dec or Np != Np_dec:
        raise ValueError(f"Inconsistent encoder/decoder shapes: encoder {enc.shape}, decoder {dec.shape}")
    velocity_dim = rnn.shape[1]
    return Ng_enc, Np, velocity_dim


def build_options(args, dims: Tuple[int, int, int], device: str, ckpt_path: str):
    """Create a simple namespace compatible with the training modules."""
    Ng, Np, velocity_dim = dims
    class Opt:
        pass
    options = Opt()
    options.batch_size = args.batch_size
    options.sequence_length = args.sequence_length
    options.Np = Np
    options.Ng = Ng
    options.place_cell_rf = args.place_cell_rf
    options.surround_scale = args.surround_scale
    options.RNN_type = 'RNN'
    options.activation = args.activation
    options.weight_decay = args.weight_decay
    options.DoG = True
    options.periodic = False
    options.box_width = args.box_width
    options.box_height = args.box_height
    options.learning_rate = args.learning_rate
    options.device = device
    options.velocity_dim = velocity_dim
    options.run_ID = 'multi_seed_analysis'
    options.save_dir = os.path.dirname(ckpt_path) or '.'
    options.trajectory_speed_scale = args.traj_speed_scale
    options.trajectory_speed_max = args.traj_speed_max
    options.trajectory_velocity_smoothing = args.traj_velocity_smoothing
    options.trajectory_turn_sigma_scale = args.traj_turn_sigma_scale
    options.trajectory_border_region = args.traj_border_region
    options.trajectory_wall_slowdown = args.traj_wall_slowdown
    options.trajectory_wall_turn_scale = args.traj_wall_turn_scale
    return options


def scatter_zero_vs_shift(zero_scores: np.ndarray,
                          shifted_scores: np.ndarray,
                          save_path: str,
                          title: str,
                          threshold: float,
                          classes: Optional[Dict[str, np.ndarray]] = None,
                          low_grid_units: Optional[np.ndarray] = None,
                          cm_per_step: Optional[float] = None,
                          best_shift_cm: Optional[np.ndarray] = None,
                          lag_cm: Optional[np.ndarray] = None,
                          min_shift_cm: Optional[float] = None) -> None:
    """Create scatter comparing zero shift vs best shifted gridness.

    Points are color-coded by their preferred shift (cm), while marker shapes
    label predictive vs retrospective vs other classes where available.
    """
    zero_scores = np.asarray(zero_scores)
    shifted_scores = np.asarray(shifted_scores)
    assert zero_scores.shape == shifted_scores.shape, "Zero vs shifted arrays must align"
    valid_mask = np.isfinite(zero_scores) & np.isfinite(shifted_scores)

    fig, ax = plt.subplots(figsize=(6.5, 5.4))

    def _filter_indices(indices: Optional[Sequence[int]]) -> np.ndarray:
        if indices is None:
            return np.array([], dtype=int)
        idxs = np.asarray(indices, dtype=int)
        if idxs.size == 0:
            return idxs
        idxs = idxs[(0 <= idxs) & (idxs < zero_scores.shape[0])]
        if idxs.size == 0:
            return idxs
        return idxs[valid_mask[idxs]]

    legend_handles: List[Any] = []
    other_mask = valid_mask.copy()
    category_defs = []
    predictive_idxs = _filter_indices(get_class_indices(classes, 'predictive'))
    retrospective_idxs = _filter_indices(get_class_indices(classes, 'phase_precession', 'retrospective'))
    zero_lag_idxs = _filter_indices(get_class_indices(classes, 'phase_locked', 'zero_lag'))
    if predictive_idxs.size or retrospective_idxs.size or zero_lag_idxs.size:
        category_defs.extend([
            ('Predictive', predictive_idxs, 'o', 80),
            ('Retrospective', retrospective_idxs, '^', 75),
            ('Zero lag', zero_lag_idxs, 's', 65),
        ])
    low_idxs = _filter_indices(low_grid_units)
    if low_idxs.size:
        category_defs.append(('Low grid', low_idxs, 'x', 55))
    for _, idxs, *_ in category_defs:
        other_mask[idxs] = False

    if other_mask.any():
        other_idxs = np.where(other_mask)[0]
        category_defs.append(('Unclassified', other_idxs, 'd', 55))

    color_mapper = None
    norm = None
    color_array = None
    limit = None
    if best_shift_cm is not None and np.isfinite(best_shift_cm).any():
        limit = float(np.nanmax(np.abs(best_shift_cm[np.isfinite(best_shift_cm)])))
        if lag_cm is not None and np.isfinite(lag_cm).any():
            limit = max(limit, float(np.nanmax(np.abs(lag_cm))))
        if not np.isfinite(limit) or limit <= 0:
            limit = 1.0
        color_mapper = plt.cm.get_cmap('coolwarm')
        norm = plt.Normalize(vmin=-limit, vmax=limit)
        color_array = np.full(best_shift_cm.shape, np.nan, dtype=float)
        color_array[:] = best_shift_cm

    def _color_slice(idxs: np.ndarray):
        if color_mapper is None or idxs.size == 0:
            return '#7a7a7a'
        vals = np.clip(color_array[idxs], -limit, limit)
        vals = np.nan_to_num(vals, nan=0.0)
        return color_mapper(norm(vals))

    if category_defs:
        for label, idxs, marker, size in category_defs:
            if idxs.size == 0:
                continue
            scat = ax.scatter(
                zero_scores[idxs],
                shifted_scores[idxs],
                s=size,
                c=_color_slice(idxs),
                marker=marker,
                edgecolors='none',
                alpha=0.85,
                label=f'{label} (n={idxs.size})',
            )
            legend_handles.append(scat)
    else:
        z = zero_scores[valid_mask]
        s = shifted_scores[valid_mask]
        ax.scatter(z, s, s=32, c='#2a6fdb', alpha=0.7, edgecolors='none', label=f'Units (n={z.size})')

    ax.axhline(threshold, color='grey', lw=1.0, ls='--', alpha=0.7)
    ax.axvline(threshold, color='grey', lw=1.0, ls='--', alpha=0.7)
    if min_shift_cm is not None and cm_per_step is not None:
        ax.text(
            threshold + 0.02,
            threshold + 0.02,
            f'|Δ| ≥ {min_shift_cm:.1f} cm',
            fontsize=8,
            color='grey',
        )
    ax.axline((0, 0), slope=1, color='black', lw=1.0, ls=':')
    ax.set_xlabel('Gridness at 0 shift (60°)')
    ax.set_ylabel('Best gridness for |shift| ≥ threshold')
    if cm_per_step is not None:
        ax.set_title(f'{title}\nΔ step ≈ {cm_per_step:.2f} cm')
    else:
        ax.set_title(title)
    ax.grid(alpha=0.25)
    if legend_handles:
        ax.legend(frameon=False, fontsize=9, loc='lower right')
    if color_mapper is not None and norm is not None:
        sm = plt.cm.ScalarMappable(cmap=color_mapper, norm=norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.02)
        cbar.set_label('Preferred shift (cm)')
    fig.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def plot_preferred_shift_distribution(best_shift_cm: np.ndarray,
                                      classes: Dict[str, np.ndarray],
                                      min_shift_cm: float,
                                      save_path: str,
                                      bins: int = 40) -> None:
    """Plot histogram/KDE-style overview of preferred shifts."""
    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    valid = np.isfinite(best_shift_cm)
    if not np.any(valid):
        ax.text(0.5, 0.5, 'No valid preferred shifts', ha='center', va='center', transform=ax.transAxes)
    else:
        overall = best_shift_cm[valid]
        span = float(np.nanmax(np.abs(overall)))
        max_range = max(span, min_shift_cm)
        bins_arr = np.linspace(-max_range, max_range, bins)
        ax.hist(overall, bins=bins_arr, color='#6baed6', alpha=0.45, label='All units')
        predictive_idxs = get_class_indices(classes, 'predictive')
        retro_idx = get_class_indices(classes, 'phase_precession', 'retrospective')
        if predictive_idxs.size:
            ax.hist(best_shift_cm[predictive_idxs], bins=bins_arr, color=COLORS['predictive'], alpha=0.6, label='Predictive')
        if retro_idx.size:
            ax.hist(best_shift_cm[retro_idx], bins=bins_arr, color='#d95f02', alpha=0.6, label='Retrospective')
        ax.axvline(0, color='k', lw=1.0)
        ax.axvspan(-min_shift_cm, min_shift_cm, color='#bbbbbb', alpha=0.3, label='± min shift')
    ax.set_xlabel('Preferred shift (cm)')
    ax.set_ylabel('Units')
    ax.set_title('Distribution of preferred shifts')
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def plot_low_grid_ratemaps(xs: np.ndarray,
                           ys: np.ndarray,
                           activations: np.ndarray,
                           scorer,
                           unit_indices: Sequence[int],
                           save_path: str,
                           max_units: int = 16) -> None:
    """Plot ratemaps for units with consistently low gridness."""
    if len(unit_indices) == 0:
        return
    idxs = list(unit_indices)[:max_units]
    n = len(idxs)
    cols = min(4, n)
    rows = int(np.ceil(n / cols))
    fig = plt.figure(figsize=(3 * cols, 3 * rows))
    gs = gridspec.GridSpec(rows, cols, figure=fig, wspace=0.15, hspace=0.25)
    flat_x = xs.reshape(-1)
    flat_y = ys.reshape(-1)
    for i, u in enumerate(idxs):
        rm = scorer.calculate_ratemap(flat_x, flat_y, activations[:, :, u].reshape(-1), statistic='mean')
        ax = fig.add_subplot(gs[i])
        im = ax.imshow(rm, cmap='jet', interpolation='nearest')
        ax.set_title(f'Unit {u}', fontsize=9)
        ax.axis('off')
    fig.suptitle('Units with low gridness', fontsize=14)
    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close(fig)


def zero_unit_weights_in_place(model: RNN, unit_indices: Sequence[int]) -> None:
    """Zero encoder/decoder/recurrent weights so selected units are silenced."""
    if not unit_indices:
        return
    idx_arr = np.array(unit_indices, dtype=int)
    idx_arr = idx_arr[(idx_arr >= 0) & (idx_arr < model.Ng)]
    if idx_arr.size == 0:
        return
    idx_arr = np.unique(idx_arr)
    device = model.decoder.weight.device
    idx = torch.as_tensor(idx_arr, dtype=torch.long, device=device)
    with torch.no_grad():
        model.decoder.weight[:, idx] = 0
        model.encoder.weight[idx, :] = 0
        model.RNN.weight_hh_l0[idx, :] = 0
        model.RNN.weight_hh_l0[:, idx] = 0
        model.RNN.weight_ih_l0[idx, :] = 0


def collect_eval_batches(traj_gen: TrajectoryGenerator, n_batches: int):
    """Cache a fixed set of evaluation batches for fair ablation comparison."""
    batches = []
    for _ in range(max(0, int(n_batches))):
        inputs, pos_batch, _ = traj_gen.get_test_batch()
        v, init = inputs
        batches.append((
            (v.detach().clone(), init.detach().clone()),
            pos_batch.detach().clone(),
        ))
    return batches


def mean_decoding_error_cm(model: RNN, eval_batches) -> float:
    """Compute average position decoding error (cm) across cached batches."""
    if not eval_batches:
        return float('nan')
    model.eval()
    errors = []
    with torch.no_grad():
        for inputs, pos_batch in eval_batches:
            preds = model.predict(inputs)
            pred_pos = model.place_cells.get_nearest_cell_pos(preds)
            err = torch.sqrt(((pos_batch - pred_pos) ** 2).sum(-1)) * 100.0
            errors.append(err.detach().cpu().numpy())
    if not errors:
        return float('nan')
    return float(np.mean(np.concatenate(errors, axis=None)))


def plot_ablation_effects(results: Dict[str, Any], save_path: str) -> None:
    """Visualize decoding error shifts for predictive vs random ablations."""
    baseline = results.get('baseline_error_cm')
    predictive = results.get('predictive_error_cm')
    random_vals = results.get('random_error_cm') or []
    random_vals = [val for val in random_vals if val is not None and math.isfinite(val)]

    valid_values = [
        val for val in (baseline, predictive) if val is not None and math.isfinite(val)
    ]
    if not valid_values and not random_vals:
        return

    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    categories = []
    colors = []
    labels = []
    if baseline is not None and math.isfinite(baseline):
        categories.append(baseline)
        colors.append('#4c72b0')
        labels.append('Baseline')
    if predictive is not None and math.isfinite(predictive):
        categories.append(predictive)
        colors.append('#d62728')
        labels.append('Predictive ablated')
    if random_vals:
        categories.append(float(np.mean(random_vals)))
        colors.append('#7f7f7f')
        labels.append('Random ablated')

    xs = np.arange(len(categories))
    if categories:
        ax.bar(xs, categories, color=colors, alpha=0.85)
        ax.set_xticks(xs, labels)
    if random_vals and labels and 'Random ablated' in labels:
        x_rand = labels.index('Random ablated')
        jitter = np.linspace(-0.12, 0.12, num=len(random_vals)) if len(random_vals) > 1 else np.array([0.0])
        ax.scatter(np.full_like(jitter, x_rand) + jitter, random_vals, color='#444444', alpha=0.7, s=30, label='Random trials')

    ax.set_ylabel('Mean decoding error (cm)')
    ax.set_title('Predictive unit ablations')
    ax.grid(axis='y', alpha=0.3)
    if random_vals:
        ax.legend(frameon=False, fontsize=9, loc='upper left')
    os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)


def best_shift_scores(s60: np.ndarray,
                      lag_cm: np.ndarray,
                      min_shift_cm: float) -> np.ndarray:
    """Return max gridness per unit for |shift| >= min_shift_cm."""
    mask = np.abs(lag_cm) >= min_shift_cm
    if not mask.any():
        return np.full(s60.shape[1], np.nan)
    shifted = s60[mask]
    scores = np.nanmax(shifted, axis=0)
    scores[~np.isfinite(scores)] = np.nan
    return scores


def run_predictive_ablation(model: RNN,
                            traj_gen: TrajectoryGenerator,
                            classes: Dict[str, np.ndarray],
                            args,
                            out_dir: str,
                            rng_seed: int,
                            analysed_units: int) -> Optional[Dict[str, Any]]:
    """Evaluate decoding error impact of ablating predictive vs random units."""
    n_batches = max(0, int(getattr(args, 'ablation_batches', 0)))
    if n_batches <= 0:
        return None

    eval_batches = collect_eval_batches(traj_gen, n_batches)
    if not eval_batches:
        return None

    baseline_raw = mean_decoding_error_cm(model, eval_batches)
    baseline = float(baseline_raw) if math.isfinite(baseline_raw) else None

    predictive_idxs = np.asarray(classes.get('predictive', []), dtype=int)
    predictive_idxs = predictive_idxs[(predictive_idxs >= 0) & (predictive_idxs < model.Ng)]
    predictive_err = None
    if predictive_idxs.size > 0:
        predictive_model = copy.deepcopy(model)
        zero_unit_weights_in_place(predictive_model, predictive_idxs)
        predictive_raw = mean_decoding_error_cm(predictive_model, eval_batches)
        predictive_err = float(predictive_raw) if math.isfinite(predictive_raw) else None

    random_trials = max(0, int(getattr(args, 'ablation_random_trials', 0)))
    random_errs: List[float] = []
    n_remove = predictive_idxs.size
    if n_remove > 0 and random_trials > 0:
        rng = np.random.default_rng(rng_seed)
        pool = np.arange(analysed_units)
        non_predictive = np.setdiff1d(pool, predictive_idxs, assume_unique=True)
        candidate_pool = non_predictive if non_predictive.size >= n_remove else pool
        for _ in range(random_trials):
            rand_idxs = rng.choice(candidate_pool, size=n_remove, replace=False)
            rand_model = copy.deepcopy(model)
            zero_unit_weights_in_place(rand_model, rand_idxs)
            rand_raw = mean_decoding_error_cm(rand_model, eval_batches)
            if math.isfinite(rand_raw):
                random_errs.append(float(rand_raw))

    results: Dict[str, Any] = {
        'batches': len(eval_batches),
        'baseline_error_cm': baseline,
        'predictive_error_cm': predictive_err,
        'random_error_cm': random_errs,
        'predictive_unit_count': int(predictive_idxs.size),
        'random_trials': random_trials,
    }

    metrics_path = os.path.join(out_dir, 'predictive_ablation_metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(results, f, indent=2)
    plot_ablation_effects(results, os.path.join(out_dir, 'predictive_ablation_effects.png'))
    return results


def analyse_checkpoint(ckpt_path: str,
                       args,
                       lags: List[int],
                       min_shift_cm: float,
                       low_grid_threshold: float,
                       rng_seed: int = 0) -> None:
    """Run full analysis for a single checkpoint."""
    print(f'\n=== Analysing {ckpt_path} ===')
    raw = torch.load(ckpt_path, map_location='cpu')
    if isinstance(raw, dict) and 'state_dict' in raw:
        state = raw['state_dict']
    elif isinstance(raw, dict) and 'model_state_dict' in raw:
        state = raw['model_state_dict']
    elif isinstance(raw, dict) and all(hasattr(v, 'shape') for v in raw.values()):
        state = raw
    else:
        raise TypeError(f'Unsupported checkpoint format: {type(raw)}')

    Ng, Np, velocity_dim = infer_dims_from_state(state)
    device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    options = build_options(args, (Ng, Np, velocity_dim), device, ckpt_path)

    if velocity_dim != 2:
        raise NotImplementedError(f'Checkpoint {ckpt_path} expects velocity_dim={velocity_dim}. Not supported in this analysis.')

    place_cells = PlaceCells(options)
    model = RNN(options, place_cells).to(options.device)
    model.load_state_dict(state)
    model.eval()
    traj_gen = TrajectoryGenerator(options, place_cells)

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
    print(f'  cm-per-step ≈ {cm_step:.3f} cm')
    print(f'  Shift range: {lag_cm[0]:.1f} .. {lag_cm[-1]:.1f} cm ({len(lag_cm)} lags)')

    zero_idx = lags.index(0)
    zero_scores = scores_60[zero_idx]
    shifted_scores = best_shift_scores(scores_60, lag_cm, min_shift_cm)
    classes = classify_units(lag_cm, scores_60, min_shift_cm, args.gridness_threshold)
    max_scores = np.nanmax(scores_60, axis=0)
    low_grid_units = np.where(max_scores < low_grid_threshold)[0]
    best_idx, _ = safe_nanargmax(scores_60)
    best_cm = np.full(best_idx.shape, np.nan)
    valid_best = best_idx >= 0
    best_cm[valid_best] = lag_cm[best_idx[valid_best]]

    out_dir = os.path.join(os.path.dirname(ckpt_path), 'analysis_outputs')
    os.makedirs(out_dir, exist_ok=True)

    scatter_zero_vs_shift(
        zero_scores,
        shifted_scores,
        save_path=os.path.join(out_dir, 'gridness_zero_vs_shift.png'),
        title='Gridness comparison (0 shift vs ≥ threshold)',
        threshold=low_grid_threshold,
        classes=classes,
        low_grid_units=low_grid_units,
        cm_per_step=cm_step,
        best_shift_cm=best_cm,
        lag_cm=lag_cm,
        min_shift_cm=min_shift_cm,
    )
    group_matrices = {key: prepare_group_matrix(scores_60, idxs, best_cm) for key, idxs in classes.items()}
    group_scores = {key: scores_60[:, idxs] for key, idxs in classes.items()}
    plot_heatmaps_and_curves(
        lag_cm,
        group_scores,
        group_matrices,
        COLORS,
        save_path=os.path.join(out_dir, 'predictive_classes.png'),
    )

    plot_preferred_shift_distribution(
        best_cm,
        classes,
        min_shift_cm,
        save_path=os.path.join(out_dir, 'preferred_shift_distribution.png'),
    )

    def _report_shift(label: str, idxs: np.ndarray) -> Tuple[float, float]:
        if idxs.size == 0:
            print(f'  {label}: no units')
            return float('nan'), float('nan')
        vals = best_cm[idxs]
        mean = float(np.nanmean(vals))
        median = float(np.nanmedian(vals))
        print(f'  {label}: mean {mean:.2f} cm, median {median:.2f} cm')
        return mean, median

    predictive_mean, predictive_median = _report_shift('Predictive preferred shift', get_class_indices(classes, 'predictive'))
    retro_idx = get_class_indices(classes, 'phase_precession', 'retrospective')
    retro_mean, retro_median = _report_shift('Retrospective preferred shift', retro_idx)

    print(f'  Units analysed: {Ng_eval}')
    print(f'  Units with max gridness < {low_grid_threshold:.2f}: {len(low_grid_units)}')

    plot_low_grid_ratemaps(
        xs,
        ys,
        activations,
        scorer,
        low_grid_units,
        save_path=os.path.join(out_dir, f'low_grid_ratemaps_lt_{low_grid_threshold:.2f}.png'),
        max_units=args.low_grid_plot_units,
    )

    ablation_results = run_predictive_ablation(
        model,
        traj_gen,
        classes,
        args,
        out_dir,
        rng_seed,
        Ng_eval,
    )
    if ablation_results:
        baseline_err = ablation_results.get('baseline_error_cm')
        predictive_err = ablation_results.get('predictive_error_cm')
        random_vals = ablation_results.get('random_error_cm') or []
        if baseline_err is not None:
            print(f'  Baseline decoding error: {baseline_err:.3f} cm')
        if ablation_results.get('predictive_unit_count', 0) == 0:
            print('  Predictive ablation skipped (no predictive units).')
        elif predictive_err is not None and baseline_err is not None:
            delta = predictive_err - baseline_err
            print(f'  Predictive ablation error: {predictive_err:.3f} cm (Δ {delta:+.3f} cm)')
        if random_vals and baseline_err is not None:
            rand_mean = float(np.mean(random_vals))
            print(f'  Random ablation mean: {rand_mean:.3f} cm (Δ {rand_mean - baseline_err:+.3f} cm)')
    else:
        random_vals = []

    ablation_baseline_metric = np.nan
    ablation_predictive_metric = np.nan
    ablation_random_metrics = np.array([], dtype=float)
    if ablation_results:
        if ablation_results.get('baseline_error_cm') is not None:
            ablation_baseline_metric = ablation_results['baseline_error_cm']
        if ablation_results.get('predictive_error_cm') is not None:
            ablation_predictive_metric = ablation_results['predictive_error_cm']
        if ablation_results.get('random_error_cm'):
            ablation_random_metrics = np.array(ablation_results['random_error_cm'], dtype=float)

    np.savez(
        os.path.join(out_dir, 'gridness_data.npz'),
        lag_cm=lag_cm,
        scores_60=scores_60,
        zero_scores=zero_scores,
        shifted_scores=shifted_scores,
        classes_predictive=np.array(classes['predictive']),
        classes_phase_precession=np.array(classes['phase_precession']),
        classes_phase_locked=np.array(classes['phase_locked']),
        best_cm=best_cm,
        low_grid_units=low_grid_units,
        ablation_baseline_error_cm=ablation_baseline_metric,
        ablation_predictive_error_cm=ablation_predictive_metric,
        ablation_random_error_cm=ablation_random_metrics,
    )

    diagnostics = {
        'cm_per_step': cm_step,
        'lag_cm': lag_cm.tolist(),
        'zero_scores_mean': float(np.nanmean(zero_scores)),
        'shifted_scores_mean': float(np.nanmean(shifted_scores)),
        'num_predictive': int(len(classes['predictive'])),
        'num_phase_precession': int(len(classes['phase_precession'])),
        'num_phase_locked': int(len(classes['phase_locked'])),
        'num_low_grid_units': int(len(low_grid_units)),
        'preferred_shift_mean_predictive_cm': predictive_mean,
        'preferred_shift_median_predictive_cm': predictive_median,
        'preferred_shift_mean_retrospective_cm': retro_mean,
        'preferred_shift_median_retrospective_cm': retro_median,
    }
    if ablation_results:
        diagnostics['ablation_batches'] = ablation_results['batches']
        diagnostics['ablation_predictive_units'] = ablation_results['predictive_unit_count']
        diagnostics['ablation_baseline_error_cm'] = ablation_results.get('baseline_error_cm')
        diagnostics['ablation_predictive_error_cm'] = ablation_results.get('predictive_error_cm')
        diagnostics['ablation_random_error_cm'] = ablation_results.get('random_error_cm')
        if ablation_results.get('random_error_cm'):
            diagnostics['ablation_random_error_cm_mean'] = float(np.mean(ablation_results['random_error_cm']))
    with open(os.path.join(out_dir, 'summary.txt'), 'w') as f:
        for k, v in diagnostics.items():
            f.write(f'{k}: {v}\n')


def expand_checkpoints(paths: Sequence[str]) -> List[str]:
    result: List[str] = []
    for p in paths:
        if any(ch in p for ch in '*?[]'):
            result.extend(sorted(glob.glob(p)))
        elif os.path.isdir(p):
            result.extend(sorted(glob.glob(os.path.join(p, '**', '*.pth'), recursive=True)))
        else:
            result.append(p)
    # Deduplicate while preserving order
    seen = set()
    ordered = []
    for p in result:
        if p not in seen:
            seen.add(p)
            ordered.append(p)
    return ordered


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint_paths', nargs='+', required=True,
                        help='List of checkpoint paths, directories, or globs.')
    parser.add_argument('--batch_size', default=100, type=int)
    parser.add_argument('--sequence_length', default=20, type=int)
    parser.add_argument('--place_cell_rf', default=0.12, type=float)
    parser.add_argument('--surround_scale', default=2.0, type=float)
    parser.add_argument('--activation', default='relu')
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--box_width', default=2.2, type=float)
    parser.add_argument('--box_height', default=2.2, type=float)
    parser.add_argument('--learning_rate', default=1e-4, type=float)
    parser.add_argument('--res', default=20, type=int)
    parser.add_argument('--n_batches', default=25, type=int)
    parser.add_argument('--Ng_use', default=512, type=int,
                        help='Number of units to analyse per checkpoint.')
    parser.add_argument('--traj_speed_scale', default=1.0, type=float,
                        help='Multiplier on the Rayleigh speed scale (1.0 = default).')
    parser.add_argument('--traj_speed_max', default=None, type=float,
                        help='Optional cap on the forward speed in m/s.')
    parser.add_argument('--traj_velocity_smoothing', default=0.0, type=float,
                        help='EMA factor (0-0.99) applied to speeds for smoother motion.')
    parser.add_argument('--traj_turn_sigma_scale', default=1.0, type=float,
                        help='Multiplier applied to the turning noise σ (lower = smoother).')
    parser.add_argument('--traj_border_region', default=0.03, type=float,
                        help='Wall avoidance triggers within this distance in meters.')
    parser.add_argument('--traj_wall_slowdown', default=0.25, type=float,
                        help='Speed multiplier applied near walls (default 0.25).')
    parser.add_argument('--traj_wall_turn_scale', default=1.0, type=float,
                        help='Scales the corrective turn applied near walls.')
    parser.add_argument('--gridness_threshold', default=0.2, type=float,
                        help='Minimum gridness for classifying units.')
    parser.add_argument('--low_grid_threshold', default=0.2, type=float,
                        help='Threshold defining “non-grid” units.')
    parser.add_argument('--low_grid_plot_units', default=16, type=int,
                        help='How many low-grid units to plot.')
    parser.add_argument('--min_shift_cm', default=5.0, type=float,
                        help='Minimum spatial shift when searching for predictive peaks.')
    parser.add_argument('--max_lag', default=20, type=int,
                        help='Evaluate lags from -max_lag .. +max_lag.')
    parser.add_argument('--device', default=None,
                        help='Override device (cpu/cuda). Defaults to best available.')
    parser.add_argument('--ablation_batches', default=8, type=int,
                        help='Number of cached batches for ablation tests (0 disables).')
    parser.add_argument('--ablation_random_trials', default=5, type=int,
                        help='How many random-unit ablations to average.')
    parser.add_argument('--rng_seed', default=0, type=int,
                        help='Base seed for reproducible sampling (offset per checkpoint).')
    args = parser.parse_args()

    checkpoints = expand_checkpoints(args.checkpoint_paths)
    if not checkpoints:
        raise FileNotFoundError('No checkpoints matched the provided paths.')

    lags = list(range(-args.max_lag, args.max_lag + 1))
    for idx, path in enumerate(checkpoints):
        analyse_checkpoint(
            ckpt_path=path,
            args=args,
            lags=lags,
            min_shift_cm=args.min_shift_cm,
            low_grid_threshold=args.low_grid_threshold,
            rng_seed=args.rng_seed + idx,
        )


if __name__ == '__main__':
    main()
