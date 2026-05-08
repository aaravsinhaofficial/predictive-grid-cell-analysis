#!/usr/bin/env python3
"""Analyze emergence of grid and predictive cells across training checkpoints."""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch

from model import RNN
from place_cells import PlaceCells
from trajectory_generator import TrajectoryGenerator
from visualize import predictive_gridness_analysis
from multi_seed_predictive_analysis import build_options, infer_dims_from_state
from replicate_predictive_grid_figure import cm_per_step_from_positions
from scores import GridScorer
from shift_utils import build_shift_values, shift_values_to_cm


def _extract_state(raw: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    if isinstance(raw, dict):
        if "state_dict" in raw:
            return raw["state_dict"]
        if "model_state_dict" in raw:
            return raw["model_state_dict"]
        if all(hasattr(v, "shape") for v in raw.values()):
            return raw
    raise TypeError(f"Unsupported checkpoint format: {type(raw)}")


def _epoch_from_path(path: str) -> int:
    match = re.search(r"epoch_(\d+)\.pth$", os.path.basename(path))
    return int(match.group(1)) if match else -1


def _safe_nanargmax(arr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    idxs = np.full(arr.shape[1], -1, dtype=int)
    vals = np.full(arr.shape[1], np.nan, dtype=float)
    for u in range(arr.shape[1]):
        col = arr[:, u]
        if np.isfinite(col).any():
            i = int(np.nanargmax(col))
            idxs[u] = i
            vals[u] = col[i]
    return idxs, vals


def _best_scores(scores: np.ndarray, lag_cm: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    if not np.any(mask):
        n_units = scores.shape[1]
        return np.full(n_units, np.nan), np.full(n_units, np.nan)
    sub = scores[mask]
    idxs, vals = _safe_nanargmax(sub)
    best_cm = np.full(vals.shape, np.nan, dtype=float)
    valid = idxs >= 0
    best_cm[valid] = lag_cm[mask][idxs[valid]]
    return vals, best_cm


def _load_series(path: Optional[str]) -> Optional[np.ndarray]:
    if not path:
        return None
    if not os.path.exists(path):
        return None
    return np.load(path)


def _infer_steps_per_epoch(series_len: int, max_epoch: int) -> Tuple[int, int]:
    if max_epoch <= 0:
        return series_len, 1
    if series_len % (max_epoch + 1) == 0:
        steps = series_len // (max_epoch + 1)
        return steps, max_epoch + 1
    if series_len % max_epoch == 0:
        steps = series_len // max_epoch
        return steps, max_epoch
    steps = int(round(series_len / (max_epoch + 1)))
    steps = max(1, steps)
    epochs = series_len // steps
    return steps, max(1, epochs)


def _epoch_means(series: np.ndarray, max_epoch: int) -> Tuple[np.ndarray, int]:
    steps, n_epochs = _infer_steps_per_epoch(len(series), max_epoch)
    usable = steps * n_epochs
    means = series[:usable].reshape(n_epochs, steps).mean(axis=1)
    return means, n_epochs


def _format_float(val: float) -> float:
    return float(val) if np.isfinite(val) else float("nan")


def collect_checkpoints(checkpoint_dir: Optional[str], checkpoint_paths: Optional[Sequence[str]]) -> List[str]:
    paths: List[str] = []
    if checkpoint_dir:
        paths.extend(sorted(Path(checkpoint_dir).glob("epoch_*.pth")))
    if checkpoint_paths:
        for p in checkpoint_paths:
            if any(ch in p for ch in "*?[]"):
                paths.extend(sorted(Path().glob(p)))
            else:
                paths.append(Path(p))
    uniq = []
    seen = set()
    for p in paths:
        p_str = str(p)
        if p_str not in seen:
            seen.add(p_str)
            uniq.append(p_str)
    uniq.sort(key=_epoch_from_path)
    return uniq


def _build_scorer(options, res: int) -> GridScorer:
    starts = [0.2] * 10
    ends = np.linspace(0.4, 1.0, num=10)
    coord_range = ((-options.box_width / 2, options.box_width / 2),
                   (-options.box_height / 2, options.box_height / 2))
    masks_parameters = zip(starts, ends.tolist())
    return GridScorer(res, coord_range, masks_parameters)


def build_cached_batches(args, checkpoint_path: str, seed: Optional[int] = None) -> Dict[str, object]:
    """Generate a fixed set of trajectories to reuse across checkpoints."""
    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)

    raw = torch.load(checkpoint_path, map_location="cpu")
    state = _extract_state(raw)
    Ng, Np, velocity_dim = infer_dims_from_state(state)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    options = build_options(args, (Ng, Np, velocity_dim), device, checkpoint_path)

    options.trajectory_style = args.trajectory_style
    options.trajectory_fixed_speed = args.trajectory_fixed_speed if args.trajectory_style == "straight" else None
    options.trajectory_dt = args.trajectory_dt
    options.trajectory_speed_scale = args.traj_speed_scale
    options.trajectory_speed_max = args.traj_speed_max
    options.trajectory_velocity_smoothing = args.traj_velocity_smoothing
    options.trajectory_turn_sigma_scale = args.traj_turn_sigma_scale
    options.trajectory_border_region = args.traj_border_region
    options.trajectory_wall_slowdown = args.traj_wall_slowdown
    options.trajectory_wall_turn_scale = args.traj_wall_turn_scale

    place_cells = PlaceCells(options)
    traj_gen = TrajectoryGenerator(options, place_cells)

    cached_batches: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
    for _ in range(args.n_batches):
        inputs, pos, _ = traj_gen.get_test_batch()
        v, init_actv = inputs
        cached_batches.append((
            v.detach().cpu(),
            init_actv.detach().cpu(),
            pos.detach().cpu(),
        ))

    return {
        "batches": cached_batches,
        "Np": int(Np),
        "batch_size": int(options.batch_size),
        "sequence_length": int(options.sequence_length),
        "box_width": float(options.box_width),
        "box_height": float(options.box_height),
    }


def compute_sequences_from_cache(model: torch.nn.Module,
                                 cached: Dict[str, object],
                                 Ng_eval: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run model on cached inputs to get activations + aligned positions."""
    device = next(model.parameters()).device
    xs_list: List[np.ndarray] = []
    ys_list: List[np.ndarray] = []
    g_list: List[np.ndarray] = []
    for v_cpu, init_actv_cpu, pos_cpu in cached["batches"]:
        v = v_cpu.to(device)
        init_actv = init_actv_cpu.to(device)
        with torch.no_grad():
            g_batch = model.g((v, init_actv)).detach().cpu().numpy()
        pos_np = pos_cpu.numpy()
        T = min(g_batch.shape[0], pos_np.shape[0])
        g_batch = g_batch[:T, :, :Ng_eval]
        pos_np = pos_np[:T]
        xs_list.append(pos_np[:, :, 0])
        ys_list.append(pos_np[:, :, 1])
        g_list.append(g_batch)

    min_T = min(arr.shape[0] for arr in xs_list) if xs_list else 0
    if min_T == 0:
        return np.zeros((0, 0)), np.zeros((0, 0)), np.zeros((0, 0, 0))
    xs = np.concatenate([arr[:min_T] for arr in xs_list], axis=1)
    ys = np.concatenate([arr[:min_T] for arr in ys_list], axis=1)
    activations = np.concatenate([arr[:min_T] for arr in g_list], axis=1)
    return xs, ys, activations


def analyze_checkpoint(ckpt_path: str,
                       args,
                       lags: List[int],
                       seed: Optional[int] = None,
                       cached: Optional[Dict[str, object]] = None) -> Dict[str, float]:
    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)

    raw = torch.load(ckpt_path, map_location="cpu")
    state = _extract_state(raw)
    Ng, Np, velocity_dim = infer_dims_from_state(state)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    options = build_options(args, (Ng, Np, velocity_dim), device, ckpt_path)

    options.trajectory_style = args.trajectory_style
    options.trajectory_fixed_speed = args.trajectory_fixed_speed if args.trajectory_style == "straight" else None
    options.trajectory_dt = args.trajectory_dt
    options.trajectory_speed_scale = args.traj_speed_scale
    options.trajectory_speed_max = args.traj_speed_max
    options.trajectory_velocity_smoothing = args.traj_velocity_smoothing
    options.trajectory_turn_sigma_scale = args.traj_turn_sigma_scale
    options.trajectory_border_region = args.traj_border_region
    options.trajectory_wall_slowdown = args.traj_wall_slowdown
    options.trajectory_wall_turn_scale = args.traj_wall_turn_scale

    place_cells = PlaceCells(options)
    traj_gen = TrajectoryGenerator(options, place_cells)
    model = RNN(options, place_cells).to(options.device)
    model.load_state_dict(state)
    model.eval()

    if args.Ng_use is None or args.Ng_use <= 0:
        Ng_eval = Ng
    else:
        Ng_eval = min(Ng, int(args.Ng_use))
    idxs = np.arange(Ng_eval)

    use_cache = cached is not None
    if use_cache:
        if cached.get("Np") != int(Np):
            use_cache = False
        if cached.get("batch_size") != int(options.batch_size):
            use_cache = False
        if cached.get("sequence_length") != int(options.sequence_length):
            use_cache = False

    if use_cache:
        xs, ys, activations = compute_sequences_from_cache(model, cached, Ng_eval)
        scorer = _build_scorer(options, args.res)
        scores_60, _ = scorer.predictive_grid_scores(
            xs,
            ys,
            activations,
            lags,
            shift_mode=args.shift_mode,
            periodic=bool(getattr(options, "periodic", False)),
            space_projection=args.space_projection,
        )
    else:
        with torch.no_grad():
            scores_60, _, xs, ys, _, _ = predictive_gridness_analysis(
                model,
                traj_gen,
                options,
                lags=lags,
                res=args.res,
                n_batches=args.n_batches,
                Ng=Ng_eval,
                idxs=idxs,
                shift_mode=args.shift_mode,
                space_projection=args.space_projection,
            )

    cm_step = cm_per_step_from_positions(xs, ys)
    lag_cm = shift_values_to_cm(lags, args.shift_mode, cm_step)

    zero_idx = np.where(np.array(lags) == 0)[0]
    zero_scores = scores_60[zero_idx[0]] if zero_idx.size else np.full(Ng_eval, np.nan)

    pos_mask = lag_cm >= args.min_shift_cm
    neg_mask = lag_cm <= -args.min_shift_cm

    best_pos_scores, best_pos_cm = _best_scores(scores_60, lag_cm, pos_mask)
    best_neg_scores, best_neg_cm = _best_scores(scores_60, lag_cm, neg_mask)

    grid_mask = np.isfinite(zero_scores) & (zero_scores >= args.gridness_threshold)
    predictive_mask = np.isfinite(best_pos_scores) & (best_pos_scores >= args.gridness_threshold)
    retro_mask = np.isfinite(best_neg_scores) & (best_neg_scores >= args.gridness_threshold)
    pred_and_grid = predictive_mask & grid_mask
    retro_and_grid = retro_mask & grid_mask

    metrics = {
        "epoch": _epoch_from_path(ckpt_path),
        "checkpoint": ckpt_path,
        "Ng_total": int(Ng),
        "Ng_eval": int(Ng_eval),
        "grid_count": int(np.sum(grid_mask)),
        "predictive_count": int(np.sum(predictive_mask)),
        "retrospective_count": int(np.sum(retro_mask)),
        "predictive_grid_count": int(np.sum(pred_and_grid)),
        "retrospective_grid_count": int(np.sum(retro_and_grid)),
        "grid_fraction": _format_float(np.mean(grid_mask)),
        "predictive_fraction": _format_float(np.mean(predictive_mask)),
        "retrospective_fraction": _format_float(np.mean(retro_mask)),
        "grid_mean": _format_float(np.nanmean(zero_scores[grid_mask]) if np.any(grid_mask) else np.nan),
        "grid_median": _format_float(np.nanmedian(zero_scores[grid_mask]) if np.any(grid_mask) else np.nan),
        "predictive_mean": _format_float(np.nanmean(best_pos_scores[predictive_mask]) if np.any(predictive_mask) else np.nan),
        "predictive_median": _format_float(np.nanmedian(best_pos_scores[predictive_mask]) if np.any(predictive_mask) else np.nan),
        "retrospective_mean": _format_float(np.nanmean(best_neg_scores[retro_mask]) if np.any(retro_mask) else np.nan),
        "retrospective_median": _format_float(np.nanmedian(best_neg_scores[retro_mask]) if np.any(retro_mask) else np.nan),
        "mean_zero_gridness_all": _format_float(np.nanmean(zero_scores)),
        "mean_best_pos_gridness_all": _format_float(np.nanmean(best_pos_scores)),
        "mean_best_neg_gridness_all": _format_float(np.nanmean(best_neg_scores)),
        "mean_best_pos_shift_cm": _format_float(np.nanmean(best_pos_cm)),
        "mean_best_neg_shift_cm": _format_float(np.nanmean(best_neg_cm)),
        "cm_per_step": _format_float(cm_step),
        "shift_mode": args.shift_mode,
        "space_projection": args.space_projection,
        "shift_values": np.asarray(lags, dtype=float).tolist(),
        "gridness_threshold": float(args.gridness_threshold),
        "min_shift_cm": float(args.min_shift_cm),
    }
    return metrics


def plot_counts(metrics: List[Dict[str, float]], save_path: str) -> None:
    metrics = sorted(metrics, key=lambda m: m["epoch"])
    epochs = np.array([m["epoch"] for m in metrics])
    grid_counts = np.array([m["grid_count"] for m in metrics])
    pred_counts = np.array([m["predictive_count"] for m in metrics])
    retro_counts = np.array([m["retrospective_count"] for m in metrics])
    pred_grid_counts = np.array([m["predictive_grid_count"] for m in metrics])

    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    ax.plot(epochs, grid_counts, marker="o", label="Grid cells (0 lag)")
    ax.plot(epochs, pred_counts, marker="o", label="Predictive cells (+lag)")
    ax.plot(epochs, retro_counts, marker="o", label="Retrospective cells (-lag)")
    ax.plot(epochs, pred_grid_counts, marker="o", linestyle="--", label="Predictive ∩ grid")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Count (evaluated units)")
    ax.set_title("Emergence of grid/predictive cells")
    ax.grid(alpha=0.3)
    ax.legend(frameon=False, ncol=2)
    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)


def plot_griddiness(metrics: List[Dict[str, float]], save_path: str) -> None:
    metrics = sorted(metrics, key=lambda m: m["epoch"])
    epochs = np.array([m["epoch"] for m in metrics])
    grid_mean = np.array([m["grid_mean"] for m in metrics])
    pred_mean = np.array([m["predictive_mean"] for m in metrics])
    retro_mean = np.array([m["retrospective_mean"] for m in metrics])
    grid_median = np.array([m["grid_median"] for m in metrics])
    pred_median = np.array([m["predictive_median"] for m in metrics])

    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    ax.plot(epochs, grid_mean, marker="o", label="Grid cells (mean 0-lag)")
    ax.plot(epochs, pred_mean, marker="o", label="Predictive cells (mean best +lag)")
    ax.plot(epochs, retro_mean, marker="o", label="Retrospective cells (mean best -lag)")
    ax.plot(epochs, grid_median, linestyle="--", color="#1f77b4", alpha=0.6, label="Grid median")
    ax.plot(epochs, pred_median, linestyle="--", color="#ff7f0e", alpha=0.6, label="Predictive median")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Gridness (60°)")
    ax.set_title("Average griddiness of classified units")
    ax.grid(alpha=0.3)
    ax.legend(frameon=False, ncol=2)
    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)


def plot_training_overlay(metrics: List[Dict[str, float]],
                          loss_path: Optional[str],
                          err_path: Optional[str],
                          save_path: str) -> None:
    loss = _load_series(loss_path)
    err = _load_series(err_path)
    if loss is None and err is None:
        return

    metrics = sorted(metrics, key=lambda m: m["epoch"])
    epochs = np.array([m["epoch"] for m in metrics])
    max_epoch = int(np.max(epochs)) if epochs.size else 0

    fig, ax1 = plt.subplots(figsize=(8.6, 4.8))
    lines = []
    labels = []

    if loss is not None:
        loss_means, _ = _epoch_means(loss, max_epoch)
        ax1.plot(np.arange(len(loss_means)), loss_means, color="#1f77b4", label="Loss")
        lines += ax1.get_lines()[-1:]
        labels += ["Loss"]
        ax1.set_ylim(bottom=0)
    if err is not None:
        err_means, _ = _epoch_means(err, max_epoch)
        ax1.plot(np.arange(len(err_means)), err_means, color="#ff7f0e", linestyle="--", label="Decoding error")
        lines += ax1.get_lines()[-1:]
        labels += ["Decoding error"]

    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Training loss / error")
    ax1.grid(alpha=0.3)

    ax2 = ax1.twinx()
    grid_frac = np.array([m["grid_fraction"] for m in metrics])
    pred_frac = np.array([m["predictive_fraction"] for m in metrics])
    ax2.plot(epochs, grid_frac, color="#9467bd", label="Grid fraction (0 lag)")
    ax2.plot(epochs, pred_frac, color="#2ca02c", label="Predictive fraction (+lag)")
    ax2.set_ylabel("Fraction of units")

    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, frameon=False, loc="upper right")
    ax1.set_title("Training regime vs predictive/grid emergence")
    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)


def save_metrics(metrics: List[Dict[str, float]], out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    json_path = os.path.join(out_dir, "metrics.json")
    with open(json_path, "w") as f:
        json.dump(metrics, f, indent=2)

    csv_path = os.path.join(out_dir, "metrics.csv")
    fieldnames = sorted(metrics[0].keys()) if metrics else []
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in metrics:
            writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint_dir", default=None, help="Directory with epoch_*.pth checkpoints.")
    parser.add_argument("--checkpoint_paths", nargs="*", default=None, help="Explicit checkpoint paths or globs.")
    parser.add_argument("--output_dir", default=None, help="Directory to save metrics/figures.")
    parser.add_argument("--loss_path", default=None, help="Optional loss.npy for training overlay plot.")
    parser.add_argument("--error_path", default=None, help="Optional decoding_error.npy for training overlay plot.")

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
    parser.add_argument("--Ng_use", default=512, type=int,
                        help="Number of units to analyze (<=0 uses all).")

    parser.add_argument("--trajectory_style", default="random_walk",
                        choices=["random_walk", "straight", "per_step_random"])
    parser.add_argument("--trajectory_fixed_speed", default=None, type=float)
    parser.add_argument("--trajectory_dt", default=0.02, type=float)
    parser.add_argument("--traj_speed_scale", default=1.0, type=float)
    parser.add_argument("--traj_speed_max", default=None, type=float)
    parser.add_argument("--traj_velocity_smoothing", default=0.0, type=float)
    parser.add_argument("--traj_turn_sigma_scale", default=1.0, type=float)
    parser.add_argument("--traj_border_region", default=0.03, type=float)
    parser.add_argument("--traj_wall_slowdown", default=0.25, type=float)
    parser.add_argument("--traj_wall_turn_scale", default=1.0, type=float)

    parser.add_argument("--gridness_threshold", default=0.3, type=float)
    parser.add_argument("--min_shift_cm", default=5.0, type=float)
    parser.add_argument("--shift_mode", default="time", choices=["time", "space"])
    parser.add_argument("--space_projection", default="path", choices=["path", "heading"])
    parser.add_argument("--max_lag", default=10, type=int)
    parser.add_argument("--max_shift_cm", default=10.0, type=float)
    parser.add_argument("--shift_step_cm", default=1.0, type=float)
    parser.add_argument("--lag_step", default=1, type=int)
    parser.add_argument("--device", default=None)
    parser.add_argument("--rng_seed", default=0, type=int)

    args = parser.parse_args()

    checkpoints = collect_checkpoints(args.checkpoint_dir, args.checkpoint_paths)
    if not checkpoints:
        raise FileNotFoundError("No checkpoints found.")

    lags = build_shift_values(
        args.shift_mode,
        max_lag=args.max_lag,
        lag_step=args.lag_step,
        max_shift_cm=args.max_shift_cm,
        shift_step_cm=args.shift_step_cm,
    )

    out_dir = args.output_dir
    if not out_dir:
        out_dir = os.path.join("analysis_outputs", Path(checkpoints[0]).parent.name, "training_emergence")

    metrics: List[Dict[str, float]] = []
    cache = build_cached_batches(args, checkpoints[0], seed=args.rng_seed)
    for idx, ckpt in enumerate(checkpoints):
        seed = args.rng_seed + idx
        print(f"Analyzing {ckpt} (seed={seed})")
        metrics.append(analyze_checkpoint(ckpt, args, lags, seed=seed, cached=cache))

    metrics = sorted(metrics, key=lambda m: m["epoch"])
    save_metrics(metrics, out_dir)

    plot_counts(metrics, os.path.join(out_dir, "counts.png"))
    plot_griddiness(metrics, os.path.join(out_dir, "griddiness.png"))
    plot_training_overlay(metrics, args.loss_path, args.error_path, os.path.join(out_dir, "training_overlay.png"))

    print(f"Saved metrics and figures to: {out_dir}")


if __name__ == "__main__":
    main()
