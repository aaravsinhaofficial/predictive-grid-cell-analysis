#!/usr/bin/env python3
"""Replicate the predictive/phase-precession/phase-locked grid cell figure.

The script loads a trained RNN checkpoint, computes predictive gridness (60°)
scores across position shifts, classifies units by their preferred shift, and
renders the heatmaps + average curves shown in the reference figure.
"""
import argparse
import os
from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch

from model import RNN
from place_cells import PlaceCells
from trajectory_generator import TrajectoryGenerator
from visualize import predictive_gridness_analysis


def cm_per_step_from_positions(xs: np.ndarray, ys: np.ndarray) -> float:
    """Estimate the average centimeters traveled per time step."""
    dx = np.diff(xs, axis=0)
    dy = np.diff(ys, axis=0)
    step = np.sqrt(dx**2 + dy**2)  # meters
    return float(np.nanmean(step) * 100.0)


@dataclass
class Options:
    """Configuration stub compatible with the training code."""

    checkpoint_path: str
    batch_size: int
    sequence_length: int
    Np: int
    Ng: int
    velocity_dim: int
    place_cell_rf: float
    surround_scale: float
    RNN_type: str
    activation: str
    weight_decay: float
    DoG: bool
    periodic: bool
    box_width: float
    box_height: float
    learning_rate: float
    device: str
    save_dir: str
    run_ID: str = "replicate_figure"


def build_options(args: argparse.Namespace, dims: Tuple[int, int, int]) -> Options:
    """Fill the minimal Options structure required by the modules."""
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    inferred_Ng, inferred_Np, inferred_vel = dims

    Np = args.Np if args.Np is not None else inferred_Np
    Ng = args.Ng if args.Ng is not None else inferred_Ng
    velocity_dim = args.velocity_dim if args.velocity_dim is not None else inferred_vel

    return Options(
        checkpoint_path=args.checkpoint_path,
        batch_size=args.batch_size,
        sequence_length=args.sequence_length,
        Np=Np,
        Ng=Ng,
        velocity_dim=velocity_dim,
        place_cell_rf=args.place_cell_rf,
        surround_scale=args.surround_scale,
        RNN_type=args.RNN_type,
        activation=args.activation,
        weight_decay=args.weight_decay,
        DoG=bool(args.DoG),
        periodic=bool(args.periodic),
        box_width=args.box_width,
        box_height=args.box_height,
        learning_rate=args.learning_rate,
        device=device,
        save_dir=os.path.dirname(args.checkpoint_path) or ".",
    )


def safe_nanargmax(arr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Return argmax + values per column, guarding all-NaN columns."""
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


def classify_units(
    lag_cm: np.ndarray,
    scores: np.ndarray,
    threshold_cm: float,
    min_gridness: float,
) -> Dict[str, np.ndarray]:
    """Assign unit indices to predictive, phase-precession, or phase-locked."""
    best_idx, best_vals = safe_nanargmax(scores)
    best_cm = np.full(best_idx.shape, np.nan)
    valid = best_idx >= 0
    best_cm[valid] = lag_cm[best_idx[valid]]

    qual = valid & np.isfinite(best_vals) & (best_vals >= min_gridness)
    predictive = qual & (best_cm >= threshold_cm)
    precession = qual & (best_cm <= -threshold_cm)
    locked = qual & ~(predictive | precession)

    return {
        "predictive": np.where(predictive)[0],
        "phase_precession": np.where(precession)[0],
        "phase_locked": np.where(locked)[0],
    }


def prepare_group_matrix(
    scores: np.ndarray,
    unit_indices: Iterable[int],
    sort_key: np.ndarray,
) -> np.ndarray:
    """Return a (#units, #lags) matrix sorted by the provided key."""
    idxs = np.array(list(unit_indices), dtype=int)
    if idxs.size == 0:
        return np.zeros((0, scores.shape[0]), dtype=float)
    order = np.argsort(sort_key[idxs])
    sorted_units = idxs[order]
    return scores[:, sorted_units].T


def plot_heatmaps_and_curves(
    lag_cm: np.ndarray,
    group_scores: Dict[str, np.ndarray],
    group_matrices: Dict[str, np.ndarray],
    colors: Dict[str, str],
    save_path: str = None,
) -> None:
    """Render the replicated figure."""
    titles = {
        "predictive": "Predictive grid cells",
        "phase_precession": "Phase-precession grid cells",
        "phase_locked": "Phase-locked grid cells",
    }

    fig, axes = plt.subplots(
        2,
        3,
        figsize=(10.5, 7.0),
        gridspec_kw={"height_ratios": [3.0, 1.2], "hspace": 0.32, "wspace": 0.28},
        sharex="col",
    )
    axes_top = axes[0]
    axes_bot = axes[1]

    vmax = max(np.nanmax(mat) for mat in group_matrices.values() if mat.size)
    vmin = min(np.nanmin(mat) for mat in group_matrices.values() if mat.size)
    vmax = 0.3 if not np.isfinite(vmax) else max(0.3, float(vmax))
    vmin = -0.1 if not np.isfinite(vmin) else min(-0.1, float(vmin))
    im_ref = None

    for ax_top, ax_bot, key in zip(axes_top, axes_bot, titles.keys()):
        mat = group_matrices[key]
        col = colors[key]
        title = titles[key]

        if mat.size:
            extent = (lag_cm[0], lag_cm[-1], 0, mat.shape[0])
            im_ref = ax_top.imshow(
                mat,
                cmap="jet",
                aspect="auto",
                origin="lower",
                extent=extent,
                vmin=vmin,
                vmax=vmax,
            )
        else:
            ax_top.imshow(np.zeros((2, 2)), cmap="jet", vmin=vmin, vmax=vmax)
        ax_top.set_title(title, fontsize=12, color=col)
        ax_top.axvline(0, color="k", lw=1.2)
        ax_top.set_yticks([0, mat.shape[0]]) if mat.size else ax_top.set_yticks([])

        curves = group_scores[key]
        if curves.size:
            counts = np.sum(np.isfinite(curves), axis=1)
            sums = np.nansum(curves, axis=1)
            mean = np.divide(sums, counts, out=np.full_like(sums, np.nan), where=counts > 0)
            centered = np.where(np.isfinite(curves), curves - mean[:, None], 0.0)
            denom = np.where(counts > 1, counts - 1, 1)
            var = np.divide(
                np.sum(centered**2, axis=1),
                denom,
                out=np.zeros_like(mean),
                where=counts > 1,
            )
            std = np.sqrt(var)
            sem = np.divide(std, np.sqrt(counts), out=np.zeros_like(std), where=counts > 0)
            ax_bot.plot(lag_cm, mean, color=col, lw=2.4)
            lower = mean - sem
            upper = mean + sem
            ax_bot.fill_between(lag_cm, lower, upper, color=col, alpha=0.25, linewidth=0)
        ax_bot.axhline(0, color="k", lw=0.8)
        ax_bot.axvline(0, color="k", lw=0.8)
        ax_bot.set_xlabel("Position shift (cm)", fontsize=11)
        ax_bot.set_xlim(lag_cm[0], lag_cm[-1])
        ax_bot.grid(alpha=0.25)

    axes_top[0].set_ylabel("Cell #", fontsize=11)
    axes_bot[0].set_ylabel("Gridness (60°)", fontsize=11)
    for ax in axes_top[1:]:
        ax.set_yticklabels([])
    for ax in axes_bot[1:]:
        ax.set_yticklabels([])

    if im_ref is not None:
        cbar = fig.colorbar(
            im_ref,
            ax=axes_top,
            fraction=0.025,
            pad=0.02,
        )
        cbar.set_label("Gridness (60°)", fontsize=11)

    fig.text(0.01, 0.97, "A", fontsize=14, fontweight="bold", ha="left", va="top")
    fig.text(0.01, 0.48, "B", fontsize=14, fontweight="bold", ha="left", va="top")

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
    else:
        plt.show()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint_path", required=True, help="Path to trained model .pth file.")
    parser.add_argument("--save_path", default=None, help="Optional output path for the figure.")
    parser.add_argument("--batch_size", default=100, type=int)
    parser.add_argument("--sequence_length", default=20, type=int)
    parser.add_argument("--Np", default=None, type=int, help="Override place cell count; inferred from checkpoint if omitted.")
    parser.add_argument("--Ng", default=None, type=int, help="Override hidden unit count; inferred from checkpoint if omitted.")
    parser.add_argument("--velocity_dim", default=None, type=int, help="Override velocity input dimension; inferred from checkpoint if omitted.")
    parser.add_argument("--place_cell_rf", default=0.12, type=float)
    parser.add_argument("--surround_scale", default=2.0, type=float)
    parser.add_argument("--RNN_type", default="RNN")
    parser.add_argument("--activation", default="relu")
    parser.add_argument("--weight_decay", default=1e-4, type=float)
    parser.add_argument("--DoG", default=True, type=bool)
    parser.add_argument("--periodic", default=False, type=bool)
    parser.add_argument("--box_width", default=2.2, type=float)
    parser.add_argument("--box_height", default=2.2, type=float)
    parser.add_argument("--learning_rate", default=1e-4, type=float)
    parser.add_argument("--device", default=None, help="Force cpu/cuda device; defaults to available GPU.")

    parser.add_argument("--res", default=20, type=int, help="Rate-map resolution used by the scorer.")
    parser.add_argument("--max_lag", default=20, type=int, help="Evaluate lags from -max_lag .. +max_lag (time steps).")
    parser.add_argument("--n_batches", default=40, type=int, help="Number of trajectory batches to aggregate.")
    parser.add_argument("--Ng_use", default=512, type=int, help="Number of units to analyze.")
    parser.add_argument("--gridness_threshold", default=0.2, type=float, help="Minimum gridness to include a unit.")
    parser.add_argument("--shift_threshold_cm", default=5.0, type=float, help="Shift (cm) separating predictive vs others.")

    args = parser.parse_args()

    if not os.path.exists(args.checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint_path}")

    def _extract_state_dict(obj):
        if isinstance(obj, dict):
            for key in ("state_dict", "model_state_dict", "model"):
                if key in obj and isinstance(obj[key], dict):
                    return obj[key]
        return obj

    raw_state = torch.load(args.checkpoint_path, map_location="cpu")
    state_dict = _extract_state_dict(raw_state)
    if not isinstance(state_dict, dict):
        raise TypeError(f"Unsupported checkpoint format: {type(raw_state)}")

    def _infer_dims(sd: Dict[str, torch.Tensor]) -> Tuple[int, int, int]:
        if "encoder.weight" not in sd or "decoder.weight" not in sd or "RNN.weight_ih_l0" not in sd:
            missing = [k for k in ("encoder.weight", "decoder.weight", "RNN.weight_ih_l0") if k not in sd]
            raise KeyError(f"Checkpoint missing required tensors: {missing}")
        enc = sd["encoder.weight"]
        dec = sd["decoder.weight"]
        rnn_ih = sd["RNN.weight_ih_l0"]
        Ng_enc, Np = enc.shape
        Np_dec, Ng_dec = dec.shape
        if Ng_enc != Ng_dec or Np != Np_dec:
            raise ValueError(f"Inconsistent encoder/decoder shapes: encoder {enc.shape}, decoder {dec.shape}")
        velocity_dim = rnn_ih.shape[1]
        return Ng_enc, Np, velocity_dim

    dims = _infer_dims(state_dict)
    options = build_options(args, dims)

    print(f"Inferred from checkpoint -> Ng: {options.Ng}, Np: {options.Np}, velocity_dim: {options.velocity_dim}")

    if options.velocity_dim != 2:
        raise NotImplementedError(
            f"Checkpoint expects velocity_dim={options.velocity_dim}. Provide a compatible trajectory generator "
            "before running predictive analysis."
        )

    place_cells = PlaceCells(options)
    model = RNN(options, place_cells).to(options.device)
    model.load_state_dict(state_dict)
    model.eval()

    traj_gen = TrajectoryGenerator(options, place_cells)
    lags = list(range(-args.max_lag, args.max_lag + 1))

    Ng_eval = min(args.Ng_use, options.Ng)

    scores_60, _, xs, ys, activations, _ = predictive_gridness_analysis(
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

    classes = classify_units(lag_cm, scores_60, args.shift_threshold_cm, args.gridness_threshold)

    best_idx, _ = safe_nanargmax(scores_60)
    best_cm = np.full(best_idx.shape, np.nan)
    mask_valid = best_idx >= 0
    best_cm[mask_valid] = lag_cm[best_idx[mask_valid]]

    group_matrices = {
        key: prepare_group_matrix(scores_60, idxs, best_cm) for key, idxs in classes.items()
    }
    group_scores = {
        key: scores_60[:, idxs] for key, idxs in classes.items()
    }
    colors = {
        "predictive": "#1f77b4",
        "phase_precession": "#d62728",
        "phase_locked": "#2ca02c",
    }

    print("Cells per class:")
    for key, idxs in classes.items():
        print(f"  {key}: {len(idxs)}")

    plot_heatmaps_and_curves(lag_cm, group_scores, group_matrices, colors, save_path=args.save_path)


if __name__ == "__main__":
    main()
