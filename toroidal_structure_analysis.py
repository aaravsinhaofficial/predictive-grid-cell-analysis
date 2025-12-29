#!/usr/bin/env python3
"""Toroidal manifold analysis with predictive/retrospective/normal ablations.

This script:
  1) extracts a torus parameterization from grid-cell rate maps,
  2) projects hidden activity onto toroidal coordinates,
  3) compares baseline vs predictive/retrospective/normal grid-cell ablations.

Outputs are written beside the checkpoint under analysis_outputs/torus/:
  - torus_comparison.png : 3D point clouds for each condition (baseline / PGC ablated / RGC ablated / normal ablated)
  - torus_metrics.json   : lattice vectors, unit counts, radius variation, phase-to-position decoding errors
"""

from __future__ import annotations

import argparse
import copy
import json
import os
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple, Optional

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch

from model import RNN
from place_cells import PlaceCells
from predictive_retrospective_ablation import classify_from_scores
from trajectory_generator import TrajectoryGenerator
from visualize import compute_ratemaps
from multi_seed_predictive_analysis import (
    build_options,
    collect_eval_batches,
    infer_dims_from_state,
    zero_unit_weights_in_place,
)

matplotlib.use("Agg")


# --------------------------------------------------------------------------------------
# Helper data structures
# --------------------------------------------------------------------------------------

@dataclass
class TorusBasis:
    k1: np.ndarray  # shape (2,), cycles / meter
    k2: np.ndarray  # shape (2,), cycles / meter
    angle_rad: float
    peak_power: float
    phase1: np.ndarray  # shape (Ng_subset,)
    phase2: np.ndarray  # shape (Ng_subset,)
    units: np.ndarray   # indices used to build the basis


@dataclass
class TorusProjection:
    theta1: np.ndarray  # shape (T,B)
    theta2: np.ndarray  # shape (T,B)
    r1: np.ndarray      # shape (T,B)
    r2: np.ndarray      # shape (T,B)
    coords3d: np.ndarray  # shape (T*B, 3)
    rmse_cm: float
    step_rmse_cm: float
    radius_cv: Dict[str, float]


# --------------------------------------------------------------------------------------
# Core computations
# --------------------------------------------------------------------------------------

def estimate_lattice_vectors(rate_maps: np.ndarray, grid_units: np.ndarray, box_width: float) -> Tuple[np.ndarray, np.ndarray, float, float]:
    """Estimate the fundamental lattice vectors (k1, k2) from grid-cell rate maps."""
    res = rate_maps.shape[1]
    if grid_units.size == 0:
        raise ValueError("No grid units provided for lattice estimation.")
    avg_map = rate_maps[grid_units].mean(axis=0)

    spectrum = np.fft.fftshift(np.fft.fft2(avg_map))
    power = np.abs(spectrum)

    # Suppress the DC component
    center = res // 2
    power[center - 1 : center + 2, center - 1 : center + 2] = 0

    peak_idx = np.unravel_index(np.argmax(power), power.shape)
    freqs = np.fft.fftshift(np.fft.fftfreq(res, d=box_width / res))
    kx = freqs[peak_idx[0]]
    ky = freqs[peak_idx[1]]
    k1 = np.array([kx, ky], dtype=float)
    k_mag = np.linalg.norm(k1)
    if k_mag == 0:
        raise ValueError("Estimated lattice vector has zero magnitude; check rate maps.")

    angle = np.arctan2(ky, kx)
    rot = angle + np.deg2rad(60.0)
    k2 = k_mag * np.array([np.cos(rot), np.sin(rot)], dtype=float)
    return k1, k2, angle, float(power[peak_idx])


def _phase_for_vector(rate_map: np.ndarray, k_vec: np.ndarray, xs: np.ndarray, ys: np.ndarray) -> float:
    """Compute the complex phase of a rate map at wavevector k."""
    phase_term = np.exp(-2j * np.pi * (k_vec[0] * xs + k_vec[1] * ys))
    coeff = np.sum(rate_map * phase_term)
    return float(np.angle(coeff))


def build_torus_basis(
    rate_maps: np.ndarray,
    grid_units: np.ndarray,
    box_width: float,
) -> TorusBasis:
    """Return torus basis (k1, k2, per-unit phases) from rate maps."""
    k1, k2, angle, peak_power = estimate_lattice_vectors(rate_maps, grid_units, box_width)
    res = rate_maps.shape[1]
    xs = np.linspace(-box_width / 2, box_width / 2, res)
    ys = np.linspace(-box_width / 2, box_width / 2, res)
    X, Y = np.meshgrid(xs, ys, indexing="ij")

    phase1 = []
    phase2 = []
    for u in grid_units:
        rm = rate_maps[u]
        phase1.append(_phase_for_vector(rm, k1, X, Y))
        phase2.append(_phase_for_vector(rm, k2, X, Y))

    return TorusBasis(
        k1=k1,
        k2=k2,
        angle_rad=angle,
        peak_power=peak_power,
        phase1=np.array(phase1, dtype=float),
        phase2=np.array(phase2, dtype=float),
        units=np.array(grid_units, dtype=int),
    )


def project_states_to_torus(states: np.ndarray, basis: TorusBasis, torus_radii: Tuple[float, float], T: int, B: int) -> TorusProjection:
    """Project hidden states onto toroidal coordinates."""
    # states: [T, B, Ng]; restrict to basis units
    flat = states[:, :, basis.units].reshape(-1, basis.units.size)
    v1 = np.cos(basis.phase1)
    w1 = np.sin(basis.phase1)
    v2 = np.cos(basis.phase2)
    w2 = np.sin(basis.phase2)

    p1 = flat @ v1
    q1 = flat @ w1
    p2 = flat @ v2
    q2 = flat @ w2

    theta1 = np.arctan2(q1, p1).reshape(T, B)
    theta2 = np.arctan2(q2, p2).reshape(T, B)
    r1 = np.sqrt(p1**2 + q1**2).reshape(T, B)
    r2 = np.sqrt(p2**2 + q2**2).reshape(T, B)

    R, r_minor = torus_radii
    coords = []
    # Use median radii to embed; variability shows up as spread around the torus
    r1_med = np.nanmedian(r1)
    r2_med = np.nanmedian(r2)
    scale_r1 = R / (r1_med + 1e-8)
    scale_r2 = r_minor / (r2_med + 1e-8)
    for th1, th2, rr1, rr2 in zip(theta1.reshape(-1), theta2.reshape(-1), r1.reshape(-1), r2.reshape(-1)):
        major = (rr1 * scale_r1) + R * 0.0  # center on R but keep amplitude spread
        minor = (rr2 * scale_r2)
        x = (R + minor * np.cos(th2)) * np.cos(th1)
        y = (R + minor * np.cos(th2)) * np.sin(th1)
        z = minor * np.sin(th2)
        coords.append((x, y, z))
    coords3d = np.array(coords, dtype=float)

    radius_cv = {
        "r1_cv": float(np.nanstd(r1) / (np.nanmean(r1) + 1e-8)),
        "r2_cv": float(np.nanstd(r2) / (np.nanmean(r2) + 1e-8)),
    }

    return TorusProjection(
        theta1=theta1,
        theta2=theta2,
        r1=r1,
        r2=r2,
        coords3d=coords3d,
        rmse_cm=np.nan,
        step_rmse_cm=np.nan,
        radius_cv=radius_cv,
    )


def decode_positions_from_phases(theta1: np.ndarray, theta2: np.ndarray, positions: np.ndarray, k1: np.ndarray, k2: np.ndarray) -> Tuple[float, float]:
    """Decode positions from torus phases and compute RMSE (cm)."""
    T, B = theta1.shape
    K = 2 * np.pi * np.array([[k1[0], k1[1]], [k2[0], k2[1]]], dtype=float)
    K_inv = np.linalg.inv(K)

    rmse_list = []
    step_rmse_list = []
    for b in range(B):
        pos = positions[:, b, :]  # meters
        t1 = np.unwrap(theta1[:, b])
        t2 = np.unwrap(theta2[:, b])
        dtheta = np.stack([np.diff(t1), np.diff(t2)], axis=1)
        delta_pos_hat = dtheta @ K_inv.T  # meters per step
        pos_hat = np.zeros_like(pos)
        pos_hat[0] = pos[0]
        pos_hat[1:] = pos_hat[0] + np.cumsum(delta_pos_hat, axis=0)

        rmse = np.sqrt(np.mean(np.sum((pos_hat - pos) ** 2, axis=1)))
        step_rmse = np.sqrt(np.mean(np.sum((delta_pos_hat - np.diff(pos, axis=0)) ** 2, axis=1)))
        rmse_list.append(rmse)
        step_rmse_list.append(step_rmse)

    return float(np.mean(rmse_list) * 100.0), float(np.mean(step_rmse_list) * 100.0)


def run_condition(
    base_state: Dict[str, torch.Tensor],
    basis: TorusBasis,
    options,
    place_cells: PlaceCells,
    traj_gen: TrajectoryGenerator,
    cached_batches,
    device: torch.device,
    label: str,
    ablate_units: Sequence[int] | None,
    torus_radii: Tuple[float, float],
) -> TorusProjection:
    """Compute torus projection (and decoding error) for one ablation condition."""
    model = RNN(options, place_cells).to(device)
    model.load_state_dict(copy.deepcopy(base_state))
    if ablate_units:
        zero_unit_weights_in_place(model, ablate_units)
    model.eval()

    g_list = []
    pos_list = []
    with torch.no_grad():
        for (inputs, pos_batch) in cached_batches:
            v, init = inputs
            v = v.to(device)
            init = init.to(device)
            g = model.g((v, init)).cpu().numpy()  # [T,B,Ng]
            pos_np = pos_batch.cpu().numpy()
            g_list.append(g)
            pos_list.append(pos_np)
    g_all = np.concatenate(g_list, axis=1)
    pos_all = np.concatenate(pos_list, axis=1)
    T, B, _ = g_all.shape

    proj = project_states_to_torus(g_all, basis, torus_radii, T, B)
    rmse_cm, step_rmse_cm = decode_positions_from_phases(proj.theta1, proj.theta2, pos_all, basis.k1, basis.k2)
    proj.rmse_cm = rmse_cm
    proj.step_rmse_cm = step_rmse_cm
    return proj


# --------------------------------------------------------------------------------------
# Plotting
# --------------------------------------------------------------------------------------

def _subsample(coords: np.ndarray, colors: Optional[np.ndarray], sample: int, seed: int = 0):
    if coords.shape[0] > sample:
        rng = np.random.default_rng(seed)
        sel = rng.choice(coords.shape[0], size=sample, replace=False)
        coords = coords[sel]
        if colors is not None:
            colors = colors[sel]
    return coords, colors


def plot_torus_comparison(
    projections: Dict[str, TorusProjection],
    positions: np.ndarray,
    save_path: str,
    sample: int = 8000,
    alt_save_path: str | None = None,
) -> None:
    """Plot 3D torus embeddings for each condition (single view + multiview grid)."""
    titles = list(projections.keys())
    cmap = plt.cm.get_cmap("viridis")
    flat_pos = positions.reshape(-1, 2)
    x_norm = (flat_pos[:, 0] - flat_pos[:, 0].min()) / (flat_pos[:, 0].ptp() + 1e-8)

    # Primary view
    fig = plt.figure(figsize=(14, 4.5))
    for idx, name in enumerate(titles):
        proj = projections[name]
        coords, color = _subsample(proj.coords3d, x_norm, sample)
        ax = fig.add_subplot(1, len(titles), idx + 1, projection="3d")
        ax.scatter(coords[:, 0], coords[:, 1], coords[:, 2], c=cmap(color), s=4, alpha=0.65)
        ax.set_title(f"{name}\nRMSE {proj.rmse_cm:.2f} cm | step {proj.step_rmse_cm:.2f} cm")
        ax.set_axis_off()
        ax.view_init(elev=20, azim=35)
    fig.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=220, bbox_inches="tight")
    plt.close(fig)

    # Multi-view grid (optional)
    if alt_save_path:
        views = [(20, 35), (20, 120), (60, 35), (70, 200)]
        fig_mv, axes = plt.subplots(len(titles), len(views), figsize=(4 * len(views), 3.5 * len(titles)), subplot_kw={"projection": "3d"})
        if len(titles) == 1:
            axes = np.array([axes])
        for i, name in enumerate(titles):
            proj = projections[name]
            coords, color = _subsample(proj.coords3d, x_norm, sample, seed=1 + i)
            for j, (elev, azim) in enumerate(views):
                ax = axes[i, j]
                ax.scatter(coords[:, 0], coords[:, 1], coords[:, 2], c=cmap(color), s=3, alpha=0.6)
                ax.set_axis_off()
                ax.view_init(elev=elev, azim=azim)
                if j == 0:
                    ax.set_title(name, fontsize=10)
        fig_mv.tight_layout()
        fig_mv.savefig(alt_save_path, dpi=220, bbox_inches="tight")
        plt.close(fig_mv)


def plot_torus_seaborn(projections: Dict[str, TorusProjection], save_path: str, sample: int = 10000) -> None:
    """2-D seaborn view of the torus using scatter + KDE."""
    sns.set_style("white")
    names = list(projections.keys())
    fig, axes = plt.subplots(len(names), 2, figsize=(10, 3.2 * len(names)))
    if len(names) == 1:
        axes = np.array([axes])
    for i, name in enumerate(names):
        proj = projections[name]
        coords, _ = _subsample(proj.coords3d, None, sample, seed=3 + i)
        df = pd.DataFrame({"x": coords[:, 0], "y": coords[:, 1], "z": coords[:, 2]})
        sns.scatterplot(ax=axes[i, 0], data=df, x="x", y="y", hue="z", palette="viridis", s=5, linewidth=0, legend=False)
        axes[i, 0].set_title(f"{name}: XY scatter (colored by z)")
        sns.kdeplot(ax=axes[i, 1], data=df, x="x", y="y", fill=True, cmap="mako")
        axes[i, 1].set_title(f"{name}: XY density")
        for ax in axes[i]:
            ax.set_xticks([])
            ax.set_yticks([])
    fig.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=220)
    plt.close(fig)


def compute_ring_metrics(coords: np.ndarray, sample: int = 12000, seed: int = 42):
    """Return statistics describing how ring-like a point cloud is."""
    if coords.shape[0] > sample:
        rng = np.random.default_rng(seed)
        coords = coords[rng.choice(coords.shape[0], size=sample, replace=False)]
    rho = np.sqrt(coords[:, 0] ** 2 + coords[:, 1] ** 2)
    R = float(np.mean(rho))
    delta = rho - R
    minor = np.sqrt(delta ** 2 + coords[:, 2] ** 2)
    stats = {
        "major_radius_mean": R,
        "major_radius_std": float(np.std(rho)),
        "major_radius_cv": float(np.std(rho) / (R + 1e-8)),
        "minor_radius_mean": float(np.mean(minor)),
        "minor_radius_std": float(np.std(minor)),
        "minor_radius_cv": float(np.std(minor) / (np.mean(minor) + 1e-8)),
    }
    return stats, rho, minor


def plot_ring_diagnostics(samples: Dict[str, Tuple[np.ndarray, np.ndarray]], save_path: str) -> None:
    """Plot histograms of major/minor radii to visualize torus tightness."""
    names = list(samples.keys())
    fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=False)
    for name, (rho, minor) in samples.items():
        axes[0].hist(rho, bins=60, alpha=0.4, label=name, density=True)
        axes[1].hist(minor, bins=60, alpha=0.4, label=name, density=True)
    axes[0].set_title("Distribution of major radius ρ = √(x² + y²)")
    axes[0].set_xlabel("ρ")
    axes[0].set_ylabel("Density")
    axes[0].legend(frameon=False)
    axes[1].set_title("Distribution of minor radius = √((ρ - R)² + z²)")
    axes[1].set_xlabel("Minor radius")
    axes[1].set_ylabel("Density")
    fig.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=200)
    plt.close(fig)


# --------------------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------------------

def load_gridness_data(ckpt_path: str) -> Dict[str, np.ndarray]:
    out_dir = os.path.join(os.path.dirname(ckpt_path), "analysis_outputs")
    path = os.path.join(out_dir, "gridness_data.npz")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing gridness_data.npz beside checkpoint: {path}. Run multi_seed_predictive_analysis.py first.")
    data = np.load(path)
    return {k: data[k] for k in data.files}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint_path", required=True, help="Path to a trained RNN checkpoint (.pth).")
    parser.add_argument("--batch_size", type=int, default=80)
    parser.add_argument("--sequence_length", type=int, default=20)
    parser.add_argument("--place_cell_rf", type=float, default=0.12)
    parser.add_argument("--surround_scale", type=float, default=2.0)
    parser.add_argument("--activation", default="relu")
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--box_width", type=float, default=2.2)
    parser.add_argument("--box_height", type=float, default=2.2)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--n_batches", type=int, default=12, help="Number of trajectory batches to project.")
    parser.add_argument("--Ng_use", type=int, default=512, help="Number of units to analyse (subset of Ng).")
    parser.add_argument("--gridness_threshold", type=float, default=0.5, help="Minimum peak gridness to treat as a grid cell.")
    parser.add_argument("--min_shift_cm", type=float, default=5.0, help="Minimum shift for predictive/retrospective classification.")
    parser.add_argument("--res", type=int, default=40, help="Ratemap resolution for phase estimation.")
    parser.add_argument("--device", default=None, help="cpu or cuda. Defaults to best available.")
    parser.add_argument("--traj_speed_scale", default=1.0, type=float, help="Multiplier on the Rayleigh speed scale.")
    parser.add_argument("--traj_speed_max", default=None, type=float, help="Optional cap on forward speed (m/s).")
    parser.add_argument("--traj_velocity_smoothing", default=0.0, type=float, help="EMA factor (0-0.99) applied to speeds.")
    parser.add_argument("--traj_turn_sigma_scale", default=1.0, type=float, help="Multiplier on turning noise sigma.")
    parser.add_argument("--traj_border_region", default=0.03, type=float, help="Wall avoidance distance (m).")
    parser.add_argument("--traj_wall_slowdown", default=0.25, type=float, help="Speed multiplier near walls.")
    parser.add_argument("--traj_wall_turn_scale", default=1.0, type=float, help="Multiplier on wall-turn correction.")
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    raw = torch.load(args.checkpoint_path, map_location="cpu")
    state = raw["state_dict"] if isinstance(raw, dict) and "state_dict" in raw else raw
    Ng, Np, velocity_dim = infer_dims_from_state(state)
    if args.Ng_use > Ng:
        args.Ng_use = Ng

    options = build_options(args, (Ng, Np, velocity_dim), device, args.checkpoint_path)
    place_cells = PlaceCells(options)
    traj_gen = TrajectoryGenerator(options, place_cells)

    grid_data = load_gridness_data(args.checkpoint_path)
    # Grid-like units for basis (best gridness >= threshold)
    best_idx = np.nanargmax(grid_data["scores_60"], axis=0)
    best_vals = grid_data["scores_60"][best_idx, np.arange(grid_data["scores_60"].shape[1])]
    grid_units_global = np.where(best_vals >= args.gridness_threshold)[0]
    if grid_units_global.size == 0:
        raise RuntimeError("No units pass the gridness threshold; lower --gridness_threshold.")
    basis_units_global = grid_units_global[: args.Ng_use]

    # Predictive / retrospective unit lists for ablation
    classes = classify_from_scores(grid_data, args.min_shift_cm, args.gridness_threshold)
    predictive_units = classes["predictive"].tolist()
    retrospective_units = classes["retrospective"].tolist()
    normal_units = classes.get("normal", np.array([], dtype=int)).tolist()

    # Rate maps (baseline) for the basis units
    model = RNN(options, place_cells).to(device)
    model.load_state_dict(state)
    model.eval()
    idxs = np.array(basis_units_global, dtype=int)
    Ng_subset = idxs.size
    rate_maps, _, _, _ = compute_ratemaps(model, traj_gen, options, res=args.res, n_avg=max(8, args.n_batches), Ng=Ng_subset, idxs=idxs)
    rate_maps = rate_maps.astype(float)

    basis = build_torus_basis(rate_maps, np.arange(Ng_subset), options.box_width)
    # Replace local unit labels with the original indices so projections slice the right neurons
    basis.units = idxs

    # Cache trajectories for fair comparisons
    cached_batches = collect_eval_batches(traj_gen, args.n_batches)
    if not cached_batches:
        raise RuntimeError("No batches collected; increase --n_batches.")

    # Set torus radii from baseline grid amplitudes
    torus_R = 1.0
    torus_r = 0.35
    torus_radii = (torus_R, torus_r)

    projections: Dict[str, TorusProjection] = {}
    projections["Baseline"] = run_condition(
        state, basis, options, place_cells, traj_gen, cached_batches, device, "baseline", [], torus_radii
    )
    projections["Predictive ablated"] = run_condition(
        state, basis, options, place_cells, traj_gen, cached_batches, device, "predictive", predictive_units, torus_radii
    )
    projections["Retrospective ablated"] = run_condition(
        state, basis, options, place_cells, traj_gen, cached_batches, device, "retrospective", retrospective_units, torus_radii
    )
    projections["Normal grid ablated"] = run_condition(
        state, basis, options, place_cells, traj_gen, cached_batches, device, "normal", normal_units, torus_radii
    )

    # Plot
    out_dir = os.path.join(os.path.dirname(args.checkpoint_path), "analysis_outputs", "torus")
    os.makedirs(out_dir, exist_ok=True)
    positions_baseline = np.concatenate([p for _, p in cached_batches], axis=1)
    plot_torus_comparison(
        projections,
        positions_baseline,
        os.path.join(out_dir, "torus_comparison.png"),
        alt_save_path=os.path.join(out_dir, "torus_multiview.png"),
    )
    plot_torus_seaborn(projections, os.path.join(out_dir, "torus_seaborn.png"))

    ring_samples: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    ring_stats: Dict[str, Dict[str, float]] = {}
    for name, proj in projections.items():
        stats, rho, minor = compute_ring_metrics(proj.coords3d)
        ring_stats[name] = stats
        ring_samples[name] = (rho, minor)
    plot_ring_diagnostics(ring_samples, os.path.join(out_dir, "torus_ring_metrics.png"))

    # Metrics
    metrics = {
        "k1_cycles_per_m": basis.k1.tolist(),
        "k2_cycles_per_m": basis.k2.tolist(),
        "peak_angle_rad": basis.angle_rad,
        "peak_power": basis.peak_power,
        "grid_units_used": int(len(basis_units_global)),
        "predictive_units": len(predictive_units),
        "retrospective_units": len(retrospective_units),
        "normal_units": len(normal_units),
    }
    for name, proj in projections.items():
        metrics[name] = {
            "rmse_cm": proj.rmse_cm,
            "step_rmse_cm": proj.step_rmse_cm,
            "radius_cv": proj.radius_cv,
        }
        metrics[name].update(ring_stats.get(name, {}))
    with open(os.path.join(out_dir, "torus_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"[toroidal_structure_analysis] Saved figures and metrics to {out_dir}")


if __name__ == "__main__":
    main()
