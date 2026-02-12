#!/usr/bin/env python3
"""Recreate Fig. 1b-style toroidal embeddings for all single-agent seeds.

For each seed, the script:
  - samples trajectories from the trained RNN,
  - projects grid-cell population states onto a torus coordinate system (3-D),
  - overlays baseline + ablation trajectories on the baseline cloud,
  - computes k-NN distance to the baseline cloud over time.

Outputs are written under analysis_outputs/<model>/<seed>/<run>/torus/:
  - fig1b_baseline_cloud_dualtraj_full_straight.png
  - fig1b_offmanifold_distance_full_straight.png
"""

from __future__ import annotations

import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

# Keep numba/matplotlib caches writable inside the repo before importing heavy deps.
os.environ.setdefault("NUMBA_CACHE_DIR", str(Path(".numba_cache").resolve()))
os.environ.setdefault("MPLCONFIGDIR", str(Path(".mplconfig").resolve()))
Path(os.environ["NUMBA_CACHE_DIR"]).mkdir(parents=True, exist_ok=True)
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize, TwoSlopeNorm
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d.art3d import Line3DCollection
import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from model import RNN
from place_cells import PlaceCells
from trajectory_generator import TrajectoryGenerator
from predictive_retrospective_ablation import classify_from_scores
from multi_seed_predictive_analysis import zero_unit_weights_in_place
from single_seed_torus_distance import build_options
from path_utils import analysis_dir_for_checkpoint
from visualize import compute_ratemaps


# --------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------

@dataclass
class EmbeddingInputs:
    states: np.ndarray  # shape (T, B, K)
    positions: np.ndarray  # shape (T, B, 2)
    dt: float
    grid_units_used: np.ndarray


@dataclass
class TorusBasis:
    k1: np.ndarray
    k2: np.ndarray
    angle_rad: float
    peak_power: float
    phase1: np.ndarray
    phase2: np.ndarray
    units: np.ndarray


@dataclass
class OverlaySpec:
    coords: np.ndarray
    times: np.ndarray
    label: str
    cmap: object | None = None
    color: str | None = None
    shadow_color: str | None = "black"
    glow_color: str | None = "white"
    shadow_width: float = 5.5
    glow_width: float = 3.8
    line_width: float = 2.6
    alpha: float = 0.95
    point_size: float = 22.0
    point_edge_color: str = "black"
    point_edge_width: float = 0.4
    start_marker: str = "o"
    start_color: str = "white"
    start_size: float = 60.0
    end_marker: str = "s"
    end_color: str = "#ffcc00"
    end_size: float = 60.0
    legend_color: str | None = None


def find_checkpoints(base: Path) -> List[Tuple[int, Path]]:
    """Locate single-agent checkpoints and return [(seed, path)]."""
    ckpts: List[Tuple[int, Path]] = []
    for ckpt in base.rglob("final_model.pth"):
        if "analysis_outputs" in str(ckpt):
            continue
        match = re.search(r"Seed\s+(\d+)", str(ckpt))
        seed = int(match.group(1)) if match else -1
        ckpts.append((seed, ckpt))
    ckpts.sort(key=lambda x: x[0])
    return ckpts


def load_grid_units(
    ckpt_path: Path,
    gridness_threshold: float,
    max_units: int,
    min_shift_cm: float,
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """Return (grid units, class indices) based on cached gridness analysis."""
    grid_path = analysis_dir_for_checkpoint(ckpt_path) / "gridness_data.npz"
    if not grid_path.exists():
        raise FileNotFoundError(f"Missing gridness_data.npz beside checkpoint: {grid_path}")
    data = np.load(grid_path)
    grid_data: Dict[str, np.ndarray] = {k: data[k] for k in data.files}

    scores = np.asarray(grid_data["scores_60"], dtype=float)
    with np.errstate(all="ignore"):
        best_vals = np.nanmax(scores, axis=0)
    grid_mask = np.isfinite(best_vals) & (best_vals >= gridness_threshold)
    candidates = np.where(grid_mask)[0]
    if candidates.size == 0:
        candidates = np.arange(scores.shape[1])
    order = np.argsort(best_vals[candidates])[::-1]
    grid_units = candidates[order][:max_units]

    classes = classify_from_scores(grid_data, min_shift_cm, gridness_threshold)
    class_indices = {
        "predictive": np.asarray(classes.get("predictive", []), dtype=int),
        "retrospective": np.asarray(classes.get("retrospective", []), dtype=int),
        "normal": np.asarray(classes.get("normal", []), dtype=int),
    }
    return grid_units, class_indices


def estimate_lattice_vectors(rate_maps: np.ndarray, grid_units: np.ndarray, box_width: float) -> Tuple[np.ndarray, np.ndarray, float, float]:
    """Estimate lattice vectors (k1, k2) from grid-cell rate maps."""
    res = rate_maps.shape[1]
    if grid_units.size == 0:
        raise ValueError("No grid units provided for lattice estimation.")
    avg_map = rate_maps[grid_units].mean(axis=0)

    spectrum = np.fft.fftshift(np.fft.fft2(avg_map))
    power = np.abs(spectrum)

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


def build_torus_basis(rate_maps: np.ndarray, grid_units: np.ndarray, box_width: float) -> TorusBasis:
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


def project_states_to_torus(states: np.ndarray, basis: TorusBasis, torus_radii: Tuple[float, float]) -> np.ndarray:
    """Project hidden states onto explicit torus coordinates."""
    flat = states[:, :, basis.units].reshape(-1, basis.units.size)
    v1 = np.cos(basis.phase1)
    w1 = np.sin(basis.phase1)
    v2 = np.cos(basis.phase2)
    w2 = np.sin(basis.phase2)

    p1 = flat @ v1
    q1 = flat @ w1
    p2 = flat @ v2
    q2 = flat @ w2

    theta1 = np.arctan2(q1, p1)
    theta2 = np.arctan2(q2, p2)
    r1 = np.sqrt(p1**2 + q1**2)
    r2 = np.sqrt(p2**2 + q2**2)

    R, r_minor = torus_radii
    r1_med = np.nanmedian(r1)
    r2_med = np.nanmedian(r2)
    scale_r1 = R / (r1_med + 1e-8)
    scale_r2 = r_minor / (r2_med + 1e-8)
    major = (r1 * scale_r1) + R * 0.0
    minor = (r2 * scale_r2)

    x = (R + minor * np.cos(theta2)) * np.cos(theta1)
    y = (R + minor * np.cos(theta2)) * np.sin(theta1)
    z = minor * np.sin(theta2)
    coords3d = np.stack([x, y, z], axis=1)
    return coords3d


def off_torus_distance(coords: np.ndarray, torus_radii: Tuple[float, float]) -> np.ndarray:
    """Return per-point deviation from a fixed torus surface."""
    R, r = torus_radii
    rho = np.sqrt(coords[:, 0] ** 2 + coords[:, 1] ** 2)
    minor = np.sqrt((rho - R) ** 2 + coords[:, 2] ** 2)
    return np.abs(minor - r)


def compute_torus_basis(
    model: RNN,
    traj_gen: TrajectoryGenerator,
    options,
    grid_units: np.ndarray,
    res: int = 40,
    n_avg: int = 12,
) -> TorusBasis:
    """Estimate a torus basis using grid-unit rate maps."""
    grid_units = np.asarray(grid_units, dtype=int)
    if grid_units.size == 0:
        raise ValueError("No grid units available for torus basis.")
    activations, _, _, _ = compute_ratemaps(
        model,
        traj_gen,
        options,
        res=res,
        n_avg=n_avg,
        Ng=grid_units.size,
        idxs=grid_units,
    )
    rate_maps = activations.astype(float)
    return build_torus_basis(rate_maps, np.arange(grid_units.size), options.box_width)


def collect_batches(traj_gen: TrajectoryGenerator, n_batches: int):
    """Cache trajectories so baseline and ablated runs see identical inputs."""
    batches = []
    for _ in range(max(1, n_batches)):
        inputs, pos, _ = traj_gen.get_test_batch()
        v, init = inputs
        batches.append(((v.cpu().clone(), init.cpu().clone()), pos.cpu().clone()))
    return batches


def collect_straight_batches(
    options,
    place_cells: PlaceCells,
    batch_size: int,
    n_batches: int,
    rng: np.random.Generator,
    speed_mps: float = 0.25,
) -> List[Tuple[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]]:
    """Create straight-line trajectories with fixed step size."""
    dt = float(getattr(options, "trajectory_dt", 0.02))
    T = int(options.sequence_length)
    min_dim = float(min(options.box_width, options.box_height))
    max_step = 0.5 * min_dim / max(1, T - 1)
    step = min(speed_mps * dt, max_step)

    batches = []
    for _ in range(max(1, n_batches)):
        headings = rng.uniform(0.0, 2 * np.pi, size=batch_size)
        dx = step * np.cos(headings)
        dy = step * np.sin(headings)
        total_dx = dx * (T - 1)
        total_dy = dy * (T - 1)

        x_min = -options.box_width / 2
        x_max = options.box_width / 2
        y_min = -options.box_height / 2
        y_max = options.box_height / 2

        start_x_min = np.where(total_dx >= 0, x_min, x_min - total_dx)
        start_x_max = np.where(total_dx >= 0, x_max - total_dx, x_max)
        start_y_min = np.where(total_dy >= 0, y_min, y_min - total_dy)
        start_y_max = np.where(total_dy >= 0, y_max - total_dy, y_max)

        start_x = rng.uniform(start_x_min, start_x_max)
        start_y = rng.uniform(start_y_min, start_y_max)
        start = np.stack([start_x, start_y], axis=-1)

        steps = (np.arange(T, dtype=np.float32) + 1.0)[:, None]
        disp = np.stack([dx, dy], axis=-1)[None, :, :]
        pos = start[None, :, :] + steps[:, None, :] * disp

        v = np.repeat(disp, T, axis=0)
        v_t = torch.tensor(v, dtype=torch.float32)
        pos_t = torch.tensor(pos, dtype=torch.float32)

        init_pos = torch.tensor(start[:, None, :], dtype=torch.float32)
        init_actv = place_cells.get_activation(init_pos).squeeze()
        batches.append(((v_t, init_actv), pos_t))
    return batches


def collect_activity_batches(
    model: RNN,
    traj_gen: TrajectoryGenerator,
    device: torch.device,
    grid_units: np.ndarray,
    n_batches: int,
    time_stride: int = 1,
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """Collect grid-unit activity + positions from multiple batches."""
    g_list = []
    pos_list = []
    grid_units = np.asarray(grid_units, dtype=int)
    model.eval()
    with torch.no_grad():
        for _ in range(max(1, n_batches)):
            inputs, pos, _ = traj_gen.get_test_batch()
            v, init = inputs
            v = v.to(device)
            init = init.to(device)
            pos = pos.to(device)
            g = model.g((v, init)).cpu().numpy()[:, :, grid_units]
            pos_np = pos.cpu().numpy()
            if time_stride > 1:
                g = g[::time_stride]
                pos_np = pos_np[::time_stride]
            g_list.append(g)
            pos_list.append(pos_np)
    return g_list, pos_list


def compute_knn_distances(cloud_z: np.ndarray, query_z: np.ndarray, k: int = 5) -> np.ndarray:
    """Compute mean k-NN distance from query points to a cloud."""
    nn = NearestNeighbors(n_neighbors=k, algorithm="auto")
    nn.fit(cloud_z)
    distances, _ = nn.kneighbors(query_z, return_distance=True)
    return distances.mean(axis=1)


def sample_random_units(
    rng: np.random.Generator,
    candidates: np.ndarray,
    count: int,
) -> np.ndarray:
    """Sample random unit indices from candidate set."""
    candidates = np.asarray(candidates, dtype=int)
    if count <= 0 or candidates.size == 0:
        return np.array([], dtype=int)
    count = min(count, candidates.size)
    return rng.choice(candidates, size=count, replace=False)


def run_condition(
    base_state: Dict[str, torch.Tensor],
    options,
    place_cells: PlaceCells,
    eval_batches,
    device: torch.device,
    ablate_units: Sequence[int] | None,
    grid_units: np.ndarray,
) -> Tuple[EmbeddingInputs, List[np.ndarray], List[np.ndarray]]:
    """Forward trajectories through the network (optionally ablated) and slice grid units."""
    model = RNN(options, place_cells).to(device)
    model.load_state_dict(base_state)
    if ablate_units is not None and len(ablate_units) > 0:
        zero_unit_weights_in_place(model, ablate_units)
    model.eval()

    g_list = []
    pos_list = []
    with torch.no_grad():
        for (v, init), pos in eval_batches:
            v = v.to(device)
            init = init.to(device)
            pos = pos.to(device)
            g = model.g((v, init)).cpu().numpy()  # [T, B, Ng]
            pos_np = pos.cpu().numpy()  # [T, B, 2]
            g_list.append(g)
            pos_list.append(pos_np)
    states_full = np.concatenate(g_list, axis=1)
    pos_full = np.concatenate(pos_list, axis=1)

    grid_units = np.asarray(grid_units, dtype=int)
    grid_units = grid_units[(grid_units >= 0) & (grid_units < states_full.shape[2])]
    if grid_units.size == 0:
        grid_units = np.arange(min(256, states_full.shape[2]), dtype=int)

    states = states_full[:, :, grid_units]
    dt = float(getattr(options, "trajectory_dt", 0.02))
    return EmbeddingInputs(states=states, positions=pos_full, dt=dt, grid_units_used=grid_units), g_list, pos_list


def select_divergent_path(
    baseline_batches: List[np.ndarray],
    ablated_batches: List[np.ndarray],
    pos_batches: List[np.ndarray],
    basis: TorusBasis,
    torus_radii: Tuple[float, float],
    dt: float,
    overlay_seconds: float,
    complexity_percentile: float = 60.0,
    complexity_penalty: float = 1.2,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pick a smooth trajectory with maximal divergence between baseline and ablated torus paths."""
    best_score = -np.inf
    best = None
    overlay_steps = int(math.floor(overlay_seconds / dt))

    def path_complexity(path_xy: np.ndarray) -> Tuple[float, float]:
        deltas = np.diff(path_xy, axis=0)
        step_len = np.linalg.norm(deltas, axis=1)
        path_len = float(np.sum(step_len))
        straight = float(np.linalg.norm(path_xy[-1] - path_xy[0]))
        tortuosity = path_len / (straight + 1e-6)
        if deltas.shape[0] >= 2:
            v1 = deltas[:-1]
            v2 = deltas[1:]
            denom = (np.linalg.norm(v1, axis=1) * np.linalg.norm(v2, axis=1)) + 1e-8
            cos = np.sum(v1 * v2, axis=1) / denom
            cos = np.clip(cos, -1.0, 1.0)
            mean_turn = float(np.mean(np.arccos(cos)))
        else:
            mean_turn = 0.0
        return tortuosity, mean_turn

    for g_base, g_ab, pos in zip(baseline_batches, ablated_batches, pos_batches):
        steps = min(overlay_steps, g_base.shape[0])
        if steps <= 1:
            continue
        base_slice = g_base[:steps]
        ab_slice = g_ab[:steps]
        pos_slice = pos[:steps]

        coords_base = project_states_to_torus(base_slice, basis, torus_radii).reshape(steps, -1, 3)
        coords_ab = project_states_to_torus(ab_slice, basis, torus_radii).reshape(steps, -1, 3)

        diff = np.linalg.norm(coords_base - coords_ab, axis=2)  # [T, B]
        mean_diff = diff.mean(axis=0)

        off_base = off_torus_distance(coords_base.reshape(-1, 3), torus_radii).reshape(steps, -1).mean(axis=0)
        off_ab = off_torus_distance(coords_ab.reshape(-1, 3), torus_radii).reshape(steps, -1).mean(axis=0)
        score = mean_diff + 0.75 * np.maximum(0.0, off_ab - off_base)

        tort = np.zeros(score.shape, dtype=float)
        turn = np.zeros(score.shape, dtype=float)
        for b in range(pos_slice.shape[1]):
            tort[b], turn[b] = path_complexity(pos_slice[:, b])
        complexity = 0.7 * (tort - 1.0) + 0.3 * (turn / np.pi)
        if np.isfinite(complexity).any():
            thresh = np.nanpercentile(complexity, complexity_percentile)
            candidates = np.where(complexity <= thresh)[0]
        else:
            candidates = np.arange(score.shape[0])
        if candidates.size == 0:
            candidates = np.arange(score.shape[0])

        score_adj = score - complexity_penalty * complexity
        idx = int(candidates[np.argmax(score_adj[candidates])])
        if score_adj[idx] > best_score:
            best_score = float(score_adj[idx])
            best = (base_slice[:, idx], ab_slice[:, idx], pos_slice[:, idx])

    if best is None:
        raise RuntimeError("No valid trajectory found for divergence search.")
    return best


def select_random_path(
    baseline_batches: List[np.ndarray],
    ablated_batches: List[np.ndarray],
    pos_batches: List[np.ndarray],
    dt: float,
    overlay_seconds: float,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pick a random trajectory shared across baseline and ablated conditions."""
    overlay_steps = int(math.floor(overlay_seconds / dt))
    valid_indices = [
        idx
        for idx, g_base in enumerate(baseline_batches)
        if g_base.shape[0] > 1
    ]
    if not valid_indices:
        raise RuntimeError("No valid trajectory batches available for random selection.")
    batch_idx = int(rng.choice(valid_indices))
    g_base = baseline_batches[batch_idx]
    g_ab = ablated_batches[batch_idx]
    pos = pos_batches[batch_idx]
    steps = min(overlay_steps, g_base.shape[0])
    if steps <= 1:
        raise RuntimeError("Selected trajectory is too short for overlay.")
    b_idx = int(rng.integers(0, g_base.shape[1]))
    return g_base[:steps, b_idx], g_ab[:steps, b_idx], pos[:steps, b_idx]


def choose_sample_indices(total_points: int, overlay_steps: int, total_B: int, rng: np.random.Generator, max_points: int) -> Tuple[np.ndarray, np.ndarray]:
    """Pick indices for embedding (overlay + random sample)."""
    overlay_idx = (np.arange(overlay_steps, dtype=int) * total_B)
    overlay_idx = overlay_idx[overlay_idx < total_points]

    if total_points <= max_points:
        return np.arange(total_points, dtype=int), overlay_idx

    remaining = np.setdiff1d(np.arange(total_points, dtype=int), overlay_idx, assume_unique=False)
    extra_needed = max_points - overlay_idx.size
    if extra_needed > 0 and remaining.size > 0:
        extra = rng.choice(remaining, size=min(extra_needed, remaining.size), replace=False)
        sample_idx = np.unique(np.concatenate([overlay_idx, extra]))
    else:
        sample_idx = overlay_idx
    return sample_idx, overlay_idx


def embed_states(
    embedding_inputs: EmbeddingInputs,
    sample_size: int,
    rng: np.random.Generator,
    basis: TorusBasis,
    torus_radii: Tuple[float, float],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute torus coordinates + PC1 colors for a point-cloud scatter."""
    T, B, K = embedding_inputs.states.shape
    flat_states = embedding_inputs.states.reshape(-1, K)
    flat_pos = embedding_inputs.positions.reshape(-1, 2)

    sample_idx = rng.choice(flat_states.shape[0], size=min(sample_size, flat_states.shape[0]), replace=False)
    sample_states = flat_states[sample_idx]
    sample_pos = flat_pos[sample_idx]

    scaler = StandardScaler()
    sample_scaled = scaler.fit_transform(sample_states)
    pca = PCA(n_components=min(3, sample_scaled.shape[1]))
    pc_scores = pca.fit_transform(sample_scaled)
    pc1 = pc_scores[:, 0] if pc_scores.shape[1] else np.zeros(sample_scaled.shape[0])

    coords_all = project_states_to_torus(embedding_inputs.states, basis, torus_radii)
    embedding = coords_all[sample_idx]

    return embedding, pc1, np.asarray(sample_pos)


def _add_overlay(ax, overlay: OverlaySpec, time_norm: Normalize) -> None:
    if overlay.coords.size <= 1 or overlay.times.size <= 1:
        return
    segments = np.stack([overlay.coords[:-1], overlay.coords[1:]], axis=1)
    if overlay.shadow_color:
        shadow = Line3DCollection(
            segments,
            colors=overlay.shadow_color,
            linewidths=overlay.shadow_width,
            alpha=0.7,
        )
        ax.add_collection3d(shadow)
    if overlay.glow_color:
        glow = Line3DCollection(
            segments,
            colors=overlay.glow_color,
            linewidths=overlay.glow_width,
            alpha=0.75,
        )
        ax.add_collection3d(glow)
    if overlay.cmap is not None:
        lc = Line3DCollection(segments, cmap=overlay.cmap, norm=time_norm)
        lc.set_array(overlay.times[:-1])
    elif overlay.color is not None:
        lc = Line3DCollection(segments, colors=overlay.color)
    else:
        return
    lc.set_linewidth(overlay.line_width)
    ax.add_collection3d(lc)

    stride = max(1, overlay.coords.shape[0] // 50)
    if overlay.cmap is not None:
        ax.scatter(
            overlay.coords[::stride, 0],
            overlay.coords[::stride, 1],
            overlay.coords[::stride, 2],
            c=overlay.times[::stride],
            cmap=overlay.cmap,
            norm=time_norm,
            s=overlay.point_size,
            edgecolors=overlay.point_edge_color,
            linewidths=overlay.point_edge_width,
            alpha=overlay.alpha,
        )
    else:
        ax.scatter(
            overlay.coords[::stride, 0],
            overlay.coords[::stride, 1],
            overlay.coords[::stride, 2],
            color=overlay.color,
            s=overlay.point_size,
            edgecolors=overlay.point_edge_color,
            linewidths=overlay.point_edge_width,
            alpha=overlay.alpha,
        )
    ax.scatter(
        overlay.coords[0, 0],
        overlay.coords[0, 1],
        overlay.coords[0, 2],
        s=overlay.start_size,
        color=overlay.start_color,
        edgecolors=overlay.point_edge_color,
        linewidths=overlay.point_edge_width + 0.2,
        marker=overlay.start_marker,
        zorder=10,
    )
    ax.scatter(
        overlay.coords[-1, 0],
        overlay.coords[-1, 1],
        overlay.coords[-1, 2],
        s=overlay.end_size,
        color=overlay.end_color,
        edgecolors=overlay.point_edge_color,
        linewidths=overlay.point_edge_width + 0.2,
        marker=overlay.end_marker,
        zorder=10,
    )


def plot_fig1b(
    embedding: np.ndarray,
    pc1: np.ndarray,
    overlays: Sequence[OverlaySpec],
    overlay_xy: np.ndarray,
    overlay_times: np.ndarray,
    pos_background: np.ndarray,
    box_limits: Tuple[float, float],
    out_path: Path,
    title: str,
):
    """Render three 3-D views plus the corresponding open-field trajectory."""
    cmap_points = plt.cm.Spectral_r
    traj_cmap = plt.cm.inferno
    max_scatter = 8000
    if embedding.shape[0] > max_scatter:
        sel = np.linspace(0, embedding.shape[0] - 1, max_scatter, dtype=int)
        emb_plot = embedding[sel]
        pc1_plot = pc1[sel]
    else:
        emb_plot = embedding
        pc1_plot = pc1
    pc_lo = np.nanpercentile(pc1, 2)
    pc_hi = np.nanpercentile(pc1, 98)
    if not np.isfinite(pc_lo):
        pc_lo = np.nanmin(pc1)
    if not np.isfinite(pc_hi):
        pc_hi = np.nanmax(pc1)
    if not np.isfinite(pc_lo) or not np.isfinite(pc_hi) or pc_lo == pc_hi:
        pc_lo, pc_hi = -1.0, 1.0
    pc_norm = TwoSlopeNorm(vcenter=0.0, vmin=float(pc_lo), vmax=float(pc_hi))
    time_norm = Normalize(vmin=0.0, vmax=float(overlay_times.max() if overlay_times.size else 1.0))

    views = [(25, 35), (25, 125), (70, 35)]
    fig = plt.figure(figsize=(12.0, 3.6))
    gs = gridspec.GridSpec(1, 4, width_ratios=[1.0, 1.0, 1.0, 1.05], wspace=0.08)
    lim = float(np.nanmax(np.abs(embedding))) if np.isfinite(embedding).any() else 1.0

    for i, (elev, azim) in enumerate(views):
        ax = fig.add_subplot(gs[0, i], projection="3d")
        ax.scatter(
            emb_plot[:, 0],
            emb_plot[:, 1],
            emb_plot[:, 2],
            c=pc1_plot,
            cmap=cmap_points,
            norm=pc_norm,
            s=1.6,
            alpha=0.18,
        )
        for overlay in overlays:
            _add_overlay(ax, overlay, time_norm)
        ax.view_init(elev=elev, azim=azim)
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_zlim(-lim, lim)
        ax.set_box_aspect([1, 1, 1])
        ax.set_axis_off()
        ax.set_title(title, fontsize=10, pad=2)

    ax2d = fig.add_subplot(gs[0, 3])
    if pos_background.size:
        ax2d.scatter(pos_background[:, 0], pos_background[:, 1], color="#b5b5b5", s=2, alpha=0.25, linewidths=0)
    if overlay_xy.size > 1 and overlay_times.size > 1:
        seg2d = np.stack([overlay_xy[:-1], overlay_xy[1:]], axis=1)
        shadow2d = LineCollection(seg2d, colors="black", linewidths=5.0, alpha=0.7)
        ax2d.add_collection(shadow2d)
        glow2d = LineCollection(seg2d, colors="white", linewidths=3.4, alpha=0.75)
        ax2d.add_collection(glow2d)
        lc2d = LineCollection(seg2d, colors="#ff7f0e", linewidths=2.4, alpha=0.95)
        ax2d.add_collection(lc2d)
        stride = max(1, overlay_xy.shape[0] // 50)
        ax2d.scatter(
            overlay_xy[::stride, 0],
            overlay_xy[::stride, 1],
            color="#ff7f0e",
            s=24,
            edgecolors="black",
            linewidths=0.4,
            alpha=0.95,
        )
        ax2d.scatter(
            overlay_xy[0, 0],
            overlay_xy[0, 1],
            s=70,
            color="#ff7f0e",
            edgecolors="black",
            linewidths=0.6,
            marker="o",
            zorder=10,
        )
        ax2d.scatter(
            overlay_xy[-1, 0],
            overlay_xy[-1, 1],
            s=70,
            color="#ff7f0e",
            edgecolors="black",
            linewidths=0.6,
            marker="s",
            zorder=10,
        )
    legend_handles = []
    for overlay in overlays:
        if overlay.label:
            legend_color = overlay.legend_color or overlay.color
            if legend_color:
                legend_handles.append(Line2D([0], [0], color=legend_color, lw=2, label=overlay.label))
    if legend_handles:
        ax2d.legend(handles=legend_handles, loc="lower left", frameon=False, fontsize=8)
    ax2d.set_xlim(-box_limits[0], box_limits[0])
    ax2d.set_ylim(-box_limits[1], box_limits[1])
    ax2d.set_aspect("equal")
    ax2d.set_xticks([])
    ax2d.set_yticks([])
    ax2d.set_title("Open-field path", fontsize=10)

    fig.subplots_adjust(left=0.02, right=0.98, top=0.88, bottom=0.08, wspace=0.08)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def plot_distance_over_time(
    times: np.ndarray,
    baseline_mean: np.ndarray,
    baseline_sem: np.ndarray,
    ablated_mean: np.ndarray,
    ablated_sem: np.ndarray,
    out_path: Path,
    title: str,
) -> None:
    fig, ax = plt.subplots(figsize=(6.8, 4.4))
    ax.plot(times, baseline_mean, color="#ff7f0e", lw=2.2, label="Baseline")
    ax.fill_between(times, baseline_mean - baseline_sem, baseline_mean + baseline_sem, color="#ff7f0e", alpha=0.25, linewidth=0)
    ax.plot(times, ablated_mean, color="#1f77b4", lw=2.2, label="Predictive ablated")
    ax.fill_between(times, ablated_mean - ablated_sem, ablated_mean + ablated_sem, color="#1f77b4", alpha=0.25, linewidth=0)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("k-NN distance to baseline cloud")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend(frameon=False)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------------------

def main():
    base = Path("Models/Single agent path integration")
    checkpoints = find_checkpoints(base)
    if not checkpoints:
        raise SystemExit("No checkpoints found under Models/Single agent path integration/")

    overlay_seconds = 5.0
    sample_size = 20000
    gridness_threshold = 0.5
    min_shift_cm = 5.0
    max_units = 512
    traj_batches = 4
    eval_batch_size = 64
    rng = np.random.default_rng(123)
    trajectory_mode = "compare_full_straight"
    cloud_trajectories = 1000
    eval_trajectories = 100
    use_full_cloud = True
    cloud_time_stride = 1 if use_full_cloud else 4
    cloud_max_points = None if use_full_cloud else 40000

    for seed, ckpt_path in checkpoints:
        print(f"[seed {seed}] Loading checkpoint {ckpt_path}")
        device = torch.device("cpu")
        raw = torch.load(ckpt_path, map_location="cpu")
        state = raw["state_dict"] if isinstance(raw, dict) and "state_dict" in raw else raw

        grid_units, class_indices = load_grid_units(ckpt_path, gridness_threshold, max_units, min_shift_cm)
        predictive_units = class_indices.get("predictive", np.array([], dtype=int))
        retrospective_units = class_indices.get("retrospective", np.array([], dtype=int))

        options = build_options(Path(ckpt_path), device=str(device))
        options.sequence_length = max(int(math.ceil(overlay_seconds / getattr(options, "trajectory_dt", 0.02))) + 10, 260)
        options.batch_size = eval_batch_size
        options.run_ID = f"fig1b_seed_{seed}"
        place_cells = PlaceCells(options)
        traj_gen = TrajectoryGenerator(options, place_cells)
        if "straight" in trajectory_mode:
            eval_batches = collect_straight_batches(
                options,
                place_cells,
                eval_batch_size,
                traj_batches,
                rng,
            )
        else:
            eval_batches = collect_batches(traj_gen, traj_batches)

        model = RNN(options, place_cells).to(device)
        model.load_state_dict(state)
        model.eval()

        rm_options = build_options(Path(ckpt_path), device=str(device))
        rm_options.sequence_length = 20
        rm_options.batch_size = 200
        traj_gen_rm = TrajectoryGenerator(rm_options, place_cells)
        torus_basis = compute_torus_basis(model, traj_gen_rm, rm_options, grid_units)
        torus_radii = (1.0, 0.35)

        baseline_inputs, baseline_batches, pos_batches = run_condition(
            state,
            options,
            place_cells,
            eval_batches,
            device,
            [],
            grid_units,
        )
        if predictive_units.size:
            ablated_inputs, ablated_batches, _ = run_condition(
                state,
                options,
                place_cells,
                eval_batches,
                device,
                predictive_units,
                grid_units,
            )
        else:
            ablated_inputs, ablated_batches, _ = run_condition(
                state,
                options,
                place_cells,
                eval_batches,
                device,
                [],
                grid_units,
            )

        out_dir = analysis_dir_for_checkpoint(ckpt_path) / "torus"
        if trajectory_mode in {"compare", "compare_full", "compare_full_straight"}:
            cloud_batches = int(math.ceil(cloud_trajectories / eval_batch_size))
            cloud_g_list, cloud_pos_list = collect_activity_batches(
                model,
                traj_gen,
                device,
                grid_units,
                cloud_batches,
                time_stride=cloud_time_stride,
            )
            cloud_states = np.concatenate(cloud_g_list, axis=1)
            cloud_positions = np.concatenate(cloud_pos_list, axis=1)

            cloud_flat = cloud_states.reshape(-1, cloud_states.shape[2]).astype(np.float32)
            mean = cloud_flat.mean(axis=0)
            std = cloud_flat.std(axis=0) + 1e-6
            cloud_z = (cloud_flat - mean) / std
            if cloud_max_points is not None and cloud_z.shape[0] > cloud_max_points:
                sel = rng.choice(cloud_z.shape[0], cloud_max_points, replace=False)
                cloud_z = cloud_z[sel]

            emb_inputs = EmbeddingInputs(
                states=cloud_states,
                positions=cloud_positions,
                dt=baseline_inputs.dt,
                grid_units_used=grid_units,
            )
            emb, pc1, pos_bg = embed_states(
                emb_inputs,
                sample_size,
                rng,
                torus_basis,
                torus_radii,
            )

            overlay_steps = min(
                int(math.floor(overlay_seconds / baseline_inputs.dt)),
                baseline_inputs.states.shape[0],
            )
            base_eval = baseline_inputs.states[:overlay_steps]
            ab_eval = ablated_inputs.states[:overlay_steps]
            pos_eval = baseline_inputs.positions[:overlay_steps]
            total_traj = base_eval.shape[1]
            n_eval = min(eval_trajectories, total_traj)
            traj_idx = rng.choice(total_traj, n_eval, replace=False)
            base_eval = base_eval[:, traj_idx]
            ab_eval = ab_eval[:, traj_idx]
            pos_eval = pos_eval[:, traj_idx]

            base_flat = base_eval.reshape(-1, base_eval.shape[2]).astype(np.float32)
            ab_flat = ab_eval.reshape(-1, ab_eval.shape[2]).astype(np.float32)
            base_z = (base_flat - mean) / std
            ab_z = (ab_flat - mean) / std
            base_dist = compute_knn_distances(cloud_z, base_z, k=5).reshape(overlay_steps, n_eval)
            ab_dist = compute_knn_distances(cloud_z, ab_z, k=5).reshape(overlay_steps, n_eval)

            times = np.arange(overlay_steps) * baseline_inputs.dt
            base_mean = base_dist.mean(axis=1)
            ab_mean = ab_dist.mean(axis=1)
            base_sem = base_dist.std(axis=1, ddof=1) / np.sqrt(n_eval)
            ab_sem = ab_dist.std(axis=1, ddof=1) / np.sqrt(n_eval)

            traj_choice = int(rng.integers(0, n_eval))
            base_traj = base_eval[:, traj_choice]
            overlay_xy = pos_eval[:, traj_choice, :2]
            overlay_emb = project_states_to_torus(base_traj[:, None, :], torus_basis, torus_radii).reshape(-1, 3)
            overlay_emb_ab = project_states_to_torus(ab_eval[:, traj_choice][:, None, :], torus_basis, torus_radii).reshape(-1, 3)

            overlay_specs = [
                OverlaySpec(
                    coords=overlay_emb_ab,
                    times=times,
                    label="Predictive ablation",
                    color="#3fb9ff",
                    shadow_color="#0b1b2b",
                    glow_color="#e3f6ff",
                    shadow_width=3.2,
                    glow_width=2.2,
                    line_width=1.4,
                    point_size=14.0,
                    start_marker="^",
                    start_color="#3fb9ff",
                    start_size=40.0,
                    end_marker="D",
                    end_color="#3fb9ff",
                    end_size=40.0,
                    legend_color="#3fb9ff",
                    alpha=0.75,
                ),
                OverlaySpec(
                    coords=overlay_emb,
                    times=times,
                    label="Baseline trajectory",
                    color="#ff7f0e",
                    shadow_width=6.5,
                    glow_width=4.8,
                    line_width=3.4,
                    point_size=26.0,
                    point_edge_width=0.5,
                    start_color="#ff7f0e",
                    start_size=70.0,
                    end_color="#ff7f0e",
                    end_size=70.0,
                    alpha=1.0,
                    legend_color="#ff7f0e",
                ),
            ]

            suffix_parts = []
            if "full" in trajectory_mode:
                suffix_parts.append("full")
            if "straight" in trajectory_mode:
                suffix_parts.append("straight")
            suffix = "_".join(suffix_parts) if suffix_parts else "compare"
            plot_fig1b(
                emb,
                pc1,
                overlay_specs,
                overlay_xy,
                times,
                pos_bg,
                box_limits=(options.box_width / 2, options.box_height / 2),
                out_path=out_dir / f"fig1b_baseline_cloud_dualtraj_{suffix}.png",
                title=f"Seed {seed} baseline cloud",
            )
            plot_distance_over_time(
                times,
                base_mean,
                base_sem,
                ab_mean,
                ab_sem,
                out_path=out_dir / f"fig1b_offmanifold_distance_{suffix}.png",
                title=f"Seed {seed} k-NN distance to baseline cloud",
            )

            random_units = sample_random_units(rng, grid_units, predictive_units.size)
            random_inputs, random_batches, _ = run_condition(
                state,
                options,
                place_cells,
                eval_batches,
                device,
                random_units,
                grid_units,
            )
            if retrospective_units.size:
                retrospective_inputs, retrospective_batches, _ = run_condition(
                    state,
                    options,
                    place_cells,
                    eval_batches,
                    device,
                    retrospective_units,
                    grid_units,
                )
            else:
                retrospective_inputs, retrospective_batches = None, None

            random_traj = random_inputs.states[:overlay_steps, traj_idx][:, traj_choice]
            random_emb = project_states_to_torus(random_traj[:, None, :], torus_basis, torus_radii).reshape(-1, 3)
            random_specs = [
                OverlaySpec(
                    coords=overlay_emb,
                    times=times,
                    label="Baseline trajectory",
                    color="#ff7f0e",
                    shadow_width=6.5,
                    glow_width=4.8,
                    line_width=3.4,
                    point_size=26.0,
                    point_edge_width=0.5,
                    start_color="#ff7f0e",
                    start_size=70.0,
                    end_color="#ff7f0e",
                    end_size=70.0,
                    alpha=1.0,
                    legend_color="#ff7f0e",
                ),
                OverlaySpec(
                    coords=random_emb,
                    times=times,
                    label="Random ablation",
                    color="#7a52ff",
                    shadow_color="#2b1d55",
                    glow_color="#e7dcff",
                    shadow_width=3.0,
                    glow_width=2.0,
                    line_width=1.6,
                    point_size=16.0,
                    start_marker="^",
                    start_color="#bfa5ff",
                    start_size=45.0,
                    end_marker="D",
                    end_color="#7a52ff",
                    end_size=45.0,
                    legend_color="#7a52ff",
                    alpha=0.8,
                ),
            ]
            plot_fig1b(
                emb,
                pc1,
                random_specs,
                overlay_xy,
                times,
                pos_bg,
                box_limits=(options.box_width / 2, options.box_height / 2),
                out_path=out_dir / f"fig1b_baseline_cloud_dualtraj_{suffix}_random.png",
                title=f"Seed {seed} baseline cloud (random ablation)",
            )

            if retrospective_inputs is not None:
                retro_traj = retrospective_inputs.states[:overlay_steps, traj_idx][:, traj_choice]
                retro_emb = project_states_to_torus(retro_traj[:, None, :], torus_basis, torus_radii).reshape(-1, 3)
                retro_specs = [
                    OverlaySpec(
                        coords=overlay_emb,
                        times=times,
                        label="Baseline trajectory",
                        color="#ff7f0e",
                        shadow_width=6.5,
                        glow_width=4.8,
                        line_width=3.4,
                        point_size=26.0,
                        point_edge_width=0.5,
                        start_color="#ff7f0e",
                        start_size=70.0,
                        end_color="#ff7f0e",
                        end_size=70.0,
                        alpha=1.0,
                        legend_color="#ff7f0e",
                    ),
                    OverlaySpec(
                        coords=retro_emb,
                        times=times,
                        label="Retrospective ablation",
                        color="#2ca02c",
                        shadow_color="#0f2b0f",
                        glow_color="#d6f5d6",
                        shadow_width=3.0,
                        glow_width=2.0,
                        line_width=1.6,
                        point_size=16.0,
                        start_marker="^",
                        start_color="#8fd98f",
                        start_size=45.0,
                        end_marker="D",
                        end_color="#2ca02c",
                        end_size=45.0,
                        legend_color="#2ca02c",
                        alpha=0.8,
                    ),
                ]
                plot_fig1b(
                    emb,
                    pc1,
                    retro_specs,
                    overlay_xy,
                    times,
                    pos_bg,
                    box_limits=(options.box_width / 2, options.box_height / 2),
                    out_path=out_dir / f"fig1b_baseline_cloud_dualtraj_{suffix}_retrospective.png",
                    title=f"Seed {seed} baseline cloud (retrospective ablation)",
                )
        else:
            emb, pc1, pos_bg = embed_states(
                baseline_inputs,
                sample_size,
                rng,
                torus_basis,
                torus_radii,
            )

            ab_emb, ab_pc1, _ = embed_states(
                ablated_inputs,
                sample_size,
                rng,
                torus_basis,
                torus_radii,
            )

            if trajectory_mode == "random":
                base_traj, ablated_traj, overlay_pos = select_random_path(
                    baseline_batches,
                    ablated_batches,
                    pos_batches,
                    baseline_inputs.dt,
                    overlay_seconds,
                    rng,
                )
            else:
                base_traj, ablated_traj, overlay_pos = select_divergent_path(
                    baseline_batches,
                    ablated_batches,
                    pos_batches,
                    torus_basis,
                    torus_radii,
                    baseline_inputs.dt,
                    overlay_seconds,
                )
            overlay_emb = project_states_to_torus(base_traj[:, None, :], torus_basis, torus_radii).reshape(-1, 3)
            overlay_emb_ab = project_states_to_torus(ablated_traj[:, None, :], torus_basis, torus_radii).reshape(-1, 3)
            overlay_steps = overlay_emb.shape[0]
            overlay_times = np.arange(overlay_steps) * baseline_inputs.dt
            overlay_xy = overlay_pos[:, :2]
            baseline_name = "fig1b_random_baseline.png" if trajectory_mode == "random" else "fig1b_baseline.png"
            overlay_specs = [
                OverlaySpec(
                    coords=overlay_emb,
                    times=overlay_times,
                    label="Baseline trajectory",
                    cmap=plt.cm.inferno,
                    legend_color="#ff7f0e",
                ),
            ]
            if predictive_units.size:
                overlay_specs.append(
                    OverlaySpec(
                        coords=overlay_emb_ab,
                        times=overlay_times,
                        label="Predictive ablation",
                        color="#3fb9ff",
                        shadow_color="#0b1b2b",
                        glow_color="#e3f6ff",
                        line_width=2.4,
                        point_size=20.0,
                        start_marker="^",
                        start_color="#bce7ff",
                        start_size=55.0,
                        end_marker="D",
                        end_color="#2f5dff",
                        end_size=55.0,
                        legend_color="#3fb9ff",
                    ),
                )
            plot_fig1b(
                emb,
                pc1,
                overlay_specs,
                overlay_xy,
                overlay_times,
                pos_bg,
                box_limits=(options.box_width / 2, options.box_height / 2),
                out_path=out_dir / baseline_name,
                title=f"Seed {seed} baseline",
            )

            if predictive_units.size:
                ablated_name = (
                    "fig1b_random_predictive_ablated.png"
                    if trajectory_mode == "random"
                    else "fig1b_predictive_ablated.png"
                )
                plot_fig1b(
                    ab_emb,
                    ab_pc1,
                    [
                        OverlaySpec(
                            coords=overlay_emb_ab,
                            times=overlay_times,
                            label="Predictive ablation",
                            cmap=plt.cm.inferno,
                            legend_color="#ff7f0e",
                        )
                    ],
                    overlay_xy,
                    overlay_times,
                    pos_bg,
                    box_limits=(options.box_width / 2, options.box_height / 2),
                    out_path=out_dir / ablated_name,
                    title=f"Seed {seed} predictive ablated",
                )
            else:
                print(f"[seed {seed}] No predictive units found above threshold; skipping ablated plot.")

        print(f"[seed {seed}] Wrote figures to {out_dir}")


if __name__ == "__main__":
    main()
