#!/usr/bin/env python3
"""Toroidal manifold analysis with toroidal/predictive/retrospective/normal ablations.

This script:
  1) extracts a torus parameterization from grid-cell rate maps,
  2) finds the toroidal-cell ensemble via rotational autocorrelogram clustering,
  3) projects hidden activity onto toroidal coordinates,
  4) compares baseline vs toroidal/predictive/retrospective/normal grid-cell ablations.

Outputs are written under analysis_outputs/<model>/<seed>/<run>/torus/:
  - torus_comparison.png : 3D point clouds for each condition (baseline / toroidal / PGC ablated / RGC ablated / normal ablated)
  - torus_metrics.json   : lattice vectors, unit counts, radius variation, phase-to-position decoding errors
  - toroidal_embedding.png: UMAP/PCA + DBSCAN view used to pick toroidal cells
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple, Optional

# Keep caches writable inside the repo before importing matplotlib/UMAP stack.
os.environ.setdefault("NUMBA_CACHE_DIR", str(Path(".numba_cache").resolve()))
os.environ.setdefault("MPLCONFIGDIR", str(Path(".mplconfig").resolve()))
Path(os.environ["NUMBA_CACHE_DIR"]).mkdir(parents=True, exist_ok=True)
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import scipy.ndimage as ndimage
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from model import RNN
from path_utils import analysis_dir_for_checkpoint
from place_cells import PlaceCells
from predictive_retrospective_ablation import classify_from_scores
from trajectory_generator import TrajectoryGenerator
from visualize import compute_ratemaps
from scores import GridScorer, band_scores
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


@dataclass
class ToroidalDetection:
    units: np.ndarray
    labels: np.ndarray
    embedding: np.ndarray
    clusters: Dict[int, Dict[str, float]]
    selected_clusters: List[int]
    angles_deg: List[float]
    ring_bounds: Tuple[float, float]
    eps: float
    min_samples: int
    embedding_method: str


def select_band_units(band_vals: np.ndarray, idxs: np.ndarray, percentile: float, threshold: Optional[float]) -> Tuple[np.ndarray, float]:
    """Select band-like units using percentile or absolute cutoff."""
    finite = band_vals[np.isfinite(band_vals)]
    if finite.size == 0:
        return np.array([], dtype=int), float("nan")
    if threshold is not None:
        cutoff = float(threshold)
    else:
        cutoff = float(np.nanpercentile(finite, percentile))
    sel = np.where(band_vals >= cutoff)[0]
    return idxs[sel].astype(int), cutoff


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
    if ablate_units is not None and len(ablate_units) > 0:
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
# Toroidal-cell detection
# --------------------------------------------------------------------------------------

def _rotation_profile_from_sac(sac: np.ndarray, angles_deg: Sequence[float], mask: np.ndarray) -> np.ndarray:
    """Correlation of SAC with rotated copies (rotational autocorrelogram)."""
    masked = sac * mask
    base = masked.ravel()
    base -= np.mean(base)
    base_norm = np.linalg.norm(base) + 1e-8
    prof = []
    for ang in angles_deg:
        rot = ndimage.rotate(sac, ang, reshape=False, order=1, mode="nearest")
        rot_masked = (rot * mask).ravel()
        rot_masked -= np.mean(rot_masked)
        denom = (np.linalg.norm(rot_masked) + 1e-8) * base_norm
        prof.append(float(np.dot(base, rot_masked) / denom))
    return np.array(prof, dtype=float)


def _mean_orientation(angles_rad: np.ndarray) -> float:
    """Circular mean for orientations (mod π)."""
    if angles_rad.size == 0:
        return float("nan")
    s = np.mean(np.sin(2 * angles_rad))
    c = np.mean(np.cos(2 * angles_rad))
    return float(0.5 * np.mod(np.arctan2(s, c), 2 * np.pi))


def compute_rotational_features(
    rate_maps: np.ndarray,
    box_width: float,
    angles_deg: Sequence[float],
    ring_bounds: Tuple[float, float],
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """Return feature matrix + metadata for rotational autocorrelograms."""
    res = rate_maps.shape[1]
    coord_range = ((-box_width / 2, box_width / 2), (-box_width / 2, box_width / 2))
    scorer = GridScorer(res, coord_range, [ring_bounds])
    mask = scorer._get_ring_mask(*ring_bounds)  # type: ignore[attr-defined]
    angles_rad = np.deg2rad(angles_deg)
    band_template = np.cos(2 * angles_rad)
    tri_template = np.cos(3 * angles_rad)

    features = []
    profiles = []
    band_scores = []
    tri_scores = []
    anisotropy = []
    peak_angles = []
    for rm in rate_maps:
        sac = scorer.calculate_sac(rm)
        prof = _rotation_profile_from_sac(sac, angles_deg, mask)
        profiles.append(prof)
        prof_norm = (prof - np.mean(prof)) / (np.std(prof) + 1e-8)
        if not np.isfinite(prof_norm).all():
            prof_norm = np.nan_to_num(prof_norm)
        band_corr = float(np.corrcoef(prof_norm, band_template)[0, 1])
        tri_corr = float(np.corrcoef(prof_norm, tri_template)[0, 1])
        if not np.isfinite(band_corr):
            band_corr = 0.0
        if not np.isfinite(tri_corr):
            tri_corr = 0.0
        band_scores.append(band_corr)
        tri_scores.append(tri_corr)

        spectrum = np.fft.fftshift(np.fft.fft2(rm - np.nanmean(rm)))
        power = np.abs(spectrum)
        center = res // 2
        power[center - 1 : center + 2, center - 1 : center + 2] = 0
        peak_idx = np.unravel_index(np.argmax(power), power.shape)
        freqs = np.fft.fftshift(np.fft.fftfreq(res, d=box_width / res))
        kx = freqs[peak_idx[0]]
        ky = freqs[peak_idx[1]]
        ang = float(np.mod(np.arctan2(ky, kx), np.pi))
        peak_angles.append(ang)
        peak_power = float(power[peak_idx])
        anis_val = peak_power / (float(np.mean(power)) + 1e-8)
        anisotropy.append(anis_val)

        feat_vec = np.concatenate(
            [
                prof_norm,
                np.array([band_corr, tri_corr, anis_val, math.sin(ang), math.cos(ang)], dtype=float),
            ]
        )
        features.append(feat_vec)

    meta = {
        "profiles": np.array(profiles, dtype=float),
        "band_score": np.array(band_scores, dtype=float),
        "grid_like_score": np.array(tri_scores, dtype=float),
        "anisotropy": np.array(anisotropy, dtype=float),
        "peak_angle": np.array(peak_angles, dtype=float),
    }
    return np.array(features, dtype=float), meta


def embed_rotations(features: np.ndarray, mode: str = "auto", random_state: int = 0) -> Tuple[np.ndarray, str]:
    """Embed rotational features via UMAP (preferred) or PCA."""
    feats_std = StandardScaler().fit_transform(np.nan_to_num(features))
    mode = (mode or "auto").lower()
    if mode == "pca":
        return PCA(n_components=2, random_state=random_state).fit_transform(feats_std), "pca"
    if mode == "umap":
        try:
            import umap  # type: ignore
        except ImportError:
            return PCA(n_components=2, random_state=random_state).fit_transform(feats_std), "pca"
        reducer = umap.UMAP(
            n_components=2,
            n_neighbors=25,
            min_dist=0.1,
            metric="cosine",
            random_state=random_state,
        )
        return reducer.fit_transform(feats_std), "umap"

    try:
        import umap  # type: ignore
        reducer = umap.UMAP(
            n_components=2,
            n_neighbors=25,
            min_dist=0.1,
            metric="cosine",
            random_state=random_state,
        )
        return reducer.fit_transform(feats_std), "umap"
    except ImportError:
        return PCA(n_components=2, random_state=random_state).fit_transform(feats_std), "pca"


def cluster_embedding(embedding: np.ndarray, min_samples: int = 6, eps: float | None = None) -> Tuple[np.ndarray, float]:
    """DBSCAN clustering with adaptive eps when not provided."""
    if embedding.shape[0] < 2:
        return np.full(embedding.shape[0], -1, dtype=int), 0.0
    emb_std = StandardScaler().fit_transform(embedding)
    if eps is None:
        k = min(10, embedding.shape[0] - 1)
        nbrs = NearestNeighbors(n_neighbors=k).fit(emb_std)
        kth = np.sort(nbrs.kneighbors(emb_std)[0][:, -1])
        eps = float(np.median(kth) * 1.2)
    labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(emb_std)
    return labels, float(eps)


def _orientation_distance(a: float, b: float) -> float:
    diff = abs(a - b)
    return min(diff, math.pi - diff)


def select_toroidal_clusters(
    labels: np.ndarray,
    meta: Dict[str, np.ndarray],
    unit_indices: np.ndarray,
    desired: int = 3,
    band_floor: float = 0.15,
    min_size: int = 4,
) -> Tuple[np.ndarray, Dict[int, Dict[str, float]], List[int]]:
    """Pick up to `desired` DBSCAN clusters that look band-like."""
    unique = [l for l in np.unique(labels) if l >= 0]
    clusters: Dict[int, Dict[str, float]] = {}
    for lab in unique:
        idxs = np.where(labels == lab)[0]
        angles = meta["peak_angle"][idxs]
        clusters[lab] = {
            "size": int(idxs.size),
            "band_score_mean": float(np.nanmean(meta["band_score"][idxs])) if idxs.size else float("nan"),
            "grid_like_mean": float(np.nanmean(meta["grid_like_score"][idxs])) if idxs.size else float("nan"),
            "anisotropy_mean": float(np.nanmean(meta["anisotropy"][idxs])) if idxs.size else float("nan"),
            "angle_mean_rad": _mean_orientation(angles) if idxs.size else float("nan"),
            "indices_local": idxs.tolist(),
            "indices_global": unit_indices[idxs].astype(int).tolist(),
        }

    def _score(lab: int) -> float:
        info = clusters[lab]
        return info["band_score_mean"] * max(info["anisotropy_mean"], 1e-4)

    candidates = sorted(unique, key=_score, reverse=True)
    selected: List[int] = []
    for lab in candidates:
        info = clusters[lab]
        if info["size"] < min_size or info["band_score_mean"] < band_floor:
            continue
        if selected and any(_orientation_distance(info["angle_mean_rad"], clusters[s]["angle_mean_rad"]) < math.radians(20) for s in selected):
            continue
        selected.append(lab)
        if len(selected) >= desired:
            break

    if len(selected) < desired:
        fillers = sorted(unique, key=lambda l: clusters[l]["size"], reverse=True)
        for lab in fillers:
            if lab not in selected:
                selected.append(lab)
            if len(selected) >= desired:
                break

    if not selected:
        return np.array([], dtype=int), clusters, selected

    toroidal_units = np.unique(np.concatenate([unit_indices[labels == lab] for lab in selected if lab in clusters], axis=0))
    return toroidal_units.astype(int), clusters, selected


def plot_toroidal_embedding(
    embedding: np.ndarray,
    labels: np.ndarray,
    toroidal_units: np.ndarray,
    unit_indices: np.ndarray,
    save_path: str,
) -> None:
    """Scatter of embedding colored by DBSCAN clusters with toroidal cells highlighted."""
    if embedding.size == 0:
        return
    uniq = np.unique(labels)
    cmap = plt.cm.get_cmap("tab10")
    fig, ax = plt.subplots(figsize=(6.0, 5.0))
    for lab in uniq:
        mask = labels == lab
        color = "#bbbbbb" if lab == -1 else cmap(lab % 10)
        label = f"Cluster {lab}" if lab >= 0 else "Noise"
        ax.scatter(embedding[mask, 0], embedding[mask, 1], s=28, c=color, alpha=0.7, edgecolors="none", label=label)
    if toroidal_units.size:
        sel = np.isin(unit_indices, toroidal_units)
        ax.scatter(
            embedding[sel, 0],
            embedding[sel, 1],
            s=60,
            facecolors="none",
            edgecolors="black",
            linewidths=1.2,
            label=f"Toroidal (n={toroidal_units.size})",
        )
    ax.set_xlabel("Embedding dim 1")
    ax.set_ylabel("Embedding dim 2")
    ax.set_title("Rotational-profile embedding (DBSCAN clusters)")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=220)
    plt.close(fig)


def identify_toroidal_cells(
    rate_maps: np.ndarray,
    unit_indices: np.ndarray,
    box_width: float,
    out_dir: str,
    angles_deg: Optional[Sequence[float]] = None,
    ring_bounds: Tuple[float, float] = (0.18, 0.55),
    random_state: int = 0,
    embed_mode: str = "auto",
) -> ToroidalDetection:
    """UMAP/DBSCAN-based toroidal ensemble finder."""
    if angles_deg is None:
        angles_deg = np.linspace(0, 180, 36, endpoint=False)
    features, meta = compute_rotational_features(rate_maps, box_width, angles_deg, ring_bounds)
    embedding, method = embed_rotations(features, mode=embed_mode, random_state=random_state)
    labels, eps = cluster_embedding(embedding)
    toroidal_units, clusters, selected = select_toroidal_clusters(labels, meta, unit_indices)
    plot_toroidal_embedding(embedding, labels, toroidal_units, unit_indices, os.path.join(out_dir, "toroidal_embedding.png"))
    return ToroidalDetection(
        units=toroidal_units,
        labels=labels,
        embedding=embedding,
        clusters=clusters,
        selected_clusters=selected,
        angles_deg=list(angles_deg),
        ring_bounds=ring_bounds,
        eps=eps,
        min_samples=6,
        embedding_method=method,
    )


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
    fig = plt.figure(figsize=(max(14, 3.6 * len(titles)), 4.5))
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
        try:
            sns.kdeplot(ax=axes[i, 1], data=df, x="x", y="y", fill=True, cmap="mako", levels=30, thresh=0.0)
            axes[i, 1].set_title(f"{name}: XY density")
        except ValueError:
            sns.scatterplot(ax=axes[i, 1], data=df, x="x", y="y", s=4, color="#4b4b4b", alpha=0.5, linewidth=0)
            axes[i, 1].set_title(f"{name}: XY density (scatter fallback)")
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
    out_dir = str(analysis_dir_for_checkpoint(Path(ckpt_path)))
    path = os.path.join(out_dir, "gridness_data.npz")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing gridness_data.npz beside checkpoint: {path}. Run multi_seed_predictive_analysis.py first.")
    data = np.load(path)
    return {k: data[k] for k in data.files}


def load_saved_class_ids(class_dir: str, suffix: str) -> Dict[str, np.ndarray]:
    """Load predictive/retrospective/grid IDs from a saved analysis-output folder."""
    class_path = Path(class_dir)

    def _load(name: str) -> np.ndarray:
        path = class_path / f"{name}_{suffix}.npy"
        if not path.exists():
            raise FileNotFoundError(f"Missing saved class file: {path}")
        arr = np.load(path)
        return np.asarray(arr, dtype=int).reshape(-1)

    predictive = _load("predictive_ids")
    retrospective = _load("retrospective_ids")
    grid = _load("grid_ids")

    dead_path = class_path / f"dead_unit_ids_{suffix}.npy"
    if dead_path.exists():
        dead_mask = np.load(dead_path).astype(bool).reshape(-1)
        if dead_mask.size:
            alive = np.where(~dead_mask)[0]
            predictive = np.intersect1d(predictive, alive, assume_unique=False)
            retrospective = np.intersect1d(retrospective, alive, assume_unique=False)
            grid = np.intersect1d(grid, alive, assume_unique=False)

    normal = np.setdiff1d(grid, np.union1d(predictive, retrospective), assume_unique=False)
    return {
        "predictive": predictive,
        "retrospective": retrospective,
        "grid": grid,
        "normal": normal,
    }


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint_path", required=True, help="Path to a trained RNN checkpoint (.pth).")
    parser.add_argument("--class_dir", default=None, help="Optional directory containing saved predictive_ids_<suffix>.npy etc.")
    parser.add_argument("--class_suffix", default="random_walk", help="Suffix used for saved class files in --class_dir.")
    parser.add_argument("--output_dir", default=None, help="Optional output directory override.")
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
    parser.add_argument("--band_percentile", type=float, default=90.0, help="Percentile cutoff for band-cell selection.")
    parser.add_argument("--band_threshold", type=float, default=None, help="Absolute band score cutoff (overrides percentile).")
    parser.add_argument("--band_k_max", type=float, default=2.0, help="Max k for band score grid.")
    parser.add_argument("--band_k_step", type=float, default=0.1, help="Step for band score k grid.")
    parser.add_argument("--device", default=None, help="cpu or cuda. Defaults to best available.")
    parser.add_argument("--traj_speed_scale", default=1.0, type=float, help="Multiplier on the Rayleigh speed scale.")
    parser.add_argument("--traj_speed_max", default=None, type=float, help="Optional cap on forward speed (m/s).")
    parser.add_argument("--traj_velocity_smoothing", default=0.0, type=float, help="EMA factor (0-0.99) applied to speeds.")
    parser.add_argument("--traj_turn_sigma_scale", default=1.0, type=float, help="Multiplier on turning noise sigma.")
    parser.add_argument("--traj_border_region", default=0.03, type=float, help="Wall avoidance distance (m).")
    parser.add_argument("--traj_wall_slowdown", default=0.25, type=float, help="Speed multiplier near walls.")
    parser.add_argument("--traj_wall_turn_scale", default=1.0, type=float, help="Multiplier on wall-turn correction.")
    parser.add_argument("--toroid_embed", default="auto", choices=["auto", "umap", "pca"], help="Embedding method for toroidal-cell detection.")
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
    if args.output_dir:
        out_dir = str(Path(args.output_dir))
    else:
        out_dir = str(analysis_dir_for_checkpoint(Path(args.checkpoint_path)) / "torus")
    os.makedirs(out_dir, exist_ok=True)

    if args.class_dir:
        classes = load_saved_class_ids(args.class_dir, args.class_suffix)
        grid_units_global = np.asarray(classes["grid"], dtype=int)
        grid_units_global = grid_units_global[(grid_units_global >= 0) & (grid_units_global < Ng)]
        if grid_units_global.size == 0:
            raise RuntimeError("No saved grid units found in --class_dir.")
        basis_units_global = np.unique(grid_units_global)[: args.Ng_use]
        predictive_units = np.asarray(classes["predictive"], dtype=int)
        predictive_units = predictive_units[(predictive_units >= 0) & (predictive_units < Ng)].tolist()
        retrospective_units = np.asarray(classes["retrospective"], dtype=int)
        retrospective_units = retrospective_units[(retrospective_units >= 0) & (retrospective_units < Ng)].tolist()
        normal_units = np.asarray(classes.get("normal", np.array([], dtype=int)), dtype=int)
        normal_units = normal_units[(normal_units >= 0) & (normal_units < Ng)].tolist()
        class_source = {
            "mode": "saved_ids",
            "class_dir": str(Path(args.class_dir).resolve()),
            "class_suffix": args.class_suffix,
        }
    else:
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
        class_source = {
            "mode": "gridness_data",
            "gridness_threshold": float(args.gridness_threshold),
            "min_shift_cm": float(args.min_shift_cm),
        }

    # Rate maps (baseline) for the basis units
    model = RNN(options, place_cells).to(device)
    model.load_state_dict(state)
    model.eval()
    idxs = np.array(basis_units_global, dtype=int)
    Ng_subset = idxs.size
    rate_maps, _, _, _ = compute_ratemaps(model, traj_gen, options, res=args.res, n_avg=max(8, args.n_batches), Ng=Ng_subset, idxs=idxs)
    rate_maps = rate_maps.astype(float)

    band_vals, band_kx, band_ky = band_scores(
        rate_maps,
        args.res,
        options.box_width,
        k_values=np.arange(0.0, args.band_k_max + 1e-6, args.band_k_step),
    )
    band_units, band_cutoff = select_band_units(band_vals, idxs, args.band_percentile, args.band_threshold)

    basis = build_torus_basis(rate_maps, np.arange(Ng_subset), options.box_width)
    # Replace local unit labels with the original indices so projections slice the right neurons
    basis.units = idxs

    toroidal_detection = identify_toroidal_cells(rate_maps, idxs, options.box_width, out_dir, embed_mode=args.toroid_embed)
    toroidal_units = toroidal_detection.units.tolist()
    predictive_grid_units = np.intersect1d(np.asarray(predictive_units, dtype=int), np.asarray(grid_units_global, dtype=int)).tolist()
    retrospective_grid_units = np.intersect1d(np.asarray(retrospective_units, dtype=int), np.asarray(grid_units_global, dtype=int)).tolist()
    predictive_basis_units = np.intersect1d(np.asarray(predictive_units, dtype=int), np.asarray(basis_units_global, dtype=int)).tolist()
    retrospective_basis_units = np.intersect1d(np.asarray(retrospective_units, dtype=int), np.asarray(basis_units_global, dtype=int)).tolist()
    predictive_toroidal_units = np.intersect1d(np.asarray(predictive_units, dtype=int), np.asarray(toroidal_units, dtype=int)).tolist()
    retrospective_toroidal_units = np.intersect1d(np.asarray(retrospective_units, dtype=int), np.asarray(toroidal_units, dtype=int)).tolist()

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
    projections["Toroidal ablated"] = run_condition(
        state, basis, options, place_cells, traj_gen, cached_batches, device, "toroidal", toroidal_units, torus_radii
    )
    projections["Predictive ablated"] = run_condition(
        state, basis, options, place_cells, traj_gen, cached_batches, device, "predictive", predictive_units, torus_radii
    )
    projections["Retrospective ablated"] = run_condition(
        state, basis, options, place_cells, traj_gen, cached_batches, device, "retrospective", retrospective_units, torus_radii
    )
    projections["Predictive∩grid ablated"] = run_condition(
        state, basis, options, place_cells, traj_gen, cached_batches, device, "predictive_grid", predictive_grid_units, torus_radii
    )
    projections["Retrospective∩grid ablated"] = run_condition(
        state, basis, options, place_cells, traj_gen, cached_batches, device, "retrospective_grid", retrospective_grid_units, torus_radii
    )
    projections["Predictive∩toroidal ablated"] = run_condition(
        state, basis, options, place_cells, traj_gen, cached_batches, device, "predictive_toroidal", predictive_toroidal_units, torus_radii
    )
    projections["Retrospective∩toroidal ablated"] = run_condition(
        state, basis, options, place_cells, traj_gen, cached_batches, device, "retrospective_toroidal", retrospective_toroidal_units, torus_radii
    )
    projections["Normal grid ablated"] = run_condition(
        state, basis, options, place_cells, traj_gen, cached_batches, device, "normal", normal_units, torus_radii
    )
    projections["Band ablated"] = run_condition(
        state, basis, options, place_cells, traj_gen, cached_batches, device, "band", band_units.tolist(), torus_radii
    )

    # Plot
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
    toroidal_clusters_json: Dict[str, Dict[str, float | int | List[int]]] = {}
    for k, v in toroidal_detection.clusters.items():
        toroidal_clusters_json[str(int(k))] = {
            "size": int(v.get("size", 0)),
            "band_score_mean": float(v.get("band_score_mean", float("nan"))),
            "grid_like_mean": float(v.get("grid_like_mean", float("nan"))),
            "anisotropy_mean": float(v.get("anisotropy_mean", float("nan"))),
            "angle_mean_rad": float(v.get("angle_mean_rad", float("nan"))),
            "indices_local": [int(x) for x in v.get("indices_local", [])],
            "indices_global": [int(x) for x in v.get("indices_global", [])],
        }

    metrics = {
        "k1_cycles_per_m": basis.k1.tolist(),
        "k2_cycles_per_m": basis.k2.tolist(),
        "peak_angle_rad": basis.angle_rad,
        "peak_power": basis.peak_power,
        "grid_units_used": int(len(basis_units_global)),
        "predictive_units": len(predictive_units),
        "retrospective_units": len(retrospective_units),
        "normal_units": len(normal_units),
        "predictive_grid_units": len(predictive_grid_units),
        "retrospective_grid_units": len(retrospective_grid_units),
        "predictive_basis_units": len(predictive_basis_units),
        "retrospective_basis_units": len(retrospective_basis_units),
        "predictive_toroidal_units": len(predictive_toroidal_units),
        "retrospective_toroidal_units": len(retrospective_toroidal_units),
        "predictive_grid_unit_ids": [int(x) for x in predictive_grid_units],
        "retrospective_grid_unit_ids": [int(x) for x in retrospective_grid_units],
        "predictive_basis_unit_ids": [int(x) for x in predictive_basis_units],
        "retrospective_basis_unit_ids": [int(x) for x in retrospective_basis_units],
        "predictive_toroidal_unit_ids": [int(x) for x in predictive_toroidal_units],
        "retrospective_toroidal_unit_ids": [int(x) for x in retrospective_toroidal_units],
        "class_source": class_source,
        "toroidal_units": int(len(toroidal_units)),
        "band_units": int(len(band_units)),
        "band_cutoff": float(band_cutoff) if band_cutoff == band_cutoff else float("nan"),
        "band_percentile": float(args.band_percentile),
        "band_threshold": None if args.band_threshold is None else float(args.band_threshold),
        "band_overlap": {
            "toroidal": int(len(set(band_units.tolist()).intersection(toroidal_units))),
            "predictive": int(len(set(band_units.tolist()).intersection(predictive_units))),
            "retrospective": int(len(set(band_units.tolist()).intersection(retrospective_units))),
            "normal": int(len(set(band_units.tolist()).intersection(normal_units))),
        },
        "toroidal_detection": {
            "embedding_method": toroidal_detection.embedding_method,
            "eps": toroidal_detection.eps,
            "min_samples": toroidal_detection.min_samples,
            "angles_deg": toroidal_detection.angles_deg,
            "ring_bounds": list(toroidal_detection.ring_bounds),
            "selected_clusters": [int(c) for c in toroidal_detection.selected_clusters],
            "clusters": toroidal_clusters_json,
        },
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
