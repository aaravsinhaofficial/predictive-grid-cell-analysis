#!/usr/bin/env python3
"""Compute topological distance-to-torus and decoding error per single-agent seed.

This script approximates the notebook 071_application_dual_agent_rnns.ipynb using the
available single-agent checkpoints in Models/Single agent path integration/. It:
  - loads each seed's final checkpoint,
  - computes rate maps from simulated trajectories,
  - embeds spatial autocorrelations with UMAP + DBSCAN for visualization,
  - computes persistent homology of the rate-map point cloud and compares it to a
    synthetic torus via bottleneck distance (H1/H2),
  - estimates decoding error on held-out trajectories,
  - saves per-seed metrics and plots.

Outputs land in analysis_outputs/<model>/summary/torus_distance_single/:
  - seed_<id>_umap.png : UMAP of spatial autocorrelations colored by grid score.
  - seed_<id>.json     : metrics (distance_to_torus, decoding_error_cm, grid scores).
  - distance_vs_error.png : twin-axis plot of decoding error vs torus distance per seed.
"""

from __future__ import annotations

import json
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

# Ensure numba/matplotlib caches are writable inside the repo.
os.environ.setdefault("NUMBA_CACHE_DIR", str(Path(".numba_cache").resolve()))
os.environ.setdefault("MPLCONFIGDIR", str(Path(".mplconfig").resolve()))
Path(os.environ["NUMBA_CACHE_DIR"]).mkdir(parents=True, exist_ok=True)
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import matplotlib.pyplot as plt
import numpy as np
import torch
import umap
from persim import bottleneck
from ripser import ripser
from sklearn.cluster import DBSCAN

from model import RNN
from place_cells import PlaceCells
from scores import GridScorer
from trajectory_generator import TrajectoryGenerator
from visualize import compute_ratemaps
from path_utils import analysis_summary_dir
from multi_seed_predictive_analysis import infer_dims_from_state


# --------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------


@dataclass
class Options:
    batch_size: int
    sequence_length: int
    learning_rate: float
    Np: int
    Ng: int
    place_cell_rf: float
    surround_scale: float
    RNN_type: str
    activation: str
    weight_decay: float
    DoG: bool
    periodic: bool
    box_width: float
    box_height: float
    device: str
    velocity_dim: int
    run_ID: str
    save_dir: str
    trajectory_speed_scale: float = 1.0
    trajectory_speed_max: float | None = None
    trajectory_velocity_smoothing: float = 0.0
    trajectory_turn_sigma_scale: float = 1.0
    trajectory_border_region: float = 0.03
    trajectory_wall_slowdown: float = 0.25
    trajectory_wall_turn_scale: float = 1.0
    trajectory_style: str = "random_walk"
    trajectory_fixed_speed: float | None = None
    trajectory_dt: float = 0.02


def _parse_from_dirname(dirname: str) -> Dict[str, float | int | str | bool]:
    """Parse key hyperparameters from the checkpoint directory name."""
    out: Dict[str, float | int | str | bool] = {}
    m = re.search(r"steps_(\d+)_batch_(\d+)", dirname)
    if m:
        out["sequence_length"] = int(m.group(1))
        out["batch_size"] = int(m.group(2))
    m = re.search(r"rf_0*([0-9]+)", dirname)
    if m:
        rf_digits = m.group(1)
        try:
            out["place_cell_rf"] = float(int(rf_digits)) / 100.0
        except ValueError:
            pass
    m = re.search(r"weight_decay_([0-9eE\\.\\-]+)", dirname)
    if m:
        try:
            out["weight_decay"] = float(m.group(1))
        except ValueError:
            pass
    m = re.search(r"lr_0*([0-9]+)", dirname)
    if m:
        lr_digits = m.group(1)
        try:
            out["learning_rate"] = float(f"0.{lr_digits}")
        except ValueError:
            pass
    m = re.search(r"DoG_(True|False)", dirname)
    if m:
        out["DoG"] = m.group(1) == "True"
    m = re.search(r"periodic_(True|False)", dirname)
    if m:
        out["periodic"] = m.group(1) == "True"
    if "relu" in dirname:
        out["activation"] = "relu"
    return out


def build_options(ckpt_path: Path, device: str = "cpu") -> Options:
    state = torch.load(ckpt_path, map_location=device)
    Ng, Np, velocity_dim = infer_dims_from_state(state)
    defaults = dict(
        batch_size=200,
        sequence_length=20,
        learning_rate=1e-4,
        place_cell_rf=0.12,
        surround_scale=2.0,
        RNN_type="RNN",
        activation="relu",
        weight_decay=1e-4,
        DoG=True,
        periodic=False,
        box_width=2.2,
        box_height=2.2,
    )
    parsed = _parse_from_dirname(ckpt_path.parent.name)
    defaults.update(parsed)
    return Options(
        batch_size=int(defaults["batch_size"]),
        sequence_length=int(defaults["sequence_length"]),
        learning_rate=float(defaults["learning_rate"]),
        Np=int(Np),
        Ng=int(Ng),
        place_cell_rf=float(defaults["place_cell_rf"]),
        surround_scale=float(defaults["surround_scale"]),
        RNN_type=str(defaults["RNN_type"]),
        activation=str(defaults["activation"]),
        weight_decay=float(defaults["weight_decay"]),
        DoG=bool(defaults["DoG"]),
        periodic=bool(defaults["periodic"]),
        box_width=float(defaults["box_width"]),
        box_height=float(defaults["box_height"]),
        device=device,
        velocity_dim=int(velocity_dim),
        run_ID=ckpt_path.parent.name,
        save_dir=str(ckpt_path.parent),
    )


def load_model_and_generators(ckpt_path: Path, device: str = "cpu"):
    options = build_options(ckpt_path, device=device)
    place_cells = PlaceCells(options)
    model = RNN(options, place_cells).to(device)
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    traj_gen = TrajectoryGenerator(options, place_cells)
    return model, options, place_cells, traj_gen


def compute_decoding_error(model: RNN, traj_gen: TrajectoryGenerator, batches: int = 5) -> float:
    errs = []
    model.eval()
    with torch.no_grad():
        for _ in range(batches):
            inputs, pos, pc_outputs = traj_gen.get_test_batch()
            loss, err = model.compute_loss(inputs, pc_outputs, pos)
            errs.append(err.item())
    return float(np.mean(errs))


def compute_spatial_autocorrelations(rate_maps: np.ndarray, scorer: GridScorer) -> Tuple[np.ndarray, np.ndarray]:
    """Return spatial autocorrelations and grid scores for each unit."""
    sacs = []
    grid_scores = []
    for rm in rate_maps:
        score_60, _, _, _, sac, _ = scorer.get_scores(rm)
        sacs.append(sac)
        grid_scores.append(score_60)
    sacs = np.array(sacs)
    grid_scores = np.array(grid_scores)
    return sacs, grid_scores


def vectorize_sacs(sacs: np.ndarray) -> np.ndarray:
    """Flatten and z-score sacs across units."""
    num_cells = sacs.shape[0]
    num_bins = sacs.shape[1] * sacs.shape[2]
    mat = sacs.reshape(num_cells, num_bins)
    mat = (mat - mat.mean(axis=0, keepdims=True)) / (mat.std(axis=0, keepdims=True) + 1e-6)
    return mat


def compute_umap(mat: np.ndarray, random_state: int = 42) -> np.ndarray:
    reducer = umap.UMAP(n_components=2, random_state=random_state)
    return reducer.fit_transform(mat)


def sample_torus(num_points: int, R: float = 1.0, r: float = 0.3, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    u = rng.uniform(0, 2 * np.pi, num_points)
    v = rng.uniform(0, 2 * np.pi, num_points)
    x = (R + r * np.cos(v)) * np.cos(u)
    y = (R + r * np.cos(v)) * np.sin(u)
    z = r * np.sin(v)
    w = r * np.cos(v)
    return np.stack([x, y, z, w], axis=1)


def persistence_distance(points: np.ndarray, torus_points: np.ndarray, maxdim: int = 2) -> float:
    """Compute bottleneck distance (max over H1/H2) between point cloud and torus reference."""
    dgms = ripser(points, maxdim=maxdim)["dgms"]
    ref = ripser(torus_points, maxdim=maxdim)["dgms"]
    dists = []
    for dim in range(1, maxdim + 1):
        diag = dgms[dim] if dim < len(dgms) else np.empty((0, 2))
        ref_diag = ref[dim] if dim < len(ref) else np.empty((0, 2))
        dist = bottleneck(diag, ref_diag)
        dists.append(dist)
    return float(max(dists))


def plot_umap(embedding: np.ndarray, labels: np.ndarray, grid_scores: np.ndarray, out_path: Path, title: str):
    plt.figure(figsize=(6, 5))
    sc = plt.scatter(embedding[:, 0], embedding[:, 1], c=grid_scores, cmap="viridis", s=10)
    plt.xlabel("UMAP 1")
    plt.ylabel("UMAP 2")
    plt.title(title)
    plt.colorbar(sc, label="Grid score (60°)")
    # Overlay DBSCAN cluster labels if provided
    uniq = np.unique(labels)
    for lab in uniq:
        sel = labels == lab
        plt.scatter(
            embedding[sel, 0],
            embedding[sel, 1],
            edgecolors="none",
            facecolors="none",
            s=80,
            linewidths=1,
            label=f"Cluster {lab}",
        )
    if uniq.size:
        plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_distance_vs_error(seeds: List[int], distances: List[float], errors: List[float], out_path: Path):
    fig, ax1 = plt.subplots(figsize=(7, 5))
    x = np.arange(len(seeds))
    color_err = "tab:blue"
    ax1.set_xlabel("Seed")
    ax1.set_ylabel("Decoding Error (cm)", color=color_err)
    ax1.plot(x, errors, color=color_err, marker="o")
    ax1.set_xticks(x)
    ax1.set_xticklabels(seeds)
    ax1.tick_params(axis="y", labelcolor=color_err)

    ax2 = ax1.twinx()
    color_topo = "tab:red"
    ax2.set_ylabel("Topological distance to Torus", color=color_topo)
    ax2.plot(x, distances, color=color_topo, marker="o")
    ax2.tick_params(axis="y", labelcolor=color_topo)
    fig.tight_layout()
    ax1.grid(True)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------


def main():
    base = Path("Models/Single agent path integration")
    seed_dirs = sorted(base.glob("Seed * weight decay 1e-06/*/"))
    if not seed_dirs:
        raise SystemExit("No single-agent checkpoints found under Models/Single agent path integration/")

    device = "cpu"
    model_name = base.name
    out_root = analysis_summary_dir(model_name, "torus_distance_single")
    seed_ids: List[int] = []
    distances: List[float] = []
    errors: List[float] = []

    # Reference torus point cloud (shared across seeds)
    torus_points = sample_torus(num_points=400, R=1.0, r=0.3, seed=0)

    for ckpt_dir in seed_dirs:
        seed_name = ckpt_dir.parent.name
        seed_match = re.search(r"Seed\s+(\d+)", seed_name)
        seed = int(seed_match.group(1)) if seed_match else -1
        ckpt_path = ckpt_dir / "final_model.pth"
        if not ckpt_path.exists():
            continue
        print(f"[seed {seed}] Loading {ckpt_path}")
        model, options, place_cells, traj_gen = load_model_and_generators(ckpt_path, device=device)

        # Compute decoding error
        decoding_err = compute_decoding_error(model, traj_gen, batches=5)

        # Rate maps and spatial autocorrelations
        res = 20
        Ng_use = min(512, options.Ng)
        activations, rate_map, _, _ = compute_ratemaps(
            model,
            traj_gen,
            options,
            res=res,
            n_avg=5,
            Ng=Ng_use,
            idxs=np.arange(Ng_use),
        )
        starts = [0.2] * 10
        ends = np.linspace(0.4, 1.0, num=10)
        coord_range = ((-options.box_width / 2, options.box_width / 2), (-options.box_height / 2, options.box_height / 2))
        scorer = GridScorer(res, coord_range, zip(starts, ends.tolist()))
        sacs, grid_scores = compute_spatial_autocorrelations(activations, scorer)
        sac_mat = vectorize_sacs(sacs)
        embedding = compute_umap(sac_mat)
        labels = DBSCAN(eps=0.5, min_samples=5).fit_predict(embedding)
        plot_umap(
            embedding,
            labels,
            grid_scores,
            out_root / f"seed_{seed}_umap.png",
            title=f"Seed {seed} spatial autocorrelations",
        )

        # Topological distance to torus
        rep = rate_map.T  # shape (positions, units)
        rep = rep / (np.linalg.norm(rep, axis=1, keepdims=True) + 1e-8)
        rep = np.nan_to_num(rep)
        topo_dist = persistence_distance(rep, torus_points, maxdim=2)

        # Persist metrics
        metrics = {
            "seed": seed,
            "distance_to_torus": topo_dist,
            "decoding_error_cm": decoding_err,
            "Ng_used": int(Ng_use),
            "num_positions": rep.shape[0],
            "umap_eps": 0.5,
            "umap_min_samples": 5,
        }
        out_root.mkdir(parents=True, exist_ok=True)
        with open(out_root / f"seed_{seed}.json", "w") as f:
            json.dump(metrics, f, indent=2)

        seed_ids.append(seed)
        distances.append(topo_dist)
        errors.append(decoding_err)

    if seed_ids:
        plot_distance_vs_error(seed_ids, distances, errors, out_root / "distance_vs_error.png")
        print(f"[done] Wrote metrics/plots to {out_root}")
    else:
        raise SystemExit("No seeds processed (check checkpoint paths).")


if __name__ == "__main__":
    main()
