#!/usr/bin/env python3
"""Measure distance to the baseline population manifold for multiple ablations.

For each seed, the script:
  - samples a baseline population-activity cloud from many trajectories,
  - samples evaluation trajectories shared across baseline + ablations,
  - computes k-NN distances to the baseline cloud at each time step,
  - saves per-seed distance traces and a summary plot.

Outputs are written under analysis_outputs/<model>/<seed>/<run>/torus/:
  - off_manifold_seed_<id>_<style>_pct<pct>_multi.npz
  - off_manifold_seed_<id>_<style>_pct<pct>_multi.json
  - off_manifold_seed_<id>_<style>_pct<pct>_multi.png
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

# Keep caches writable inside the repo.
os.environ.setdefault("NUMBA_CACHE_DIR", str(Path(".numba_cache").resolve()))
os.environ.setdefault("MPLCONFIGDIR", str(Path(".mplconfig").resolve()))
Path(os.environ["NUMBA_CACHE_DIR"]).mkdir(parents=True, exist_ok=True)
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from model import RNN
from place_cells import PlaceCells
from predictive_retrospective_ablation import classify_from_scores, class_score_vectors, select_top_percentile
from trajectory_generator import TrajectoryGenerator
from multi_seed_predictive_analysis import zero_unit_weights_in_place
from single_seed_torus_distance import build_options
from path_utils import analysis_dir_for_checkpoint
from replicate_predictive_grid_figure import cm_per_step_from_positions
from shift_utils import build_shift_values
from visualize import predictive_gridness_analysis


# --------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------


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


def extract_state(raw: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    if isinstance(raw, dict) and "state_dict" in raw:
        return raw["state_dict"]
    if isinstance(raw, dict) and "model_state_dict" in raw:
        return raw["model_state_dict"]
    if isinstance(raw, dict) and all(hasattr(v, "shape") for v in raw.values()):
        return raw
    return raw


def load_grid_units(
    ckpt_path: Path, gridness_threshold: float, max_units: int, min_shift_cm: float
) -> Tuple[np.ndarray, Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """Return (grid units, class indices, score vectors) from cached gridness analysis."""
    grid_path = analysis_dir_for_checkpoint(ckpt_path) / "gridness_data.npz"
    if not grid_path.exists():
        raise FileNotFoundError(f"Missing gridness_data.npz for checkpoint: {grid_path}")
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
    score_vectors = class_score_vectors(grid_data, min_shift_cm)
    return grid_units, classes, score_vectors


def load_saved_class_ids(class_dir: Path, suffix: str, max_units: int) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """Load predictive/retrospective/grid IDs from a saved analysis-output folder."""
    def _load(name: str) -> np.ndarray:
        path = class_dir / f"{name}_{suffix}.npy"
        if not path.exists():
            raise FileNotFoundError(f"Missing saved class file: {path}")
        return np.asarray(np.load(path), dtype=int).reshape(-1)

    predictive = _load("predictive_ids")
    retrospective = _load("retrospective_ids")
    grid = _load("grid_ids")

    dead_path = class_dir / f"dead_unit_ids_{suffix}.npy"
    if dead_path.exists():
        dead_mask = np.asarray(np.load(dead_path), dtype=bool).reshape(-1)
        if dead_mask.size:
            alive = np.where(~dead_mask)[0]
            predictive = np.intersect1d(predictive, alive, assume_unique=False)
            retrospective = np.intersect1d(retrospective, alive, assume_unique=False)
            grid = np.intersect1d(grid, alive, assume_unique=False)

    normal = np.setdiff1d(grid, np.union1d(predictive, retrospective), assume_unique=False)
    grid_units = np.asarray(grid, dtype=int).reshape(-1)[:max_units]
    if grid_units.size == 0:
        raise RuntimeError(f"No saved grid units found in {class_dir} ({suffix}).")
    return grid_units, {
        "predictive": np.asarray(predictive, dtype=int).reshape(-1),
        "retrospective": np.asarray(retrospective, dtype=int).reshape(-1),
        "normal": np.asarray(normal, dtype=int).reshape(-1),
    }


def sanitize_units(units: Sequence[int], num_units: int, fallback: bool = False) -> np.ndarray:
    units = np.asarray(units, dtype=int)
    units = units[(units >= 0) & (units < num_units)]
    if fallback and units.size == 0:
        units = np.arange(min(256, num_units), dtype=int)
    return units


def sample_random_units(candidates: np.ndarray, count: int, rng: np.random.Generator) -> np.ndarray:
    if count <= 0 or candidates.size == 0:
        return np.array([], dtype=int)
    if count >= candidates.size:
        return candidates.astype(int, copy=False)
    return rng.choice(candidates, size=count, replace=False)


def select_percentile_units(indices: Sequence[int], score_vec: np.ndarray, percentile: float) -> np.ndarray:
    return select_top_percentile(indices, score_vec, percentile)


def format_percent(pct: float) -> str:
    if math.isfinite(pct) and abs(pct - round(pct)) < 1e-6:
        return str(int(round(pct)))
    return str(pct).replace(".", "p")


def configure_trajectory_options(
    options,
    style: str,
    fixed_speed: float | None,
    speed_scale: float | None,
    turn_sigma_scale: float | None,
    velocity_smoothing: float | None,
    speed_max: float | None,
) -> None:
    options.trajectory_style = style
    if speed_scale is not None:
        options.trajectory_speed_scale = float(speed_scale)
    if turn_sigma_scale is not None:
        options.trajectory_turn_sigma_scale = float(turn_sigma_scale)
    if velocity_smoothing is not None:
        options.trajectory_velocity_smoothing = float(velocity_smoothing)
    if speed_max is not None:
        options.trajectory_speed_max = float(speed_max)
    if style == "straight":
        if fixed_speed is None:
            fixed_speed = 0.2
        options.trajectory_fixed_speed = float(fixed_speed)
    else:
        options.trajectory_fixed_speed = None


def collect_eval_batches(
    traj_gen: TrajectoryGenerator,
    total_trajectories: int,
    batch_size: int,
) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    """Cache evaluation inputs for baseline/ablation comparisons."""
    batches: List[Tuple[torch.Tensor, torch.Tensor]] = []
    remaining = int(total_trajectories)
    while remaining > 0:
        bs = min(batch_size, remaining)
        inputs, _, _ = traj_gen.get_test_batch(batch_size=bs)
        v, init = inputs
        batches.append((v.detach().cpu().clone(), init.detach().cpu().clone()))
        remaining -= bs
    return batches


def run_batches(
    model: RNN,
    batches: List[Tuple[torch.Tensor, torch.Tensor]],
    device: torch.device,
    grid_units: np.ndarray,
) -> List[np.ndarray]:
    """Forward cached batches and return grid-unit states per batch."""
    outs: List[np.ndarray] = []
    with torch.no_grad():
        for v, init in batches:
            v = v.to(device)
            init = init.to(device)
            g = model.g((v, init)).cpu().numpy()
            g = g[:, :, grid_units]
            outs.append(g.astype(np.float32, copy=False))
    return outs


def build_baseline_cloud(
    model: RNN,
    traj_gen: TrajectoryGenerator,
    grid_units: np.ndarray,
    total_trajectories: int,
    batch_size: int,
    max_points: int | None,
    rng: np.random.Generator,
    device: torch.device,
) -> np.ndarray:
    """Sample baseline population activity for the manifold cloud."""
    total_trajectories = int(total_trajectories)
    if total_trajectories <= 0:
        raise ValueError("total_trajectories must be positive.")
    num_batches = int(math.ceil(total_trajectories / batch_size))
    per_batch = None
    if max_points is not None:
        per_batch = max(1, int(math.ceil(max_points / num_batches)))
    samples = []
    model.eval()
    with torch.no_grad():
        remaining = total_trajectories
        while remaining > 0:
            bs = min(batch_size, remaining)
            inputs, _, _ = traj_gen.get_test_batch(batch_size=bs)
            v, init = inputs
            v = v.to(device)
            init = init.to(device)
            g = model.g((v, init)).cpu().numpy()
            g = g[:, :, grid_units]
            flat = g.reshape(-1, g.shape[2])
            if per_batch is not None and flat.shape[0] > per_batch:
                idx = rng.choice(flat.shape[0], size=per_batch, replace=False)
                flat = flat[idx]
            samples.append(flat.astype(np.float32, copy=False))
            remaining -= bs
    cloud = np.concatenate(samples, axis=0) if samples else np.empty((0, len(grid_units)), dtype=np.float32)
    if max_points is not None and cloud.shape[0] > max_points:
        idx = rng.choice(cloud.shape[0], size=max_points, replace=False)
        cloud = cloud[idx]
    return cloud


def prepare_transform(
    cloud: np.ndarray,
    pca_components: int | None,
    zscore: bool,
) -> Tuple[np.ndarray, StandardScaler | None, PCA | None]:
    scaler = None
    pca = None
    transformed = cloud
    if zscore:
        scaler = StandardScaler()
        transformed = scaler.fit_transform(transformed)
    if pca_components is not None and pca_components > 0:
        n_components = min(int(pca_components), transformed.shape[1])
        pca = PCA(n_components=n_components, random_state=0)
        transformed = pca.fit_transform(transformed)
    return transformed, scaler, pca


def transform_states(
    states: np.ndarray,
    scaler: StandardScaler | None,
    pca: PCA | None,
) -> np.ndarray:
    transformed = states
    if scaler is not None:
        transformed = scaler.transform(transformed)
    if pca is not None:
        transformed = pca.transform(transformed)
    return transformed


def distances_to_cloud(
    nn: NearestNeighbors,
    batch_states: List[np.ndarray],
    scaler: StandardScaler | None,
    pca: PCA | None,
    knn_k: int,
) -> np.ndarray:
    n_neighbors = max(1, min(int(knn_k), int(getattr(nn, "n_samples_fit_", knn_k) or knn_k)))
    dist_batches = []
    for g in batch_states:
        flat = g.reshape(-1, g.shape[2])
        flat = transform_states(flat, scaler, pca)
        dist = nn.kneighbors(flat, n_neighbors=n_neighbors, return_distance=True)[0].mean(axis=1)
        dist_batches.append(dist.reshape(g.shape[0], g.shape[1]))
    return np.concatenate(dist_batches, axis=1)


def summarize_distances(distances: np.ndarray) -> Dict[str, np.ndarray]:
    mean = np.nanmean(distances, axis=1)
    p10 = np.nanpercentile(distances, 10, axis=1)
    p90 = np.nanpercentile(distances, 90, axis=1)
    return {"mean": mean, "p10": p10, "p90": p90}


def plot_distance_traces_multi(
    time_s: np.ndarray,
    stats_by_label: Dict[str, Dict[str, np.ndarray]],
    colors_by_label: Dict[str, str],
    out_path: Path,
    title: str,
) -> None:
    plt.figure(figsize=(6.6, 4.4))
    for label, stats in stats_by_label.items():
        color = colors_by_label.get(label, "#333333")
        plt.fill_between(
            time_s,
            stats["p10"],
            stats["p90"],
            color=color,
            alpha=0.16,
            linewidth=0,
        )
        plt.plot(time_s, stats["mean"], color=color, label=label)
    plt.xlabel("Time (s)")
    plt.ylabel("Distance to baseline manifold")
    plt.title(title)
    plt.grid(alpha=0.3)
    plt.legend(frameon=False, fontsize=8)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200)
    plt.close()


# --------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute off-manifold distances for baseline, predictive, random, and retrospective ablations.")
    parser.add_argument("--base-dir", type=str, default="Models/Single agent path integration")
    parser.add_argument("--checkpoint", type=str, default=None, help="Optional single checkpoint to process.")
    parser.add_argument("--class-dir", type=str, default=None, help="Optional directory containing saved predictive_ids_<suffix>.npy etc.")
    parser.add_argument("--class-suffix", type=str, default="random_walk", help="Suffix used for saved class files in --class-dir.")
    parser.add_argument("--output-dir", type=str, default=None, help="Optional output directory override.")
    parser.add_argument("--seed", type=int, default=None, help="Optional seed id to filter checkpoints.")
    parser.add_argument("--cloud-trajectories", type=int, default=1000)
    parser.add_argument("--eval-trajectories", type=int, default=100)
    parser.add_argument("--cloud-batch-size", type=int, default=100)
    parser.add_argument("--eval-batch-size", type=int, default=100)
    parser.add_argument("--sequence-seconds", type=float, default=5.0)
    parser.add_argument("--gridness-threshold", type=float, default=0.5)
    parser.add_argument("--min-shift-cm", type=float, default=5.0)
    parser.add_argument("--max-units", type=int, default=512)
    parser.add_argument("--max-cloud-points", type=int, default=80000)
    parser.add_argument("--pca-components", type=int, default=0, help="Set to 0 to disable PCA.")
    parser.add_argument("--zscore", action="store_true", help="Alias for enabling z-scoring (on by default).")
    parser.add_argument("--no-zscore", action="store_true", help="Disable z-scoring before distance calculations.")
    parser.add_argument("--nn-algorithm", type=str, default="ball_tree", choices=["auto", "ball_tree", "kd_tree", "brute"])
    parser.add_argument("--leaf-size", type=int, default=40)
    parser.add_argument("--knn-k", type=int, default=5, help="Number of neighbors used for k-NN distance.")
    parser.add_argument(
        "--ablation-percentages",
        type=str,
        default="25,75,100",
        help="Comma-separated list of percent ablations within each class.",
    )
    parser.add_argument(
        "--trajectory-styles",
        type=str,
        default="random_walk",
        help="Comma-separated list: random_walk, straight, per_step_random.",
    )
    parser.add_argument("--trajectory-fixed-speed", type=float, default=None)
    parser.add_argument("--trajectory-speed-scale", type=float, default=None)
    parser.add_argument("--trajectory-speed-max", type=float, default=None)
    parser.add_argument("--trajectory-turn-sigma-scale", type=float, default=None)
    parser.add_argument("--trajectory-velocity-smoothing", type=float, default=None)
    parser.add_argument("--score-batches", type=int, default=8, help="Batches used to rank saved predictive/retrospective units.")
    parser.add_argument("--max-lag", type=int, default=20, help="Time-lag sweep used when ranking saved class IDs.")
    parser.add_argument("--rng-seed", type=int, default=123)
    args = parser.parse_args()

    if args.cloud_trajectories <= 0 or args.eval_trajectories <= 0:
        raise SystemExit("cloud/eval trajectories must be positive.")
    if args.knn_k <= 0:
        raise SystemExit("--knn-k must be positive.")

    pct_list = [p for p in re.split(r"[,\s]+", str(args.ablation_percentages).strip()) if p]
    ablation_percentages = [float(p) for p in pct_list] if pct_list else [100.0]

    style_list = [s for s in re.split(r"[,\s]+", str(args.trajectory_styles).strip()) if s]
    trajectory_styles = [s.lower() for s in style_list] if style_list else ["random_walk"]
    valid_styles = {"random_walk", "straight", "per_step_random"}
    unknown_styles = [s for s in trajectory_styles if s not in valid_styles]
    if unknown_styles:
        raise SystemExit(f"Unknown trajectory style(s): {unknown_styles}.")

    np.random.seed(args.rng_seed)
    rng = np.random.default_rng(args.rng_seed)

    if args.checkpoint:
        ckpt_path = Path(args.checkpoint)
        checkpoints = [(args.seed or -1, ckpt_path)]
    else:
        base = Path(args.base_dir)
        checkpoints = find_checkpoints(base)
        if args.seed is not None:
            checkpoints = [pair for pair in checkpoints if pair[0] == args.seed]

    if not checkpoints:
        raise SystemExit("No checkpoints found for off-manifold analysis.")

    device = torch.device("cpu")

    for seed, ckpt_path in checkpoints:
        print(f"[seed {seed}] Loading {ckpt_path}")
        raw = torch.load(ckpt_path, map_location="cpu")
        state = extract_state(raw)

        options = build_options(Path(ckpt_path), device=str(device))
        dt = float(getattr(options, "trajectory_dt", 0.02))

        for style in trajectory_styles:
            options_style = build_options(Path(ckpt_path), device=str(device))
            options_style.sequence_length = max(2, int(math.ceil(args.sequence_seconds / dt)))
            options_style.batch_size = max(args.cloud_batch_size, args.eval_batch_size)
            options_style.run_ID = f"off_manifold_seed_{seed}_{style}"
            configure_trajectory_options(
                options_style,
                style,
                args.trajectory_fixed_speed,
                args.trajectory_speed_scale,
                args.trajectory_turn_sigma_scale,
                args.trajectory_velocity_smoothing,
                args.trajectory_speed_max,
            )
            place_cells = PlaceCells(options_style)
            traj_gen = TrajectoryGenerator(options_style, place_cells)

            model = RNN(options_style, place_cells).to(device)
            model.load_state_dict(state)
            model.eval()

            if args.class_dir:
                grid_units, class_indices = load_saved_class_ids(
                    Path(args.class_dir),
                    args.class_suffix,
                    args.max_units,
                )
                ranking_units = np.unique(
                    np.concatenate(
                        [
                            np.asarray(class_indices.get("predictive", []), dtype=int),
                            np.asarray(class_indices.get("retrospective", []), dtype=int),
                        ]
                    )
                )
                ranking_units = ranking_units[(ranking_units >= 0) & (ranking_units < model.Ng)]
                score_vectors = {
                    "predictive": np.full(model.Ng, np.nan, dtype=float),
                    "retrospective": np.full(model.Ng, np.nan, dtype=float),
                    "normal": np.full(model.Ng, np.nan, dtype=float),
                }
                if ranking_units.size:
                    lags = build_shift_values("time", max_lag=args.max_lag)
                    scores_60, _, xs, ys, _, _ = predictive_gridness_analysis(
                        model,
                        traj_gen,
                        options_style,
                        lags=lags,
                        res=20,
                        n_batches=args.score_batches,
                        Ng=int(ranking_units.size),
                        idxs=ranking_units,
                    )
                    lag_cm = np.asarray(lags, dtype=float) * float(cm_per_step_from_positions(xs, ys))
                    partial_scores = class_score_vectors(
                        {"lag_cm": lag_cm, "scores_60": np.asarray(scores_60, dtype=float)},
                        args.min_shift_cm,
                    )
                    score_vectors["predictive"][ranking_units] = partial_scores["predictive"]
                    score_vectors["retrospective"][ranking_units] = partial_scores["retrospective"]
                class_source = {
                    "mode": "saved_ids",
                    "class_dir": str(Path(args.class_dir).resolve()),
                    "class_suffix": args.class_suffix,
                }
            else:
                grid_units, class_indices, score_vectors = load_grid_units(
                    ckpt_path,
                    gridness_threshold=args.gridness_threshold,
                    max_units=args.max_units,
                    min_shift_cm=args.min_shift_cm,
                )
                class_source = {
                    "mode": "gridness_data",
                    "gridness_threshold": float(args.gridness_threshold),
                    "min_shift_cm": float(args.min_shift_cm),
                }

            grid_units = sanitize_units(grid_units, model.Ng, fallback=True)
            predictive_units = sanitize_units(class_indices.get("predictive", []), model.Ng, fallback=False)
            retrospective_units = sanitize_units(class_indices.get("retrospective", []), model.Ng, fallback=False)

            max_cloud_points = args.max_cloud_points if args.max_cloud_points and args.max_cloud_points > 0 else None
            cloud = build_baseline_cloud(
                model,
                traj_gen,
                grid_units,
                args.cloud_trajectories,
                args.cloud_batch_size,
                max_cloud_points,
                rng,
                device,
            )
            if cloud.shape[0] < 2:
                raise RuntimeError(f"[seed {seed}] Baseline cloud is empty; increase cloud trajectories or max points.")

            use_pca = args.pca_components if args.pca_components and args.pca_components > 0 else None
            use_zscore = bool(args.zscore or not args.no_zscore or use_pca is not None)
            cloud_transformed, scaler, pca = prepare_transform(cloud, use_pca, use_zscore)

            knn_k = min(int(args.knn_k), cloud_transformed.shape[0])
            nn = NearestNeighbors(
                n_neighbors=knn_k,
                algorithm=args.nn_algorithm,
                leaf_size=args.leaf_size,
                n_jobs=-1,
            )
            nn.fit(cloud_transformed)

            eval_batches = collect_eval_batches(traj_gen, args.eval_trajectories, args.eval_batch_size)
            baseline_states = run_batches(model, eval_batches, device, grid_units)
            baseline_dist = distances_to_cloud(nn, baseline_states, scaler, pca, knn_k)
            time_s = np.arange(baseline_dist.shape[0]) * dt
            baseline_stats = summarize_distances(baseline_dist)

            for pct in ablation_percentages:
                pct_label = format_percent(pct)
                pred_units = select_percentile_units(predictive_units, score_vectors["predictive"], pct)
                retro_units = select_percentile_units(retrospective_units, score_vectors["retrospective"], pct)
                rand_units = sample_random_units(grid_units, int(pred_units.size), rng)

                stats_by_label = {"Baseline": baseline_stats}
                dist_by_label = {"baseline": baseline_dist}

                if pred_units.size:
                    pred_model = RNN(options_style, place_cells).to(device)
                    pred_model.load_state_dict(state)
                    zero_unit_weights_in_place(pred_model, pred_units)
                    pred_model.eval()
                    pred_states = run_batches(pred_model, eval_batches, device, grid_units)
                    pred_dist = distances_to_cloud(nn, pred_states, scaler, pca, knn_k)
                else:
                    pred_dist = baseline_dist
                stats_by_label["Predictive ablation"] = summarize_distances(pred_dist)
                dist_by_label["predictive"] = pred_dist

                if rand_units.size:
                    rand_model = RNN(options_style, place_cells).to(device)
                    rand_model.load_state_dict(state)
                    zero_unit_weights_in_place(rand_model, rand_units)
                    rand_model.eval()
                    rand_states = run_batches(rand_model, eval_batches, device, grid_units)
                    rand_dist = distances_to_cloud(nn, rand_states, scaler, pca, knn_k)
                else:
                    rand_dist = baseline_dist
                stats_by_label["Random ablation"] = summarize_distances(rand_dist)
                dist_by_label["random"] = rand_dist

                if retro_units.size:
                    retro_model = RNN(options_style, place_cells).to(device)
                    retro_model.load_state_dict(state)
                    zero_unit_weights_in_place(retro_model, retro_units)
                    retro_model.eval()
                    retro_states = run_batches(retro_model, eval_batches, device, grid_units)
                    retro_dist = distances_to_cloud(nn, retro_states, scaler, pca, knn_k)
                else:
                    retro_dist = baseline_dist
                stats_by_label["Retrospective ablation"] = summarize_distances(retro_dist)
                dist_by_label["retrospective"] = retro_dist

                if args.output_dir:
                    out_dir = Path(args.output_dir)
                else:
                    out_dir = analysis_dir_for_checkpoint(ckpt_path) / "torus"
                out_dir.mkdir(parents=True, exist_ok=True)
                stem = f"off_manifold_seed_{seed}_{style}_pct{pct_label}_multi"

                np.savez(
                    out_dir / f"{stem}.npz",
                    time_s=time_s,
                    baseline_distances=baseline_dist.astype(np.float32),
                    predictive_distances=dist_by_label["predictive"].astype(np.float32),
                    random_distances=dist_by_label["random"].astype(np.float32),
                    retrospective_distances=dist_by_label["retrospective"].astype(np.float32),
                    baseline_mean=stats_by_label["Baseline"]["mean"],
                    baseline_p10=stats_by_label["Baseline"]["p10"],
                    baseline_p90=stats_by_label["Baseline"]["p90"],
                    predictive_mean=stats_by_label["Predictive ablation"]["mean"],
                    predictive_p10=stats_by_label["Predictive ablation"]["p10"],
                    predictive_p90=stats_by_label["Predictive ablation"]["p90"],
                    random_mean=stats_by_label["Random ablation"]["mean"],
                    random_p10=stats_by_label["Random ablation"]["p10"],
                    random_p90=stats_by_label["Random ablation"]["p90"],
                    retrospective_mean=stats_by_label["Retrospective ablation"]["mean"],
                    retrospective_p10=stats_by_label["Retrospective ablation"]["p10"],
                    retrospective_p90=stats_by_label["Retrospective ablation"]["p90"],
                    grid_units=grid_units,
                    predictive_units=pred_units,
                    random_units=rand_units,
                    retrospective_units=retro_units,
                )

                summary = {
                    "seed": seed,
                    "checkpoint": str(ckpt_path),
                    "class_source": class_source,
                    "trajectory_style": style,
                    "ablation_percent": float(pct),
                    "cloud_trajectories": int(args.cloud_trajectories),
                    "eval_trajectories": int(args.eval_trajectories),
                    "sequence_length": int(baseline_dist.shape[0]),
                    "dt": dt,
                    "cloud_points": int(cloud.shape[0]),
                    "grid_units_used": int(grid_units.size),
                    "predictive_units_total": int(predictive_units.size),
                    "retrospective_units_total": int(retrospective_units.size),
                    "predictive_units_ablated": int(pred_units.size),
                    "retrospective_units_ablated": int(retro_units.size),
                    "random_units_ablated": int(rand_units.size),
                    "gridness_threshold": float(args.gridness_threshold),
                    "min_shift_cm": float(args.min_shift_cm),
                    "max_units": int(args.max_units),
                    "max_cloud_points": int(max_cloud_points) if max_cloud_points is not None else None,
                    "pca_components": int(use_pca) if use_pca is not None else None,
                    "zscore": bool(use_zscore),
                    "nn_algorithm": str(args.nn_algorithm),
                    "leaf_size": int(args.leaf_size),
                    "knn_k": int(knn_k),
                    "baseline_mean_distance": float(np.nanmean(baseline_dist)),
                    "predictive_mean_distance": float(np.nanmean(dist_by_label["predictive"])),
                    "random_mean_distance": float(np.nanmean(dist_by_label["random"])),
                    "retrospective_mean_distance": float(np.nanmean(dist_by_label["retrospective"])),
                }
                with open(out_dir / f"{stem}.json", "w") as f:
                    json.dump(summary, f, indent=2)

                colors_by_label = {
                    "Baseline": "#ff7f0e",
                    "Predictive ablation": "#1f77b4",
                    "Random ablation": "#7f7f7f",
                    "Retrospective ablation": "#2ca02c",
                }
                plot_distance_traces_multi(
                    time_s,
                    stats_by_label,
                    colors_by_label,
                    out_dir / f"{stem}.png",
                    title=f"Seed {seed}: distance to baseline manifold ({style}, {pct_label}%)",
                )
                print(f"[seed {seed}] Wrote off-manifold outputs to {out_dir} ({style}, {pct_label}%).")


if __name__ == "__main__":
    main()
