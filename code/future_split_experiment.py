#!/usr/bin/env python3
"""Matched-state future-split experiment for cued predictive grid cells.

This script adds a compact T-maze/fork task where paired left/right trials share
the same position and velocity through the stem, after an early route cue has
disappeared.  A future-place auxiliary loss encourages the RNN state to carry
the selected future before the body diverges.

Subcommands:
  train      Train a cued fork RNN and save full + core checkpoints.
  classify   Classify predictive/retrospective/standard grid units on zero-cue
             open-field trajectories.
  decode     Decode future route, position, and torus phase from matched fork
             trials and controls.
  intervene  Ablate/scramble/swap predictive-unit activity before the branch.
  smoke      Tiny CPU-only end-to-end smoke test.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import os
import sys
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/future_split_mplconfig")
os.environ.setdefault("NUMBA_CACHE_DIR", "/private/tmp/future_split_numba_cache")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
Path(os.environ["NUMBA_CACHE_DIR"]).mkdir(parents=True, exist_ok=True)

import numpy as np
import torch
import torch.nn.functional as F

try:
    from sklearn.linear_model import LogisticRegression, Ridge
    from sklearn.metrics import accuracy_score, roc_auc_score
    from sklearn.preprocessing import StandardScaler
except Exception as exc:  # pragma: no cover - smoke environment should have sklearn.
    raise RuntimeError(
        "future_split_experiment.py needs scikit-learn for decoding controls."
    ) from exc

from model import RNN as CoreRNN
from place_cells import PlaceCells
from scores import GridScorer, band_scores
from trajectory_generator import TrajectoryGenerator


# --------------------------------------------------------------------------------------
# Small utilities
# --------------------------------------------------------------------------------------


LOGGER = logging.getLogger("future_split")


def format_duration(seconds: float) -> str:
    seconds = float(seconds)
    if not np.isfinite(seconds) or seconds < 0:
        return "?"
    total = int(round(seconds))
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours:d}h {minutes:02d}m {secs:02d}s"
    if minutes:
        return f"{minutes:d}m {secs:02d}s"
    return f"{secs:d}s"


def configure_logging(out_dir: Path | str, filename: str) -> Path:
    """Configure timestamped console + file logging for one subcommand."""
    out_dir = ensure_dir(out_dir)
    log_path = out_dir / filename
    LOGGER.handlers.clear()
    LOGGER.setLevel(logging.INFO)
    LOGGER.propagate = False
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(formatter)
    file_handler = logging.FileHandler(log_path, mode="a")
    file_handler.setFormatter(formatter)
    LOGGER.addHandler(console)
    LOGGER.addHandler(file_handler)
    LOGGER.info("logging to %s", log_path)
    return log_path


def log_info(message: str, *args) -> None:
    if LOGGER.handlers:
        LOGGER.info(message, *args)
    elif args:
        print(message % args, flush=True)
    else:
        print(message, flush=True)


def parameter_count(model: torch.nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters()))


def set_seed(seed: int) -> np.random.Generator:
    """Set NumPy/Torch seeds and return a NumPy Generator."""
    seed = int(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    return np.random.default_rng(seed)


def ensure_dir(path: Path | str) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def json_safe(obj):
    if isinstance(obj, dict):
        return {str(k): json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_safe(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return json_safe(obj.tolist())
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, Path):
        return str(obj)
    return obj


def write_json(path: Path, payload: Dict) -> None:
    ensure_dir(path.parent)
    with open(path, "w") as f:
        json.dump(json_safe(payload), f, indent=2)


def write_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    ensure_dir(path.parent)
    if not rows:
        rows = [{"status": "empty"}]
    fields: List[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fields:
                fields.append(key)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _to_float_list(text_or_values: str | Sequence[float]) -> List[float]:
    if isinstance(text_or_values, str):
        pieces = [p for p in text_or_values.replace(",", " ").split() if p]
        return [float(p) for p in pieces]
    return [float(x) for x in text_or_values]


def _safe_nanargmax(arr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    arr = np.asarray(arr, dtype=float)
    idxs = np.full(arr.shape[1], -1, dtype=int)
    vals = np.full(arr.shape[1], np.nan, dtype=float)
    for u in range(arr.shape[1]):
        col = arr[:, u]
        if np.isfinite(col).any():
            idx = int(np.nanargmax(col))
            idxs[u] = idx
            vals[u] = col[idx]
    return idxs, vals


def cm_per_step(xs: np.ndarray, ys: np.ndarray) -> float:
    dx = np.diff(xs, axis=0)
    dy = np.diff(ys, axis=0)
    step = np.sqrt(dx**2 + dy**2)
    return float(np.nanmean(step) * 100.0)


def angular_distance(theta_a: np.ndarray, theta_b: np.ndarray) -> np.ndarray:
    """Absolute circular distance between angles in radians."""
    return np.abs(np.angle(np.exp(1j * (theta_a - theta_b))))


def phase_feature_error(pred: np.ndarray, target: np.ndarray) -> float:
    """Mean two-angle phase error in radians for [cos1,sin1,cos2,sin2]."""
    pred = np.asarray(pred, dtype=float)
    target = np.asarray(target, dtype=float)
    if pred.size == 0 or target.size == 0:
        return float("nan")
    th1_p = np.arctan2(pred[:, 1], pred[:, 0])
    th1_t = np.arctan2(target[:, 1], target[:, 0])
    th2_p = np.arctan2(pred[:, 3], pred[:, 2])
    th2_t = np.arctan2(target[:, 3], target[:, 2])
    return float(np.mean(0.5 * (angular_distance(th1_p, th1_t) + angular_distance(th2_p, th2_t))))


def phase_feature_distance(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Per-sample two-angle distance for [cos1,sin1,cos2,sin2]."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    th1_a = np.arctan2(a[:, 1], a[:, 0])
    th1_b = np.arctan2(b[:, 1], b[:, 0])
    th2_a = np.arctan2(a[:, 3], a[:, 2])
    th2_b = np.arctan2(b[:, 3], b[:, 2])
    return 0.5 * (angular_distance(th1_a, th1_b) + angular_distance(th2_a, th2_b))


# --------------------------------------------------------------------------------------
# Options and model
# --------------------------------------------------------------------------------------


COMMON_DEFAULTS = {
    "batch_size": 64,
    "sequence_length": 32,
    "Np": 256,
    "Ng": 512,
    "velocity_dim": 3,
    "place_cell_rf": 0.12,
    "surround_scale": 2.0,
    "RNN_type": "RNN",
    "activation": "relu",
    "weight_decay": 1e-4,
    "DoG": True,
    "periodic": False,
    "box_width": 2.2,
    "box_height": 2.2,
    "learning_rate": 1e-4,
    "device": "cpu",
    "trajectory_dt": 0.02,
    "trajectory_style": "random_walk",
    "trajectory_speed_scale": 1.0,
    "trajectory_speed_max": None,
    "trajectory_velocity_smoothing": 0.0,
    "trajectory_turn_sigma_scale": 1.0,
    "trajectory_border_region": 0.03,
    "trajectory_wall_slowdown": 0.25,
    "trajectory_wall_turn_scale": 1.0,
    "save_dir": ".",
    "run_ID": "future_split",
    "fork_cue_steps": 4,
    "fork_branch_step": 20,
    "fork_stem_start_y": -0.85,
    "fork_branch_y": 0.0,
    "fork_arm_length": 0.75,
    "fork_cue_scale": 1.0,
    "fork_lateral_jitter": 0.0,
    "future_horizon": 8,
    "future_loss_weight": 0.5,
}


def make_options(**kwargs) -> SimpleNamespace:
    cfg = dict(COMMON_DEFAULTS)
    cfg.update({k: v for k, v in kwargs.items() if v is not None})
    cfg["velocity_dim"] = 3
    cfg["DoG"] = bool(cfg["DoG"])
    cfg["periodic"] = bool(cfg["periodic"])
    cfg["device"] = str(cfg["device"])
    return SimpleNamespace(**cfg)


def options_to_dict(options: SimpleNamespace) -> Dict[str, object]:
    out = {}
    for key, val in vars(options).items():
        if isinstance(val, (str, int, float, bool)) or val is None:
            out[key] = val
    return out


class FutureSplitRNN(torch.nn.Module):
    """RNN with current-place and future-place readouts."""

    def __init__(self, options: SimpleNamespace, place_cells: PlaceCells):
        super().__init__()
        self.Ng = int(options.Ng)
        self.Np = int(options.Np)
        self.velocity_dim = int(getattr(options, "velocity_dim", 3))
        self.sequence_length = int(options.sequence_length)
        self.weight_decay = float(options.weight_decay)
        self.place_cells = place_cells

        self.encoder = torch.nn.Linear(self.Np, self.Ng, bias=False)
        self.RNN = torch.nn.RNN(
            input_size=self.velocity_dim,
            hidden_size=self.Ng,
            nonlinearity=options.activation,
            bias=False,
        )
        self.decoder = torch.nn.Linear(self.Ng, self.Np, bias=False)
        self.future_decoder = torch.nn.Linear(self.Ng, self.Np, bias=False)

    def g(self, inputs: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        v, p0 = inputs
        init_state = self.encoder(p0)[None]
        g, _ = self.RNN(v, init_state)
        return g

    def predict(self, inputs: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        return self.decoder(self.g(inputs))

    def future_predict(self, inputs: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        return self.future_decoder(self.g(inputs))

    def compute_loss(
        self,
        inputs: Tuple[torch.Tensor, torch.Tensor],
        current_pc: torch.Tensor,
        future_pc: torch.Tensor,
        future_mask: torch.Tensor,
        future_weight: float,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        states = self.g(inputs)
        current_logits = self.decoder(states)
        future_logits = self.future_decoder(states)

        current_logp = F.log_softmax(current_logits, dim=-1)
        current_ce = -(current_pc * current_logp).sum(-1).mean()

        future_logp = F.log_softmax(future_logits, dim=-1)
        future_ce_per = -(future_pc * future_logp).sum(-1)
        mask = future_mask.to(future_ce_per.device).float()
        denom = torch.clamp(mask.sum(), min=1.0)
        future_ce = (future_ce_per * mask).sum() / denom

        reg = self.weight_decay * (self.RNN.weight_hh_l0**2).sum()
        loss = current_ce + float(future_weight) * future_ce + reg
        terms = {
            "loss": float(loss.detach().cpu()),
            "current_ce": float(current_ce.detach().cpu()),
            "future_ce": float(future_ce.detach().cpu()),
            "rec_reg": float(reg.detach().cpu()),
        }
        return loss, terms

    def core_state_dict(self) -> Dict[str, torch.Tensor]:
        """State dict compatible with code/model.py, excluding future_decoder."""
        return {
            "encoder.weight": self.encoder.weight.detach().cpu(),
            "RNN.weight_ih_l0": self.RNN.weight_ih_l0.detach().cpu(),
            "RNN.weight_hh_l0": self.RNN.weight_hh_l0.detach().cpu(),
            "decoder.weight": self.decoder.weight.detach().cpu(),
        }


def load_future_model(checkpoint_path: str | Path, device: str = "cpu") -> Tuple[FutureSplitRNN, PlaceCells, SimpleNamespace, Dict]:
    """Load a full future-split checkpoint."""
    payload = torch.load(checkpoint_path, map_location=device)
    if not (isinstance(payload, dict) and "model_state_dict" in payload):
        raise ValueError(
            f"{checkpoint_path} is not a full future-split checkpoint. "
            "Use the *_full.pth checkpoint for decode/intervene."
        )
    config = dict(COMMON_DEFAULTS)
    config.update(payload.get("config", {}))
    config["device"] = device
    options = make_options(**config)
    place_cells = PlaceCells(options)
    model = FutureSplitRNN(options, place_cells).to(device)
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    return model, place_cells, options, payload


def load_core_model(checkpoint_path: str | Path, options: SimpleNamespace, place_cells: PlaceCells) -> CoreRNN:
    """Load only the encoder/RNN/decoder into the existing CoreRNN type."""
    raw = torch.load(checkpoint_path, map_location=options.device)
    if isinstance(raw, dict) and "core_state_dict" in raw:
        state = raw["core_state_dict"]
    elif isinstance(raw, dict) and "model_state_dict" in raw:
        state = {
            k: v
            for k, v in raw["model_state_dict"].items()
            if not k.startswith("future_decoder.")
        }
    elif isinstance(raw, dict) and all(hasattr(v, "shape") for v in raw.values()):
        state = raw
    else:
        raise TypeError(f"Unsupported checkpoint format: {type(raw)}")
    model = CoreRNN(options, place_cells).to(options.device)
    model.load_state_dict(state)
    model.eval()
    return model


# --------------------------------------------------------------------------------------
# Fork and open-field data
# --------------------------------------------------------------------------------------


@dataclass
class ForkBatch:
    inputs: Tuple[torch.Tensor, torch.Tensor]
    pos: torch.Tensor
    place_outputs: torch.Tensor
    future_place_outputs: torch.Tensor
    future_mask: torch.Tensor
    routes: np.ndarray
    pair_ids: np.ndarray
    positions_np: np.ndarray
    velocity_np: np.ndarray
    cue_np: np.ndarray


def generate_fork_batch(
    options: SimpleNamespace,
    place_cells: PlaceCells,
    batch_size: Optional[int] = None,
    rng: Optional[np.random.Generator] = None,
    paired: bool = True,
    future_horizon: Optional[int] = None,
) -> ForkBatch:
    """Generate paired T-maze/fork trajectories.

    Paired mode orders trials as [left_0, right_0, left_1, right_1, ...].
    Positions and velocities are identical within each pair until branch_step.
    """
    rng = rng or np.random.default_rng()
    B = int(batch_size or options.batch_size)
    if paired:
        B = max(2, B)
        if B % 2:
            B -= 1
        pair_count = B // 2
        routes = np.tile(np.array([-1, 1], dtype=float), pair_count)
        pair_ids = np.repeat(np.arange(pair_count, dtype=int), 2)
        pair_x = rng.normal(0.0, float(options.fork_lateral_jitter), size=pair_count)
        x0 = np.repeat(pair_x, 2)
        pair_y0 = np.full(pair_count, float(options.fork_stem_start_y))
        y0 = np.repeat(pair_y0, 2)
    else:
        routes = rng.choice(np.array([-1.0, 1.0]), size=B)
        pair_ids = np.arange(B, dtype=int)
        x0 = rng.normal(0.0, float(options.fork_lateral_jitter), size=B)
        y0 = np.full(B, float(options.fork_stem_start_y))

    T = int(options.sequence_length)
    branch_step = int(np.clip(options.fork_branch_step, 2, T - 2))
    cue_steps = int(np.clip(options.fork_cue_steps, 1, branch_step - 1))
    branch_y = float(options.fork_branch_y)
    arm_length = float(options.fork_arm_length)
    horizon = int(future_horizon if future_horizon is not None else options.future_horizon)

    pos = np.zeros((T, B, 2), dtype=np.float32)
    for t in range(T):
        if t < branch_step:
            frac = t / max(branch_step - 1, 1)
            pos[t, :, 0] = x0
            pos[t, :, 1] = y0 + frac * (branch_y - y0)
        else:
            frac = (t - branch_step + 1) / max(T - branch_step, 1)
            pos[t, :, 0] = x0 + routes * arm_length * frac
            pos[t, :, 1] = branch_y

    vel_xy = np.zeros_like(pos)
    vel_xy[1:] = pos[1:] - pos[:-1]
    cue = np.zeros((T, B, 1), dtype=np.float32)
    cue[:cue_steps, :, 0] = (routes * float(options.fork_cue_scale))[None, :]
    v = np.concatenate([vel_xy, cue], axis=-1)

    future_idx = np.minimum(np.arange(T) + horizon, T - 1)
    future_pos = pos[future_idx]
    future_mask = ((np.arange(T) + horizon) < T).astype(np.float32)[:, None]
    future_mask = np.repeat(future_mask, B, axis=1)

    device = torch.device(options.device)
    pos_t = torch.tensor(pos, dtype=torch.float32, device=device)
    v_t = torch.tensor(v, dtype=torch.float32, device=device)
    init_pos = torch.tensor(pos[0], dtype=torch.float32, device=device)[:, None, :]
    init_actv = place_cells.get_activation(init_pos).squeeze(1)
    place_outputs = place_cells.get_activation(pos_t)
    future_place_outputs = place_cells.get_activation(torch.tensor(future_pos, dtype=torch.float32, device=device))
    future_mask_t = torch.tensor(future_mask, dtype=torch.float32, device=device)

    return ForkBatch(
        inputs=(v_t, init_actv),
        pos=pos_t,
        place_outputs=place_outputs,
        future_place_outputs=future_place_outputs,
        future_mask=future_mask_t,
        routes=routes.astype(int),
        pair_ids=pair_ids,
        positions_np=pos,
        velocity_np=vel_xy,
        cue_np=cue,
    )


def assert_matched_prebranch(batch: ForkBatch, options: SimpleNamespace) -> None:
    cue_steps = int(options.fork_cue_steps)
    branch_step = int(options.fork_branch_step)
    for pair_id in np.unique(batch.pair_ids):
        idx = np.where(batch.pair_ids == pair_id)[0]
        if idx.size != 2:
            continue
        a, b = idx
        pos_a = batch.positions_np[cue_steps:branch_step, a]
        pos_b = batch.positions_np[cue_steps:branch_step, b]
        vel_a = batch.velocity_np[cue_steps:branch_step, a]
        vel_b = batch.velocity_np[cue_steps:branch_step, b]
        cue_a = batch.cue_np[cue_steps:branch_step, a]
        cue_b = batch.cue_np[cue_steps:branch_step, b]
        if not (np.allclose(pos_a, pos_b) and np.allclose(vel_a, vel_b) and np.allclose(cue_a, 0) and np.allclose(cue_b, 0)):
            raise AssertionError("Fork-pair present-state matching failed after cue and before branch.")


def pad_zero_cue(v2: torch.Tensor) -> torch.Tensor:
    z = torch.zeros((*v2.shape[:2], 1), dtype=v2.dtype, device=v2.device)
    return torch.cat([v2, z], dim=-1)


def collect_open_field_sequences(
    model: FutureSplitRNN | CoreRNN,
    options: SimpleNamespace,
    place_cells: PlaceCells,
    n_batches: int,
    batch_size: int,
    Ng_use: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Collect zero-cue open-field activity for gridness classification."""
    traj_options = make_options(**options_to_dict(options))
    traj_options.batch_size = int(batch_size)
    traj_options.velocity_dim = 3
    traj_gen = TrajectoryGenerator(traj_options, place_cells)
    xs_list, ys_list, g_list = [], [], []
    Ng_use = min(int(Ng_use), int(options.Ng))
    idxs = np.arange(Ng_use, dtype=int)
    model.eval()
    with torch.no_grad():
        for _ in range(max(1, int(n_batches))):
            inputs2, pos_batch, _ = traj_gen.get_test_batch(batch_size=batch_size)
            v2, init = inputs2
            v3 = pad_zero_cue(v2.to(options.device))
            init = init.to(options.device)
            g = model.g((v3, init)).detach().cpu().numpy()[:, :, idxs]
            pos_np = pos_batch.detach().cpu().numpy()
            xs_list.append(pos_np[:, :, 0])
            ys_list.append(pos_np[:, :, 1])
            g_list.append(g)
    return (
        np.concatenate(xs_list, axis=1),
        np.concatenate(ys_list, axis=1),
        np.concatenate(g_list, axis=1),
    )


def compute_rate_maps(
    xs: np.ndarray,
    ys: np.ndarray,
    activations: np.ndarray,
    scorer: GridScorer,
) -> np.ndarray:
    T, B, Ng = activations.shape
    flat_x = xs.reshape(-1)
    flat_y = ys.reshape(-1)
    maps = np.zeros((Ng, scorer._nbins, scorer._nbins), dtype=np.float32)
    for u in range(Ng):
        maps[u] = scorer.calculate_ratemap(flat_x, flat_y, activations[:, :, u].reshape(-1), statistic="mean")
    return maps


# --------------------------------------------------------------------------------------
# Torus phase helpers
# --------------------------------------------------------------------------------------


@dataclass
class PhaseProjector:
    status: str
    basis: object | None
    units: np.ndarray
    torus_radii: Tuple[float, float] = (1.0, 0.35)


def build_phase_projector(
    rate_maps: Optional[np.ndarray],
    grid_units: np.ndarray,
    options: SimpleNamespace,
    min_units: int = 4,
) -> PhaseProjector:
    """Build a torus phase projector, falling back for tiny smoke models."""
    grid_units = np.asarray(grid_units, dtype=int)
    if rate_maps is None or grid_units.size < min_units:
        units = grid_units if grid_units.size else np.arange(min(min_units, int(options.Ng)), dtype=int)
        return PhaseProjector(status="fallback_insufficient_units", basis=None, units=units)
    try:
        from toroidal_structure_analysis import build_torus_basis

        basis = build_torus_basis(rate_maps, grid_units, float(options.box_width))
        return PhaseProjector(status="torus_basis", basis=basis, units=np.asarray(basis.units, dtype=int))
    except Exception as exc:
        units = grid_units[: min(max(min_units, 4), grid_units.size)]
        return PhaseProjector(status=f"fallback_torus_failed:{type(exc).__name__}", basis=None, units=units)


def phase_features_from_states(states: np.ndarray, projector: PhaseProjector) -> np.ndarray:
    """Return [T,B,4] = cos/sin features for two torus angles."""
    T, B, Ng = states.shape
    if projector.basis is not None:
        try:
            from toroidal_structure_analysis import project_states_to_torus

            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                proj = project_states_to_torus(states, projector.basis, projector.torus_radii, T, B)
            features = np.stack(
                [
                    np.cos(proj.theta1),
                    np.sin(proj.theta1),
                    np.cos(proj.theta2),
                    np.sin(proj.theta2),
                ],
                axis=-1,
            ).astype(np.float32)
            if np.isfinite(features).all():
                return features
        except Exception:
            pass

    units = np.asarray(projector.units, dtype=int)
    units = units[(units >= 0) & (units < Ng)]
    if units.size == 0:
        units = np.arange(min(4, Ng), dtype=int)
    vals = states[:, :, units]
    if vals.shape[-1] < 4:
        vals = np.pad(vals, ((0, 0), (0, 0), (0, 4 - vals.shape[-1])), mode="constant")
    theta1 = np.arctan2(vals[:, :, 1], vals[:, :, 0] + 1e-8)
    theta2 = np.arctan2(vals[:, :, 3], vals[:, :, 2] + 1e-8)
    return np.stack([np.cos(theta1), np.sin(theta1), np.cos(theta2), np.sin(theta2)], axis=-1).astype(np.float32)


# --------------------------------------------------------------------------------------
# Classification
# --------------------------------------------------------------------------------------


def classify_units_from_scores(
    lag_cm: np.ndarray,
    scores_60: np.ndarray,
    min_shift_cm: float,
    gridness_threshold: float,
) -> Dict[str, np.ndarray]:
    best_idx, best_vals = _safe_nanargmax(scores_60)
    best_cm = np.full(best_idx.shape, np.nan, dtype=float)
    valid = best_idx >= 0
    best_cm[valid] = lag_cm[best_idx[valid]]
    qual = valid & np.isfinite(best_vals) & (best_vals >= gridness_threshold)
    predictive = qual & (best_cm >= min_shift_cm)
    retrospective = qual & (best_cm <= -min_shift_cm)
    standard = qual & ~(predictive | retrospective)
    low_grid = ~qual
    return {
        "predictive": np.where(predictive)[0],
        "retrospective": np.where(retrospective)[0],
        "standard": np.where(standard)[0],
        "low_grid": np.where(low_grid)[0],
        "best_cm": best_cm,
        "best_scores": best_vals,
    }


def run_classify(args) -> Path:
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    model, place_cells, options, payload = load_future_model(args.checkpoint_path, device=device)
    options.batch_size = int(args.batch_size)
    options.sequence_length = int(args.sequence_length or options.sequence_length)
    options.device = device

    out_dir = ensure_dir(Path(args.output_dir) if args.output_dir else Path(payload.get("output_dir", ".")).parent / "future_split_classify")
    configure_logging(out_dir, "classify.log")
    Ng_use = min(int(args.Ng_use), int(options.Ng))
    log_info("[classify] checkpoint=%s", args.checkpoint_path)
    log_info(
        "[classify] device=%s Ng_use=%d batches=%d batch_size=%d res=%d max_lag=%d",
        device,
        Ng_use,
        int(args.n_batches),
        int(args.batch_size),
        int(args.res),
        int(args.max_lag),
    )

    start_time = time.time()
    log_info("[classify] collecting zero-cue open-field sequences")
    xs, ys, activations = collect_open_field_sequences(
        model,
        options,
        place_cells,
        n_batches=int(args.n_batches),
        batch_size=int(args.batch_size),
        Ng_use=Ng_use,
    )
    log_info(
        "[classify] collected xs=%s activations=%s in %s",
        tuple(xs.shape),
        tuple(activations.shape),
        format_duration(time.time() - start_time),
    )
    coord_range = ((-options.box_width / 2, options.box_width / 2), (-options.box_height / 2, options.box_height / 2))
    starts = [0.2] * 10
    ends = np.linspace(0.4, 1.0, num=10)
    scorer = GridScorer(int(args.res), coord_range, zip(starts, ends.tolist()))
    lags = list(range(-int(args.max_lag), int(args.max_lag) + 1))
    log_info("[classify] scoring %d lags across %d units", len(lags), Ng_use)
    scores_60, scores_90 = scorer.predictive_grid_scores(xs, ys, activations, lags, shift_mode="time")
    cm_step = cm_per_step(xs, ys)
    lag_cm = np.asarray(lags, dtype=float) * cm_step
    classes = classify_units_from_scores(lag_cm, scores_60, args.min_shift_cm, args.gridness_threshold)
    log_info("[classify] computing rate maps and band controls")
    rate_maps = compute_rate_maps(xs, ys, activations, scorer)

    band_vals, band_kx, band_ky = band_scores(rate_maps, int(args.res), float(options.box_width))
    finite_band = band_vals[np.isfinite(band_vals)]
    if args.band_threshold is not None:
        band_cutoff = float(args.band_threshold)
    elif finite_band.size:
        band_cutoff = float(np.nanpercentile(finite_band, float(args.band_percentile)))
    else:
        band_cutoff = float("nan")
    band_units = np.where(band_vals >= band_cutoff)[0] if np.isfinite(band_cutoff) else np.array([], dtype=int)

    np.savez_compressed(
        out_dir / "gridness_data.npz",
        shift_mode=np.array("time"),
        shift_values=np.asarray(lags, dtype=float),
        lag_cm=lag_cm,
        scores_60=scores_60,
        scores_90=scores_90,
        classes_predictive=classes["predictive"],
        classes_phase_precession=classes["retrospective"],
        classes_retrospective=classes["retrospective"],
        classes_phase_locked=classes["standard"],
        classes_standard=classes["standard"],
        low_grid_units=classes["low_grid"],
        best_cm=classes["best_cm"],
        best_scores=classes["best_scores"],
        rate_maps=rate_maps,
        band_scores=band_vals,
        band_kx=band_kx,
        band_ky=band_ky,
        band_units=band_units,
        band_cutoff=np.array(band_cutoff),
        Ng_use=np.array(Ng_use),
        res=np.array(int(args.res)),
    )
    summary = {
        "checkpoint_path": str(args.checkpoint_path),
        "Ng_use": int(Ng_use),
        "cm_per_step": cm_step,
        "lags": lags,
        "lag_cm": lag_cm,
        "gridness_threshold": float(args.gridness_threshold),
        "min_shift_cm": float(args.min_shift_cm),
        "counts": {
            "predictive": int(classes["predictive"].size),
            "retrospective": int(classes["retrospective"].size),
            "standard": int(classes["standard"].size),
            "low_grid": int(classes["low_grid"].size),
            "band": int(band_units.size),
        },
        "gridness_path": str(out_dir / "gridness_data.npz"),
    }
    write_json(out_dir / "classify_summary.json", summary)
    log_info(
        "[classify] counts predictive=%d retrospective=%d standard=%d low_grid=%d band=%d",
        int(classes["predictive"].size),
        int(classes["retrospective"].size),
        int(classes["standard"].size),
        int(classes["low_grid"].size),
        int(band_units.size),
    )
    log_info("[classify] wrote %s", out_dir)
    return out_dir


# --------------------------------------------------------------------------------------
# Decoding
# --------------------------------------------------------------------------------------


def load_gridness_payload(path: str | Path) -> Dict[str, np.ndarray]:
    with np.load(path, allow_pickle=True) as data:
        return {k: data[k] for k in data.files}


def class_arrays(grid_data: Dict[str, np.ndarray], Ng: int, fallback: bool = True) -> Dict[str, np.ndarray]:
    def get(*names):
        for name in names:
            if name in grid_data:
                arr = np.asarray(grid_data[name], dtype=int).reshape(-1)
                arr = arr[(arr >= 0) & (arr < Ng)]
                if arr.size:
                    return np.unique(arr)
        return np.array([], dtype=int)

    pred = get("classes_predictive", "predictive")
    retro = get("classes_retrospective", "classes_phase_precession", "retrospective")
    standard = get("classes_standard", "classes_phase_locked", "classes_normal", "normal")
    low = get("low_grid_units", "classes_low_grid", "non_grid")
    band = get("band_units", "classes_band")
    if fallback and pred.size == 0:
        best_cm = np.asarray(grid_data.get("best_cm", np.full(Ng, np.nan)), dtype=float)
        best_scores = np.asarray(grid_data.get("best_scores", np.zeros(Ng)), dtype=float)
        candidates = np.where(np.isfinite(best_cm) & (best_cm > 0))[0]
        if candidates.size == 0:
            candidates = np.arange(min(Ng, 8), dtype=int)
        order = np.argsort(np.nan_to_num(best_scores[candidates], nan=-np.inf))[::-1]
        pred = candidates[order[: min(8, candidates.size)]]
    if fallback and standard.size == 0:
        used = np.union1d(pred, retro)
        standard = np.setdiff1d(np.arange(min(Ng, 32), dtype=int), used)[: min(8, Ng)]
    if fallback and low.size == 0:
        low = np.setdiff1d(np.arange(min(Ng, 32), dtype=int), np.union1d(pred, standard))[: min(8, Ng)]
    return {
        "predictive": pred,
        "retrospective": retro,
        "standard": standard,
        "low_grid": low,
        "band": band,
    }


def collect_fork_activity(
    model: FutureSplitRNN,
    options: SimpleNamespace,
    place_cells: PlaceCells,
    n_batches: int,
    batch_size: int,
    rng: np.random.Generator,
    future_horizon: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    g_list, pos_list, vel_list, routes_list, pair_list = [], [], [], [], []
    pair_offset = 0
    model.eval()
    with torch.no_grad():
        for _ in range(max(1, int(n_batches))):
            batch = generate_fork_batch(
                options,
                place_cells,
                batch_size=batch_size,
                rng=rng,
                paired=True,
                future_horizon=future_horizon,
            )
            states = model.g(batch.inputs).detach().cpu().numpy()
            g_list.append(states)
            pos_list.append(batch.positions_np)
            vel_list.append(batch.velocity_np)
            routes_list.append(batch.routes)
            pair_list.append(batch.pair_ids + pair_offset)
            pair_offset += int(batch.pair_ids.max()) + 1
    return (
        np.concatenate(g_list, axis=1),
        np.concatenate(pos_list, axis=1),
        np.concatenate(vel_list, axis=1),
        np.concatenate(routes_list, axis=0),
        np.concatenate(pair_list, axis=0),
        np.arange(g_list[0].shape[0], dtype=int),
    )


def matched_sample_times(options: SimpleNamespace, horizon: int) -> np.ndarray:
    cue_steps = int(options.fork_cue_steps)
    branch_step = int(options.fork_branch_step)
    T = int(options.sequence_length)
    start = max(cue_steps, branch_step - int(horizon))
    stop = min(branch_step, T - int(horizon))
    if stop <= start:
        start = cue_steps
        stop = min(branch_step, T - int(horizon))
    return np.arange(start, max(start, stop), dtype=int)


def kinematic_features(pos: np.ndarray, vel: np.ndarray, t: int, history_steps: int) -> np.ndarray:
    """Build current kinematic/recent-history features for all trials at time t."""
    B = pos.shape[1]
    v = vel[t]
    speed = np.linalg.norm(v, axis=1, keepdims=True)
    heading = np.zeros((B, 2), dtype=float)
    good = speed[:, 0] > 1e-12
    heading[good, 0] = v[good, 0] / speed[good, 0]
    heading[good, 1] = v[good, 1] / speed[good, 0]
    acc = np.zeros_like(v)
    if t > 0:
        acc = vel[t] - vel[t - 1]
    hist = []
    for k in range(int(history_steps)):
        tt = max(0, t - k)
        hist.append(vel[tt])
    hist_arr = np.concatenate(hist, axis=1) if hist else np.zeros((B, 0), dtype=float)
    return np.concatenate([pos[t], v, speed, heading, acc, hist_arr], axis=1).astype(np.float32)


def make_decode_dataset(
    states: np.ndarray,
    pos: np.ndarray,
    vel: np.ndarray,
    phase: np.ndarray,
    routes: np.ndarray,
    pair_ids: np.ndarray,
    units: Dict[str, np.ndarray],
    options: SimpleNamespace,
    horizon: int,
    history_steps: int,
) -> Dict[str, np.ndarray]:
    times = matched_sample_times(options, horizon)
    X_kin, X_pred, X_std, X_low, X_band = [], [], [], [], []
    y_route, y_pos, y_phase, sample_pairs, sample_times = [], [], [], [], []
    for t in times:
        fut_t = t + int(horizon)
        kin = kinematic_features(pos, vel, int(t), history_steps)
        X_kin.append(kin)
        X_pred.append(states[t][:, units["predictive"]] if units["predictive"].size else np.zeros((states.shape[1], 0)))
        X_std.append(states[t][:, units["standard"]] if units["standard"].size else np.zeros((states.shape[1], 0)))
        X_low.append(states[t][:, units["low_grid"]] if units["low_grid"].size else np.zeros((states.shape[1], 0)))
        X_band.append(states[t][:, units["band"]] if units["band"].size else np.zeros((states.shape[1], 0)))
        y_route.append((routes > 0).astype(int))
        y_pos.append(pos[fut_t])
        y_phase.append(phase[fut_t])
        sample_pairs.append(pair_ids)
        sample_times.append(np.full(routes.shape, t, dtype=int))

    def cat(rows, axis=0):
        return np.concatenate(rows, axis=axis) if rows else np.zeros((0, 0), dtype=np.float32)

    return {
        "X_kinematics": cat(X_kin),
        "X_predictive": cat(X_pred),
        "X_standard": cat(X_std),
        "X_low_grid": cat(X_low),
        "X_band": cat(X_band),
        "y_route": np.concatenate(y_route, axis=0) if y_route else np.array([], dtype=int),
        "y_pos": cat(y_pos),
        "y_phase": cat(y_phase),
        "pair_ids": np.concatenate(sample_pairs, axis=0) if sample_pairs else np.array([], dtype=int),
        "times": np.concatenate(sample_times, axis=0) if sample_times else np.array([], dtype=int),
    }


def split_by_pair(pair_ids: np.ndarray, test_fraction: float, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    pairs = np.unique(pair_ids)
    rng.shuffle(pairs)
    n_test = max(1, int(round(pairs.size * float(test_fraction))))
    test_pairs = set(int(x) for x in pairs[:n_test])
    test_mask = np.array([int(p) in test_pairs for p in pair_ids], dtype=bool)
    train_mask = ~test_mask
    return train_mask, test_mask


def standardize_train_test(X: np.ndarray, train_mask: np.ndarray, test_mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    X = np.asarray(X, dtype=np.float32)
    if X.ndim != 2 or X.shape[1] == 0:
        return np.zeros((int(train_mask.sum()), 0), dtype=np.float32), np.zeros((int(test_mask.sum()), 0), dtype=np.float32)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X[train_mask])
    X_test = scaler.transform(X[test_mask])
    return X_train, X_test


def residualize(X: np.ndarray, nuisance: np.ndarray, train_mask: np.ndarray, alpha: float) -> np.ndarray:
    if X.shape[1] == 0 or nuisance.shape[1] == 0:
        return X
    scaler_n = StandardScaler()
    scaler_x = StandardScaler()
    N_train = scaler_n.fit_transform(nuisance[train_mask])
    X_train = scaler_x.fit_transform(X[train_mask])
    reg = Ridge(alpha=float(alpha))
    reg.fit(N_train, X_train)
    N_all = scaler_n.transform(nuisance)
    pred_scaled = reg.predict(N_all)
    pred = scaler_x.inverse_transform(pred_scaled)
    return X - pred


def evaluate_decoder_group(
    name: str,
    X: np.ndarray,
    y_route: np.ndarray,
    y_pos: np.ndarray,
    y_phase: np.ndarray,
    train_mask: np.ndarray,
    test_mask: np.ndarray,
    ridge_alpha: float,
) -> Dict[str, object]:
    row: Dict[str, object] = {"signal": name, "n_features": int(X.shape[1]) if X.ndim == 2 else 0}
    if X.ndim != 2 or X.shape[1] == 0 or train_mask.sum() < 2 or test_mask.sum() < 1:
        row.update({"status": "skipped_no_features"})
        return row

    X_train, X_test = standardize_train_test(X, train_mask, test_mask)
    y_train = y_route[train_mask]
    y_test = y_route[test_mask]
    if np.unique(y_train).size >= 2 and np.unique(y_test).size >= 1:
        try:
            clf = LogisticRegression(max_iter=1000, solver="liblinear")
            clf.fit(X_train, y_train)
            prob = clf.predict_proba(X_test)[:, 1]
            pred = (prob >= 0.5).astype(int)
            row["route_accuracy"] = float(accuracy_score(y_test, pred))
            row["route_auc"] = float(roc_auc_score(y_test, prob)) if np.unique(y_test).size == 2 else float("nan")
        except Exception as exc:
            row["route_error"] = type(exc).__name__
    else:
        row["route_accuracy"] = float("nan")
        row["route_auc"] = float("nan")

    try:
        reg_pos = Ridge(alpha=float(ridge_alpha))
        reg_pos.fit(X_train, y_pos[train_mask])
        pos_hat = reg_pos.predict(X_test)
        row["future_position_rmse_cm"] = float(np.sqrt(np.mean(np.sum((pos_hat - y_pos[test_mask]) ** 2, axis=1))) * 100.0)
    except Exception as exc:
        row["future_position_error"] = type(exc).__name__

    try:
        reg_phase = Ridge(alpha=float(ridge_alpha))
        reg_phase.fit(X_train, y_phase[train_mask])
        phase_hat = reg_phase.predict(X_test)
        row["future_torus_phase_error_rad"] = phase_feature_error(phase_hat, y_phase[test_mask])
    except Exception as exc:
        row["future_torus_error"] = type(exc).__name__
    row.setdefault("status", "ok")
    return row


def run_decode(args) -> Path:
    rng = set_seed(int(args.random_seed))
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = ensure_dir(args.output_dir)
    configure_logging(out_dir, "decode.log")
    log_info("[decode] checkpoint=%s", args.checkpoint_path)
    log_info("[decode] gridness=%s", args.gridness_path)
    model, place_cells, options, _ = load_future_model(args.checkpoint_path, device=device)
    grid_data = load_gridness_payload(args.gridness_path)
    units = class_arrays(grid_data, int(options.Ng), fallback=bool(args.allow_fallback_units))
    rate_maps = np.asarray(grid_data["rate_maps"], dtype=float) if "rate_maps" in grid_data else None
    grid_units = np.unique(np.concatenate([units["predictive"], units["standard"], units["retrospective"]]))
    if grid_units.size == 0:
        grid_units = np.arange(min(int(options.Ng), int(args.Ng_torus_fallback)), dtype=int)
    projector = build_phase_projector(rate_maps, grid_units, options)
    log_info(
        "[decode] device=%s torus_projector=%s class_counts=%s",
        device,
        projector.status,
        {k: int(v.size) for k, v in units.items()},
    )

    options.batch_size = int(args.batch_size)
    start_time = time.time()
    log_info("[decode] collecting matched fork activity batches=%d batch_size=%d", int(args.n_batches), int(args.batch_size))
    states, pos, vel, routes, pair_ids, _ = collect_fork_activity(
        model,
        options,
        place_cells,
        n_batches=int(args.n_batches),
        batch_size=int(args.batch_size),
        rng=rng,
        future_horizon=int(args.main_horizon),
    )
    log_info("[decode] collected states=%s in %s", tuple(states.shape), format_duration(time.time() - start_time))
    phase = phase_features_from_states(states, projector)

    horizons = [int(h) for h in _to_float_list(args.horizons)]
    rows: List[Dict[str, object]] = []
    for horizon in horizons:
        horizon_start = time.time()
        dataset = make_decode_dataset(
            states,
            pos,
            vel,
            phase,
            routes,
            pair_ids,
            units,
            options,
            horizon=horizon,
            history_steps=int(args.history_steps),
        )
        if dataset["y_route"].size == 0:
            rows.append({"horizon": int(horizon), "status": "skipped_no_matched_samples"})
            log_info("[decode] horizon=%d skipped: no matched samples", int(horizon))
            continue
        train_mask, test_mask = split_by_pair(dataset["pair_ids"], float(args.test_fraction), rng)
        nuisance = np.concatenate([dataset["X_kinematics"], dataset["X_standard"]], axis=1)
        X_pred_resid = residualize(dataset["X_predictive"], nuisance, train_mask, float(args.ridge_alpha))
        groups = {
            "kinematics": dataset["X_kinematics"],
            "predictive": dataset["X_predictive"],
            "standard_grid": dataset["X_standard"],
            "standard_grid_plus_kinematics": nuisance,
            "non_grid": dataset["X_low_grid"],
            "band": dataset["X_band"],
            "predictive_residualized": X_pred_resid,
        }
        for name, X in groups.items():
            row = evaluate_decoder_group(
                name,
                X,
                dataset["y_route"],
                dataset["y_pos"],
                dataset["y_phase"],
                train_mask,
                test_mask,
                float(args.ridge_alpha),
            )
            row["horizon"] = int(horizon)
            rows.append(row)
        best = [
            row for row in rows
            if row.get("horizon") == int(horizon)
            and row.get("signal") == "predictive"
            and row.get("status") == "ok"
        ]
        if best:
            log_info(
                "[decode] horizon=%d predictive_acc=%.3f predictive_phase_err=%.3f samples=%d elapsed=%s",
                int(horizon),
                float(best[-1].get("route_accuracy", np.nan)),
                float(best[-1].get("future_torus_phase_error_rad", np.nan)),
                int(dataset["y_route"].size),
                format_duration(time.time() - horizon_start),
            )
        else:
            log_info("[decode] horizon=%d completed in %s", int(horizon), format_duration(time.time() - horizon_start))

    pred_rows = [r for r in rows if r.get("signal") == "predictive" and r.get("status") == "ok"]
    peak_route = None
    finite_route = [r for r in pred_rows if np.isfinite(float(r.get("route_accuracy", np.nan)))]
    if finite_route:
        peak = max(finite_route, key=lambda r: float(r.get("route_accuracy", np.nan)))
        peak_route = int(peak["horizon"])

    summary = {
        "checkpoint_path": str(args.checkpoint_path),
        "gridness_path": str(args.gridness_path),
        "torus_projector_status": projector.status,
        "class_counts": {k: int(v.size) for k, v in units.items()},
        "n_trials": int(states.shape[1]),
        "horizons": horizons,
        "predictive_peak_route_horizon": peak_route,
        "rows": rows,
    }
    write_json(out_dir / "decode_metrics.json", summary)
    write_csv(out_dir / "decode_metrics.csv", rows)
    log_info("[decode] predictive_peak_route_horizon=%s", str(peak_route))
    log_info("[decode] wrote %s", out_dir)
    return out_dir


# --------------------------------------------------------------------------------------
# Interventions
# --------------------------------------------------------------------------------------


def rollout_with_intervention(
    model: FutureSplitRNN,
    inputs: Tuple[torch.Tensor, torch.Tensor],
    units: np.ndarray,
    pair_ids: np.ndarray,
    window: Tuple[int, int],
    intervention: str,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, torch.Tensor, torch.Tensor]:
    """Roll out RNN while modifying selected hidden units inside a time window."""
    v, init = inputs
    device = v.device
    units_t = torch.as_tensor(np.asarray(units, dtype=int), dtype=torch.long, device=device)
    h = model.encoder(init)[None]
    states = []
    start, stop = window
    pair_index = {}
    for pid in np.unique(pair_ids):
        idx = np.where(pair_ids == pid)[0]
        if idx.size == 2:
            pair_index[int(pid)] = (int(idx[0]), int(idx[1]))

    with torch.no_grad():
        for t in range(v.shape[0]):
            _, h = model.RNN(v[t : t + 1], h)
            if units_t.numel() > 0 and start <= t < stop:
                h = h.clone()
                if intervention == "ablate":
                    h[0, :, units_t] = 0.0
                elif intervention == "scramble":
                    for u in units_t:
                        perm = torch.as_tensor(rng.permutation(v.shape[1]), dtype=torch.long, device=device)
                        h[0, :, u] = h[0, perm, u]
                elif intervention == "swap":
                    for a, b in pair_index.values():
                        tmp = h[0, a, units_t].clone()
                        h[0, a, units_t] = h[0, b, units_t]
                        h[0, b, units_t] = tmp
            states.append(h[0].clone())
        states_t = torch.stack(states, dim=0)
        current_logits = model.decoder(states_t)
        future_logits = model.future_decoder(states_t)
    return states_t.detach().cpu().numpy(), current_logits, future_logits


def intervention_metrics(
    name: str,
    baseline: Dict[str, object],
    condition: Dict[str, object],
    batch: ForkBatch,
    phase_baseline: np.ndarray,
    phase_condition: np.ndarray,
    options: SimpleNamespace,
    horizon: int,
) -> Dict[str, object]:
    times = matched_sample_times(options, horizon)
    if times.size == 0:
        times = np.arange(int(options.fork_cue_steps), int(options.fork_branch_step), dtype=int)
    future_pred = condition["future_pos"]
    future_base = baseline["future_pos"]
    own_future = batch.positions_np[np.minimum(times + horizon, int(options.sequence_length) - 1)]
    routes = batch.routes
    donor_routes = -routes
    route_target = (routes > 0).astype(int)

    cond_samples = future_pred[times].reshape(-1, 2)
    base_samples = future_base[times].reshape(-1, 2)
    target_samples = own_future.reshape(-1, 2)
    route_repeated = np.tile(route_target, times.size)
    donor_repeated = np.tile(donor_routes, times.size)

    rmse = float(np.sqrt(np.mean(np.sum((cond_samples - target_samples) ** 2, axis=1))) * 100.0)
    route_acc = float(np.mean((cond_samples[:, 0] > 0).astype(int) == route_repeated))
    donor_pull_cm = float(np.mean((cond_samples[:, 0] - base_samples[:, 0]) * donor_repeated) * 100.0)

    own_phase = phase_baseline[np.minimum(times + horizon, int(options.sequence_length) - 1)].reshape(-1, 4)
    cond_phase = phase_condition[times].reshape(-1, 4)
    donor_phase_rows = []
    for t in times:
        fut_t = min(int(t + horizon), int(options.sequence_length) - 1)
        donor = phase_baseline[fut_t].copy()
        for pid in np.unique(batch.pair_ids):
            idx = np.where(batch.pair_ids == pid)[0]
            if idx.size == 2:
                donor[idx[0]] = phase_baseline[fut_t, idx[1]]
                donor[idx[1]] = phase_baseline[fut_t, idx[0]]
        donor_phase_rows.append(donor)
    donor_phase = np.concatenate(donor_phase_rows, axis=0) if donor_phase_rows else own_phase
    baseline_phase_here = phase_baseline[times].reshape(-1, 4)
    base_margin = np.mean(phase_feature_distance(baseline_phase_here, donor_phase) - phase_feature_distance(baseline_phase_here, own_phase))
    cond_margin = np.mean(phase_feature_distance(cond_phase, donor_phase) - phase_feature_distance(cond_phase, own_phase))
    donor_phase_pull_rad = float(base_margin - cond_margin)

    branch_eval_t = min(int(options.fork_branch_step) + int(horizon), int(options.sequence_length) - 1)
    current_cond = condition["current_pos"][branch_eval_t]
    current_base = baseline["current_pos"][branch_eval_t]
    downstream_pull_cm = float(np.mean((current_cond[:, 0] - current_base[:, 0]) * donor_routes) * 100.0)
    return {
        "condition": name,
        "future_decoder_rmse_cm": rmse,
        "future_decoder_route_accuracy": route_acc,
        "future_decoder_donor_pull_cm": donor_pull_cm,
        "future_torus_donor_pull_rad": donor_phase_pull_rad,
        "downstream_position_donor_pull_cm": downstream_pull_cm,
    }


def run_intervene(args) -> Path:
    rng = set_seed(int(args.random_seed))
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = ensure_dir(args.output_dir)
    configure_logging(out_dir, "intervene.log")
    log_info("[intervene] checkpoint=%s", args.checkpoint_path)
    log_info("[intervene] gridness=%s", args.gridness_path)
    model, place_cells, options, _ = load_future_model(args.checkpoint_path, device=device)
    grid_data = load_gridness_payload(args.gridness_path)
    units = class_arrays(grid_data, int(options.Ng), fallback=bool(args.allow_fallback_units))
    pred_units = units["predictive"]
    rate_maps = np.asarray(grid_data["rate_maps"], dtype=float) if "rate_maps" in grid_data else None
    grid_units = np.unique(np.concatenate([units["predictive"], units["standard"], units["retrospective"]]))
    if grid_units.size == 0:
        grid_units = np.arange(min(int(options.Ng), int(args.Ng_torus_fallback)), dtype=int)
    projector = build_phase_projector(rate_maps, grid_units, options)
    log_info(
        "[intervene] device=%s torus_projector=%s predictive_units=%d",
        device,
        projector.status,
        int(pred_units.size),
    )

    options.batch_size = int(args.batch_size)
    horizon = int(args.main_horizon)
    batch = generate_fork_batch(options, place_cells, batch_size=args.batch_size, rng=rng, paired=True, future_horizon=horizon)
    assert_matched_prebranch(batch, options)
    window = (int(args.window_start) if args.window_start is not None else int(options.fork_cue_steps), int(args.window_stop) if args.window_stop is not None else int(options.fork_branch_step))
    log_info(
        "[intervene] generated matched batch trials=%d branch_step=%d window=%s horizon=%d",
        int(batch.routes.size),
        int(options.fork_branch_step),
        list(window),
        horizon,
    )

    rows: List[Dict[str, object]] = []
    rollouts: Dict[str, Dict[str, object]] = {}
    for name in ("baseline", "ablate", "scramble", "swap"):
        step_start = time.time()
        intervention = "none" if name == "baseline" else name
        states, current_logits, future_logits = rollout_with_intervention(
            model,
            batch.inputs,
            pred_units,
            batch.pair_ids,
            window,
            intervention,
            rng,
        )
        current_pos = place_cells.get_nearest_cell_pos(current_logits).detach().cpu().numpy()
        future_pos = place_cells.get_nearest_cell_pos(future_logits).detach().cpu().numpy()
        phase = phase_features_from_states(states, projector)
        rollouts[name] = {
            "states": states,
            "current_pos": current_pos,
            "future_pos": future_pos,
            "phase": phase,
        }
        log_info("[intervene] rollout=%s states=%s elapsed=%s", name, tuple(states.shape), format_duration(time.time() - step_start))
    for name in ("ablate", "scramble", "swap"):
        rows.append(
            intervention_metrics(
                name,
                rollouts["baseline"],
                rollouts[name],
                batch,
                rollouts["baseline"]["phase"],
                rollouts[name]["phase"],
                options,
                horizon,
            )
        )

    summary = {
        "checkpoint_path": str(args.checkpoint_path),
        "gridness_path": str(args.gridness_path),
        "torus_projector_status": projector.status,
        "predictive_unit_count": int(pred_units.size),
        "window": list(window),
        "main_horizon": horizon,
        "rows": rows,
    }
    write_json(out_dir / "intervention_metrics.json", summary)
    write_csv(out_dir / "intervention_metrics.csv", rows)
    for row in rows:
        log_info(
            "[intervene] %s donor_pull_cm=%.3f phase_pull_rad=%.3f downstream_pull_cm=%.3f",
            row["condition"],
            float(row.get("future_decoder_donor_pull_cm", np.nan)),
            float(row.get("future_torus_donor_pull_rad", np.nan)),
            float(row.get("downstream_position_donor_pull_cm", np.nan)),
        )
    log_info("[intervene] wrote %s", out_dir)
    return out_dir


# --------------------------------------------------------------------------------------
# Training and smoke
# --------------------------------------------------------------------------------------


def run_train(args) -> Path:
    rng = set_seed(int(args.random_seed))
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = ensure_dir(args.output_dir)
    configure_logging(out_dir, "train.log")
    options = make_options(
        batch_size=args.batch_size,
        sequence_length=args.sequence_length,
        Np=args.Np,
        Ng=args.Ng,
        place_cell_rf=args.place_cell_rf,
        surround_scale=args.surround_scale,
        activation=args.activation,
        weight_decay=args.weight_decay,
        box_width=args.box_width,
        box_height=args.box_height,
        learning_rate=args.learning_rate,
        device=device,
        save_dir=str(args.output_dir),
        run_ID="future_split",
        fork_cue_steps=args.fork_cue_steps,
        fork_branch_step=args.fork_branch_step,
        fork_stem_start_y=args.fork_stem_start_y,
        fork_branch_y=args.fork_branch_y,
        fork_arm_length=args.fork_arm_length,
        fork_cue_scale=args.fork_cue_scale,
        future_horizon=args.future_horizon,
        future_loss_weight=args.future_loss_weight,
    )
    place_cells = PlaceCells(options)
    model = FutureSplitRNN(options, place_cells).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=float(args.learning_rate))
    log_every = max(1, int(getattr(args, "log_every_steps", 50)))
    total_steps = max(1, int(args.n_epochs) * int(args.n_steps))
    train_start = time.time()
    log_info("[train] starting future-split training")
    log_info("[train] output_dir=%s", out_dir)
    log_info("[train] device=%s cuda_available=%s", device, torch.cuda.is_available())
    log_info("[train] parameters=%d Ng=%d Np=%d input_dim=%d", parameter_count(model), int(options.Ng), int(options.Np), int(options.velocity_dim))
    log_info(
        "[train] epochs=%d steps_per_epoch=%d batch_size=%d sequence_length=%d total_steps=%d",
        int(args.n_epochs),
        int(args.n_steps),
        int(args.batch_size),
        int(args.sequence_length),
        total_steps,
    )
    log_info(
        "[train] fork cue_steps=%d branch_step=%d future_horizon=%d future_loss_weight=%.4g",
        int(args.fork_cue_steps),
        int(args.fork_branch_step),
        int(args.future_horizon),
        float(args.future_loss_weight),
    )
    log_info("[train] config=%s", json.dumps(json_safe(options_to_dict(options)), sort_keys=True))
    history = []
    model.train()
    for epoch in range(int(args.n_epochs)):
        epoch_start = time.time()
        epoch_terms: List[Dict[str, float]] = []
        for step_idx in range(int(args.n_steps)):
            step_start = time.time()
            batch = generate_fork_batch(options, place_cells, rng=rng, paired=True)
            opt.zero_grad()
            loss, terms = model.compute_loss(
                batch.inputs,
                batch.place_outputs,
                batch.future_place_outputs,
                batch.future_mask,
                future_weight=float(args.future_loss_weight),
            )
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite future-split training loss.")
            loss.backward()
            opt.step()
            epoch_terms.append(terms)
            completed = epoch * int(args.n_steps) + step_idx + 1
            if completed == 1 or completed % log_every == 0 or step_idx + 1 == int(args.n_steps):
                recent = epoch_terms[-min(len(epoch_terms), log_every):]
                recent_loss = float(np.mean([row["loss"] for row in recent]))
                recent_current = float(np.mean([row["current_ce"] for row in recent]))
                recent_future = float(np.mean([row["future_ce"] for row in recent]))
                elapsed = time.time() - train_start
                steps_per_sec = completed / max(elapsed, 1e-9)
                eta = (total_steps - completed) / max(steps_per_sec, 1e-9)
                log_info(
                    "[train] step=%d/%d epoch=%d/%d step_in_epoch=%d/%d "
                    "loss=%.4f current_ce=%.4f future_ce=%.4f step_time=%s throughput=%.2f steps/s eta=%s",
                    completed,
                    total_steps,
                    epoch + 1,
                    int(args.n_epochs),
                    step_idx + 1,
                    int(args.n_steps),
                    recent_loss,
                    recent_current,
                    recent_future,
                    format_duration(time.time() - step_start),
                    steps_per_sec,
                    format_duration(eta),
                )
        mean_terms = {
            key: float(np.mean([row[key] for row in epoch_terms if key in row]))
            for key in epoch_terms[0].keys()
        }
        mean_terms["epoch"] = epoch
        mean_terms["elapsed_seconds"] = float(time.time() - train_start)
        history.append(mean_terms)
        log_info(
            "[train] epoch=%d/%d summary loss=%.4f current_ce=%.4f future_ce=%.4f epoch_time=%s elapsed=%s",
            epoch + 1,
            int(args.n_epochs),
            mean_terms["loss"],
            mean_terms["current_ce"],
            mean_terms["future_ce"],
            format_duration(time.time() - epoch_start),
            format_duration(time.time() - train_start),
        )

    full_path = out_dir / "future_split_full.pth"
    core_path = out_dir / "future_split_core.pth"
    payload = {
        "model_state_dict": model.state_dict(),
        "core_state_dict": model.core_state_dict(),
        "config": options_to_dict(options),
        "history": history,
        "output_dir": str(out_dir),
    }
    torch.save(payload, full_path)
    torch.save(model.core_state_dict(), core_path)
    write_json(
        out_dir / "training_metrics.json",
        {
            "full_checkpoint": str(full_path),
            "core_checkpoint": str(core_path),
            "config": options_to_dict(options),
            "history": history,
        },
    )
    log_info("[train] wrote %s", full_path)
    log_info("[train] wrote %s", core_path)
    log_info("[train] completed in %s", format_duration(time.time() - train_start))
    return out_dir


def run_smoke(args) -> Path:
    out_dir = ensure_dir(Path(args.output_dir))
    configure_logging(out_dir, "smoke.log")
    log_info("[smoke] starting CPU-only smoke test")
    train_args = argparse.Namespace(
        output_dir=out_dir,
        n_epochs=1,
        n_steps=3,
        log_every_steps=1,
        batch_size=8,
        sequence_length=18,
        Np=32,
        Ng=24,
        place_cell_rf=0.18,
        surround_scale=2.0,
        activation="relu",
        weight_decay=1e-5,
        box_width=2.2,
        box_height=2.2,
        learning_rate=5e-3,
        device="cpu",
        random_seed=7,
        fork_cue_steps=3,
        fork_branch_step=10,
        fork_stem_start_y=-0.8,
        fork_branch_y=0.0,
        fork_arm_length=0.55,
        fork_cue_scale=1.0,
        future_horizon=4,
        future_loss_weight=0.5,
    )
    run_train(train_args)
    configure_logging(out_dir, "smoke.log")
    full_ckpt = out_dir / "future_split_full.pth"

    model, place_cells, options, _ = load_future_model(full_ckpt, device="cpu")
    smoke_rng = set_seed(11)
    batch = generate_fork_batch(options, place_cells, batch_size=8, rng=smoke_rng, paired=True, future_horizon=4)
    assert_matched_prebranch(batch, options)
    log_info("[smoke] matched pre-branch assertions passed")
    with torch.no_grad():
        loss, terms = model.compute_loss(batch.inputs, batch.place_outputs, batch.future_place_outputs, batch.future_mask, 0.5)
    if not torch.isfinite(loss):
        raise AssertionError("Smoke loss is non-finite.")
    log_info("[smoke] finite loss %.4f", float(loss.detach().cpu()))

    classify_args = argparse.Namespace(
        checkpoint_path=str(full_ckpt),
        output_dir=str(out_dir / "classify"),
        device="cpu",
        batch_size=8,
        sequence_length=18,
        n_batches=1,
        Ng_use=16,
        res=8,
        max_lag=2,
        min_shift_cm=0.1,
        gridness_threshold=-1.0,
        band_percentile=80.0,
        band_threshold=None,
    )
    classify_dir = run_classify(classify_args)
    configure_logging(out_dir, "smoke.log")
    gridness_path = classify_dir / "gridness_data.npz"

    decode_args = argparse.Namespace(
        checkpoint_path=str(full_ckpt),
        gridness_path=str(gridness_path),
        output_dir=str(out_dir / "decode"),
        device="cpu",
        batch_size=8,
        n_batches=1,
        horizons="1 2 4",
        main_horizon=4,
        history_steps=2,
        test_fraction=0.5,
        random_seed=13,
        ridge_alpha=1.0,
        allow_fallback_units=True,
        Ng_torus_fallback=8,
    )
    decode_dir = run_decode(decode_args)
    configure_logging(out_dir, "smoke.log")

    intervene_args = argparse.Namespace(
        checkpoint_path=str(full_ckpt),
        gridness_path=str(gridness_path),
        output_dir=str(out_dir / "intervene"),
        device="cpu",
        batch_size=8,
        main_horizon=4,
        random_seed=17,
        allow_fallback_units=True,
        Ng_torus_fallback=8,
        window_start=None,
        window_stop=None,
    )
    intervene_dir = run_intervene(intervene_args)
    configure_logging(out_dir, "smoke.log")

    required = [
        out_dir / "training_metrics.json",
        gridness_path,
        decode_dir / "decode_metrics.json",
        decode_dir / "decode_metrics.csv",
        intervene_dir / "intervention_metrics.json",
        intervene_dir / "intervention_metrics.csv",
    ]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise AssertionError(f"Smoke outputs missing: {missing}")
    write_json(
        out_dir / "smoke_summary.json",
        {
            "status": "ok",
            "matched_prebranch_checked": True,
            "finite_loss": float(loss.detach().cpu()),
            "loss_terms": terms,
            "outputs": [str(p) for p in required],
        },
    )
    log_info("[smoke] ok; wrote %s", out_dir)
    return out_dir


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------


def add_model_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--sequence_length", type=int, default=32)
    parser.add_argument("--Np", type=int, default=256)
    parser.add_argument("--Ng", type=int, default=512)
    parser.add_argument("--place_cell_rf", type=float, default=0.12)
    parser.add_argument("--surround_scale", type=float, default=2.0)
    parser.add_argument("--activation", default="relu", choices=["relu", "tanh"])
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--box_width", type=float, default=2.2)
    parser.add_argument("--box_height", type=float, default=2.2)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--device", default=None)
    parser.add_argument("--random_seed", type=int, default=0)
    parser.add_argument("--fork_cue_steps", type=int, default=4)
    parser.add_argument("--fork_branch_step", type=int, default=20)
    parser.add_argument("--fork_stem_start_y", type=float, default=-0.85)
    parser.add_argument("--fork_branch_y", type=float, default=0.0)
    parser.add_argument("--fork_arm_length", type=float, default=0.75)
    parser.add_argument("--fork_cue_scale", type=float, default=1.0)
    parser.add_argument("--future_horizon", type=int, default=8)
    parser.add_argument("--future_loss_weight", type=float, default=0.5)


def parse_args(argv: Optional[Sequence[str]] = None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_train = sub.add_parser("train", help="Train a cued future-split RNN.")
    add_model_args(p_train)
    p_train.add_argument("--output_dir", required=True)
    p_train.add_argument("--n_epochs", type=int, default=50)
    p_train.add_argument("--n_steps", type=int, default=250)
    p_train.add_argument(
        "--log_every_steps",
        type=int,
        default=50,
        help="Write rolling training metrics every N optimizer steps.",
    )

    p_class = sub.add_parser("classify", help="Classify grid cells on zero-cue open-field data.")
    p_class.add_argument("--checkpoint_path", required=True)
    p_class.add_argument("--output_dir", default=None)
    p_class.add_argument("--device", default=None)
    p_class.add_argument("--batch_size", type=int, default=80)
    p_class.add_argument("--sequence_length", type=int, default=None)
    p_class.add_argument("--n_batches", type=int, default=10)
    p_class.add_argument("--Ng_use", type=int, default=512)
    p_class.add_argument("--res", type=int, default=20)
    p_class.add_argument("--max_lag", type=int, default=12)
    p_class.add_argument("--min_shift_cm", type=float, default=5.0)
    p_class.add_argument("--gridness_threshold", type=float, default=0.2)
    p_class.add_argument("--band_percentile", type=float, default=90.0)
    p_class.add_argument("--band_threshold", type=float, default=None)

    p_decode = sub.add_parser("decode", help="Decode future route/position/torus phase on matched fork trials.")
    p_decode.add_argument("--checkpoint_path", required=True)
    p_decode.add_argument("--gridness_path", required=True)
    p_decode.add_argument("--output_dir", required=True)
    p_decode.add_argument("--device", default=None)
    p_decode.add_argument("--batch_size", type=int, default=80)
    p_decode.add_argument("--n_batches", type=int, default=8)
    p_decode.add_argument("--horizons", default="1 2 4 6 8 10 12")
    p_decode.add_argument("--main_horizon", type=int, default=8)
    p_decode.add_argument("--history_steps", type=int, default=4)
    p_decode.add_argument("--test_fraction", type=float, default=0.35)
    p_decode.add_argument("--random_seed", type=int, default=0)
    p_decode.add_argument("--ridge_alpha", type=float, default=1.0)
    p_decode.add_argument("--allow_fallback_units", action=argparse.BooleanOptionalAction, default=True)
    p_decode.add_argument("--Ng_torus_fallback", type=int, default=32)

    p_int = sub.add_parser("intervene", help="Ablate, scramble, and swap predictive activity pre-branch.")
    p_int.add_argument("--checkpoint_path", required=True)
    p_int.add_argument("--gridness_path", required=True)
    p_int.add_argument("--output_dir", required=True)
    p_int.add_argument("--device", default=None)
    p_int.add_argument("--batch_size", type=int, default=80)
    p_int.add_argument("--main_horizon", type=int, default=8)
    p_int.add_argument("--random_seed", type=int, default=0)
    p_int.add_argument("--allow_fallback_units", action=argparse.BooleanOptionalAction, default=True)
    p_int.add_argument("--Ng_torus_fallback", type=int, default=32)
    p_int.add_argument("--window_start", type=int, default=None)
    p_int.add_argument("--window_stop", type=int, default=None)

    p_smoke = sub.add_parser("smoke", help="Run a tiny CPU-only smoke test.")
    p_smoke.add_argument("--output_dir", default="/private/tmp/future_split_smoke")

    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    if args.cmd == "train":
        run_train(args)
    elif args.cmd == "classify":
        run_classify(args)
    elif args.cmd == "decode":
        run_decode(args)
    elif args.cmd == "intervene":
        run_intervene(args)
    elif args.cmd == "smoke":
        run_smoke(args)
    else:  # pragma: no cover
        raise SystemExit(f"Unknown command: {args.cmd}")


if __name__ == "__main__":
    main(sys.argv[1:])
