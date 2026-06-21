#!/usr/bin/env python3
"""Matched-state future-split experiments for cued predictive grid cells.

This script includes the original compact binary T-maze/fork task plus a harder
multi-route graph-maze task.  In the graph task, many future trajectories share
the same cued stem and the same post-cue local velocity history, so future state
cannot be reduced to instantaneous kinematics or a one-bit left/right route.

Subcommands:
  train      Train a cued fork/graph RNN and save full + core checkpoints.
  classify   Classify predictive/retrospective/standard grid units on zero-cue
             open-field trajectories.
  decode     Decode future route, position, and torus phase from matched
             trials and controls.
  crossing   Compare predictive and standard-grid population similarity at
             matched X-crossings with different travel directions.
  plot_crossing
             Plot crossing correlation summaries from a crossing output dir.
  crossing_step_sweep
             Run crossing over multiple crossing timepoints and plot the curve.
  plot_crossing_step_sweeps
             Overlay crossing-step sweep curves from multiple output dirs.
  crossing --pairing_mode same_bin
             Pasted-script-style same-spatial-bin population correlations.
  intervene  Ablate/scramble/swap matched unit groups before the branch.
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
import tempfile
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

_CACHE_ROOT = Path(os.environ.get("FUTURE_SPLIT_CACHE_DIR", tempfile.gettempdir()))
for _env_key, _dirname in (
    ("MPLCONFIGDIR", "future_split_mplconfig"),
    ("NUMBA_CACHE_DIR", "future_split_numba_cache"),
):
    os.environ.setdefault(_env_key, str(_CACHE_ROOT / _dirname))
    try:
        Path(os.environ[_env_key]).mkdir(parents=True, exist_ok=True)
    except OSError:
        fallback = Path.cwd() / ".future_split_cache" / _dirname
        fallback.mkdir(parents=True, exist_ok=True)
        os.environ[_env_key] = str(fallback)

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.append(str(_REPO_ROOT))

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
from shift_utils import build_shift_values, shift_config_label, shift_values_to_cm
from trajectory_generator import TrajectoryGenerator

try:
    from LowRankRNN import LowRankRNN
except Exception:  # pragma: no cover - only needed for --recurrent_type low_rank.
    LowRankRNN = None


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


def read_json(path: Path | str) -> Dict:
    with open(path) as f:
        return json.load(f)


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


def read_csv_rows(path: Path | str) -> List[Dict[str, str]]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def _to_float_list(text_or_values: str | Sequence[float]) -> List[float]:
    if isinstance(text_or_values, str):
        pieces = [p for p in text_or_values.replace(",", " ").split() if p]
        return [float(p) for p in pieces]
    return [float(x) for x in text_or_values]


def _to_int_list(text_or_values: str | Sequence[int]) -> List[int]:
    return [int(round(x)) for x in _to_float_list(text_or_values)]


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
    "task": "binary_fork",
    "velocity_dim": 3,
    "cue_dim": 1,
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
    "num_routes": 2,
    "cue_steps": None,
    "branch_step": None,
    "velocity_frame": "local_graph",
    "open_field_mix": 0.0,
    "activity_l1": 0.0,
    "delay_noise_std": 0.0,
    "cue_dropout": 0.0,
    "cue_noise_std": 0.0,
    "hidden_dropout": 0.0,
    "recurrent_type": "full",
    "rank": 128,
    "future_horizon": 8,
    "future_loss_weight": 0.5,
}


def make_options(**kwargs) -> SimpleNamespace:
    cfg = dict(COMMON_DEFAULTS)
    cfg.update({k: v for k, v in kwargs.items() if v is not None})
    task = str(cfg.get("task", "binary_fork"))
    cfg["task"] = task
    if cfg.get("cue_steps") is not None:
        cfg["fork_cue_steps"] = int(cfg["cue_steps"])
    else:
        cfg["cue_steps"] = int(cfg["fork_cue_steps"])
    if cfg.get("branch_step") is not None:
        cfg["fork_branch_step"] = int(cfg["branch_step"])
    else:
        cfg["branch_step"] = int(cfg["fork_branch_step"])
    if task == "multi_route_graph" and int(cfg.get("cue_dim", 1)) == 1:
        cfg["cue_dim"] = 2
    if task == "binary_fork":
        cfg["num_routes"] = 2
    cfg["cue_dim"] = max(0, int(cfg.get("cue_dim", 1)))
    cfg["velocity_dim"] = 2 + cfg["cue_dim"]
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


def extract_state_dict(raw: object) -> Dict[str, torch.Tensor]:
    """Normalize common core RNN checkpoint formats."""
    if isinstance(raw, dict):
        for key in ("core_state_dict", "state_dict", "model_state_dict", "model"):
            state = raw.get(key)
            if isinstance(state, dict) and all(hasattr(v, "shape") for v in state.values()):
                return state
        if all(hasattr(v, "shape") for v in raw.values()):
            return raw
    raise TypeError(f"Unsupported checkpoint format: {type(raw)}")


def infer_dims_from_state_dict(state: Dict[str, torch.Tensor]) -> Tuple[int, int, int]:
    missing = [key for key in ("encoder.weight", "decoder.weight", "RNN.weight_ih_l0") if key not in state]
    if missing:
        raise KeyError(f"Checkpoint missing required tensors: {missing}")
    enc = state["encoder.weight"]
    dec = state["decoder.weight"]
    rnn_ih = state["RNN.weight_ih_l0"]
    Ng_enc, Np = enc.shape
    Np_dec, Ng_dec = dec.shape
    if Ng_enc != Ng_dec or Np != Np_dec:
        raise ValueError(f"Inconsistent encoder/decoder shapes: encoder {tuple(enc.shape)}, decoder {tuple(dec.shape)}")
    return int(Ng_enc), int(Np), int(rnn_ih.shape[1])


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
        self.options = options
        self.recurrent_type = str(getattr(options, "recurrent_type", "full")).lower()

        self.encoder = torch.nn.Linear(self.Np, self.Ng, bias=False)
        if self.recurrent_type == "low_rank":
            if LowRankRNN is None:
                raise RuntimeError("--recurrent_type low_rank requires LowRankRNN.py to be importable.")
            self.RNN = LowRankRNN(
                input_size=self.velocity_dim,
                hidden_size=self.Ng,
                k=int(getattr(options, "rank", 128)),
                nonlinearity=options.activation,
                factor_init=getattr(options, "low_rank_factor_init", "balanced"),
                recurrent_gain=getattr(options, "low_rank_recurrent_gain", 1.0),
                input_init_scale=getattr(options, "low_rank_input_init_scale", 1.0),
            )
        elif self.recurrent_type == "full":
            self.RNN = torch.nn.RNN(
                input_size=self.velocity_dim,
                hidden_size=self.Ng,
                nonlinearity=options.activation,
                bias=False,
            )
        else:
            raise ValueError("recurrent_type must be 'full' or 'low_rank'.")
        self.decoder = torch.nn.Linear(self.Ng, self.Np, bias=False)
        self.future_decoder = torch.nn.Linear(self.Ng, self.Np, bias=False)

    def _recurrent_matrix(self) -> torch.Tensor:
        if self.recurrent_type == "low_rank":
            return self.RNN.recurrent_gain * (self.RNN.M @ self.RNN.N / self.Ng)
        return self.RNN.weight_hh_l0

    def _anti_broadcast_window(self, T: int) -> Tuple[int, int]:
        start = int(getattr(self, "anti_broadcast_start", 0))
        stop = int(getattr(self, "anti_broadcast_stop", T))
        start = max(0, min(start, T))
        stop = max(start, min(stop, T))
        return start, stop

    def _rollout_states(self, inputs: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        v, p0 = inputs
        h = self.encoder(p0)[None]
        states = []
        start = int(task_cue_steps(self.options))
        stop = int(task_branch_step(self.options))
        noise_std = float(getattr(self.options, "delay_noise_std", 0.0))
        hidden_dropout = float(getattr(self.options, "hidden_dropout", 0.0))
        for t in range(v.shape[0]):
            _, h = self.RNN(v[t : t + 1], h)
            if self.training and start <= t < stop:
                if noise_std > 0:
                    h = h + torch.randn_like(h) * noise_std
                if hidden_dropout > 0:
                    keep = 1.0 - hidden_dropout
                    if keep <= 0:
                        h = torch.zeros_like(h)
                    else:
                        mask = torch.empty_like(h).bernoulli_(keep) / keep
                        h = h * mask
            states.append(h[0])
        return torch.stack(states, dim=0)

    def g(self, inputs: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        return self._rollout_states(inputs)

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

        current_entropy = -(current_pc * torch.log(current_pc.clamp_min(1e-12))).sum(-1).mean()
        future_entropy = -(future_pc * torch.log(future_pc.clamp_min(1e-12))).sum(-1)
        future_entropy = (future_entropy * mask).sum() / denom
        current_pos = self.place_cells.get_nearest_cell_pos(current_logits)
        future_pos = self.place_cells.get_nearest_cell_pos(future_logits)
        target_pos = self.place_cells.get_nearest_cell_pos(current_pc)
        target_future_pos = self.place_cells.get_nearest_cell_pos(future_pc)
        current_rmse = torch.sqrt(((current_pos - target_pos) ** 2).sum(-1)).mean() * 100.0
        future_rmse = torch.sqrt(((future_pos - target_future_pos) ** 2).sum(-1)).mean() * 100.0

        reg = self.weight_decay * (self._recurrent_matrix() ** 2).sum()
        activity_l1 = float(getattr(self.options, "activity_l1", 0.0)) * states.abs().mean()
        loss = current_ce + float(future_weight) * future_ce + reg + activity_l1
        terms = {
            "loss": float(loss.detach().cpu()),
            "current_ce": float(current_ce.detach().cpu()),
            "future_ce": float(future_ce.detach().cpu()),
            "current_target_entropy": float(current_entropy.detach().cpu()),
            "future_target_entropy": float(future_entropy.detach().cpu()),
            "current_excess_ce": float((current_ce - current_entropy).detach().cpu()),
            "future_excess_ce": float((future_ce - future_entropy).detach().cpu()),
            "current_rmse_cm": float(current_rmse.detach().cpu()),
            "future_rmse_cm": float(future_rmse.detach().cpu()),
            "activity_l1": float(activity_l1.detach().cpu()),
            "activity_sparsity": float((states.detach().abs() < 1e-6).float().mean().cpu()),
            "rec_reg": float(reg.detach().cpu()),
        }
        return loss, terms

    def core_state_dict(self) -> Dict[str, torch.Tensor]:
        """State dict compatible with code/model.py, excluding future_decoder."""
        return {
            "encoder.weight": self.encoder.weight.detach().cpu(),
            "RNN.weight_ih_l0": (
                self.RNN.weight_ih.detach().cpu()
                if self.recurrent_type == "low_rank"
                else self.RNN.weight_ih_l0.detach().cpu()
            ),
            "RNN.weight_hh_l0": self._recurrent_matrix().detach().cpu(),
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


def load_analysis_model(
    checkpoint_path: str | Path,
    device: str = "cpu",
    sequence_length: Optional[int] = None,
    batch_size: Optional[int] = None,
) -> Tuple[torch.nn.Module, PlaceCells, SimpleNamespace, Dict]:
    """Load either a future-split full checkpoint or a plain path-integration checkpoint."""
    payload = torch.load(checkpoint_path, map_location=device)
    if (
        isinstance(payload, dict)
        and "model_state_dict" in payload
        and any(str(k).startswith("future_decoder.") for k in payload["model_state_dict"].keys())
    ):
        model, place_cells, options, full_payload = load_future_model(checkpoint_path, device=device)
        if sequence_length is not None:
            options.sequence_length = int(sequence_length)
        if batch_size is not None:
            options.batch_size = int(batch_size)
        return model, place_cells, options, {"format": "future_split_full", **full_payload}

    state = extract_state_dict(payload)
    state = {k: v for k, v in state.items() if not str(k).startswith("future_decoder.")}
    Ng, Np, velocity_dim = infer_dims_from_state_dict(state)
    cue_dim = max(0, int(velocity_dim) - 2)
    options = make_options(
        task="binary_fork",
        batch_size=int(batch_size or COMMON_DEFAULTS["batch_size"]),
        sequence_length=int(sequence_length or COMMON_DEFAULTS["sequence_length"]),
        Np=Np,
        Ng=Ng,
        cue_dim=cue_dim,
        device=device,
        save_dir=str(Path(checkpoint_path).parent),
        run_ID="crossing_analysis",
    )
    options.velocity_dim = int(velocity_dim)
    options.cue_dim = cue_dim
    place_cells = PlaceCells(options)
    model = CoreRNN(options, place_cells).to(device)
    model.load_state_dict(state)
    model.eval()
    return model, place_cells, options, {"format": "core_state_dict", "checkpoint_path": str(checkpoint_path)}


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
    route_ids: np.ndarray
    pair_ids: np.ndarray
    positions_np: np.ndarray
    velocity_np: np.ndarray
    input_velocity_np: np.ndarray
    cue_np: np.ndarray
    task: str = "binary_fork"


@dataclass
class CrossingBatch:
    # RNN inputs are (velocity_sequence, initial_place_cell_activity), matching
    # the interface used by the path-integration RNN.
    inputs: Tuple[torch.Tensor, torch.Tensor]
    # Physical positions for all generated trajectories, shape [T, B, 2].
    positions_np: np.ndarray
    # Physical velocity increments derived from positions, shape [T, B, 2].
    velocity_np: np.ndarray
    # Heading angle for each trajectory in radians, shape [B].
    headings_np: np.ndarray
    # Every two trajectories share a pair id and cross at the same point.
    pair_ids: np.ndarray
    # Absolute heading separation for each pair in degrees.
    angle_sep_deg: np.ndarray
    # Time index at which each trajectory pair occupies the same position.
    crossing_step: int
    # Metadata: the future offset used to record future physical separation.
    future_horizon: int


def task_cue_steps(options: SimpleNamespace) -> int:
    return int(getattr(options, "cue_steps", getattr(options, "fork_cue_steps", 4)))


def task_branch_step(options: SimpleNamespace) -> int:
    return int(getattr(options, "branch_step", getattr(options, "fork_branch_step", 20)))


def route_cue_vectors(route_ids: np.ndarray, num_routes: int, cue_dim: int, scale: float) -> np.ndarray:
    """Compact route cue code.  A 2-D circular code is the graph-task default."""
    route_ids = np.asarray(route_ids, dtype=int)
    cue_dim = int(cue_dim)
    if cue_dim <= 0:
        return np.zeros((route_ids.size, 0), dtype=np.float32)
    if cue_dim == 1:
        if int(num_routes) <= 2:
            vals = np.where(route_ids <= 0, -1.0, 1.0)
        else:
            denom = max(1, int(num_routes) - 1)
            vals = 2.0 * route_ids.astype(float) / denom - 1.0
        return (float(scale) * vals[:, None]).astype(np.float32)

    theta = 2.0 * np.pi * route_ids.astype(float) / max(1, int(num_routes))
    cols = [np.cos(theta), np.sin(theta)]
    harmonic = 2
    while len(cols) < cue_dim:
        cols.append(np.cos(harmonic * theta))
        if len(cols) < cue_dim:
            cols.append(np.sin(harmonic * theta))
        harmonic += 1
    return (float(scale) * np.stack(cols[:cue_dim], axis=1)).astype(np.float32)


def pad_zero_cue(v2: torch.Tensor, cue_dim: int = 1) -> torch.Tensor:
    cue_dim = int(cue_dim)
    if cue_dim <= 0:
        return v2
    z = torch.zeros((*v2.shape[:2], cue_dim), dtype=v2.dtype, device=v2.device)
    return torch.cat([v2, z], dim=-1)


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
    branch_step = int(np.clip(task_branch_step(options), 2, T - 2))
    cue_steps = int(np.clip(task_cue_steps(options), 1, branch_step - 1))
    branch_y = float(options.fork_branch_y)
    arm_length = float(options.fork_arm_length)
    horizon = int(future_horizon if future_horizon is not None else options.future_horizon)
    route_ids = (routes > 0).astype(int)

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
    cue_dim = int(getattr(options, "cue_dim", 1))
    cue = np.zeros((T, B, cue_dim), dtype=np.float32)
    cue[:cue_steps] = route_cue_vectors(route_ids, 2, cue_dim, float(options.fork_cue_scale))[None]
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
        route_ids=route_ids.astype(int),
        pair_ids=pair_ids,
        positions_np=pos,
        velocity_np=vel_xy,
        input_velocity_np=v,
        cue_np=cue,
        task="binary_fork",
    )


def _graph_route_points(route_id: int, num_routes: int, branch_y: float) -> np.ndarray:
    """Control points for mixed split/reconvergent graph routes inside the box."""
    r = int(route_id)
    n = max(2, int(num_routes))
    first_groups = min(4, n)
    first = r % first_groups
    recon = r % 3
    theta = 2.0 * np.pi * r / n
    start = np.array([0.0, branch_y], dtype=np.float32)
    p1 = np.array([
        -0.48 + 0.96 * first / max(1, first_groups - 1),
        branch_y + 0.28,
    ], dtype=np.float32)
    p2 = np.array([
        -0.42 + 0.42 * recon,
        branch_y + 0.60,
    ], dtype=np.float32)
    end = np.array([
        0.72 * np.cos(theta),
        0.22 + 0.62 * np.sin(theta),
    ], dtype=np.float32)
    return np.stack([start, p1, p2, np.clip(end, -0.95, 0.95)], axis=0)


def _piecewise_route_position(points: np.ndarray, progress: float) -> np.ndarray:
    progress = float(np.clip(progress, 0.0, 1.0))
    n_seg = points.shape[0] - 1
    scaled = progress * n_seg
    seg = min(n_seg - 1, int(np.floor(scaled)))
    frac = scaled - seg
    return ((1.0 - frac) * points[seg] + frac * points[seg + 1]).astype(np.float32)


def generate_graph_batch(
    options: SimpleNamespace,
    place_cells: PlaceCells,
    batch_size: Optional[int] = None,
    rng: Optional[np.random.Generator] = None,
    paired: bool = True,
    future_horizon: Optional[int] = None,
) -> ForkBatch:
    """Generate matched multi-route delayed-cue graph-maze trajectories.

    Paired mode orders trials as donor pairs.  Each pair has identical present
    position and local velocity through the post-cue pre-branch window, but the
    remembered route cue selects different physical futures.
    """
    rng = rng or np.random.default_rng()
    B = int(batch_size or options.batch_size)
    num_routes = max(2, int(getattr(options, "num_routes", 12)))
    T = int(options.sequence_length)
    branch_step = int(np.clip(task_branch_step(options), 2, T - 2))
    cue_steps = int(np.clip(task_cue_steps(options), 1, branch_step - 1))
    branch_y = float(options.fork_branch_y)
    stem_start_y = float(options.fork_stem_start_y)
    horizon = int(future_horizon if future_horizon is not None else options.future_horizon)

    if paired:
        B = max(2, B)
        if B % 2:
            B -= 1
        pair_count = B // 2
        route_a = rng.integers(0, num_routes, size=pair_count)
        offset = rng.integers(1, num_routes, size=pair_count)
        route_b = (route_a + offset) % num_routes
        route_ids = np.empty(B, dtype=int)
        route_ids[0::2] = route_a
        route_ids[1::2] = route_b
        pair_ids = np.repeat(np.arange(pair_count, dtype=int), 2)
        stem_x = np.repeat(rng.normal(0.0, float(options.fork_lateral_jitter), size=pair_count), 2)
    else:
        route_ids = rng.integers(0, num_routes, size=B)
        pair_ids = np.arange(B, dtype=int)
        stem_x = rng.normal(0.0, float(options.fork_lateral_jitter), size=B)

    pos = np.zeros((T, B, 2), dtype=np.float32)
    local_pos = np.zeros((T, B, 2), dtype=np.float32)
    for t in range(T):
        if t < branch_step:
            frac = t / max(branch_step - 1, 1)
            y = stem_start_y + frac * (branch_y - stem_start_y)
            pos[t, :, 0] = stem_x
            pos[t, :, 1] = y
            local_pos[t, :, 0] = 0.0
            local_pos[t, :, 1] = y - stem_start_y
        else:
            progress = (t - branch_step + 1) / max(T - branch_step, 1)
            local_pos[t, :, 0] = 0.0
            local_pos[t, :, 1] = (branch_y - stem_start_y) + progress
            for b, rid in enumerate(route_ids):
                points = _graph_route_points(int(rid), num_routes, branch_y)
                p = _piecewise_route_position(points, progress)
                pos[t, b] = p + np.array([stem_x[b], 0.0], dtype=np.float32)

    physical_vel = np.zeros_like(pos)
    physical_vel[1:] = pos[1:] - pos[:-1]
    local_vel = np.zeros_like(local_pos)
    local_vel[1:] = local_pos[1:] - local_pos[:-1]
    if str(getattr(options, "velocity_frame", "local_graph")) == "local_graph":
        vel_input_xy = local_vel
    else:
        vel_input_xy = physical_vel

    cue_dim = int(getattr(options, "cue_dim", 2))
    cue = np.zeros((T, B, cue_dim), dtype=np.float32)
    cue_code = route_cue_vectors(route_ids, num_routes, cue_dim, float(options.fork_cue_scale))
    cue[:cue_steps] = cue_code[None]
    cue_dropout = float(getattr(options, "cue_dropout", 0.0))
    if cue_dropout > 0:
        drop = rng.random((cue_steps, B, 1)) < cue_dropout
        cue[:cue_steps] = cue[:cue_steps] * (~drop)
    cue_noise = float(getattr(options, "cue_noise_std", 0.0))
    if cue_noise > 0:
        cue[:cue_steps] = cue[:cue_steps] + rng.normal(0.0, cue_noise, size=cue[:cue_steps].shape).astype(np.float32)
    v = np.concatenate([vel_input_xy, cue], axis=-1)

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

    route_angles = 2.0 * np.pi * route_ids / num_routes
    return ForkBatch(
        inputs=(v_t, init_actv),
        pos=pos_t,
        place_outputs=place_outputs,
        future_place_outputs=future_place_outputs,
        future_mask=future_mask_t,
        routes=np.stack([np.cos(route_angles), np.sin(route_angles)], axis=1).astype(np.float32),
        route_ids=route_ids.astype(int),
        pair_ids=pair_ids,
        positions_np=pos,
        velocity_np=physical_vel,
        input_velocity_np=v,
        cue_np=cue,
        task="multi_route_graph",
    )


def generate_task_batch(
    options: SimpleNamespace,
    place_cells: PlaceCells,
    batch_size: Optional[int] = None,
    rng: Optional[np.random.Generator] = None,
    paired: bool = True,
    future_horizon: Optional[int] = None,
) -> ForkBatch:
    if str(getattr(options, "task", "binary_fork")) == "multi_route_graph":
        return generate_graph_batch(options, place_cells, batch_size, rng, paired, future_horizon)
    return generate_fork_batch(options, place_cells, batch_size, rng, paired, future_horizon)


def generate_open_field_batch(
    options: SimpleNamespace,
    place_cells: PlaceCells,
    traj_gen: TrajectoryGenerator,
    batch_size: Optional[int] = None,
    future_horizon: Optional[int] = None,
) -> ForkBatch:
    B = int(batch_size or options.batch_size)
    horizon = int(future_horizon if future_horizon is not None else options.future_horizon)
    inputs2, pos_t, place_outputs = traj_gen.get_test_batch(batch_size=B)
    v2, init_actv = inputs2
    v = pad_zero_cue(v2.to(options.device), int(getattr(options, "cue_dim", 1)))
    pos_np = pos_t.detach().cpu().numpy().astype(np.float32)
    vel_np = np.zeros_like(pos_np)
    vel_np[1:] = pos_np[1:] - pos_np[:-1]
    T = pos_np.shape[0]
    future_idx = np.minimum(np.arange(T) + horizon, T - 1)
    future_pos = torch.tensor(pos_np[future_idx], dtype=torch.float32, device=options.device)
    future_place_outputs = place_cells.get_activation(future_pos)
    future_mask = ((np.arange(T) + horizon) < T).astype(np.float32)[:, None]
    future_mask = np.repeat(future_mask, B, axis=1)
    cue = np.zeros((T, B, int(getattr(options, "cue_dim", 1))), dtype=np.float32)
    return ForkBatch(
        inputs=(v, init_actv.to(options.device)),
        pos=pos_t.to(options.device),
        place_outputs=place_outputs,
        future_place_outputs=future_place_outputs,
        future_mask=torch.tensor(future_mask, dtype=torch.float32, device=options.device),
        routes=np.zeros(B, dtype=int),
        route_ids=np.zeros(B, dtype=int),
        pair_ids=np.arange(B, dtype=int),
        positions_np=pos_np,
        velocity_np=vel_np,
        input_velocity_np=v.detach().cpu().numpy(),
        cue_np=cue,
        task="open_field",
    )


def generate_crossing_batch(
    options: SimpleNamespace,
    place_cells: PlaceCells,
    batch_size: int,
    rng: np.random.Generator,
    crossing_step: Optional[int] = None,
    min_angle_deg: float = 30.0,
    max_angle_deg: float = 150.0,
    line_extent: float = 0.75,
    future_horizon: Optional[int] = None,
) -> CrossingBatch:
    """Generate paired straight-line trajectories that cross as an X."""
    # The analysis is pair-based, so enforce an even batch size.
    B = max(2, int(batch_size))
    if B % 2:
        B -= 1
    pair_count = B // 2

    # Use the model's configured sequence length and put the crossing in the
    # middle unless the caller requested a specific crossing time.
    T = int(options.sequence_length)
    cross_t = int(crossing_step if crossing_step is not None else T // 2)
    cross_t = int(np.clip(cross_t, 1, T - 2))
    horizon = int(future_horizon if future_horizon is not None else getattr(options, "future_horizon", 8))

    # Convert the requested angle range into radians. If min == max, this
    # produces fixed-angle crossings such as exactly 90 degrees.
    min_angle = np.deg2rad(float(min_angle_deg))
    max_angle = np.deg2rad(max(float(max_angle_deg), float(min_angle_deg)))
    max_angle = min(max_angle, np.pi)

    # Each pair gets one shared crossing center. Both trajectories pass through
    # this same center at cross_t, so current physical position is matched.
    centers = rng.uniform(-0.15, 0.15, size=(pair_count, 2)).astype(np.float32)

    # Draw heading A freely, then draw heading B at the requested angular
    # separation clockwise or counterclockwise from heading A.
    theta_a = rng.uniform(-np.pi, np.pi, size=pair_count)
    sep = rng.uniform(min_angle, max_angle, size=pair_count)
    sep *= rng.choice(np.array([-1.0, 1.0]), size=pair_count)
    theta_b = theta_a + sep

    # Flatten paired headings into the batch dimension: [pair0_A, pair0_B,
    # pair1_A, pair1_B, ...].
    headings_pair = np.stack([theta_a, theta_b], axis=1)
    headings = headings_pair.reshape(-1)
    pair_ids = np.repeat(np.arange(pair_count, dtype=int), 2)
    centers_trials = np.repeat(centers, 2, axis=0)
    directions = np.stack([np.cos(headings), np.sin(headings)], axis=1).astype(np.float32)

    # offsets[cross_t] is exactly zero. Therefore both members of a pair have
    # identical position at cross_t, but their past/future paths differ by angle.
    denom = float(max(cross_t, T - 1 - cross_t, 1))
    offsets = ((np.arange(T, dtype=np.float32) - float(cross_t)) / denom) * float(line_extent)
    pos = centers_trials[None] + offsets[:, None, None] * directions[None]

    # Keep trajectories inside the square arena used by the original RNN.
    pos[:, :, 0] = np.clip(pos[:, :, 0], -float(options.box_width) / 2 + 0.02, float(options.box_width) / 2 - 0.02)
    pos[:, :, 1] = np.clip(pos[:, :, 1], -float(options.box_height) / 2 + 0.02, float(options.box_height) / 2 - 0.02)

    # The RNN consumes velocity increments, not absolute positions. vel[0] is
    # copied from vel[1] so the first step has a valid heading/magnitude.
    vel = np.zeros_like(pos, dtype=np.float32)
    vel[1:] = pos[1:] - pos[:-1]
    if T > 1:
        vel[0] = vel[1]

    # Future-split checkpoints may expect extra cue channels; the crossing task
    # supplies zero cue channels so it behaves like ordinary path integration.
    cue_dim = int(getattr(options, "cue_dim", max(0, int(getattr(options, "velocity_dim", 2)) - 2)))
    v = np.concatenate([vel, np.zeros((T, B, cue_dim), dtype=np.float32)], axis=-1)
    device = torch.device(options.device)
    pos_t = torch.tensor(pos, dtype=torch.float32, device=device)
    v_t = torch.tensor(v, dtype=torch.float32, device=device)

    # Initialize the RNN with place-cell activity for the first physical
    # position of each trajectory, matching the standard path-integration setup.
    init_pos = torch.tensor(pos[0], dtype=torch.float32, device=device)[:, None, :]
    init_actv = place_cells.get_activation(init_pos).squeeze(1)

    # Store absolute separation in [0, 180] degrees for logging and QC.
    angle_sep = np.abs(np.rad2deg(np.angle(np.exp(1j * (theta_b - theta_a)))))
    angle_sep = np.minimum(angle_sep, 360.0 - angle_sep)
    return CrossingBatch(
        inputs=(v_t, init_actv),
        positions_np=pos.astype(np.float32),
        velocity_np=vel.astype(np.float32),
        headings_np=headings.astype(np.float32),
        pair_ids=pair_ids,
        angle_sep_deg=angle_sep.astype(np.float32),
        crossing_step=cross_t,
        future_horizon=horizon,
    )


def assert_crossing_batch(batch: CrossingBatch, min_angle_deg: float) -> None:
    # This is the central matched-state assertion: both trajectories in a pair
    # must occupy the same physical x-y position at the crossing time.
    t = int(batch.crossing_step)
    for pair_id in np.unique(batch.pair_ids):
        idx = np.where(batch.pair_ids == pair_id)[0]
        if idx.size != 2:
            continue
        if not np.allclose(batch.positions_np[t, idx[0]], batch.positions_np[t, idx[1]], atol=1e-5):
            raise AssertionError("Crossing pair positions are not matched at crossing_step.")

    # The pair should also satisfy the requested direction-separation constraint.
    if np.nanmin(batch.angle_sep_deg) < float(min_angle_deg) - 1e-5:
        raise AssertionError("Crossing pair angle separation is below the requested minimum.")


def assert_matched_prebranch(batch: ForkBatch, options: SimpleNamespace) -> None:
    cue_steps = int(task_cue_steps(options))
    branch_step = int(task_branch_step(options))
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
            v3 = pad_zero_cue(v2.to(options.device), int(getattr(options, "cue_dim", 1)))
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
    progress_every: int = 0,
) -> np.ndarray:
    T, B, Ng = activations.shape
    flat_x = xs.reshape(-1)
    flat_y = ys.reshape(-1)
    maps = np.zeros((Ng, scorer._nbins, scorer._nbins), dtype=np.float32)
    start_time = time.time()
    progress_every = max(0, int(progress_every))
    for u in range(Ng):
        maps[u] = scorer.calculate_ratemap(flat_x, flat_y, activations[:, :, u].reshape(-1), statistic="mean")
        if progress_every and ((u + 1) == 1 or (u + 1) % progress_every == 0 or (u + 1) == Ng):
            elapsed = time.time() - start_time
            rate = (u + 1) / max(elapsed, 1e-9)
            eta = (Ng - (u + 1)) / max(rate, 1e-9)
            log_info(
                "[classify:ratemaps] %d/%d units (%.1f%%) elapsed=%s eta=%s",
                u + 1,
                Ng,
                100.0 * float(u + 1) / max(1.0, float(Ng)),
                format_duration(elapsed),
                format_duration(eta),
            )
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
    # For each unit, find the shift at which its 60-degree gridness score is
    # highest. Positive best_cm means the unit's grid map aligns best with a
    # future/downstream position; near-zero best_cm means ordinary zero-shift
    # grid coding.
    best_idx, best_vals = _safe_nanargmax(scores_60)
    best_cm = np.full(best_idx.shape, np.nan, dtype=float)
    valid = best_idx >= 0
    best_cm[valid] = lag_cm[best_idx[valid]]

    # Only units above the gridness threshold are allowed to be called grid-like.
    qual = valid & np.isfinite(best_vals) & (best_vals >= gridness_threshold)

    # Classification is intentionally simple and interpretable:
    # predictive = high-gridness units shifted forward by at least min_shift_cm;
    # retrospective = shifted backward by at least min_shift_cm;
    # standard = high-gridness units whose best shift remains near zero.
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
    # load_analysis_model supports both future_split_full.pth and older
    # plain path-integration checkpoints such as epoch_50.pth.
    model, place_cells, options, payload = load_analysis_model(
        args.checkpoint_path,
        device=device,
        sequence_length=args.sequence_length,
        batch_size=args.batch_size,
    )
    options.batch_size = int(args.batch_size)
    options.sequence_length = int(args.sequence_length or options.sequence_length)
    options.device = device

    # The classifier writes gridness_data.npz, which later crossing/decode
    # commands use as the source of unit-class labels.
    out_dir = ensure_dir(Path(args.output_dir) if args.output_dir else Path(payload.get("output_dir", ".")).parent / "future_split_classify")
    configure_logging(out_dir, "classify.log")
    Ng_use_arg = str(args.Ng_use).lower()
    Ng_use = int(options.Ng) if Ng_use_arg == "all" else min(int(float(args.Ng_use)), int(options.Ng))
    log_info("[classify] checkpoint=%s", args.checkpoint_path)
    log_info(
        "[classify] device=%s Ng_use=%d batches=%d batch_size=%d res=%d shift_mode=%s max_lag=%d max_shift_cm=%.2f",
        device,
        Ng_use,
        int(args.n_batches),
        int(args.batch_size),
        int(args.res),
        str(getattr(args, "shift_mode", "time")),
        int(args.max_lag),
        float(getattr(args, "max_shift_cm", 20.0)),
    )
    log_info("[classify] checkpoint_format=%s velocity_dim=%d", payload.get("format", "?"), int(options.velocity_dim))

    start_time = time.time()
    log_info("[classify] collecting zero-cue open-field sequences")
    # Open-field data keeps classification independent from the later crossing
    # trajectories; this avoids defining PGCs on the same data used for testing.
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

    shift_mode = str(getattr(args, "shift_mode", "time"))
    space_projection = str(getattr(args, "space_projection", "path"))
    shifts = build_shift_values(
        shift_mode,
        max_lag=int(args.max_lag),
        max_shift_cm=float(getattr(args, "max_shift_cm", 20.0)),
        shift_step_cm=float(getattr(args, "shift_step_cm", 1.0)),
    )
    score_log_every = int(getattr(args, "score_log_every_units", 25))
    log_info(
        "[classify] scoring %d %s shifts (%s) across %d units; progress every %d units",
        len(shifts),
        shift_mode,
        shift_config_label(shift_mode, space_projection),
        Ng_use,
        score_log_every,
    )
    scores_60, scores_90 = scorer.predictive_grid_scores(
        xs,
        ys,
        activations,
        shifts,
        shift_mode=shift_mode,
        periodic=bool(getattr(options, "periodic", False)),
        space_projection=space_projection,
        progress=True,
        progress_label="classify:gridness",
        progress_every=score_log_every,
        progress_callback=lambda msg: log_info("%s", msg),
    )
    cm_step = cm_per_step(xs, ys)
    lag_cm = shift_values_to_cm(shifts, shift_mode, cm_step)

    # Convert shifted gridness scores into predictive/retrospective/standard
    # unit-index arrays.
    classes = classify_units_from_scores(lag_cm, scores_60, args.min_shift_cm, args.gridness_threshold)
    best_idx, _ = _safe_nanargmax(scores_60)
    best_lag_steps = np.full(best_idx.shape, np.nan, dtype=float)
    best_shift_values = np.full(best_idx.shape, np.nan, dtype=float)
    valid_best = best_idx >= 0
    shift_values_arr = np.asarray(shifts, dtype=float)
    best_shift_values[valid_best] = shift_values_arr[best_idx[valid_best]]
    if shift_mode == "time":
        best_lag_steps[valid_best] = shift_values_arr[best_idx[valid_best]]
    log_info("[classify] computing rate maps and band controls")

    # Band units are a control population selected from rate-map structure,
    # independent of the predictive-vs-zero-lag split.
    rate_maps = compute_rate_maps(xs, ys, activations, scorer, progress_every=score_log_every)

    band_vals, band_kx, band_ky = band_scores(rate_maps, int(args.res), float(options.box_width))
    finite_band = band_vals[np.isfinite(band_vals)]
    if args.band_threshold is not None:
        band_cutoff = float(args.band_threshold)
    elif finite_band.size:
        band_cutoff = float(np.nanpercentile(finite_band, float(args.band_percentile)))
    else:
        band_cutoff = float("nan")
    band_units = np.where(band_vals >= band_cutoff)[0] if np.isfinite(band_cutoff) else np.array([], dtype=int)

    # Save both raw scores and derived class labels so later analyses can be
    # rerun without repeating the slow gridness scoring step.
    np.savez_compressed(
        out_dir / "gridness_data.npz",
        shift_mode=np.array(shift_mode),
        space_projection=np.array(space_projection),
        shift_values=shift_values_arr,
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
        best_lag_steps=best_lag_steps,
        best_shift_values=best_shift_values,
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
        "shift_mode": shift_mode,
        "space_projection": space_projection if shift_mode == "space" else None,
        "shift_config": shift_config_label(shift_mode, space_projection),
        "Ng_use": int(Ng_use),
        "cm_per_step": cm_step,
        "shift_values": shift_values_arr,
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
    log_info(
        "[classify] shift axis=%s range=%.2f..%.2f cm samples=%d",
        shift_config_label(shift_mode, space_projection),
        float(np.nanmin(lag_cm)) if lag_cm.size else float("nan"),
        float(np.nanmax(lag_cm)) if lag_cm.size else float("nan"),
        int(lag_cm.size),
    )
    log_info("[classify] wrote %s", out_dir)
    return out_dir


# --------------------------------------------------------------------------------------
# Decoding
# --------------------------------------------------------------------------------------


def load_gridness_payload(path: str | Path) -> Dict[str, np.ndarray]:
    # Read the compressed classification output produced by run_classify.
    with np.load(path, allow_pickle=True) as data:
        return {k: data[k] for k in data.files}


def class_arrays(grid_data: Dict[str, np.ndarray], Ng: int, fallback: bool = True) -> Dict[str, np.ndarray]:
    # Different scripts historically used slightly different key names. This
    # helper accepts all known aliases and normalizes them into one dictionary.
    def get(*names):
        for name in names:
            if name in grid_data:
                arr = np.asarray(grid_data[name], dtype=int).reshape(-1)
                # Keep only valid unit indices for the currently loaded model.
                arr = arr[(arr >= 0) & (arr < Ng)]
                if arr.size:
                    return np.unique(arr)
        return np.array([], dtype=int)

    # Primary classes used by crossing: predictive PGCs and standard/zero-lag GCs.
    pred = get("classes_predictive", "predictive")
    retro = get("classes_retrospective", "classes_phase_precession", "retrospective")
    standard = get("classes_standard", "classes_phase_locked", "classes_normal", "normal")
    low = get("low_grid_units", "classes_low_grid", "non_grid")
    band = get("band_units", "classes_band")

    # Fallbacks keep smoke tests and tiny toy models from crashing. For real
    # analyses with valid classification files, these fallbacks should not fire.
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
            batch = generate_task_batch(
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
            routes_list.append(batch.route_ids)
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
    cue_steps = int(task_cue_steps(options))
    branch_step = int(task_branch_step(options))
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


def route_one_hot(route_ids: np.ndarray, num_routes: int) -> np.ndarray:
    route_ids = np.asarray(route_ids, dtype=int).reshape(-1)
    n = max(int(num_routes), int(route_ids.max()) + 1 if route_ids.size else 1)
    out = np.zeros((route_ids.size, n), dtype=np.float32)
    if route_ids.size:
        out[np.arange(route_ids.size), np.clip(route_ids, 0, n - 1)] = 1.0
    return out


def unit_features_for_horizon(
    states: np.ndarray,
    options: SimpleNamespace,
    horizon: int,
    units: np.ndarray,
) -> np.ndarray:
    times = matched_sample_times(options, horizon)
    units = np.asarray(units, dtype=int)
    if units.size == 0 or times.size == 0:
        return np.zeros((states.shape[1] * max(1, times.size), 0), dtype=np.float32)
    rows = [states[int(t)][:, units] for t in times]
    return np.concatenate(rows, axis=0).astype(np.float32)


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
    X_kin, X_pred, X_std, X_low, X_band, X_route = [], [], [], [], [], []
    y_route, y_pos, y_phase, sample_pairs, sample_times = [], [], [], [], []
    num_routes = int(getattr(options, "num_routes", int(np.max(routes)) + 1 if np.asarray(routes).size else 2))
    for t in times:
        fut_t = t + int(horizon)
        kin = kinematic_features(pos, vel, int(t), history_steps)
        X_kin.append(kin)
        X_pred.append(states[t][:, units["predictive"]] if units["predictive"].size else np.zeros((states.shape[1], 0)))
        X_std.append(states[t][:, units["standard"]] if units["standard"].size else np.zeros((states.shape[1], 0)))
        X_low.append(states[t][:, units["low_grid"]] if units["low_grid"].size else np.zeros((states.shape[1], 0)))
        X_band.append(states[t][:, units["band"]] if units["band"].size else np.zeros((states.shape[1], 0)))
        X_route.append(route_one_hot(routes, num_routes))
        y_route.append(routes.astype(int))
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
        "X_route_id": cat(X_route),
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
            solver = "liblinear" if np.unique(y_train).size == 2 else "lbfgs"
            clf = LogisticRegression(max_iter=1000, solver=solver)
            clf.fit(X_train, y_train)
            prob = clf.predict_proba(X_test)
            pred = clf.classes_[np.argmax(prob, axis=1)]
            row["route_accuracy"] = float(accuracy_score(y_test, pred))
            if np.unique(y_train).size == 2 and np.unique(y_test).size == 2:
                pos_col = int(np.where(clf.classes_ == np.max(clf.classes_))[0][0])
                row["route_auc"] = float(roc_auc_score(y_test, prob[:, pos_col]))
            elif np.unique(y_test).size > 2:
                row["route_auc"] = float(roc_auc_score(y_test, prob, multi_class="ovr", labels=clf.classes_))
            else:
                row["route_auc"] = float("nan")
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


def preferred_lag_steps(grid_data: Dict[str, np.ndarray], Ng: int) -> np.ndarray:
    if "best_lag_steps" in grid_data:
        vals = np.asarray(grid_data["best_lag_steps"], dtype=float).reshape(-1)
        if vals.size < Ng:
            vals = np.pad(vals, (0, Ng - vals.size), constant_values=np.nan)
        if np.isfinite(vals[:Ng]).any():
            return vals[:Ng]
    shift_mode = str(np.asarray(grid_data.get("shift_mode", "time")).reshape(()))
    if shift_mode == "space":
        return np.full(Ng, np.nan, dtype=float)
    best_cm = np.asarray(grid_data.get("best_cm", np.full(Ng, np.nan)), dtype=float).reshape(-1)
    lag_cm = np.asarray(grid_data.get("lag_cm", []), dtype=float).reshape(-1)
    shifts = np.asarray(grid_data.get("shift_values", []), dtype=float).reshape(-1)
    if best_cm.size < Ng:
        best_cm = np.pad(best_cm, (0, Ng - best_cm.size), constant_values=np.nan)
    if lag_cm.size >= 2 and shifts.size == lag_cm.size:
        cm_per = float(np.nanmean(np.abs(np.diff(lag_cm))) / max(np.nanmean(np.abs(np.diff(shifts))), 1e-9))
        if np.isfinite(cm_per) and cm_per > 0:
            return best_cm[:Ng] / cm_per
    return np.full(Ng, np.nan, dtype=float)


def predictive_units_for_preferred_horizon(
    units: Dict[str, np.ndarray],
    best_lag: np.ndarray,
    horizon: int,
    min_units: int = 2,
) -> np.ndarray:
    pred = np.asarray(units.get("predictive", []), dtype=int)
    if pred.size == 0:
        return pred
    vals = best_lag[pred]
    finite = np.isfinite(vals) & (vals > 0)
    if not finite.any():
        return pred[: min(pred.size, max(min_units, 8))]
    candidates = pred[finite]
    dist = np.abs(best_lag[candidates] - float(horizon))
    cutoff = max(1.0, 0.25 * float(horizon))
    selected = candidates[dist <= cutoff]
    if selected.size >= min_units:
        return selected
    order = np.argsort(dist)
    return candidates[order[: min(candidates.size, max(min_units, 8))]]


def safe_pearson_corr(a: np.ndarray, b: np.ndarray) -> float:
    # Correlate two population activity vectors after mean-centering each one.
    # In crossing, a and b are same-class activity vectors from trajectory A/B.
    a = np.asarray(a, dtype=float).reshape(-1)
    b = np.asarray(b, dtype=float).reshape(-1)
    if a.size < 2 or b.size < 2:
        return float("nan")
    a = a - np.nanmean(a)
    b = b - np.nanmean(b)
    denom = float(np.sqrt(np.nansum(a * a) * np.nansum(b * b)))
    if denom <= 1e-12 or not np.isfinite(denom):
        return float("nan")
    return float(np.nansum(a * b) / denom)


def bootstrap_ci(values: np.ndarray, rng: np.random.Generator, n_boot: int = 2000) -> Tuple[float, float]:
    # Nonparametric 95% CI over pair-level effects or group-level correlations.
    vals = np.asarray(values, dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return float("nan"), float("nan")
    if vals.size == 1:
        return float(vals[0]), float(vals[0])
    means = np.empty(int(n_boot), dtype=float)
    for i in range(int(n_boot)):
        means[i] = float(np.mean(rng.choice(vals, size=vals.size, replace=True)))
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(lo), float(hi)


def summarize_corr_rows(rows: Sequence[Dict[str, object]], rng: np.random.Generator) -> Tuple[List[Dict[str, object]], Dict[str, object]]:
    # First summarize each signal class independently: mean GC correlation,
    # mean PGC correlation, band control correlation, etc.
    signals = sorted({str(row.get("signal")) for row in rows})
    summary_rows: List[Dict[str, object]] = []
    for signal in signals:
        vals = np.asarray([float(row.get("correlation", np.nan)) for row in rows if row.get("signal") == signal], dtype=float)
        vals = vals[np.isfinite(vals)]
        lo, hi = bootstrap_ci(vals, rng, n_boot=1000)
        summary_rows.append(
            {
                "signal": signal,
                "n_pairs": int(vals.size),
                "mean_correlation": float(np.mean(vals)) if vals.size else float("nan"),
                "sem_correlation": float(np.std(vals, ddof=1) / np.sqrt(vals.size)) if vals.size > 1 else float("nan"),
                "ci95_low": lo,
                "ci95_high": hi,
            }
        )

    # Then build the paired contrast. Each pair contributes one number:
    #   corr(zero-lag GC population) - corr(PGC population)
    # Positive values mean standard grid cells overlap more than PGCs at the
    # same crossing position.
    pred_by_pair = {
        int(row["pair_id"]): float(row["correlation"])
        for row in rows
        if row.get("signal") == "predictive" and np.isfinite(float(row.get("correlation", np.nan)))
    }
    std_by_pair = {
        int(row["pair_id"]): float(row["correlation"])
        for row in rows
        if row.get("signal") == "standard_grid" and np.isfinite(float(row.get("correlation", np.nan)))
    }
    common = sorted(set(pred_by_pair) & set(std_by_pair))
    diffs = np.asarray([std_by_pair[p] - pred_by_pair[p] for p in common], dtype=float)
    lo, hi = bootstrap_ci(diffs, rng, n_boot=2000)
    comparison = {
        "n_matched_pairs": int(diffs.size),
        "mean_standard_minus_predictive_corr": float(np.mean(diffs)) if diffs.size else float("nan"),
        "ci95_low": lo,
        "ci95_high": hi,
        "fraction_standard_gt_predictive": float(np.mean(diffs > 0)) if diffs.size else float("nan"),
    }
    return summary_rows, comparison


def run_crossing(args) -> Path:
    if str(getattr(args, "pairing_mode", "x_crossing")) == "same_bin":
        return run_same_bin_population_corr(args)

    rng = set_seed(int(args.random_seed))
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = ensure_dir(args.output_dir)
    configure_logging(out_dir, "crossing.log")
    log_info("[crossing] checkpoint=%s", args.checkpoint_path)
    log_info("[crossing] gridness=%s", args.gridness_path)

    # Load the trained RNN and set its sequence length long enough to contain
    # the crossing plus the requested future-horizon metadata.
    min_sequence_length = max(int(args.future_horizon) * 3 + 2, 32)
    if getattr(args, "crossing_step", None) is not None:
        min_sequence_length = max(min_sequence_length, int(args.crossing_step) + int(args.future_horizon) + 2)
    requested_sequence_length = getattr(args, "sequence_length", None)
    sequence_length = max(min_sequence_length, int(requested_sequence_length)) if requested_sequence_length else min_sequence_length
    model, place_cells, options, payload = load_analysis_model(
        args.checkpoint_path,
        device=device,
        sequence_length=sequence_length,
        batch_size=args.batch_size,
    )

    # Load unit classes from gridness_data.npz: predictive, standard, band, etc.
    grid_data = load_gridness_payload(args.gridness_path)
    units = class_arrays(grid_data, int(options.Ng), fallback=bool(args.allow_fallback_units))
    log_info(
        "[crossing] device=%s class_counts=%s min_angle_deg=%.1f",
        device,
        {k: int(v.size) for k, v in units.items()},
        float(args.min_angle_deg),
    )
    log_info("[crossing] checkpoint_format=%s velocity_dim=%d", payload.get("format", "?"), int(options.velocity_dim))

    # These are the within-population vectors that will be correlated between
    # the two trajectories in each matched pair.
    groups: Dict[str, np.ndarray] = {
        "predictive": units["predictive"],
        "standard_grid": units["standard"],
        "retrospective": units["retrospective"],
        "band": units["band"],
    }
    pred_count = max(1, int(units["predictive"].size))

    # Optional count-matched standard-grid control. It uses the same number of
    # units as the predictive group when enough standard units are available.
    if units["standard"].size:
        groups["standard_grid_matched_count"] = units["standard"][: min(pred_count, units["standard"].size)]

    # Random matched-count controls sample non-predictive units repeatedly.
    random_candidates = np.setdiff1d(np.arange(int(options.Ng), dtype=int), units["predictive"])

    rows: List[Dict[str, object]] = []
    pair_offset = 0
    model.eval()
    start_time = time.time()
    with torch.no_grad():
        for batch_idx in range(max(1, int(args.n_batches))):
            # Generate many independent X-crossing pairs. Within each pair,
            # current position at crossing_step is identical while heading differs.
            batch = generate_crossing_batch(
                options,
                place_cells,
                batch_size=int(args.batch_size),
                rng=rng,
                crossing_step=args.crossing_step,
                min_angle_deg=float(args.min_angle_deg),
                max_angle_deg=float(args.max_angle_deg),
                line_extent=float(args.line_extent),
                future_horizon=int(args.future_horizon),
            )
            assert_crossing_batch(batch, float(args.min_angle_deg))

            # Run the RNN and extract the hidden grid population over time.
            states = model.g(batch.inputs).detach().cpu().numpy()

            # The main analysis is only at this single matched crossing time.
            t = int(batch.crossing_step)
            fut_t = min(t + int(args.future_horizon), states.shape[0] - 1)
            for local_pid in np.unique(batch.pair_ids):
                idx = np.where(batch.pair_ids == local_pid)[0]
                if idx.size != 2:
                    continue
                a, b = int(idx[0]), int(idx[1])
                pair_id = pair_offset + int(local_pid)

                # This is not used in the correlation itself; it documents how
                # far apart the two future physical positions become.
                future_sep_cm = float(np.linalg.norm(batch.positions_np[fut_t, a] - batch.positions_np[fut_t, b]) * 100.0)
                base = {
                    "pair_id": pair_id,
                    "batch": int(batch_idx),
                    "crossing_step": int(t),
                    "future_horizon": int(args.future_horizon),
                    "angle_sep_deg": float(batch.angle_sep_deg[int(local_pid)]),
                    "future_position_sep_cm": future_sep_cm,
                }
                for signal, unit_idx in groups.items():
                    unit_idx = np.asarray(unit_idx, dtype=int)

                    # Core statistic: correlate same-class population vectors
                    # between trajectory A and trajectory B at the crossing point.
                    corr = safe_pearson_corr(states[t, a, unit_idx], states[t, b, unit_idx]) if unit_idx.size else float("nan")
                    rows.append({**base, "signal": signal, "n_units": int(unit_idx.size), "correlation": corr})
                if random_candidates.size:
                    random_corrs = []
                    for _ in range(max(1, int(args.matched_random_repeats))):
                        n = min(pred_count, random_candidates.size)
                        draw = rng.choice(random_candidates, size=n, replace=False)
                        random_corrs.append(safe_pearson_corr(states[t, a, draw], states[t, b, draw]))

                    # Store the average over repeated random draws as one
                    # random-control value for this trajectory pair.
                    rows.append(
                        {
                            **base,
                            "signal": "random_matched_count",
                            "n_units": int(min(pred_count, random_candidates.size)),
                            "correlation": float(np.nanmean(random_corrs)),
                        }
                    )
            pair_offset += int(batch.pair_ids.max()) + 1
            if (batch_idx + 1) % max(1, int(args.log_every_batches)) == 0:
                log_info(
                    "[crossing] batch=%d/%d pairs=%d elapsed=%s",
                    batch_idx + 1,
                    int(args.n_batches),
                    pair_offset,
                    format_duration(time.time() - start_time),
                )

    # Compute per-signal means and the paired GC-minus-PGC comparison.
    summary_rows, comparison = summarize_corr_rows(rows, rng)

    # Pair-level CSV is the most detailed output; summary CSV/JSON are the
    # convenient reporting layer.
    write_csv(out_dir / "crossing_pair_metrics.csv", rows)
    write_csv(out_dir / "crossing_summary.csv", summary_rows)
    payload = {
        "checkpoint_path": str(args.checkpoint_path),
        "gridness_path": str(args.gridness_path),
        "n_pairs": int(pair_offset),
        "class_counts": {k: int(v.size) for k, v in units.items()},
        "comparison": comparison,
        "summary_rows": summary_rows,
        "pair_metrics_path": str(out_dir / "crossing_pair_metrics.csv"),
        "summary_path": str(out_dir / "crossing_summary.csv"),
    }
    write_json(out_dir / "crossing_metrics.json", payload)
    write_json(out_dir / "summary.json", payload)
    log_info(
        "[crossing] mean standard-minus-predictive corr=%.4f ci95=[%.4f, %.4f] pairs=%d",
        float(comparison.get("mean_standard_minus_predictive_corr", np.nan)),
        float(comparison.get("ci95_low", np.nan)),
        float(comparison.get("ci95_high", np.nan)),
        int(comparison.get("n_matched_pairs", 0)),
    )
    log_info("[crossing] wrote %s", out_dir)
    return out_dir


def _group_unit_indices_for_population_corr(units: Dict[str, np.ndarray], Ng: int) -> Tuple[Dict[str, np.ndarray], np.ndarray, int]:
    groups: Dict[str, np.ndarray] = {
        "predictive": np.asarray(units["predictive"], dtype=int),
        "standard_grid": np.asarray(units["standard"], dtype=int),
        "retrospective": np.asarray(units["retrospective"], dtype=int),
        "band": np.asarray(units["band"], dtype=int),
    }
    pred_count = max(1, int(groups["predictive"].size))
    if groups["standard_grid"].size:
        groups["standard_grid_matched_count"] = groups["standard_grid"][: min(pred_count, groups["standard_grid"].size)]
    random_candidates = np.setdiff1d(np.arange(int(Ng), dtype=int), groups["predictive"])
    return groups, random_candidates, pred_count


def movement_angle_sep_deg(v_a: np.ndarray, v_b: np.ndarray) -> float:
    v_a = np.asarray(v_a, dtype=float).reshape(-1)
    v_b = np.asarray(v_b, dtype=float).reshape(-1)
    norm = float(np.linalg.norm(v_a) * np.linalg.norm(v_b))
    if norm <= 1e-12 or not np.isfinite(norm):
        return float("nan")
    cosang = float(np.clip(np.dot(v_a, v_b) / norm, -1.0, 1.0))
    return float(np.rad2deg(np.arccos(cosang)))


def summarize_same_bin_rows(
    rows: Sequence[Dict[str, object]],
    res: int,
    movement_bins: int,
    rng: np.random.Generator,
) -> Tuple[List[Dict[str, object]], Dict[str, np.ndarray], List[Dict[str, object]]]:
    summary_rows, _ = summarize_corr_rows(rows, rng)
    heatmaps: Dict[str, np.ndarray] = {}
    for signal in sorted({str(row.get("signal")) for row in rows}):
        arr = np.full((int(res), int(res)), np.nan, dtype=np.float32)
        for i in range(int(res)):
            for j in range(int(res)):
                vals = [
                    float(row.get("correlation", np.nan))
                    for row in rows
                    if row.get("signal") == signal and int(row.get("bin_x", -1)) == i and int(row.get("bin_y", -1)) == j
                ]
                vals = np.asarray(vals, dtype=float)
                vals = vals[np.isfinite(vals)]
                if vals.size:
                    arr[i, j] = float(np.nanmedian(vals))
        heatmaps[signal] = arr

    std_by_pair = {
        int(row["pair_id"]): float(row["correlation"])
        for row in rows
        if row.get("signal") == "standard_grid" and np.isfinite(float(row.get("correlation", np.nan)))
    }
    pred_by_pair = {
        int(row["pair_id"]): float(row["correlation"])
        for row in rows
        if row.get("signal") == "predictive" and np.isfinite(float(row.get("correlation", np.nan)))
    }
    movement_by_pair = {
        int(row["pair_id"]): float(row.get("movement_vector_distance", np.nan))
        for row in rows
        if row.get("signal") == "standard_grid"
    }
    common = sorted(set(std_by_pair) & set(pred_by_pair) & set(movement_by_pair))
    movement = np.asarray([movement_by_pair[p] for p in common], dtype=float)
    std_vals = np.asarray([std_by_pair[p] for p in common], dtype=float)
    pred_vals = np.asarray([pred_by_pair[p] for p in common], dtype=float)
    finite = np.isfinite(movement) & np.isfinite(std_vals) & np.isfinite(pred_vals)
    movement = movement[finite]
    std_vals = std_vals[finite]
    pred_vals = pred_vals[finite]

    binned_rows: List[Dict[str, object]] = []
    if movement.size:
        percentiles = np.linspace(0, 100, int(movement_bins) + 1)
        edges = np.nanpercentile(movement, percentiles)
        edges = np.unique(edges)
        if edges.size < 2:
            edges = np.asarray([float(np.nanmin(movement)) - 1e-9, float(np.nanmax(movement)) + 1e-9])
        for b in range(edges.size - 1):
            lo, hi = float(edges[b]), float(edges[b + 1])
            if b == edges.size - 2:
                mask = (movement >= lo) & (movement <= hi)
            else:
                mask = (movement >= lo) & (movement < hi)
            if not mask.any():
                continue
            binned_rows.append(
                {
                    "bin": int(b),
                    "movement_distance_low": lo,
                    "movement_distance_high": hi,
                    "movement_distance_mid": float(0.5 * (lo + hi)),
                    "n_pairs": int(np.sum(mask)),
                    "median_standard_grid_corr": float(np.nanmedian(std_vals[mask])),
                    "q25_standard_grid_corr": float(np.nanpercentile(std_vals[mask], 25)),
                    "q75_standard_grid_corr": float(np.nanpercentile(std_vals[mask], 75)),
                    "median_predictive_corr": float(np.nanmedian(pred_vals[mask])),
                    "q25_predictive_corr": float(np.nanpercentile(pred_vals[mask], 25)),
                    "q75_predictive_corr": float(np.nanpercentile(pred_vals[mask], 75)),
                    "median_standard_minus_predictive_corr": float(np.nanmedian(std_vals[mask] - pred_vals[mask])),
                }
            )
    return summary_rows, heatmaps, binned_rows


def plot_same_bin_population_corr(
    out_dir: Path,
    rows: Sequence[Dict[str, object]],
    heatmaps: Dict[str, np.ndarray],
    binned_rows: Sequence[Dict[str, object]],
    title: Optional[str],
    dpi: int,
) -> Dict[str, Optional[str]]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        raise RuntimeError("same-bin population correlation plotting needs matplotlib installed.") from exc

    try:
        plt.style.use("seaborn-v0_8-whitegrid")
    except Exception:
        pass

    paths: Dict[str, Optional[str]] = {
        "distribution_plot": None,
        "heatmap_plot": None,
        "movement_scatter_plot": None,
        "movement_binned_plot": None,
    }
    std_vals = np.asarray(
        [float(row.get("correlation", np.nan)) for row in rows if row.get("signal") == "standard_grid"],
        dtype=float,
    )
    pred_vals = np.asarray(
        [float(row.get("correlation", np.nan)) for row in rows if row.get("signal") == "predictive"],
        dtype=float,
    )
    std_vals = std_vals[np.isfinite(std_vals)]
    pred_vals = pred_vals[np.isfinite(pred_vals)]
    colors = {"standard_grid": "#2a9d8f", "predictive": "#e76f51"}
    if std_vals.size and pred_vals.size:
        combined = np.concatenate([std_vals, pred_vals])
        bins = np.linspace(float(np.nanmin(combined)), float(np.nanmax(combined)), 40)
        if np.allclose(bins[0], bins[-1]):
            bins = np.linspace(bins[0] - 0.05, bins[0] + 0.05, 40)
        fig, ax = plt.subplots(figsize=(6.4, 4.4))
        ax.hist(std_vals, bins=bins, density=True, alpha=0.45, color=colors["standard_grid"], label=f"Grid median={np.nanmedian(std_vals):.3f}")
        ax.hist(pred_vals, bins=bins, density=True, alpha=0.45, color=colors["predictive"], label=f"Predictive median={np.nanmedian(pred_vals):.3f}")
        ax.set_xlabel("Population correlation")
        ax.set_ylabel("Density")
        ax.set_title(title or "Same-bin population correlation")
        ax.legend(frameon=True)
        fig.tight_layout()
        path = out_dir / "same_bin_gc_pgc_correlation_distributions.png"
        fig.savefig(path, dpi=int(dpi))
        plt.close(fig)
        paths["distribution_plot"] = str(path)

    if "standard_grid" in heatmaps and "predictive" in heatmaps:
        fig, axes = plt.subplots(1, 2, figsize=(8.5, 4.0))
        for ax, signal, label in zip(axes, ["standard_grid", "predictive"], ["Grid units", "Predictive grid units"]):
            im = ax.imshow(heatmaps[signal].T, origin="lower", vmin=0.5, vmax=1.0, cmap="viridis")
            ax.set_title(label)
            ax.set_xlabel("x bin")
            ax.set_ylabel("y bin")
        fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.035, pad=0.04, label="Median correlation")
        fig.suptitle(title or "Same-bin median population correlation")
        path = out_dir / "same_bin_spatial_median_heatmaps.png"
        fig.savefig(path, dpi=int(dpi), bbox_inches="tight")
        plt.close(fig)
        paths["heatmap_plot"] = str(path)

    std_by_pair = {
        int(row["pair_id"]): float(row["correlation"])
        for row in rows
        if row.get("signal") == "standard_grid" and np.isfinite(float(row.get("correlation", np.nan)))
    }
    pred_by_pair = {
        int(row["pair_id"]): float(row["correlation"])
        for row in rows
        if row.get("signal") == "predictive" and np.isfinite(float(row.get("correlation", np.nan)))
    }
    movement_by_pair = {
        int(row["pair_id"]): float(row.get("movement_vector_distance", np.nan))
        for row in rows
        if row.get("signal") == "standard_grid"
    }
    common = sorted(set(std_by_pair) & set(pred_by_pair) & set(movement_by_pair))
    movement = np.asarray([movement_by_pair[p] for p in common], dtype=float)
    std = np.asarray([std_by_pair[p] for p in common], dtype=float)
    pred = np.asarray([pred_by_pair[p] for p in common], dtype=float)
    finite = np.isfinite(movement) & np.isfinite(std) & np.isfinite(pred)
    if finite.any():
        fig, axes = plt.subplots(1, 2, figsize=(8.5, 4.0))
        axes[0].plot(movement[finite], std[finite], "o", color="#222222", alpha=0.25, markersize=2.5)
        axes[0].set_title("Grid units")
        axes[1].plot(movement[finite], pred[finite], "o", color="#222222", alpha=0.25, markersize=2.5)
        axes[1].set_title("Predictive grid units")
        for ax in axes:
            ax.set_xlabel("Movement vector distance")
            ax.set_ylabel("Population correlation")
        fig.suptitle(title or "Movement distance vs population correlation")
        fig.tight_layout()
        path = out_dir / "same_bin_corr_vs_movement_distance.png"
        fig.savefig(path, dpi=int(dpi))
        plt.close(fig)
        paths["movement_scatter_plot"] = str(path)

    if binned_rows:
        x = np.asarray([float(row["movement_distance_mid"]) for row in binned_rows], dtype=float)
        std_med = np.asarray([float(row["median_standard_grid_corr"]) for row in binned_rows], dtype=float)
        std_q25 = np.asarray([float(row["q25_standard_grid_corr"]) for row in binned_rows], dtype=float)
        std_q75 = np.asarray([float(row["q75_standard_grid_corr"]) for row in binned_rows], dtype=float)
        pred_med = np.asarray([float(row["median_predictive_corr"]) for row in binned_rows], dtype=float)
        pred_q25 = np.asarray([float(row["q25_predictive_corr"]) for row in binned_rows], dtype=float)
        pred_q75 = np.asarray([float(row["q75_predictive_corr"]) for row in binned_rows], dtype=float)
        fig, ax = plt.subplots(figsize=(6.4, 4.3))
        ax.errorbar(x, std_med, yerr=np.vstack([std_med - std_q25, std_q75 - std_med]), fmt="o-", color=colors["standard_grid"], label="Grid")
        ax.errorbar(x, pred_med, yerr=np.vstack([pred_med - pred_q25, pred_q75 - pred_med]), fmt="o-", color=colors["predictive"], label="Predictive grid")
        ax.set_xlabel("Movement vector distance")
        ax.set_ylabel("Population correlation")
        ax.set_title(title or "Same-bin correlation by movement distance")
        ax.legend(frameon=True)
        fig.tight_layout()
        path = out_dir / "same_bin_corr_vs_movement_distance_binned.png"
        fig.savefig(path, dpi=int(dpi))
        plt.close(fig)
        paths["movement_binned_plot"] = str(path)
    return paths


def run_same_bin_population_corr(args) -> Path:
    rng = set_seed(int(args.random_seed))
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = ensure_dir(args.output_dir)
    configure_logging(out_dir, "crossing.log")
    log_info("[same_bin] checkpoint=%s", args.checkpoint_path)
    log_info("[same_bin] gridness=%s", args.gridness_path)

    # same_bin is the pasted-code-style analysis path. By default it samples
    # ordinary open-field trajectories and pairs trials that land in the same
    # spatial bin. The x_crossing source keeps the pasted-style outputs, but
    # replaces those random same-bin pairs with our exact controlled X pairs.
    trajectory_source = str(getattr(args, "same_bin_trajectory_source", "random_walk"))
    timestep_arg = getattr(args, "same_bin_timestep", None)
    if timestep_arg is None:
        timestep_arg = getattr(args, "crossing_step", None)
    min_sequence_length = max(2, int(args.sequence_length or 0), int(timestep_arg if timestep_arg is not None else 0) + 1)
    if trajectory_source == "x_crossing":
        # The X generator needs room after the crossing for the requested future
        # horizon metadata; otherwise it will clip the later future point.
        if timestep_arg is not None:
            min_sequence_length = max(min_sequence_length, int(timestep_arg) + int(args.future_horizon) + 2)
        else:
            min_sequence_length = max(min_sequence_length, int(args.future_horizon) * 3 + 2)
    model, place_cells, options, payload = load_analysis_model(
        args.checkpoint_path,
        device=device,
        sequence_length=min_sequence_length if min_sequence_length > 2 else None,
        batch_size=args.batch_size,
    )
    options.batch_size = int(args.batch_size)
    options.sequence_length = int(max(min_sequence_length, int(options.sequence_length)))
    options.device = device
    timestep = int(timestep_arg if timestep_arg is not None else options.sequence_length // 2)
    timestep = int(np.clip(timestep, 0, int(options.sequence_length) - 1))

    grid_data = load_gridness_payload(args.gridness_path)
    units = class_arrays(grid_data, int(options.Ng), fallback=bool(args.allow_fallback_units))
    groups, random_candidates, pred_count = _group_unit_indices_for_population_corr(units, int(options.Ng))
    log_info(
        "[same_bin] device=%s checkpoint_format=%s class_counts=%s source=%s timestep=%d res=%d pairs_per_bin=%d",
        device,
        payload.get("format", "?"),
        {k: int(v.size) for k, v in units.items()},
        trajectory_source,
        timestep,
        int(args.same_bin_res),
        int(args.same_bin_pairs_per_bin),
    )

    traj_gen: Optional[TrajectoryGenerator] = None
    if trajectory_source == "random_walk":
        traj_options = make_options(**options_to_dict(options))
        traj_options.batch_size = int(args.batch_size)
        traj_options.sequence_length = int(options.sequence_length)
        traj_options.device = device
        traj_gen = TrajectoryGenerator(traj_options, place_cells)
    rows: List[Dict[str, object]] = []
    pair_offset = 0
    res = int(args.same_bin_res)
    border_bins = int(np.ceil(float(args.same_bin_border_buffer) / (float(options.box_width) / float(res))))
    valid_start = max(0, border_bins)
    valid_stop = min(res, res - border_bins)
    start_time = time.time()
    model.eval()
    with torch.no_grad():
        for batch_idx in range(max(1, int(args.n_batches))):
            if trajectory_source == "x_crossing":
                # Exact controlled-X source: every stored pair is already
                # position-matched at timestep, with angle controlled by the
                # same min/max angle flags used by the original crossing mode.
                batch = generate_crossing_batch(
                    options,
                    place_cells,
                    batch_size=int(args.batch_size),
                    rng=rng,
                    crossing_step=timestep,
                    min_angle_deg=float(args.min_angle_deg),
                    max_angle_deg=float(args.max_angle_deg),
                    line_extent=float(args.line_extent),
                    future_horizon=int(args.future_horizon),
                )
                assert_crossing_batch(batch, float(args.min_angle_deg))
                states = model.g(batch.inputs).detach().cpu().numpy()
                pos_np = batch.positions_np
                vel_np = batch.velocity_np
            else:
                assert traj_gen is not None
                inputs2, pos_t, _ = traj_gen.get_test_batch(batch_size=int(args.batch_size))
                v2, init = inputs2
                v = pad_zero_cue(v2.to(device), int(getattr(options, "cue_dim", 0)))
                states = model.g((v, init.to(device))).detach().cpu().numpy()
                pos_np = pos_t.detach().cpu().numpy().astype(np.float32)
                vel_np = v2.detach().cpu().numpy().astype(np.float32)
            p = pos_np[timestep]
            g = states[timestep]
            mov = vel_np[timestep]
            binned_pos = np.floor(((p + float(options.box_width) / 2.0) / float(options.box_width)) * float(res)).astype(int)
            binned_pos = np.clip(binned_pos, 0, res - 1)

            if trajectory_source == "x_crossing":
                for local_pid in np.unique(batch.pair_ids):
                    idx = np.where(batch.pair_ids == local_pid)[0]
                    if idx.size != 2:
                        continue
                    a, b = int(idx[0]), int(idx[1])
                    i, j = int(binned_pos[a, 0]), int(binned_pos[a, 1])
                    if not (valid_start <= i < valid_stop and valid_start <= j < valid_stop):
                        continue
                    pair_id = pair_offset
                    pair_offset += 1
                    fut_t = min(timestep + int(args.future_horizon), pos_np.shape[0] - 1)
                    movement_distance = float(np.linalg.norm(mov[a] - mov[b]))
                    pos_distance_cm = float(np.linalg.norm(p[a] - p[b]) * 100.0)
                    base = {
                        "pair_id": int(pair_id),
                        "batch": int(batch_idx),
                        "pairing_mode": "same_bin",
                        "trajectory_source": trajectory_source,
                        "crossing_step": int(timestep),
                        "timestep": int(timestep),
                        "bin_x": int(i),
                        "bin_y": int(j),
                        "position_distance_cm": pos_distance_cm,
                        "movement_vector_distance": movement_distance,
                        "movement_angle_sep_deg": movement_angle_sep_deg(mov[a], mov[b]),
                        "angle_sep_deg": float(batch.angle_sep_deg[int(local_pid)]),
                        "future_position_sep_cm": float(np.linalg.norm(pos_np[fut_t, a] - pos_np[fut_t, b]) * 100.0),
                    }
                    for signal, unit_idx in groups.items():
                        unit_idx = np.asarray(unit_idx, dtype=int)
                        corr = safe_pearson_corr(g[a, unit_idx], g[b, unit_idx]) if unit_idx.size else float("nan")
                        rows.append({**base, "signal": signal, "n_units": int(unit_idx.size), "correlation": corr})
                    if random_candidates.size:
                        random_corrs = []
                        for _rep in range(max(1, int(args.matched_random_repeats))):
                            n = min(pred_count, random_candidates.size)
                            draw = rng.choice(random_candidates, size=n, replace=False)
                            random_corrs.append(safe_pearson_corr(g[a, draw], g[b, draw]))
                        rows.append(
                            {
                                **base,
                                "signal": "random_matched_count",
                                "n_units": int(min(pred_count, random_candidates.size)),
                                "correlation": float(np.nanmean(random_corrs)),
                            }
                        )
            else:
                for i in range(valid_start, valid_stop):
                    for j in range(valid_start, valid_stop):
                        ids = np.where((binned_pos[:, 0] == i) & (binned_pos[:, 1] == j))[0]
                        if ids.size < 2:
                            continue
                        n_draws = min(int(args.same_bin_pairs_per_bin), int(ids.size * (ids.size - 1) // 2))
                        for _ in range(n_draws):
                            a, b = rng.choice(ids, size=2, replace=False)
                            pair_id = pair_offset
                            pair_offset += 1
                            movement_distance = float(np.linalg.norm(mov[a] - mov[b]))
                            pos_distance_cm = float(np.linalg.norm(p[a] - p[b]) * 100.0)
                            base = {
                                "pair_id": int(pair_id),
                                "batch": int(batch_idx),
                                "pairing_mode": "same_bin",
                                "trajectory_source": trajectory_source,
                                "crossing_step": int(timestep),
                                "timestep": int(timestep),
                                "bin_x": int(i),
                                "bin_y": int(j),
                                "position_distance_cm": pos_distance_cm,
                                "movement_vector_distance": movement_distance,
                                "movement_angle_sep_deg": movement_angle_sep_deg(mov[a], mov[b]),
                            }
                            for signal, unit_idx in groups.items():
                                unit_idx = np.asarray(unit_idx, dtype=int)
                                corr = safe_pearson_corr(g[a, unit_idx], g[b, unit_idx]) if unit_idx.size else float("nan")
                                rows.append({**base, "signal": signal, "n_units": int(unit_idx.size), "correlation": corr})
                            if random_candidates.size:
                                random_corrs = []
                                for _rep in range(max(1, int(args.matched_random_repeats))):
                                    n = min(pred_count, random_candidates.size)
                                    draw = rng.choice(random_candidates, size=n, replace=False)
                                    random_corrs.append(safe_pearson_corr(g[a, draw], g[b, draw]))
                                rows.append(
                                    {
                                        **base,
                                        "signal": "random_matched_count",
                                        "n_units": int(min(pred_count, random_candidates.size)),
                                        "correlation": float(np.nanmean(random_corrs)),
                                    }
                                )
            if (batch_idx + 1) % max(1, int(args.log_every_batches)) == 0:
                log_info(
                    "[same_bin] batch=%d/%d sampled_pairs=%d rows=%d elapsed=%s",
                    batch_idx + 1,
                    int(args.n_batches),
                    pair_offset,
                    len(rows),
                    format_duration(time.time() - start_time),
                )

    summary_rows, comparison = summarize_corr_rows(rows, rng)
    same_bin_summary_rows, heatmaps, movement_binned_rows = summarize_same_bin_rows(
        rows,
        res=res,
        movement_bins=int(args.same_bin_movement_bins),
        rng=rng,
    )
    if same_bin_summary_rows:
        summary_rows = same_bin_summary_rows
    write_csv(out_dir / "crossing_pair_metrics.csv", rows)
    write_csv(out_dir / "crossing_summary.csv", summary_rows)
    write_csv(out_dir / "same_bin_movement_binned_metrics.csv", movement_binned_rows)
    np.savez_compressed(out_dir / "same_bin_heatmaps.npz", **{f"{k}_median": v for k, v in heatmaps.items()})
    plot_paths: Dict[str, Optional[str]] = {}
    if bool(args.same_bin_plot):
        plot_paths = plot_same_bin_population_corr(
            out_dir,
            rows,
            heatmaps,
            movement_binned_rows,
            title=getattr(args, "title", None),
            dpi=int(getattr(args, "dpi", 200)),
        )
    payload = {
        "pairing_mode": "same_bin",
        "same_bin_trajectory_source": trajectory_source,
        "checkpoint_path": str(args.checkpoint_path),
        "gridness_path": str(args.gridness_path),
        "n_pairs": int(pair_offset),
        "n_rows": int(len(rows)),
        "class_counts": {k: int(v.size) for k, v in units.items()},
        "comparison": comparison,
        "summary_rows": summary_rows,
        "movement_binned_rows": movement_binned_rows,
        "plot_paths": plot_paths,
        "pair_metrics_path": str(out_dir / "crossing_pair_metrics.csv"),
        "summary_path": str(out_dir / "crossing_summary.csv"),
    }
    write_json(out_dir / "crossing_metrics.json", payload)
    write_json(out_dir / "summary.json", payload)
    log_info(
        "[same_bin] mean standard-minus-predictive corr=%.4f ci95=[%.4f, %.4f] pairs=%d",
        float(comparison.get("mean_standard_minus_predictive_corr", np.nan)),
        float(comparison.get("ci95_low", np.nan)),
        float(comparison.get("ci95_high", np.nan)),
        int(comparison.get("n_matched_pairs", 0)),
    )
    log_info("[same_bin] wrote %s", out_dir)
    return out_dir


def _csv_float(row: Dict[str, str], key: str, default: float = float("nan")) -> float:
    try:
        return float(row.get(key, ""))
    except (TypeError, ValueError):
        return default


def _friendly_signal_name(signal: str) -> str:
    names = {
        "predictive": "Predictive",
        "standard_grid": "Zero-lag grid",
        "standard_grid_matched_count": "Zero-lag grid\nmatched n",
        "retrospective": "Retrospective",
        "band": "Band",
        "random_matched_count": "Random\nmatched n",
    }
    return names.get(signal, signal.replace("_", " ").title())


def run_plot_crossing(args) -> Path:
    # Plotting is a pure post-processing step: it reads the CSVs written by
    # run_crossing and does not rerun the model.
    crossing_dir = Path(args.crossing_dir)
    out_dir = ensure_dir(Path(args.output_dir) if args.output_dir else crossing_dir)
    configure_logging(out_dir, "plot_crossing.log")

    # Detailed pair-level rows contain one correlation per pair per signal.
    # Summary rows contain means/CIs per signal.
    pair_path = crossing_dir / "crossing_pair_metrics.csv"
    summary_path = crossing_dir / "crossing_summary.csv"
    if not pair_path.exists():
        raise FileNotFoundError(f"Missing crossing pair metrics: {pair_path}")
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing crossing summary: {summary_path}")

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        raise RuntimeError("plot_crossing needs matplotlib installed in the active environment.") from exc

    pair_rows = read_csv_rows(pair_path)
    summary_rows = read_csv_rows(summary_path)

    # Keep the visual order stable across runs, even if some control groups are
    # absent because a classification produced too few units.
    signal_order = [
        "standard_grid",
        "standard_grid_matched_count",
        "predictive",
        "retrospective",
        "band",
        "random_matched_count",
    ]
    present_signals = {str(row.get("signal")) for row in summary_rows}
    ordered_signals = [s for s in signal_order if s in present_signals]
    ordered_signals.extend(sorted(present_signals - set(ordered_signals)))

    try:
        plt.style.use("seaborn-v0_8-whitegrid")
    except Exception:
        pass

    # Build the bar plot of mean within-population crossing correlations.
    summary_by_signal = {str(row.get("signal")): row for row in summary_rows}
    means = np.asarray([_csv_float(summary_by_signal[s], "mean_correlation") for s in ordered_signals], dtype=float)
    ci_low = np.asarray([_csv_float(summary_by_signal[s], "ci95_low") for s in ordered_signals], dtype=float)
    ci_high = np.asarray([_csv_float(summary_by_signal[s], "ci95_high") for s in ordered_signals], dtype=float)
    lower_err = np.maximum(0.0, means - ci_low)
    upper_err = np.maximum(0.0, ci_high - means)
    colors = {
        "standard_grid": "#2a9d8f",
        "standard_grid_matched_count": "#76c7ba",
        "predictive": "#e76f51",
        "retrospective": "#8d99ae",
        "band": "#577590",
        "random_matched_count": "#b8b8b8",
    }

    fig, ax = plt.subplots(figsize=(9.0, 4.8))
    x = np.arange(len(ordered_signals))
    ax.bar(x, means, color=[colors.get(s, "#6c757d") for s in ordered_signals], edgecolor="black", linewidth=0.7)
    ax.errorbar(x, means, yerr=np.vstack([lower_err, upper_err]), fmt="none", color="black", capsize=4, linewidth=1.0)
    ax.axhline(0.0, color="#333333", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([_friendly_signal_name(s) for s in ordered_signals], rotation=0)
    ax.set_ylabel("Matched crossing correlation")
    ax.set_title(args.title or "X-crossing population similarity")
    ax.margins(x=0.03)
    fig.tight_layout()
    summary_png = out_dir / "crossing_correlation_summary.png"
    fig.savefig(summary_png, dpi=int(args.dpi))
    plt.close(fig)

    # Pull out pair-matched predictive and standard-grid correlations so the
    # distribution and paired-difference plots compare the same trajectory pairs.
    pred_by_pair = {
        int(row["pair_id"]): _csv_float(row, "correlation")
        for row in pair_rows
        if row.get("signal") == "predictive" and np.isfinite(_csv_float(row, "correlation"))
    }
    std_by_pair = {
        int(row["pair_id"]): _csv_float(row, "correlation")
        for row in pair_rows
        if row.get("signal") == "standard_grid" and np.isfinite(_csv_float(row, "correlation"))
    }
    common_pairs = sorted(set(pred_by_pair) & set(std_by_pair))
    pred = np.asarray([pred_by_pair[p] for p in common_pairs], dtype=float)
    std = np.asarray([std_by_pair[p] for p in common_pairs], dtype=float)
    diffs = std - pred
    rng = np.random.default_rng(int(args.random_seed))
    lo, hi = bootstrap_ci(diffs, rng, n_boot=2000)
    mean_diff = float(np.mean(diffs)) if diffs.size else float("nan")

    # New interpretability plot: show the raw GC and PGC correlation
    # distributions directly, instead of only showing their difference.
    distribution_png = None
    diff_png = None
    scatter_png = None
    finite_pred = pred[np.isfinite(pred)]
    finite_std = std[np.isfinite(std)]
    if finite_pred.size and finite_std.size:
        combined = np.concatenate([finite_std, finite_pred])

        # Use shared bins so the two histograms are visually comparable.
        bins = np.linspace(float(np.min(combined)), float(np.max(combined)), max(5, int(args.hist_bins)))
        if np.allclose(bins[0], bins[-1]):
            bins = np.linspace(bins[0] - 0.05, bins[0] + 0.05, max(5, int(args.hist_bins)))

        fig, axes = plt.subplots(
            1,
            2,
            figsize=(10.5, 4.8),
            gridspec_kw={"width_ratios": [1.45, 0.8]},
        )
        ax = axes[0]
        ax.hist(
            finite_std,
            bins=bins,
            density=True,
            alpha=0.52,
            color=colors["standard_grid"],
            edgecolor="white",
            linewidth=0.5,
            label=f"Zero-lag grid mean={np.mean(finite_std):.3f}",
        )
        ax.hist(
            finite_pred,
            bins=bins,
            density=True,
            alpha=0.52,
            color=colors["predictive"],
            edgecolor="white",
            linewidth=0.5,
            label=f"Predictive mean={np.mean(finite_pred):.3f}",
        )
        ax.axvline(float(np.mean(finite_std)), color=colors["standard_grid"], linewidth=1.8)
        ax.axvline(float(np.mean(finite_pred)), color=colors["predictive"], linewidth=1.8)
        ax.set_xlabel("Population-vector correlation at crossing")
        ax.set_ylabel("Density")
        ax.set_title("Raw correlation distributions")
        ax.legend(frameon=True)

        ax = axes[1]
        parts = ax.violinplot([finite_std, finite_pred], showmeans=True, showmedians=True, widths=0.72)
        for body, color in zip(parts["bodies"], [colors["standard_grid"], colors["predictive"]]):
            body.set_facecolor(color)
            body.set_edgecolor("black")
            body.set_alpha(0.65)
            body.set_linewidth(0.7)
        for key in ("cmeans", "cmedians", "cbars", "cmins", "cmaxes"):
            if key in parts:
                parts[key].set_color("#222222")
                parts[key].set_linewidth(1.0)
        ax.set_xticks([1, 2])
        ax.set_xticklabels(["Zero-lag\ngrid", "Predictive"])
        ax.set_ylabel("Correlation")
        ax.set_title("Matched-pair values")
        fig.suptitle(args.title or "X-crossing GC vs PGC similarity", y=1.02)
        fig.tight_layout()
        distribution_png = out_dir / "crossing_gc_pgc_correlation_distributions.png"
        fig.savefig(distribution_png, dpi=int(args.dpi), bbox_inches="tight")
        plt.close(fig)

    # Paired-difference plot: one value per trajectory pair, equal to
    # corr(GC_A, GC_B) - corr(PGC_A, PGC_B).
    if diffs.size:
        fig, ax = plt.subplots(figsize=(7.0, 4.6))
        ax.hist(diffs, bins=int(args.hist_bins), color="#5c677d", alpha=0.82, edgecolor="white")
        ax.axvline(0.0, color="#222222", linewidth=1.0, linestyle="--", label="No difference")
        ax.axvline(mean_diff, color="#e76f51", linewidth=1.8, label=f"Mean = {mean_diff:.4f}")
        if np.isfinite(lo) and np.isfinite(hi):
            ax.axvspan(lo, hi, color="#e76f51", alpha=0.18, label=f"95% CI [{lo:.4f}, {hi:.4f}]")
        ax.set_xlabel("Zero-lag grid corr - predictive corr")
        ax.set_ylabel("Crossing pairs")
        ax.set_title(args.title or "Paired crossing difference")
        ax.legend(frameon=True)
        fig.tight_layout()
        diff_png = out_dir / "crossing_standard_minus_predictive_hist.png"
        fig.savefig(diff_png, dpi=int(args.dpi))
        plt.close(fig)

        # Scatter plot: points above the diagonal are pairs where GC similarity
        # exceeds PGC similarity.
        fig, ax = plt.subplots(figsize=(5.4, 5.2))
        ax.scatter(pred, std, s=8, alpha=0.35, color="#4a5568", linewidths=0)
        finite = np.concatenate([pred[np.isfinite(pred)], std[np.isfinite(std)]])
        if finite.size:
            lo_lim, hi_lim = np.percentile(finite, [1, 99])
            pad = max(0.02, 0.08 * float(hi_lim - lo_lim))
            lo_lim -= pad
            hi_lim += pad
            ax.set_xlim(lo_lim, hi_lim)
            ax.set_ylim(lo_lim, hi_lim)
            ax.plot([lo_lim, hi_lim], [lo_lim, hi_lim], color="#222222", linewidth=1.0, linestyle="--")
        ax.set_xlabel("Predictive correlation")
        ax.set_ylabel("Zero-lag grid correlation")
        ax.set_title("Matched pairs")
        fig.tight_layout()
        scatter_png = out_dir / "crossing_predictive_vs_standard_scatter.png"
        fig.savefig(scatter_png, dpi=int(args.dpi))
        plt.close(fig)

    plot_summary = {
        "crossing_dir": str(crossing_dir),
        "summary_plot": str(summary_png),
        "distribution_plot": str(distribution_png) if distribution_png else None,
        "difference_plot": str(diff_png) if diff_png else None,
        "scatter_plot": str(scatter_png) if scatter_png else None,
        "n_matched_pairs": int(diffs.size),
        "mean_standard_minus_predictive_corr": mean_diff,
        "ci95_low": lo,
        "ci95_high": hi,
        "signals": ordered_signals,
    }
    write_json(out_dir / "crossing_plot_summary.json", plot_summary)
    log_info(
        "[plot_crossing] wrote %s, mean standard-minus-predictive=%.4f ci95=[%.4f, %.4f]",
        out_dir,
        mean_diff,
        lo,
        hi,
    )
    return out_dir


def _plot_crossing_step_sweep_curves(
    curves: Dict[str, Sequence[Dict[str, object]]],
    out_dir: Path,
    title: Optional[str],
    dpi: int,
) -> Optional[Path]:
    """Plot one or more crossing-step sweep curves."""
    if not curves:
        return None
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        raise RuntimeError("crossing-step sweep plotting needs matplotlib installed.") from exc

    try:
        plt.style.use("seaborn-v0_8-whitegrid")
    except Exception:
        pass

    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    palette = ["#2a9d8f", "#e76f51", "#577590", "#8d99ae", "#f4a261", "#6d597a"]
    for i, (label, rows) in enumerate(curves.items()):
        finite_rows = []
        for row in rows:
            step = _csv_float(row, "crossing_step")
            mean = _csv_float(row, "mean_standard_minus_predictive_corr")
            lo = _csv_float(row, "ci95_low")
            hi = _csv_float(row, "ci95_high")
            if np.isfinite(step) and np.isfinite(mean):
                finite_rows.append((int(step), float(mean), float(lo), float(hi)))
        finite_rows.sort(key=lambda x: x[0])
        if not finite_rows:
            continue
        steps = np.asarray([x[0] for x in finite_rows], dtype=float)
        means = np.asarray([x[1] for x in finite_rows], dtype=float)
        lows = np.asarray([x[2] for x in finite_rows], dtype=float)
        highs = np.asarray([x[3] for x in finite_rows], dtype=float)
        color = palette[i % len(palette)]
        ax.plot(steps, means, marker="o", linewidth=2.0, markersize=4.5, color=color, label=label)
        if np.isfinite(lows).all() and np.isfinite(highs).all():
            ax.fill_between(steps, lows, highs, color=color, alpha=0.18, linewidth=0)
    ax.axhline(0.0, color="#222222", linewidth=0.9, linestyle="--")
    ax.set_xlabel("Crossing step")
    ax.set_ylabel("Mean zero-lag GC corr - PGC corr")
    ax.set_title(title or "X-crossing step sweep")
    ax.legend(frameon=True)
    fig.tight_layout()
    plot_path = out_dir / "crossing_step_sweep.png"
    fig.savefig(plot_path, dpi=int(dpi))
    plt.close(fig)
    return plot_path


def run_crossing_step_sweep(args) -> Path:
    steps = _to_int_list(args.crossing_steps)
    if not steps:
        raise ValueError("--crossing_steps must contain at least one step.")
    out_dir = ensure_dir(args.output_dir)
    configure_logging(out_dir, "crossing_step_sweep.log")
    log_info("[crossing_step_sweep] checkpoint=%s", args.checkpoint_path)
    log_info("[crossing_step_sweep] gridness=%s", args.gridness_path)
    log_info("[crossing_step_sweep] steps=%s", " ".join(str(s) for s in steps))

    rows: List[Dict[str, object]] = []
    start_time = time.time()
    for i, step in enumerate(steps):
        step_dir = out_dir / f"crossing_step_{int(step):03d}"
        step_args = argparse.Namespace(**vars(args))
        step_args.output_dir = str(step_dir)
        step_args.crossing_step = int(step)
        step_args.log_every_batches = int(args.log_every_batches)

        # Keep the RNN rollout long enough that the requested crossing_step is
        # not clipped by generate_crossing_batch.
        min_len = max(32, int(step) + int(args.future_horizon) + 2)
        requested_len = getattr(args, "sequence_length", None)
        step_args.sequence_length = max(min_len, int(requested_len)) if requested_len else min_len

        log_info("[crossing_step_sweep] running step=%d (%d/%d)", int(step), i + 1, len(steps))
        run_crossing(step_args)
        metrics = read_json(step_dir / "summary.json")
        comparison = metrics.get("comparison", {})
        summary_by_signal = {
            str(row.get("signal")): row
            for row in metrics.get("summary_rows", [])
            if isinstance(row, dict)
        }

        def signal_mean(signal: str) -> float:
            row = summary_by_signal.get(signal, {})
            return float(row.get("mean_correlation", np.nan)) if row else float("nan")

        rows.append(
            {
                "crossing_step": int(step),
                "n_pairs": int(metrics.get("n_pairs", 0)),
                "mean_standard_minus_predictive_corr": float(
                    comparison.get("mean_standard_minus_predictive_corr", np.nan)
                ),
                "ci95_low": float(comparison.get("ci95_low", np.nan)),
                "ci95_high": float(comparison.get("ci95_high", np.nan)),
                "fraction_standard_gt_predictive": float(
                    comparison.get("fraction_standard_gt_predictive", np.nan)
                ),
                "mean_standard_grid_corr": signal_mean("standard_grid"),
                "mean_predictive_corr": signal_mean("predictive"),
                "mean_band_corr": signal_mean("band"),
                "step_output_dir": str(step_dir),
            }
        )

        # Restore sweep logging after run_crossing temporarily reconfigured the
        # logger to write inside the per-step output directory.
        configure_logging(out_dir, "crossing_step_sweep.log")
        elapsed = format_duration(time.time() - start_time)
        log_info(
            "[crossing_step_sweep] step=%d mean=%.4f ci95=[%.4f, %.4f] elapsed=%s",
            int(step),
            float(rows[-1]["mean_standard_minus_predictive_corr"]),
            float(rows[-1]["ci95_low"]),
            float(rows[-1]["ci95_high"]),
            elapsed,
        )

    write_csv(out_dir / "crossing_step_sweep_metrics.csv", rows)
    plot_path = _plot_crossing_step_sweep_curves(
        {str(args.label or Path(args.checkpoint_path).stem): rows},
        out_dir,
        args.title,
        int(args.dpi),
    )
    payload = {
        "checkpoint_path": str(args.checkpoint_path),
        "gridness_path": str(args.gridness_path),
        "crossing_steps": steps,
        "metrics_path": str(out_dir / "crossing_step_sweep_metrics.csv"),
        "plot_path": str(plot_path) if plot_path else None,
        "rows": rows,
    }
    write_json(out_dir / "crossing_step_sweep_summary.json", payload)
    log_info("[crossing_step_sweep] wrote %s", out_dir)
    return out_dir


def run_plot_crossing_step_sweeps(args) -> Path:
    out_dir = ensure_dir(args.output_dir)
    configure_logging(out_dir, "plot_crossing_step_sweeps.log")
    curves: Dict[str, Sequence[Dict[str, object]]] = {}
    for spec in args.sweep_dir:
        label, path_text = (spec.split("=", 1) if "=" in spec else (Path(spec).name, spec))
        sweep_dir = Path(path_text)
        metrics_path = sweep_dir / "crossing_step_sweep_metrics.csv"
        if not metrics_path.exists():
            raise FileNotFoundError(f"Missing sweep metrics: {metrics_path}")
        curves[label] = read_csv_rows(metrics_path)
        log_info("[plot_crossing_step_sweeps] loaded %s from %s", label, metrics_path)
    plot_path = _plot_crossing_step_sweep_curves(curves, out_dir, args.title, int(args.dpi))
    combined_rows: List[Dict[str, object]] = []
    for label, rows in curves.items():
        for row in rows:
            combined_rows.append({"label": label, **row})
    write_csv(out_dir / "crossing_step_sweep_comparison_metrics.csv", combined_rows)
    write_json(
        out_dir / "crossing_step_sweep_comparison_summary.json",
        {
            "sweep_dirs": list(args.sweep_dir),
            "plot_path": str(plot_path) if plot_path else None,
            "metrics_path": str(out_dir / "crossing_step_sweep_comparison_metrics.csv"),
        },
    )
    log_info("[plot_crossing_step_sweeps] wrote %s", out_dir)
    return out_dir


def run_decode(args) -> Path:
    rng = set_seed(int(args.random_seed))
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = ensure_dir(args.output_dir)
    configure_logging(out_dir, "decode.log")
    log_info("[decode] checkpoint=%s", args.checkpoint_path)
    log_info("[decode] gridness=%s", args.gridness_path)
    model, place_cells, options, _ = load_future_model(args.checkpoint_path, device=device)
    if getattr(args, "task", None):
        options.task = str(args.task)
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
    log_info(
        "[decode] collecting matched %s activity batches=%d batch_size=%d",
        str(getattr(options, "task", "binary_fork")),
        int(args.n_batches),
        int(args.batch_size),
    )
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
    route_rows: List[Dict[str, object]] = []
    horizon_rows: List[Dict[str, object]] = []
    best_lag = preferred_lag_steps(grid_data, int(options.Ng))
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
        base_nuisance = np.concatenate([dataset["X_kinematics"], dataset["X_standard"]], axis=1)
        nuisance = base_nuisance
        if bool(getattr(args, "route_controls", False)):
            nuisance = np.concatenate([base_nuisance, dataset["X_route_id"]], axis=1)
        X_pred_resid = residualize(dataset["X_predictive"], nuisance, train_mask, float(args.ridge_alpha))
        groups = {
            "kinematics": dataset["X_kinematics"],
            "predictive": dataset["X_predictive"],
            "standard_grid": dataset["X_standard"],
            "standard_grid_plus_kinematics": base_nuisance,
            "non_grid": dataset["X_low_grid"],
            "band": dataset["X_band"],
            "predictive_residualized": X_pred_resid,
        }
        if bool(getattr(args, "route_controls", False)):
            groups.update(
                {
                    "route_id_only": dataset["X_route_id"],
                    "route_id_plus_kinematics": np.concatenate([dataset["X_route_id"], dataset["X_kinematics"]], axis=1),
                    "standard_grid_plus_kinematics_plus_route_id": nuisance,
                }
            )
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
            if name in {"route_id_only", "route_id_plus_kinematics", "standard_grid_plus_kinematics_plus_route_id", "predictive_residualized"}:
                route_rows.append(row.copy())

        if bool(getattr(args, "horizon_specificity", False)):
            for pref_h in horizons:
                pref_units = predictive_units_for_preferred_horizon(units, best_lag, int(pref_h), min_units=2)
                X_pref = unit_features_for_horizon(states, options, horizon, pref_units)
                hrow = evaluate_decoder_group(
                    f"predictive_pref_h{int(pref_h)}",
                    X_pref,
                    dataset["y_route"],
                    dataset["y_pos"],
                    dataset["y_phase"],
                    train_mask,
                    test_mask,
                    float(args.ridge_alpha),
                )
                hrow["horizon"] = int(horizon)
                hrow["preferred_horizon"] = int(pref_h)
                hrow["n_units"] = int(pref_units.size)
                horizon_rows.append(hrow)
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
        "task": str(getattr(options, "task", "binary_fork")),
        "torus_projector_status": projector.status,
        "class_counts": {k: int(v.size) for k, v in units.items()},
        "n_trials": int(states.shape[1]),
        "horizons": horizons,
        "predictive_peak_route_horizon": peak_route,
        "route_controls": bool(getattr(args, "route_controls", False)),
        "horizon_specificity": bool(getattr(args, "horizon_specificity", False)),
        "rows": rows,
        "horizon_specificity_rows": horizon_rows,
    }
    write_json(out_dir / "decode_metrics.json", summary)
    write_csv(out_dir / "decode_metrics.csv", rows)
    write_csv(out_dir / "route_control_metrics.csv", route_rows)
    write_csv(out_dir / "horizon_specificity.csv", horizon_rows)
    write_json(
        out_dir / "summary.json",
        {
            "task": str(getattr(options, "task", "binary_fork")),
            "class_counts": {k: int(v.size) for k, v in units.items()},
            "horizons": horizons,
            "predictive_peak_route_horizon": peak_route,
            "route_control_rows": len(route_rows),
            "horizon_specificity_rows": len(horizon_rows),
        },
    )
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


@dataclass
class PosthocFutureDecoders:
    scaler: StandardScaler
    pos_reg: Ridge
    phase_reg: Ridge
    route_clf: Optional[LogisticRegression]
    horizon: int


def fit_posthoc_future_decoders(
    model: FutureSplitRNN,
    options: SimpleNamespace,
    place_cells: PlaceCells,
    projector: PhaseProjector,
    rng: np.random.Generator,
    horizon: int,
    n_batches: int,
    batch_size: int,
    ridge_alpha: float,
) -> PosthocFutureDecoders:
    states, pos, _vel, routes, pair_ids, _ = collect_fork_activity(
        model,
        options,
        place_cells,
        n_batches=max(1, int(n_batches)),
        batch_size=int(batch_size),
        rng=rng,
        future_horizon=int(horizon),
    )
    phase = phase_features_from_states(states, projector)
    times = matched_sample_times(options, int(horizon))
    if times.size == 0:
        times = np.arange(task_cue_steps(options), task_branch_step(options), dtype=int)
    X_rows, y_pos, y_phase, y_route = [], [], [], []
    for t in times:
        fut_t = min(int(t + horizon), states.shape[0] - 1)
        X_rows.append(states[int(t)])
        y_pos.append(pos[fut_t])
        y_phase.append(phase[fut_t])
        y_route.append(routes.astype(int))
    X = np.concatenate(X_rows, axis=0).astype(np.float32)
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    pos_reg = Ridge(alpha=float(ridge_alpha)).fit(Xs, np.concatenate(y_pos, axis=0))
    phase_reg = Ridge(alpha=float(ridge_alpha)).fit(Xs, np.concatenate(y_phase, axis=0))
    route_y = np.concatenate(y_route, axis=0)
    route_clf: Optional[LogisticRegression] = None
    if np.unique(route_y).size >= 2:
        solver = "liblinear" if np.unique(route_y).size == 2 else "lbfgs"
        route_clf = LogisticRegression(max_iter=1000, solver=solver).fit(Xs, route_y)
    return PosthocFutureDecoders(scaler=scaler, pos_reg=pos_reg, phase_reg=phase_reg, route_clf=route_clf, horizon=int(horizon))


def apply_posthoc_future_decoders(decoders: PosthocFutureDecoders, states: np.ndarray) -> Dict[str, np.ndarray]:
    T, B, Ng = states.shape
    X = states.reshape(T * B, Ng).astype(np.float32)
    Xs = decoders.scaler.transform(X)
    future_pos = decoders.pos_reg.predict(Xs).reshape(T, B, 2).astype(np.float32)
    future_phase = decoders.phase_reg.predict(Xs).reshape(T, B, 4).astype(np.float32)
    route_pred = None
    route_prob = None
    if decoders.route_clf is not None:
        prob = decoders.route_clf.predict_proba(Xs)
        route_prob = prob.reshape(T, B, prob.shape[1]).astype(np.float32)
        route_pred = decoders.route_clf.classes_[np.argmax(prob, axis=1)].reshape(T, B)
    return {"future_pos": future_pos, "future_phase": future_phase, "route_pred": route_pred, "route_prob": route_prob}


def donor_indices_from_pairs(pair_ids: np.ndarray) -> np.ndarray:
    donor = np.arange(pair_ids.size, dtype=int)
    for pid in np.unique(pair_ids):
        idx = np.where(pair_ids == pid)[0]
        if idx.size == 2:
            donor[idx[0]] = idx[1]
            donor[idx[1]] = idx[0]
    return donor


def intervention_metrics(
    name: str,
    baseline: Dict[str, object],
    condition: Dict[str, object],
    batch: ForkBatch,
    options: SimpleNamespace,
    horizon: int,
) -> Dict[str, object]:
    times = matched_sample_times(options, horizon)
    if times.size == 0:
        times = np.arange(int(task_cue_steps(options)), int(task_branch_step(options)), dtype=int)
    future_pred = condition["future_pos"]
    future_base = baseline["future_pos"]
    donor_idx = donor_indices_from_pairs(batch.pair_ids)
    fut_times = np.minimum(times + horizon, int(options.sequence_length) - 1)
    own_future = batch.positions_np[fut_times]
    donor_future = batch.positions_np[fut_times][:, donor_idx]
    route_target = batch.route_ids.astype(int)

    cond_samples = future_pred[times].reshape(-1, 2)
    base_samples = future_base[times].reshape(-1, 2)
    target_samples = own_future.reshape(-1, 2)
    donor_samples = donor_future.reshape(-1, 2)
    route_repeated = np.tile(route_target, times.size)

    rmse = float(np.sqrt(np.mean(np.sum((cond_samples - target_samples) ** 2, axis=1))) * 100.0)
    route_pred = condition.get("route_pred")
    route_acc = float("nan")
    if route_pred is not None:
        route_acc = float(np.mean(route_pred[times].reshape(-1).astype(int) == route_repeated))
    base_pos_margin = np.mean(np.linalg.norm(base_samples - donor_samples, axis=1) - np.linalg.norm(base_samples - target_samples, axis=1))
    cond_pos_margin = np.mean(np.linalg.norm(cond_samples - donor_samples, axis=1) - np.linalg.norm(cond_samples - target_samples, axis=1))
    donor_pull_cm = float((base_pos_margin - cond_pos_margin) * 100.0)

    own_phase = baseline["actual_phase"][fut_times].reshape(-1, 4)
    donor_phase = baseline["actual_phase"][fut_times][:, donor_idx].reshape(-1, 4)
    baseline_phase_here = baseline["future_phase"][times].reshape(-1, 4)
    cond_phase = condition["future_phase"][times].reshape(-1, 4)
    base_margin = np.mean(phase_feature_distance(baseline_phase_here, donor_phase) - phase_feature_distance(baseline_phase_here, own_phase))
    cond_margin = np.mean(phase_feature_distance(cond_phase, donor_phase) - phase_feature_distance(cond_phase, own_phase))
    donor_phase_pull_rad = float(base_margin - cond_margin)

    branch_eval_t = min(int(task_branch_step(options)) + int(horizon), int(options.sequence_length) - 1)
    current_cond = condition["current_pos"][branch_eval_t]
    current_base = baseline["current_pos"][branch_eval_t]
    own_downstream = batch.positions_np[branch_eval_t]
    donor_downstream = batch.positions_np[branch_eval_t, donor_idx]
    base_downstream_margin = np.mean(np.linalg.norm(current_base - donor_downstream, axis=1) - np.linalg.norm(current_base - own_downstream, axis=1))
    cond_downstream_margin = np.mean(np.linalg.norm(current_cond - donor_downstream, axis=1) - np.linalg.norm(current_cond - own_downstream, axis=1))
    downstream_pull_cm = float((base_downstream_margin - cond_downstream_margin) * 100.0)
    return {
        "condition": name,
        "future_decoder_rmse_cm": rmse,
        "future_decoder_route_accuracy": route_acc,
        "future_decoder_donor_pull_cm": donor_pull_cm,
        "future_torus_donor_pull_rad": donor_phase_pull_rad,
        "downstream_position_donor_pull_cm": downstream_pull_cm,
    }


def unit_match_features(states: np.ndarray, model: FutureSplitRNN) -> np.ndarray:
    flat = states.reshape(-1, states.shape[-1])
    mean = flat.mean(axis=0)
    var = flat.var(axis=0)
    decoder_norm = model.decoder.weight.detach().cpu().numpy()
    decoder_norm = np.linalg.norm(decoder_norm, axis=0)
    feats = np.stack([mean, var, decoder_norm], axis=1).astype(np.float32)
    scale = np.nanstd(feats, axis=0)
    scale[scale < 1e-8] = 1.0
    return feats / scale[None]


def select_matched_unit_group(
    group_name: str,
    units: Dict[str, np.ndarray],
    Ng: int,
    stats: np.ndarray,
    rng: np.random.Generator,
    repeat_idx: int,
) -> np.ndarray:
    pred = np.asarray(units.get("predictive", []), dtype=int)
    if pred.size == 0:
        pred = np.arange(min(Ng, 8), dtype=int)
    if group_name in {"predictive", "predictive_grid"}:
        return pred
    if group_name in {"standard", "standard_grid"}:
        candidates = np.setdiff1d(np.asarray(units.get("standard", []), dtype=int), pred)
    elif group_name == "band":
        candidates = np.setdiff1d(np.asarray(units.get("band", []), dtype=int), pred)
    elif group_name == "random":
        candidates = np.setdiff1d(np.arange(Ng, dtype=int), pred)
    else:
        candidates = np.setdiff1d(np.arange(Ng, dtype=int), pred)
    candidates = candidates[(candidates >= 0) & (candidates < Ng)]
    if candidates.size == 0:
        candidates = np.setdiff1d(np.arange(Ng, dtype=int), pred)
    if candidates.size == 0:
        return pred
    n_select = min(pred.size, candidates.size)
    if n_select <= 0:
        return np.array([], dtype=int)
    ref = np.nanmean(stats[pred], axis=0)
    dist = np.linalg.norm(stats[candidates] - ref[None], axis=1)
    pool_n = min(candidates.size, max(n_select, n_select * 4))
    pool = candidates[np.argsort(dist)[:pool_n]]
    if group_name == "random" or repeat_idx > 0:
        return np.sort(rng.choice(pool, size=n_select, replace=False))
    return np.sort(pool[:n_select])


def run_intervene(args) -> Path:
    rng = set_seed(int(args.random_seed))
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = ensure_dir(args.output_dir)
    configure_logging(out_dir, "intervene.log")
    log_info("[intervene] checkpoint=%s", args.checkpoint_path)
    log_info("[intervene] gridness=%s", args.gridness_path)
    model, place_cells, options, _ = load_future_model(args.checkpoint_path, device=device)
    if getattr(args, "task", None):
        options.task = str(args.task)
    grid_data = load_gridness_payload(args.gridness_path)
    units = class_arrays(grid_data, int(options.Ng), fallback=bool(args.allow_fallback_units))
    pred_units = units["predictive"]
    rate_maps = np.asarray(grid_data["rate_maps"], dtype=float) if "rate_maps" in grid_data else None
    grid_units = np.unique(np.concatenate([units["predictive"], units["standard"], units["retrospective"]]))
    if grid_units.size == 0:
        grid_units = np.arange(min(int(options.Ng), int(args.Ng_torus_fallback)), dtype=int)
    projector = build_phase_projector(rate_maps, grid_units, options)
    log_info("[intervene] fitting post-hoc future decoders from held-out baseline states")

    options.batch_size = int(args.batch_size)
    horizon = int(args.main_horizon)
    decoders = fit_posthoc_future_decoders(
        model,
        options,
        place_cells,
        projector,
        rng,
        horizon,
        n_batches=int(getattr(args, "decoder_train_batches", 4)),
        batch_size=int(args.batch_size),
        ridge_alpha=float(getattr(args, "ridge_alpha", 1.0)),
    )
    batch = generate_task_batch(options, place_cells, batch_size=args.batch_size, rng=rng, paired=True, future_horizon=horizon)
    assert_matched_prebranch(batch, options)
    window = (
        int(args.window_start) if args.window_start is not None else int(task_cue_steps(options)),
        int(args.window_stop) if args.window_stop is not None else int(task_branch_step(options)),
    )
    log_info(
        "[intervene] device=%s task=%s torus_projector=%s predictive_units=%d trials=%d branch_step=%d window=%s horizon=%d",
        device,
        str(getattr(options, "task", "binary_fork")),
        projector.status,
        int(pred_units.size),
        int(batch.route_ids.size),
        int(task_branch_step(options)),
        list(window),
        horizon,
    )

    rows: List[Dict[str, object]] = []
    groups = [g for g in str(getattr(args, "intervention_groups", "predictive")).replace(",", " ").split() if g]
    group_repeats = max(1, int(getattr(args, "matched_group_repeats", 1)))
    swap_repeats = max(1, int(getattr(args, "swap_repeats", 1)))

    base_states, base_current_logits, _ = rollout_with_intervention(
        model,
        batch.inputs,
        np.array([], dtype=int),
        batch.pair_ids,
        window,
        "none",
        rng,
    )
    baseline_decoded = apply_posthoc_future_decoders(decoders, base_states)
    baseline = {
        "states": base_states,
        "current_pos": place_cells.get_nearest_cell_pos(base_current_logits).detach().cpu().numpy(),
        "actual_phase": phase_features_from_states(base_states, projector),
        **baseline_decoded,
    }
    stats = unit_match_features(base_states, model)
    log_info("[intervene] baseline states=%s groups=%s", tuple(base_states.shape), " ".join(groups))

    for group in groups:
        reps = group_repeats if group != "predictive" else 1
        for repeat_idx in range(reps):
            selected = select_matched_unit_group(group, units, int(options.Ng), stats, rng, repeat_idx)
            for name in ("ablate", "scramble", "swap"):
                local_repeats = swap_repeats if name == "swap" else 1
                for swap_idx in range(local_repeats):
                    step_start = time.time()
                    states, current_logits, _ = rollout_with_intervention(
                        model,
                        batch.inputs,
                        selected,
                        batch.pair_ids,
                        window,
                        name,
                        rng,
                    )
                    decoded = apply_posthoc_future_decoders(decoders, states)
                    condition = {
                        "states": states,
                        "current_pos": place_cells.get_nearest_cell_pos(current_logits).detach().cpu().numpy(),
                        "actual_phase": phase_features_from_states(states, projector),
                        **decoded,
                    }
                    row = intervention_metrics(
                        f"{group}_{name}",
                        baseline,
                        condition,
                        batch,
                        options,
                        horizon,
                    )
                    row["unit_group"] = group
                    row["intervention"] = name
                    row["repeat"] = int(repeat_idx)
                    row["swap_repeat"] = int(swap_idx)
                    row["n_units"] = int(selected.size)
                    rows.append(row)
                    log_info(
                        "[intervene] group=%s intervention=%s repeat=%d/%d units=%d donor_pull_cm=%.3f phase_pull_rad=%.3f elapsed=%s",
                        group,
                        name,
                        repeat_idx + 1,
                        reps,
                        int(selected.size),
                        float(row.get("future_decoder_donor_pull_cm", np.nan)),
                        float(row.get("future_torus_donor_pull_rad", np.nan)),
                        format_duration(time.time() - step_start),
                    )

    summary = {
        "checkpoint_path": str(args.checkpoint_path),
        "gridness_path": str(args.gridness_path),
        "task": str(getattr(options, "task", "binary_fork")),
        "torus_projector_status": projector.status,
        "predictive_unit_count": int(pred_units.size),
        "window": list(window),
        "main_horizon": horizon,
        "intervention_groups": groups,
        "matched_group_repeats": group_repeats,
        "swap_repeats": swap_repeats,
        "rows": rows,
    }
    write_json(out_dir / "intervention_metrics.json", summary)
    write_csv(out_dir / "intervention_metrics.csv", rows)
    write_json(
        out_dir / "summary.json",
        {
            "task": str(getattr(options, "task", "binary_fork")),
            "predictive_unit_count": int(pred_units.size),
            "intervention_groups": groups,
            "rows": len(rows),
        },
    )
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
    task = str(getattr(args, "task", "binary_fork"))
    cue_steps = getattr(args, "cue_steps", None)
    branch_step = getattr(args, "branch_step", None)
    options = make_options(
        task=task,
        batch_size=args.batch_size,
        sequence_length=args.sequence_length,
        Np=args.Np,
        Ng=args.Ng,
        cue_dim=getattr(args, "cue_dim", None),
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
        cue_steps=cue_steps,
        branch_step=branch_step,
        fork_stem_start_y=args.fork_stem_start_y,
        fork_branch_y=args.fork_branch_y,
        fork_arm_length=args.fork_arm_length,
        fork_cue_scale=args.fork_cue_scale,
        num_routes=getattr(args, "num_routes", None),
        velocity_frame=getattr(args, "velocity_frame", None),
        open_field_mix=getattr(args, "open_field_mix", None),
        activity_l1=getattr(args, "activity_l1", None),
        delay_noise_std=getattr(args, "delay_noise_std", None),
        cue_dropout=getattr(args, "cue_dropout", None),
        cue_noise_std=getattr(args, "cue_noise_std", None),
        hidden_dropout=getattr(args, "hidden_dropout", None),
        recurrent_type=getattr(args, "recurrent_type", None),
        rank=getattr(args, "rank", None),
        future_horizon=args.future_horizon,
        future_loss_weight=args.future_loss_weight,
    )
    place_cells = PlaceCells(options)
    model = FutureSplitRNN(options, place_cells).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=float(args.learning_rate))
    log_every = max(1, int(getattr(args, "log_every_steps", 50)))
    total_steps = max(1, int(args.n_epochs) * int(args.n_steps))
    train_start = time.time()
    traj_gen = TrajectoryGenerator(options, place_cells)
    log_info("[train] starting future-split training")
    log_info("[train] output_dir=%s", out_dir)
    log_info("[train] device=%s cuda_available=%s", device, torch.cuda.is_available())
    log_info(
        "[train] parameters=%d task=%s recurrent=%s Ng=%d Np=%d input_dim=%d",
        parameter_count(model),
        str(options.task),
        str(options.recurrent_type),
        int(options.Ng),
        int(options.Np),
        int(options.velocity_dim),
    )
    log_info(
        "[train] epochs=%d steps_per_epoch=%d batch_size=%d sequence_length=%d total_steps=%d",
        int(args.n_epochs),
        int(args.n_steps),
        int(args.batch_size),
        int(args.sequence_length),
        total_steps,
    )
    log_info(
        "[train] task cue_steps=%d branch_step=%d num_routes=%d velocity_frame=%s "
        "future_horizon=%d future_loss_weight=%.4g open_field_mix=%.3f",
        int(task_cue_steps(options)),
        int(task_branch_step(options)),
        int(getattr(options, "num_routes", 2)),
        str(getattr(options, "velocity_frame", "physical")),
        int(args.future_horizon),
        float(args.future_loss_weight),
        float(getattr(options, "open_field_mix", 0.0)),
    )
    log_info("[train] config=%s", json.dumps(json_safe(options_to_dict(options)), sort_keys=True))
    history = []
    model.train()
    for epoch in range(int(args.n_epochs)):
        epoch_start = time.time()
        epoch_terms: List[Dict[str, float]] = []
        for step_idx in range(int(args.n_steps)):
            step_start = time.time()
            if rng.random() < float(getattr(options, "open_field_mix", 0.0)):
                batch = generate_open_field_batch(options, place_cells, traj_gen, batch_size=args.batch_size)
            else:
                batch = generate_task_batch(options, place_cells, rng=rng, paired=True)
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
            grad_norm_sq = 0.0
            for param in model.parameters():
                if param.grad is not None:
                    grad_norm_sq += float(param.grad.detach().pow(2).sum().cpu())
            terms["grad_norm"] = float(math.sqrt(grad_norm_sq))
            opt.step()
            epoch_terms.append(terms)
            completed = epoch * int(args.n_steps) + step_idx + 1
            if completed == 1 or completed % log_every == 0 or step_idx + 1 == int(args.n_steps):
                recent = epoch_terms[-min(len(epoch_terms), log_every):]
                recent_loss = float(np.mean([row["loss"] for row in recent]))
                recent_current = float(np.mean([row["current_ce"] for row in recent]))
                recent_future = float(np.mean([row["future_ce"] for row in recent]))
                recent_excess = float(np.mean([row.get("current_excess_ce", np.nan) for row in recent]))
                recent_rmse = float(np.mean([row.get("current_rmse_cm", np.nan) for row in recent]))
                recent_sparsity = float(np.mean([row.get("activity_sparsity", np.nan) for row in recent]))
                recent_grad = float(np.mean([row.get("grad_norm", np.nan) for row in recent]))
                elapsed = time.time() - train_start
                steps_per_sec = completed / max(elapsed, 1e-9)
                eta = (total_steps - completed) / max(steps_per_sec, 1e-9)
                log_info(
                    "[train] step=%d/%d epoch=%d/%d step_in_epoch=%d/%d "
                    "loss=%.4f current_ce=%.4f excess_ce=%.4f current_rmse_cm=%.2f "
                    "future_ce=%.4f sparsity=%.3f grad_norm=%.3f step_time=%s throughput=%.2f steps/s eta=%s",
                    completed,
                    total_steps,
                    epoch + 1,
                    int(args.n_epochs),
                    step_idx + 1,
                    int(args.n_steps),
                    recent_loss,
                    recent_current,
                    recent_excess,
                    recent_rmse,
                    recent_future,
                    recent_sparsity,
                    recent_grad,
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
            "[train] epoch=%d/%d summary loss=%.4f current_ce=%.4f excess_ce=%.4f "
            "current_rmse_cm=%.2f future_ce=%.4f epoch_time=%s elapsed=%s",
            epoch + 1,
            int(args.n_epochs),
            mean_terms["loss"],
            mean_terms["current_ce"],
            mean_terms.get("current_excess_ce", float("nan")),
            mean_terms.get("current_rmse_cm", float("nan")),
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
    smoke_task = str(getattr(args, "task", "multi_route_graph"))
    log_info("[smoke] starting CPU-only smoke test task=%s", smoke_task)
    train_args = argparse.Namespace(
        task=smoke_task,
        output_dir=out_dir,
        n_epochs=1,
        n_steps=3,
        log_every_steps=1,
        batch_size=8,
        sequence_length=28 if smoke_task == "multi_route_graph" else 18,
        Np=32,
        Ng=24,
        cue_dim=2 if smoke_task == "multi_route_graph" else 1,
        num_routes=6 if smoke_task == "multi_route_graph" else 2,
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
        fork_branch_step=12 if smoke_task == "multi_route_graph" else 10,
        cue_steps=3,
        branch_step=12 if smoke_task == "multi_route_graph" else 10,
        fork_stem_start_y=-0.8,
        fork_branch_y=0.0,
        fork_arm_length=0.55,
        fork_cue_scale=1.0,
        velocity_frame="local_graph",
        open_field_mix=0.5 if smoke_task == "multi_route_graph" else 0.0,
        activity_l1=1e-5,
        delay_noise_std=0.01 if smoke_task == "multi_route_graph" else 0.0,
        cue_dropout=0.1 if smoke_task == "multi_route_graph" else 0.0,
        cue_noise_std=0.01 if smoke_task == "multi_route_graph" else 0.0,
        hidden_dropout=0.0,
        recurrent_type="full",
        rank=8,
        future_horizon=4,
        future_loss_weight=0.0 if smoke_task == "multi_route_graph" else 0.5,
    )
    run_train(train_args)
    configure_logging(out_dir, "smoke.log")
    full_ckpt = out_dir / "future_split_full.pth"

    model, place_cells, options, _ = load_future_model(full_ckpt, device="cpu")
    smoke_rng = set_seed(11)
    batch = generate_task_batch(options, place_cells, batch_size=8, rng=smoke_rng, paired=True, future_horizon=4)
    assert_matched_prebranch(batch, options)
    log_info("[smoke] matched pre-branch assertions passed")
    with torch.no_grad():
        loss, terms = model.compute_loss(
            batch.inputs,
            batch.place_outputs,
            batch.future_place_outputs,
            batch.future_mask,
            float(train_args.future_loss_weight),
        )
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
        shift_mode="time",
        space_projection="path",
        max_shift_cm=4.0,
        shift_step_cm=2.0,
        min_shift_cm=0.1,
        gridness_threshold=-1.0,
        band_percentile=80.0,
        band_threshold=None,
        score_log_every_units=4,
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
        task=smoke_task,
        route_controls=True,
        horizon_specificity=True,
        matched_group_repeats=2,
    )
    decode_dir = run_decode(decode_args)
    configure_logging(out_dir, "smoke.log")

    crossing_args = argparse.Namespace(
        checkpoint_path=str(full_ckpt),
        gridness_path=str(gridness_path),
        output_dir=str(out_dir / "crossing"),
        device="cpu",
        batch_size=8,
        n_batches=1,
        sequence_length=None,
        pairing_mode="x_crossing",
        crossing_step=None,
        future_horizon=4,
        min_angle_deg=30.0,
        max_angle_deg=140.0,
        line_extent=0.55,
        matched_random_repeats=3,
        log_every_batches=1,
        random_seed=15,
        allow_fallback_units=True,
        same_bin_trajectory_source="random_walk",
        same_bin_timestep=None,
        same_bin_res=8,
        same_bin_pairs_per_bin=2,
        same_bin_border_buffer=0.0,
        same_bin_movement_bins=4,
        same_bin_plot=True,
        title=None,
        dpi=100,
    )
    crossing_dir = run_crossing(crossing_args)
    configure_logging(out_dir, "smoke.log")

    same_bin_args = argparse.Namespace(
        checkpoint_path=str(full_ckpt),
        gridness_path=str(gridness_path),
        output_dir=str(out_dir / "same_bin"),
        device="cpu",
        batch_size=64,
        n_batches=1,
        sequence_length=12,
        pairing_mode="same_bin",
        crossing_step=None,
        future_horizon=4,
        min_angle_deg=30.0,
        max_angle_deg=140.0,
        line_extent=0.55,
        matched_random_repeats=2,
        log_every_batches=1,
        random_seed=16,
        allow_fallback_units=True,
        same_bin_trajectory_source="random_walk",
        same_bin_timestep=6,
        same_bin_res=5,
        same_bin_pairs_per_bin=1,
        same_bin_border_buffer=0.0,
        same_bin_movement_bins=4,
        same_bin_plot=True,
        title="Smoke same-bin population correlation",
        dpi=100,
    )
    same_bin_dir = run_crossing(same_bin_args)
    configure_logging(out_dir, "smoke.log")

    same_bin_x_args = argparse.Namespace(
        checkpoint_path=str(full_ckpt),
        gridness_path=str(gridness_path),
        output_dir=str(out_dir / "same_bin_x_crossing"),
        device="cpu",
        batch_size=8,
        n_batches=1,
        sequence_length=12,
        pairing_mode="same_bin",
        crossing_step=None,
        future_horizon=4,
        min_angle_deg=60.0,
        max_angle_deg=90.0,
        line_extent=0.55,
        matched_random_repeats=2,
        log_every_batches=1,
        random_seed=18,
        allow_fallback_units=True,
        same_bin_trajectory_source="x_crossing",
        same_bin_timestep=6,
        same_bin_res=5,
        same_bin_pairs_per_bin=1,
        same_bin_border_buffer=0.0,
        same_bin_movement_bins=4,
        same_bin_plot=True,
        title="Smoke same-bin exact-X population correlation",
        dpi=100,
    )
    same_bin_x_dir = run_crossing(same_bin_x_args)
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
        task=smoke_task,
        horizons="1 2 4",
        intervention_groups="predictive standard_grid band random",
        swap_repeats=1,
        matched_group_repeats=2,
        decoder_train_batches=1,
        ridge_alpha=1.0,
    )
    intervene_dir = run_intervene(intervene_args)
    configure_logging(out_dir, "smoke.log")

    required = [
        out_dir / "training_metrics.json",
        gridness_path,
        decode_dir / "decode_metrics.json",
        decode_dir / "decode_metrics.csv",
        decode_dir / "route_control_metrics.csv",
        decode_dir / "horizon_specificity.csv",
        crossing_dir / "crossing_metrics.json",
        crossing_dir / "crossing_pair_metrics.csv",
        crossing_dir / "crossing_summary.csv",
        same_bin_dir / "crossing_metrics.json",
        same_bin_dir / "crossing_pair_metrics.csv",
        same_bin_dir / "crossing_summary.csv",
        same_bin_dir / "same_bin_movement_binned_metrics.csv",
        same_bin_dir / "same_bin_heatmaps.npz",
        same_bin_x_dir / "crossing_metrics.json",
        same_bin_x_dir / "crossing_pair_metrics.csv",
        same_bin_x_dir / "crossing_summary.csv",
        same_bin_x_dir / "same_bin_movement_binned_metrics.csv",
        same_bin_x_dir / "same_bin_heatmaps.npz",
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
            "task": smoke_task,
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
    parser.add_argument("--task", default="binary_fork", choices=["binary_fork", "multi_route_graph"])
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--sequence_length", type=int, default=32)
    parser.add_argument("--Np", type=int, default=256)
    parser.add_argument("--Ng", type=int, default=512)
    parser.add_argument("--cue_dim", type=int, default=1)
    parser.add_argument("--num_routes", type=int, default=12)
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
    parser.add_argument("--cue_steps", type=int, default=None)
    parser.add_argument("--branch_step", type=int, default=None)
    parser.add_argument("--fork_stem_start_y", type=float, default=-0.85)
    parser.add_argument("--fork_branch_y", type=float, default=0.0)
    parser.add_argument("--fork_arm_length", type=float, default=0.75)
    parser.add_argument("--fork_cue_scale", type=float, default=1.0)
    parser.add_argument("--velocity_frame", default="local_graph", choices=["physical", "local_graph"])
    parser.add_argument("--open_field_mix", type=float, default=0.0)
    parser.add_argument("--activity_l1", type=float, default=0.0)
    parser.add_argument("--delay_noise_std", type=float, default=0.0)
    parser.add_argument("--cue_dropout", type=float, default=0.0)
    parser.add_argument("--cue_noise_std", type=float, default=0.0)
    parser.add_argument("--hidden_dropout", type=float, default=0.0)
    parser.add_argument("--recurrent_type", default="full", choices=["full", "low_rank"])
    parser.add_argument("--rank", type=int, default=128)
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
    p_class.add_argument("--Ng_use", default="512")
    p_class.add_argument("--res", type=int, default=20)
    p_class.add_argument("--max_lag", type=int, default=12)
    p_class.add_argument(
        "--shift_mode",
        default="time",
        choices=["time", "space"],
        help="Classify by aligning activity to shifted positions by timestep or by true spatial distance.",
    )
    p_class.add_argument(
        "--space_projection",
        default="path",
        choices=["path", "heading"],
        help="When --shift_mode space, use arc-length along the realized path or local heading projection.",
    )
    p_class.add_argument(
        "--max_shift_cm",
        type=float,
        default=20.0,
        help="When --shift_mode space, evaluate shifts from -max_shift_cm to +max_shift_cm.",
    )
    p_class.add_argument(
        "--shift_step_cm",
        type=float,
        default=1.0,
        help="When --shift_mode space, spacing between evaluated spatial shifts in cm.",
    )
    p_class.add_argument("--min_shift_cm", type=float, default=5.0)
    p_class.add_argument("--gridness_threshold", type=float, default=0.2)
    p_class.add_argument("--band_percentile", type=float, default=90.0)
    p_class.add_argument("--band_threshold", type=float, default=None)
    p_class.add_argument(
        "--score_log_every_units",
        type=int,
        default=25,
        help="During classification, log gridness/rate-map scoring progress every N units.",
    )

    p_decode = sub.add_parser("decode", help="Decode future route/position/torus phase on matched fork trials.")
    p_decode.add_argument("--checkpoint_path", required=True)
    p_decode.add_argument("--gridness_path", required=True)
    p_decode.add_argument("--output_dir", required=True)
    p_decode.add_argument("--device", default=None)
    p_decode.add_argument("--task", default=None, choices=["binary_fork", "multi_route_graph"])
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
    p_decode.add_argument("--route_controls", action="store_true")
    p_decode.add_argument("--horizon_specificity", action="store_true")
    p_decode.add_argument("--matched_group_repeats", type=int, default=1)

    p_cross = sub.add_parser("crossing", help="Compare grid class correlations on matched X-crossing trajectories.")
    p_cross.add_argument("--checkpoint_path", required=True)
    p_cross.add_argument("--gridness_path", required=True)
    p_cross.add_argument("--output_dir", required=True)
    p_cross.add_argument("--device", default=None)
    p_cross.add_argument("--batch_size", type=int, default=128)
    p_cross.add_argument("--n_batches", type=int, default=8)
    p_cross.add_argument("--sequence_length", type=int, default=None)
    p_cross.add_argument(
        "--pairing_mode",
        default="x_crossing",
        choices=["x_crossing", "same_bin"],
        help="x_crossing keeps the exact controlled-X analysis; same_bin runs the pasted-style same-spatial-bin analysis.",
    )
    p_cross.add_argument("--crossing_step", type=int, default=None)
    p_cross.add_argument("--future_horizon", type=int, default=8)
    p_cross.add_argument("--min_angle_deg", type=float, default=30.0)
    p_cross.add_argument("--max_angle_deg", type=float, default=150.0)
    p_cross.add_argument("--line_extent", type=float, default=0.75)
    p_cross.add_argument("--matched_random_repeats", type=int, default=20)
    p_cross.add_argument("--log_every_batches", type=int, default=5)
    p_cross.add_argument("--random_seed", type=int, default=0)
    p_cross.add_argument("--allow_fallback_units", action=argparse.BooleanOptionalAction, default=True)
    p_cross.add_argument(
        "--same_bin_trajectory_source",
        default="random_walk",
        choices=["random_walk", "x_crossing"],
        help="For --pairing_mode same_bin, choose ordinary same-bin random-walk pairs or exact controlled-X pairs passed through the same-bin summaries.",
    )
    p_cross.add_argument("--same_bin_timestep", type=int, default=None)
    p_cross.add_argument("--same_bin_res", type=int, default=44)
    p_cross.add_argument("--same_bin_pairs_per_bin", type=int, default=20)
    p_cross.add_argument("--same_bin_border_buffer", type=float, default=0.2)
    p_cross.add_argument("--same_bin_movement_bins", type=int, default=20)
    p_cross.add_argument("--same_bin_plot", action=argparse.BooleanOptionalAction, default=True)
    p_cross.add_argument("--title", default=None)
    p_cross.add_argument("--dpi", type=int, default=200)

    p_plot_cross = sub.add_parser("plot_crossing", help="Plot summary figures from a crossing output directory.")
    p_plot_cross.add_argument("--crossing_dir", required=True)
    p_plot_cross.add_argument("--output_dir", default=None)
    p_plot_cross.add_argument("--title", default=None)
    p_plot_cross.add_argument("--dpi", type=int, default=200)
    p_plot_cross.add_argument("--hist_bins", type=int, default=50)
    p_plot_cross.add_argument("--random_seed", type=int, default=0)

    p_step_sweep = sub.add_parser("crossing_step_sweep", help="Run X-crossing analysis across multiple crossing steps.")
    p_step_sweep.add_argument("--checkpoint_path", required=True)
    p_step_sweep.add_argument("--gridness_path", required=True)
    p_step_sweep.add_argument("--output_dir", required=True)
    p_step_sweep.add_argument("--device", default=None)
    p_step_sweep.add_argument("--batch_size", type=int, default=128)
    p_step_sweep.add_argument("--n_batches", type=int, default=8)
    p_step_sweep.add_argument("--sequence_length", type=int, default=None)
    p_step_sweep.add_argument("--crossing_steps", default="8 10 12 14 15 16 18 20 22")
    p_step_sweep.add_argument("--future_horizon", type=int, default=8)
    p_step_sweep.add_argument("--min_angle_deg", type=float, default=90.0)
    p_step_sweep.add_argument("--max_angle_deg", type=float, default=90.0)
    p_step_sweep.add_argument("--line_extent", type=float, default=0.75)
    p_step_sweep.add_argument("--matched_random_repeats", type=int, default=20)
    p_step_sweep.add_argument("--log_every_batches", type=int, default=5)
    p_step_sweep.add_argument("--random_seed", type=int, default=0)
    p_step_sweep.add_argument("--allow_fallback_units", action=argparse.BooleanOptionalAction, default=True)
    p_step_sweep.add_argument("--label", default=None)
    p_step_sweep.add_argument("--title", default=None)
    p_step_sweep.add_argument("--dpi", type=int, default=200)

    p_step_sweep_plot = sub.add_parser("plot_crossing_step_sweeps", help="Overlay one or more crossing-step sweep outputs.")
    p_step_sweep_plot.add_argument(
        "--sweep_dir",
        action="append",
        required=True,
        help="Sweep directory, optionally as LABEL=DIR. Repeat for trained/random overlays.",
    )
    p_step_sweep_plot.add_argument("--output_dir", required=True)
    p_step_sweep_plot.add_argument("--title", default=None)
    p_step_sweep_plot.add_argument("--dpi", type=int, default=300)

    p_int = sub.add_parser("intervene", help="Ablate, scramble, and swap predictive activity pre-branch.")
    p_int.add_argument("--checkpoint_path", required=True)
    p_int.add_argument("--gridness_path", required=True)
    p_int.add_argument("--output_dir", required=True)
    p_int.add_argument("--device", default=None)
    p_int.add_argument("--task", default=None, choices=["binary_fork", "multi_route_graph"])
    p_int.add_argument("--batch_size", type=int, default=80)
    p_int.add_argument("--horizons", default="1 2 4 6 8 10 12")
    p_int.add_argument("--main_horizon", type=int, default=8)
    p_int.add_argument("--random_seed", type=int, default=0)
    p_int.add_argument("--allow_fallback_units", action=argparse.BooleanOptionalAction, default=True)
    p_int.add_argument("--Ng_torus_fallback", type=int, default=32)
    p_int.add_argument("--window_start", type=int, default=None)
    p_int.add_argument("--window_stop", type=int, default=None)
    p_int.add_argument("--intervention_groups", default="predictive")
    p_int.add_argument("--swap_repeats", type=int, default=1)
    p_int.add_argument("--matched_group_repeats", type=int, default=1)
    p_int.add_argument("--decoder_train_batches", type=int, default=4)
    p_int.add_argument("--ridge_alpha", type=float, default=1.0)

    p_smoke = sub.add_parser("smoke", help="Run a tiny CPU-only smoke test.")
    p_smoke.add_argument("--output_dir", default="/private/tmp/future_split_smoke")
    p_smoke.add_argument("--task", default="multi_route_graph", choices=["binary_fork", "multi_route_graph"])

    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    if args.cmd == "train":
        run_train(args)
    elif args.cmd == "classify":
        run_classify(args)
    elif args.cmd == "decode":
        run_decode(args)
    elif args.cmd == "crossing":
        run_crossing(args)
    elif args.cmd == "plot_crossing":
        run_plot_crossing(args)
    elif args.cmd == "crossing_step_sweep":
        run_crossing_step_sweep(args)
    elif args.cmd == "plot_crossing_step_sweeps":
        run_plot_crossing_step_sweeps(args)
    elif args.cmd == "intervene":
        run_intervene(args)
    elif args.cmd == "smoke":
        run_smoke(args)
    else:  # pragma: no cover
        raise SystemExit(f"Unknown command: {args.cmd}")


if __name__ == "__main__":
    main(sys.argv[1:])
