"""Shared foundation for the PGC rigor-upgrade pipeline.

Every new analysis module (classifier, covariates, matched ablation, torus
topology, intervention) imports from here so that they all:

  * load checkpoints the same way (reusing the existing helpers), and
  * derive per-unit quantities from ONE seeded activation collection with ONE
    unit-indexing convention (units 0..Ng_use-1).

This deliberately reuses the tested helpers in ``multi_seed_predictive_analysis``
(``infer_dims_from_state``, ``build_options``, ``zero_unit_weights_in_place``,
``collect_eval_batches``, ``mean_decoding_error_cm``) rather than re-implementing
model loading, so the new pipeline cannot drift from the existing one.
"""
from __future__ import annotations

import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import torch

# Make ``code/`` importable no matter the CWD.
_CODE_DIR = os.path.dirname(os.path.abspath(__file__))
if _CODE_DIR not in sys.path:
    sys.path.insert(0, _CODE_DIR)

from model import RNN  # noqa: E402
from place_cells import PlaceCells  # noqa: E402
from trajectory_generator import TrajectoryGenerator  # noqa: E402
from multi_seed_predictive_analysis import (  # noqa: E402
    infer_dims_from_state,
    build_options,
    zero_unit_weights_in_place,
    collect_eval_batches,
    mean_decoding_error_cm,
)

SEED_RE = re.compile(r"[Ss]eed[ _]?(\d+)")


def seed_id_from_path(path: str) -> int:
    """Extract the integer seed id from a checkpoint path (or -1 if absent)."""
    m = SEED_RE.search(str(path))
    return int(m.group(1)) if m else -1


@dataclass
class AnalysisArgs:
    """Namespace compatible with ``build_options`` + ``PlaceCells``.

    Defaults match the frozen canonical recipe (RNN 4096, DoG place cells,
    2.2 m box, rf 0.12). ``sequence_length`` defaults to 40 for analysis (longer
    rollouts give cleaner rate maps) even though training used 20.
    """
    batch_size: int = 200
    sequence_length: int = 40
    place_cell_rf: float = 0.12
    surround_scale: float = 2.0
    activation: str = "relu"
    weight_decay: float = 1e-6
    box_width: float = 2.2
    box_height: float = 2.2
    learning_rate: float = 1e-4
    # trajectory knobs (defaults reproduce main.py)
    traj_speed_scale: float = 1.0
    traj_speed_max: Optional[float] = None
    traj_velocity_smoothing: float = 0.0
    traj_turn_sigma_scale: float = 1.0
    traj_border_region: float = 0.03
    traj_wall_slowdown: float = 0.25
    traj_wall_turn_scale: float = 1.0


@dataclass
class LoadedModel:
    model: RNN
    options: object
    place_cells: PlaceCells
    traj_gen: TrajectoryGenerator
    checkpoint_path: str
    Ng: int
    Np: int
    seed_id: int
    device: str


def load_model(checkpoint_path: str,
               device: str = "cuda",
               args: Optional[AnalysisArgs] = None) -> LoadedModel:
    """Load a checkpoint into an RNN + trajectory generator (eval mode)."""
    if args is None:
        args = AnalysisArgs()
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    state = torch.load(checkpoint_path, map_location=device)
    if hasattr(state, "state_dict"):
        state = state.state_dict()
    dims = infer_dims_from_state(state)  # (Ng, Np, velocity_dim)
    options = build_options(args, dims, device, checkpoint_path)
    place_cells = PlaceCells(options)
    model = RNN(options, place_cells).to(device)
    model.load_state_dict(state)
    model.eval()
    traj_gen = TrajectoryGenerator(options, place_cells)
    return LoadedModel(
        model=model, options=options, place_cells=place_cells, traj_gen=traj_gen,
        checkpoint_path=str(checkpoint_path), Ng=dims[0], Np=dims[1],
        seed_id=seed_id_from_path(checkpoint_path), device=device,
    )


@dataclass
class ActivationBundle:
    """One reproducible activation collection everything derives from.

    All per-unit covariates and classifications MUST come from a bundle so they
    share the same trajectories and unit indexing (0..Ng_use-1).
    """
    xs: np.ndarray            # [T, B]
    ys: np.ndarray            # [T, B]
    activations: np.ndarray   # [T, B, Ng_use]  time-resolved
    rate_maps: np.ndarray     # [Ng_use, res, res]
    g_flat: np.ndarray        # [N, Ng_use]  raw states (for variance/topology)
    pos_flat: np.ndarray      # [N, 2]
    idxs: np.ndarray          # unit indices, arange(Ng_use)
    res: int
    collection_seed: int


def collect_bundle(lm: LoadedModel,
                   n_batches: int = 20,
                   Ng_use: int = 512,
                   res: int = 20,
                   collection_seed: int = 1234,
                   with_ratemaps: bool = True) -> ActivationBundle:
    """Seeded collection of time-resolved activations (+ optional rate maps).

    ``get_test_batch`` draws from the global numpy RNG, so we seed it here (after
    PlaceCells has already fixed its own layout via its internal seed(0)) to make
    the collection reproducible and to let callers request disjoint held-out sets
    by passing a different ``collection_seed``.

    ``with_ratemaps=False`` skips the second forward pass that builds rate maps —
    the classifier only needs the time-resolved activations, so skipping it halves
    the (GPU) forward cost.
    """
    from visualize import collect_sequences, compute_ratemaps  # local import

    Ng_use = int(min(Ng_use, lm.Ng))
    idxs = np.arange(Ng_use)

    np.random.seed(collection_seed)
    xs, ys, activations = collect_sequences(
        lm.model, lm.traj_gen, lm.options, n_batches=n_batches, Ng=Ng_use, idxs=idxs
    )
    if with_ratemaps:
        # Rate maps from a second seeded pass (same seed stream continues).
        activations_rm, rate_map, g_flat, pos_flat = compute_ratemaps(
            lm.model, lm.traj_gen, lm.options, res=res, n_avg=n_batches, Ng=Ng_use, idxs=idxs
        )
    else:
        activations_rm = np.empty((Ng_use, res, res), dtype=float)
        g_flat = activations.reshape(-1, Ng_use)
        pos_flat = np.stack([xs.reshape(-1), ys.reshape(-1)], axis=1)
    return ActivationBundle(
        xs=xs, ys=ys, activations=activations, rate_maps=activations_rm,
        g_flat=g_flat, pos_flat=pos_flat, idxs=idxs, res=res,
        collection_seed=collection_seed,
    )


def cm_per_step(xs: np.ndarray, ys: np.ndarray) -> float:
    """Mean per-step displacement in cm (for time->cm shift conversion)."""
    dx = np.diff(xs, axis=0)
    dy = np.diff(ys, axis=0)
    return float(np.nanmean(np.sqrt(dx ** 2 + dy ** 2)) * 100.0)


# ---------------------------------------------------------------------------
# npz schemas (documented so modules written independently stay aligned).
# ---------------------------------------------------------------------------
# pgc_classification.npz  (from pgc_classifier.py)
#   labels            [Ng]  int: 0 normal/zero-lag, 1 predictive, 2 retrospective, -1 non-grid
#   best_cm           [Ng]  float: preferred shift (cm) at argmax gridness
#   best_gridness     [Ng]  float
#   pval              [Ng]  float: block-shuffle significance of best-shift gridness
#   edge_flag         [Ng]  bool: argmax sat at sweep edge (excluded)
#   heldout_confirmed [Ng]  bool: sign(best_cm) reproduced on held-out trajectories
#   lag_cm            [L]   float: the shift sweep (cm)
#   scores_60         [L,Ng] float
CLASS_LABELS = {"non_grid": -1, "normal": 0, "predictive": 1, "retrospective": 2}
CLASS_NAMES = {v: k for k, v in CLASS_LABELS.items()}

# pgc_covariates.npz  (from pgc_covariates.py)  -- all length Ng_use, index 0..Ng_use-1
COVARIATE_KEYS = (
    "module",        # int module label from grid-spacing clustering (-1 = non-grid)
    "gridness",      # best 60-deg gridness
    "bandness",      # band score
    "rate_var",      # firing-rate variance
    "hd_r",          # head-direction Rayleigh vector length
    "decoder_mag",   # ||decoder.weight[:,u]||
    "rec_in",        # ||W_hh[u,:]||  (incoming recurrent)
    "rec_out",       # ||W_hh[:,u]||  (outgoing recurrent)
)


if __name__ == "__main__":  # smoke test
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--Ng_use", type=int, default=64)
    p.add_argument("--n_batches", type=int, default=4)
    a = p.parse_args()
    lm = load_model(a.checkpoint, device=a.device)
    print(f"loaded Ng={lm.Ng} Np={lm.Np} seed_id={lm.seed_id} device={lm.device}")
    b = collect_bundle(lm, n_batches=a.n_batches, Ng_use=a.Ng_use, res=20)
    print(f"activations {b.activations.shape} rate_maps {b.rate_maps.shape} "
          f"g_flat {b.g_flat.shape} cm/step={cm_per_step(b.xs, b.ys):.2f}")
    err = mean_decoding_error_cm(lm.model, collect_eval_batches(lm.traj_gen, 4))
    print(f"baseline decoding error = {err:.2f} cm")
