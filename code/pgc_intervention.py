#!/usr/bin/env python3
"""Causal intervention on the predictive-grid-cell (PGC) subspace.

This is the flagship *causal* experiment of the PGC rigor pipeline.  Every
prior module only *observes* predictive units; here we manipulate them and read
out the consequence on the toroidal phase and on position decoding.

Mechanism -- a MANUAL, step-by-step re-implementation of the RNN recurrence so
that arbitrary activity can be injected at every timestep (``torch.nn.RNN``
consumes a whole sequence at once and gives no per-step hook):

    h_0 = encoder(p0)                                        # [B, Ng]
    h_t = relu( v_t @ W_ih.T + h_{t-1} @ W_hh.T + drive_t )  # t = 1..T
    g[t] = h_t

With ``drive == 0`` this reproduces ``model.g((v, p0))`` exactly (validated in
the self-test: bit-exact in float64, ~1e-4 in float32/GPU).

Three experiments (all read toroidal phase from the *intact* assumed-hexagonal
Fourier basis via ``build_torus_basis`` / ``project_states_to_torus`` from
``toroidal_structure_analysis``):

  1. STIMULATE / dose curve -- on the INTACT network, start from rest (v=0) and
     inject a constant drive ``alpha * d`` along the PGC subspace.  Sweep the
     commanded magnitude ``alpha`` and show it drives a monotonic net toroidal
     phase displacement.  Report the drive<->displacement correlation.

  2. FREEZE -- ablate the predictive units (``zero_unit_weights_in_place``) and
     confirm the torus phase velocity collapses and decoding error rises.

  3. RESCUE / replay -- inject the predictive subspace's *own downstream
     recurrent current*, taken from the intact network on the SAME trajectory,
     into the ablated network's rollout and show phase velocity / decoding error
     partially recover.  A direct-into-subspace injection is run as a control
     (it is expected to do nothing, because ablation severs the predictive
     units' outgoing weights).

Outputs (under ``analysis_dir_for_checkpoint(ckpt)/<out_subdir>/``):
    pgc_intervention.npz
    pgc_intervention_summary.json
    pgc_intervention.png   (bars per condition + dose curve)

CLI:
    python code/pgc_intervention.py --checkpoint <ckpt.pth> [--device cuda] ...
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import sys
from pathlib import Path

# Writable caches inside the repo, before matplotlib / numba get imported by the
# toroidal module we depend on.
os.environ.setdefault("NUMBA_CACHE_DIR", str(Path(".numba_cache").resolve()))
os.environ.setdefault("MPLCONFIGDIR", str(Path(".mplconfig").resolve()))
for _d in (os.environ["NUMBA_CACHE_DIR"], os.environ["MPLCONFIGDIR"]):
    Path(_d).mkdir(parents=True, exist_ok=True)

import numpy as np
import torch

# Make ``code/`` importable regardless of CWD (mirrors the other pgc_* modules).
_CODE_DIR = os.path.dirname(os.path.abspath(__file__))
if _CODE_DIR not in sys.path:
    sys.path.insert(0, _CODE_DIR)

import pgc_common as C  # noqa: E402
from path_utils import analysis_dir_for_checkpoint  # noqa: E402
# Reuse the torus phase read-back (assumed-hexagonal Fourier basis).
from toroidal_structure_analysis import (  # noqa: E402
    build_torus_basis,
    project_states_to_torus,
)

_TWO_PI = 2.0 * np.pi
_TORUS_RADII = (1.0, 0.35)  # embedding radii; irrelevant to theta1/theta2 read-out


# ---------------------------------------------------------------------------
# Core mechanism: manual step-by-step RNN rollout with per-step drive injection
# ---------------------------------------------------------------------------
def manual_rollout(model, v: torch.Tensor, p0: torch.Tensor,
                   drive: torch.Tensor | None = None) -> torch.Tensor:
    """Re-implement the RNN recurrence so a drive can be added each timestep.

    Args:
        model: an ``RNN`` (weights: ``RNN.weight_ih_l0`` [Ng,vd],
            ``RNN.weight_hh_l0`` [Ng,Ng], ``encoder.weight`` [Ng,Np]).
        v:  velocity input, shape [T, B, vd] (seq-first, matching ``model.g``).
        p0: initial place-cell code, shape [B, Np].
        drive: None, a [Ng] tensor (broadcast to every t & b), or a [T, B, Ng]
            tensor of per-step injections added to the pre-activation.

    Returns:
        g: hidden states, shape [T, B, Ng]  (``g[t]`` = state after input v[t],
           with h_{-1} = encoder(p0)); identical to ``model.g((v, p0))`` when
           ``drive`` is None.
    """
    W_ih = model.RNN.weight_ih_l0
    W_hh = model.RNN.weight_hh_l0
    T = int(v.shape[0])
    h = model.encoder(p0)  # [B, Ng]
    per_step = drive is not None and drive.dim() == 3
    outs = []
    for t in range(T):
        pre = v[t] @ W_ih.t() + h @ W_hh.t()
        if drive is not None:
            pre = pre + (drive[t] if per_step else drive)
        h = torch.relu(pre)
        outs.append(h)
    return torch.stack(outs, dim=0)


def validate_rollout(model, v: torch.Tensor, p0: torch.Tensor) -> dict:
    """Confirm the drive=0 manual rollout matches ``model.g``.

    Checks in the run dtype/device (informational, float32/GPU ~1e-4) AND in
    float64 on CPU (the exact algebraic check; must be ~0).
    """
    with torch.no_grad():
        g_ref = model.g((v, p0))
        g_man = manual_rollout(model, v, p0, drive=None)
        run_diff = float((g_man - g_ref).abs().max().item())
        ref_mag = float(g_ref.abs().max().item()) + 1e-12
        run_rel = run_diff / ref_mag

        # Exact float64 CPU cross-check on a small slice.
        m64 = copy.deepcopy(model).double().cpu()
        vv = v.detach().double().cpu()[:, : min(16, v.shape[1])]
        pp = p0.detach().double().cpu()[: min(16, p0.shape[0])]
        g_ref64 = m64.g((vv, pp))
        g_man64 = manual_rollout(m64, vv, pp, drive=None)
        f64_diff = float((g_man64 - g_ref64).abs().max().item())
        del m64
    return {"run_max_abs": run_diff, "run_rel": run_rel, "float64_cpu_max_abs": f64_diff}


# ---------------------------------------------------------------------------
# Toroidal phase metrics
# ---------------------------------------------------------------------------
def _wrap(a: np.ndarray) -> np.ndarray:
    return (a + np.pi) % _TWO_PI - np.pi


def phase_metrics(theta1: np.ndarray, theta2: np.ndarray) -> dict:
    """Phase velocity and net displacement from two circular coordinates [T,B]."""
    d1 = _wrap(np.diff(theta1, axis=0))
    d2 = _wrap(np.diff(theta2, axis=0))
    speed = np.sqrt(d1 ** 2 + d2 ** 2)  # rad/step, combined circles
    uw1 = np.unwrap(theta1, axis=0)
    uw2 = np.unwrap(theta2, axis=0)
    net1 = uw1[-1] - uw1[0]
    net2 = uw2[-1] - uw2[0]
    disp = np.sqrt(net1 ** 2 + net2 ** 2)
    return {
        "phase_velocity": float(np.nanmean(speed)),
        "phase_velocity_c1": float(np.nanmean(np.abs(d1))),
        "phase_velocity_c2": float(np.nanmean(np.abs(d2))),
        "net_displacement": float(np.nanmean(disp)),
        "net_disp_c1": float(np.nanmean(np.abs(net1))),
        "net_disp_c2": float(np.nanmean(np.abs(net2))),
    }


def _project(g_np: np.ndarray, basis, T: int, B: int):
    """theta1/theta2 [T,B] from full-Ng states via the intact torus basis."""
    proj = project_states_to_torus(g_np, basis, _TORUS_RADII, T, B)
    return proj.theta1, proj.theta2


def decode_error_cm(model, g: torch.Tensor, pos: torch.Tensor) -> np.ndarray:
    """Per-(t,b) decoding error (cm) from hidden states through model's decoder."""
    with torch.no_grad():
        logits = model.decoder(g)  # [T,B,Np]
        pred = model.place_cells.get_nearest_cell_pos(logits)  # [T,B,2]
        err = torch.sqrt(((pos - pred) ** 2).sum(-1)) * 100.0
    return err.detach().cpu().numpy()


# ---------------------------------------------------------------------------
# PGC subspace / drive templates
# ---------------------------------------------------------------------------
def pgc_drive_axis(g_pred_flat: np.ndarray, pred_units: np.ndarray, Ng: int,
                   mode: str, rank: int) -> np.ndarray:
    """Unit-norm drive direction in R^Ng, supported on predictive units.

    ``standard`` -> the mean predictive activity pattern (a specific point in the
    standard-basis span of the predictive units).  ``pca`` -> the top principal
    direction of predictive-unit activity.  Both live in the PGC subspace.
    """
    if g_pred_flat.shape[0] == 0 or pred_units.size == 0:
        return np.zeros(Ng, dtype=np.float32)
    if mode == "pca":
        X = g_pred_flat - g_pred_flat.mean(axis=0, keepdims=True)
        # top right-singular vector = dominant activity mode in pred-unit space
        _, _, Vt = np.linalg.svd(X, full_matrices=False)
        axis_small = Vt[0]
    else:  # "standard": mean firing pattern of the predictive ensemble
        axis_small = g_pred_flat.mean(axis=0)
    n = np.linalg.norm(axis_small)
    if n < 1e-12:
        return np.zeros(Ng, dtype=np.float32)
    axis_small = axis_small / n
    d = np.zeros(Ng, dtype=np.float32)
    d[pred_units] = axis_small.astype(np.float32)
    return d


# ---------------------------------------------------------------------------
# Basis / unit selection helpers
# ---------------------------------------------------------------------------
def _find_npz(explicit, analysis_dir: Path, class_subdir: str, name: str) -> Path | None:
    if explicit:
        p = Path(explicit)
        return p if p.exists() else None
    p = analysis_dir / class_subdir / name
    return p if p.exists() else None


def select_basis_units(cov_npz, pred_units: np.ndarray, Ng_avail: int,
                       rate_maps: np.ndarray, box_width: float) -> tuple[np.ndarray, str]:
    """Pick the torus-basis ensemble: the dominant grid module MINUS predictive
    units (so a phase 'freeze' cannot be a trivial consequence of zeroing the
    very units we read phase from).  Falls back to any grid cells, then to
    gridness thresholding if covariates are unavailable.
    """
    pred_set = set(int(x) for x in pred_units.tolist())

    def _drop_pred_and_bound(idx):
        idx = np.array(sorted({int(i) for i in idx
                               if 0 <= int(i) < Ng_avail and int(i) not in pred_set}), dtype=int)
        return idx

    if cov_npz is not None and "module" in cov_npz and "is_grid" in cov_npz:
        module = np.asarray(cov_npz["module"]).reshape(-1)
        is_grid = np.asarray(cov_npz["is_grid"]).reshape(-1).astype(bool)
        grid_idx = np.where(is_grid)[0]
        if grid_idx.size:
            mods = module[grid_idx]
            valid = mods[mods >= 0]
            if valid.size:
                labels, counts = np.unique(valid, return_counts=True)
                dom = labels[int(np.argmax(counts))]
                cand = grid_idx[module[grid_idx] == dom]
                units = _drop_pred_and_bound(cand)
                if units.size >= 5:
                    return units, f"dominant_grid_module({int(dom)})_minus_predictive"
            units = _drop_pred_and_bound(grid_idx)
            if units.size >= 5:
                return units, "all_grid_cells_minus_predictive"

    # Fallback: gridness from rate maps.
    try:
        import pgc_fastscore as F
        gridness, _ = F.parallel_ratemap_grid_scores(
            rate_maps, box_width=box_width, box_height=box_width, res=rate_maps.shape[1],
            n_workers=0)
        cand = np.where(gridness > 0.20)[0]
        units = _drop_pred_and_bound(cand)
        if units.size >= 5:
            return units, "gridness>0.20_minus_predictive"
    except Exception:
        pass
    # Last resort: everything except predictive.
    units = _drop_pred_and_bound(np.arange(Ng_avail))
    return units, "all_available_units_minus_predictive"


# ---------------------------------------------------------------------------
# Rollout drivers over the cached eval batches
# ---------------------------------------------------------------------------
def _rollout_condition(model, batches, basis, want_pred=None, drive_list=None,
                       B_full=None):
    """Roll a model over ``batches`` (list of ((v,p0),pos)); return concatenated
    theta1/theta2, decode-error array, and optionally per-batch predictive
    activity (``want_pred`` = array of predictive indices)."""
    th1_all, th2_all, err_all = [], [], []
    pred_cache = [] if want_pred is not None else None
    with torch.no_grad():
        for bi, ((v, p0), pos) in enumerate(batches):
            drive = None if drive_list is None else drive_list[bi]
            g = manual_rollout(model, v, p0, drive=drive)  # [T,B,Ng] device
            T, B, _ = g.shape
            err_all.append(decode_error_cm(model, g, pos))
            if want_pred is not None:
                pred_cache.append(g[:, :, want_pred].detach().clone())
            g_np = g.detach().cpu().numpy()
            del g
            th1, th2 = _project(g_np, basis, T, B)
            th1_all.append(th1)
            th2_all.append(th2)
    theta1 = np.concatenate(th1_all, axis=1)
    theta2 = np.concatenate(th2_all, axis=1)
    err = np.concatenate([e.reshape(-1) for e in err_all])
    return theta1, theta2, err, pred_cache


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------
def run_intervention(checkpoint: str,
                     device: str = "cpu",
                     out_subdir: str = "pgc_rigor",
                     class_subdir: str = "pgc_rigor",
                     classification: str | None = None,
                     covariates: str | None = None,
                     n_batches: int = 8,
                     ratemap_Ng: int = 256,
                     batch_size: int = 200,
                     sequence_length: int = 40,
                     projector: str = "standard",
                     pca_r: int = 8,
                     dose_points: int = 6,
                     dose_max_mult: float = 4.0,
                     dose_B: int = 64,
                     collection_seed: int = 1234,
                     n_workers: int = 0,
                     make_plot: bool = True) -> dict:
    if n_workers and n_workers > 0:
        torch.set_num_threads(int(n_workers))

    args = C.AnalysisArgs(batch_size=batch_size, sequence_length=sequence_length)
    lm = C.load_model(checkpoint, device=device, args=args)
    dev = lm.device
    Ng = lm.Ng
    box_width = float(lm.options.box_width)

    analysis_dir = analysis_dir_for_checkpoint(Path(checkpoint))
    out_dir = analysis_dir / out_subdir
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- classification: predictive units -------------------------------
    class_path = _find_npz(classification, analysis_dir, class_subdir, "pgc_classification.npz")
    if class_path is None:
        raise FileNotFoundError(
            "pgc_classification.npz not found (looked in "
            f"{analysis_dir / class_subdir}); pass --classification.")
    cl = np.load(class_path)
    labels = np.asarray(cl["labels"]).reshape(-1)
    Ng_cls = labels.shape[0]
    pred_units = np.where(labels == C.CLASS_LABELS["predictive"])[0].astype(int)
    pred_units = pred_units[pred_units < Ng]
    if pred_units.size == 0:
        raise RuntimeError("No predictive units in classification -- nothing to intervene on.")

    cov_path = _find_npz(covariates, analysis_dir, class_subdir, "pgc_covariates.npz")
    cov_npz = np.load(cov_path) if cov_path is not None else None

    ratemap_Ng = int(min(max(ratemap_Ng, Ng_cls), Ng))

    # ---- rate maps + intact torus basis ---------------------------------
    bundle = C.collect_bundle(lm, n_batches=max(2, n_batches // 2 or 2),
                              Ng_use=ratemap_Ng, res=20,
                              collection_seed=collection_seed, with_ratemaps=True)
    basis_units, basis_desc = select_basis_units(
        cov_npz, pred_units, min(ratemap_Ng, Ng), bundle.rate_maps, box_width)
    if basis_units.size < 3:
        raise RuntimeError("Too few basis units to define a torus phase read-out.")
    basis = build_torus_basis(bundle.rate_maps, basis_units, box_width)

    # ---- fixed eval trajectories (shared across all conditions) ---------
    raw_batches = C.collect_eval_batches(lm.traj_gen, n_batches)
    batches = []
    for (v, p0), pos in raw_batches:
        batches.append(((v.to(dev), p0.to(dev)), pos.to(dev)))

    # ---- rollout-match validation (drive = 0) ---------------------------
    (v0, p00), _ = batches[0]
    rollout_check = validate_rollout(lm.model, v0, p00)

    # ---- INTACT ---------------------------------------------------------
    Whh = lm.model.RNN.weight_hh_l0.detach()
    pred_t = torch.as_tensor(pred_units, dtype=torch.long, device=dev)
    th1_i, th2_i, err_i, pred_cache = _rollout_condition(
        lm.model, batches, basis, want_pred=pred_t)
    pm_intact = phase_metrics(th1_i, th2_i)
    err_intact = float(np.mean(err_i))

    # PGC subspace drive axis + natural scale (from intact predictive activity)
    g_pred_flat = np.concatenate(
        [g.reshape(-1, pred_t.numel()).detach().cpu().numpy() for g in pred_cache], axis=0)
    d_axis = pgc_drive_axis(g_pred_flat, pred_units, Ng, projector, pca_r)
    nat_scale = float(np.mean(np.linalg.norm(g_pred_flat, axis=1)))
    if not np.isfinite(nat_scale) or nat_scale <= 0:
        nat_scale = 1.0

    # ---- ABLATED (predictive units silenced) ----------------------------
    ablated = copy.deepcopy(lm.model)
    C.zero_unit_weights_in_place(ablated, pred_units)
    ablated.eval()
    th1_a, th2_a, err_a, _ = _rollout_condition(ablated, batches, basis)
    pm_ablated = phase_metrics(th1_a, th2_a)
    err_ablated = float(np.mean(err_a))

    # ---- RESCUE (downstream recurrent replay of the PGC subspace) -------
    # The current the predictive ensemble injected into the rest of the network
    # in the INTACT rollout: drive_t = g_intact[t, pred] @ W_hh[:, pred].T .
    # (Injecting directly *into* the predictive units of the ablated network is
    #  null -- their outgoing weights W_hh[:, pred] were zeroed -- so we restore
    #  their downstream effect instead.)
    Whh_out_pred = Whh[:, pred_t]  # [Ng, n_pred]
    rescue_drives = []
    for g_pred in pred_cache:
        T, B, _ = g_pred.shape
        drive = g_pred.reshape(T * B, -1) @ Whh_out_pred.t()
        rescue_drives.append(drive.reshape(T, B, Ng))
    th1_r, th2_r, err_r, _ = _rollout_condition(
        ablated, batches, basis, drive_list=rescue_drives)
    pm_rescue = phase_metrics(th1_r, th2_r)
    err_rescue = float(np.mean(err_r))

    # ---- RESCUE-DIRECT control (inject into the subspace itself) ---------
    # Expected to be ~inert because the ablated units are disconnected downstream.
    direct_drives = []
    for g_pred in pred_cache:
        T, B, _ = g_pred.shape
        d = torch.zeros(T, B, Ng, dtype=g_pred.dtype, device=dev)
        d[:, :, pred_t] = g_pred
        direct_drives.append(d)
    th1_rd, th2_rd, err_rd, _ = _rollout_condition(
        ablated, batches, basis, drive_list=direct_drives)
    pm_rescue_direct = phase_metrics(th1_rd, th2_rd)
    err_rescue_direct = float(np.mean(err_rd))

    # ---- STIMULATION dose curve (intact, from rest, v = 0) --------------
    (v_b0, p0_b0), _ = batches[0]
    Bd = int(min(dose_B, v_b0.shape[1]))
    v_rest = torch.zeros_like(v_b0[:, :Bd])
    p0_rest = p0_b0[:Bd]
    d_axis_t = torch.as_tensor(d_axis, dtype=v_b0.dtype, device=dev)
    alpha_mults = np.linspace(0.0, float(dose_max_mult), int(dose_points))
    dose_alpha = alpha_mults * nat_scale
    dose_disp, dose_vel = [], []
    with torch.no_grad():
        for a in dose_alpha:
            drive = (float(a) * d_axis_t)  # [Ng], broadcast over t & b
            g = manual_rollout(lm.model, v_rest, p0_rest, drive=drive)
            T, B, _ = g.shape
            th1, th2 = _project(g.detach().cpu().numpy(), basis, T, B)
            pm = phase_metrics(th1, th2)
            dose_disp.append(pm["net_displacement"])
            dose_vel.append(pm["phase_velocity"])
    dose_disp = np.array(dose_disp)
    dose_vel = np.array(dose_vel)
    # drive <-> displacement correlation
    try:
        from scipy.stats import pearsonr, spearmanr
        if np.ptp(dose_disp) > 0 and dose_alpha.size > 2:
            pear_r, pear_p = pearsonr(dose_alpha, dose_disp)
            spear_r, spear_p = spearmanr(dose_alpha, dose_disp)
        else:
            pear_r = pear_p = spear_r = spear_p = float("nan")
    except Exception:
        pear_r = pear_p = spear_r = spear_p = float("nan")
    monotonic = bool(np.all(np.diff(dose_disp) >= -1e-9))

    # ---- effect sizes ---------------------------------------------------
    def _frac(recovered, ablated_v, intact_v):
        denom = ablated_v - intact_v
        return float((ablated_v - recovered) / denom) if abs(denom) > 1e-12 else float("nan")

    err_rescue_frac = _frac(err_rescue, err_ablated, err_intact)
    pv_i, pv_a, pv_r = (pm_intact["phase_velocity"], pm_ablated["phase_velocity"],
                        pm_rescue["phase_velocity"])
    pv_denom = pv_i - pv_a
    pv_rescue_frac = float((pv_r - pv_a) / pv_denom) if abs(pv_denom) > 1e-12 else float("nan")

    conditions = ["intact", "ablated", "rescued", "rescued_direct"]
    phase_velocity = np.array([pm_intact["phase_velocity"], pm_ablated["phase_velocity"],
                               pm_rescue["phase_velocity"], pm_rescue_direct["phase_velocity"]])
    decode_error = np.array([err_intact, err_ablated, err_rescue, err_rescue_direct])
    net_disp = np.array([pm_intact["net_displacement"], pm_ablated["net_displacement"],
                         pm_rescue["net_displacement"], pm_rescue_direct["net_displacement"]])

    summary = {
        "checkpoint": str(checkpoint),
        "seed_id": lm.seed_id,
        "device": dev,
        "Ng": Ng,
        "n_predictive_units": int(pred_units.size),
        "predictive_units": pred_units.tolist(),
        "basis_units_desc": basis_desc,
        "n_basis_units": int(basis_units.size),
        "projector": projector,
        "nat_pred_scale": nat_scale,
        "n_batches": n_batches,
        "batch_size": batch_size,
        "sequence_length": sequence_length,
        "rollout_match": rollout_check,
        "phase_velocity": {c: float(v) for c, v in zip(conditions, phase_velocity)},
        "decode_error_cm": {c: float(v) for c, v in zip(conditions, decode_error)},
        "net_displacement": {c: float(v) for c, v in zip(conditions, net_disp)},
        "rescue_effect": {
            "decode_error_recovered_fraction": err_rescue_frac,
            "phase_velocity_recovered_fraction": pv_rescue_frac,
            "decode_error_ablation_gap_cm": float(err_ablated - err_intact),
            "decode_error_rescue_gain_cm": float(err_ablated - err_rescue),
            "phase_velocity_freeze_ratio": float(pv_a / pv_i) if pv_i > 1e-12 else float("nan"),
        },
        "dose_curve": {
            "alpha_multiples": alpha_mults.tolist(),
            "alpha": dose_alpha.tolist(),
            "displacement": dose_disp.tolist(),
            "phase_velocity": dose_vel.tolist(),
            "pearson_r": float(pear_r),
            "pearson_p": float(pear_p),
            "spearman_r": float(spear_r),
            "spearman_p": float(spear_p),
            "monotonic": monotonic,
        },
    }

    # ---- persist --------------------------------------------------------
    npz_path = out_dir / "pgc_intervention.npz"
    np.savez(
        npz_path,
        conditions=np.array(conditions),
        phase_velocity=phase_velocity,
        phase_velocity_c1=np.array([pm_intact["phase_velocity_c1"], pm_ablated["phase_velocity_c1"],
                                    pm_rescue["phase_velocity_c1"], pm_rescue_direct["phase_velocity_c1"]]),
        decode_error_cm=decode_error,
        net_displacement=net_disp,
        dose_alpha=dose_alpha,
        dose_alpha_mult=alpha_mults,
        dose_displacement=dose_disp,
        dose_phase_velocity=dose_vel,
        predictive_units=pred_units,
        basis_units=basis_units,
        drive_axis=d_axis,
        labels=labels,
        pearson_r=np.array([pear_r]),
        spearman_r=np.array([spear_r]),
        rescue_decode_frac=np.array([err_rescue_frac]),
        rescue_pv_frac=np.array([pv_rescue_frac]),
        rollout_run_max_abs=np.array([rollout_check["run_max_abs"]]),
        rollout_float64_max_abs=np.array([rollout_check["float64_cpu_max_abs"]]),
    )
    json_path = out_dir / "pgc_intervention_summary.json"
    with open(json_path, "w") as fh:
        json.dump(summary, fh, indent=2)

    png_path = None
    if make_plot:
        try:
            png_path = _make_plot(out_dir, conditions, phase_velocity, decode_error,
                                  dose_alpha, dose_disp, summary)
        except Exception as exc:  # plotting must never sink the experiment
            print(f"[warn] plot failed: {exc}")

    summary["outputs"] = {
        "npz": str(npz_path), "json": str(json_path),
        "png": str(png_path) if png_path else None,
    }
    return summary


def _make_plot(out_dir, conditions, phase_velocity, decode_error,
               dose_alpha, dose_disp, summary):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    colors = {"intact": "#4c72b0", "ablated": "#c44e52",
              "rescued": "#55a868", "rescued_direct": "#8c8c8c"}
    cvec = [colors.get(c, "#333") for c in conditions]

    axes[0].bar(conditions, phase_velocity, color=cvec)
    axes[0].set_ylabel("phase velocity (rad/step)")
    axes[0].set_title("Toroidal phase velocity")
    axes[0].tick_params(axis="x", rotation=20)

    axes[1].bar(conditions, decode_error, color=cvec)
    axes[1].set_ylabel("decoding error (cm)")
    axes[1].set_title("Position decoding error")
    axes[1].tick_params(axis="x", rotation=20)

    axes[2].plot(dose_alpha, dose_disp, "o-", color="#4c72b0")
    axes[2].set_xlabel("commanded PGC drive  |alpha|")
    axes[2].set_ylabel("net phase displacement (rad)")
    dc = summary["dose_curve"]
    axes[2].set_title(f"Stimulation dose curve\nPearson r={dc['pearson_r']:.3f}")

    rf = summary["rescue_effect"]["decode_error_recovered_fraction"]
    fig.suptitle(f"PGC causal intervention (seed {summary['seed_id']})  |  "
                 f"decode-error rescued: {rf*100:.0f}%", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    png_path = out_dir / "pgc_intervention.png"
    fig.savefig(png_path, dpi=130)
    plt.close(fig)
    return png_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--device", default="cpu")
    p.add_argument("--out_subdir", default="pgc_rigor")
    p.add_argument("--class_subdir", default="pgc_rigor",
                   help="subdir (under the analysis dir) holding pgc_classification.npz")
    p.add_argument("--classification", default=None, help="explicit path to pgc_classification.npz")
    p.add_argument("--covariates", default=None, help="explicit path to pgc_covariates.npz")
    p.add_argument("--n_batches", type=int, default=8)
    p.add_argument("--ratemap_Ng", type=int, default=256)
    p.add_argument("--batch_size", type=int, default=200)
    p.add_argument("--sequence_length", type=int, default=40)
    p.add_argument("--projector", choices=["standard", "pca"], default="standard")
    p.add_argument("--pca_r", type=int, default=8)
    p.add_argument("--dose_points", type=int, default=6)
    p.add_argument("--dose_max_mult", type=float, default=4.0)
    p.add_argument("--dose_B", type=int, default=64)
    p.add_argument("--collection_seed", type=int, default=1234)
    p.add_argument("--n_workers", type=int, default=0)
    p.add_argument("--no_plot", action="store_true")
    return p


def main(argv=None):
    a = _build_argparser().parse_args(argv)
    summary = run_intervention(
        checkpoint=a.checkpoint, device=a.device, out_subdir=a.out_subdir,
        class_subdir=a.class_subdir, classification=a.classification,
        covariates=a.covariates, n_batches=a.n_batches, ratemap_Ng=a.ratemap_Ng,
        batch_size=a.batch_size, sequence_length=a.sequence_length,
        projector=a.projector, pca_r=a.pca_r, dose_points=a.dose_points,
        dose_max_mult=a.dose_max_mult, dose_B=a.dose_B,
        collection_seed=a.collection_seed, n_workers=a.n_workers,
        make_plot=not a.no_plot)

    rc = summary["rollout_match"]
    print(f"[rollout] drive=0 match: run(float32) max_abs={rc['run_max_abs']:.2e} "
          f"(rel={rc['run_rel']:.2e}); float64/CPU max_abs={rc['float64_cpu_max_abs']:.2e}")
    pv = summary["phase_velocity"]
    de = summary["decode_error_cm"]
    print(f"[phase velocity rad/step] intact={pv['intact']:.4f} ablated={pv['ablated']:.4f} "
          f"rescued={pv['rescued']:.4f} rescued_direct={pv['rescued_direct']:.4f}")
    print(f"[decode error cm]        intact={de['intact']:.2f} ablated={de['ablated']:.2f} "
          f"rescued={de['rescued']:.2f} rescued_direct={de['rescued_direct']:.2f}")
    re = summary["rescue_effect"]
    print(f"[rescue] decode-error recovered fraction = {re['decode_error_recovered_fraction']:.3f}  "
          f"(gap {re['decode_error_ablation_gap_cm']:.2f} cm, gain {re['decode_error_rescue_gain_cm']:.2f} cm); "
          f"phase-velocity recovered fraction = {re['phase_velocity_recovered_fraction']:.3f}; "
          f"freeze ratio pv_ablated/pv_intact = {re['phase_velocity_freeze_ratio']:.3f}")
    dc = summary["dose_curve"]
    print(f"[dose] alpha={['%.3f'%x for x in dc['alpha']]}")
    print(f"[dose] displacement={['%.3f'%x for x in dc['displacement']]} "
          f"pearson_r={dc['pearson_r']:.3f} spearman_r={dc['spearman_r']:.3f} monotonic={dc['monotonic']}")
    print(f"[outputs] {summary['outputs']}")
    return summary


if __name__ == "__main__":
    main()
