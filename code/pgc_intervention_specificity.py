#!/usr/bin/env python3
"""SKEPTICAL specificity controls for the PGC 'rescue' claim.

The flagship claim (code/pgc_intervention.py) is that replaying the predictive
ensemble's downstream recurrent current
    drive_t = g_intact[t, pred] @ W_hh[:, pred].T
into a pred-ablated network rescues decoding, and that this is PGC-SPECIFIC.

The module's own 'rescued_direct' control only shows the INJECTION SITE matters
(injecting into the disconnected pred units does nothing). It does NOT test
whether the injected SIGNAL has to be the correct predictive computation.

This script adds the missing specificity controls, per seed:
  (a) ablated          -- no drive (baseline high error)
  (b) TRUE rescue      -- g_intact[t,pred] @ W_hh[:,pred].T
  (c) SCRAMBLED replay -- TIME-shuffle g_intact[:,pred] per trajectory, THEN
                          @ W_hh[:,pred].T  (same units, same weights, same
                          per-vector magnitude, destroyed timing)
  (d) RANDOM subspace  -- g_intact[t,rand] @ W_hh[:,rand].T for a random set of
                          n_pred OTHER grid units (equal count, different subspace)
  (e) intact           -- floor

Decision rule: the claim is PGC-specific only if TRUE rescue (b) recovers
substantially MORE of the ablation gap than BOTH (c) and (d).

Reuses code/pgc_intervention.py internals (manual_rollout, decode_error_cm,
select_basis_units) and pgc_common (load_model, collect_bundle, ablation).
"""
from __future__ import annotations

import os
import sys
import json
import copy
from pathlib import Path

os.environ.setdefault("NUMBA_CACHE_DIR", str(Path(".numba_cache").resolve()))
os.environ.setdefault("MPLCONFIGDIR", str(Path(".mplconfig").resolve()))
for _d in (os.environ["NUMBA_CACHE_DIR"], os.environ["MPLCONFIGDIR"]):
    Path(_d).mkdir(parents=True, exist_ok=True)

import numpy as np
import torch

_CODE_DIR = os.path.dirname(os.path.abspath(__file__))
if _CODE_DIR not in sys.path:
    sys.path.insert(0, _CODE_DIR)

import pgc_common as C  # noqa: E402
import pgc_intervention as PI  # noqa: E402
from path_utils import analysis_dir_for_checkpoint  # noqa: E402


def _mean_err(err_list):
    return float(np.mean(np.concatenate([e.reshape(-1) for e in err_list])))


def _drive_from_units(full_g_cpu, unit_idx, Whh, dev):
    """Build per-batch drive [T,B,Ng] = g_intact[:,:,unit_idx] @ Whh[:,unit_idx].T"""
    u_cpu = torch.as_tensor(np.asarray(unit_idx, dtype=np.int64))  # index CPU cache
    u_t = u_cpu.to(dev)
    Whh_out = Whh[:, u_t]  # [Ng, k]
    Ng = Whh.shape[0]
    drives = []
    for g in full_g_cpu:
        T, B, _ = g.shape
        g_sub = g[:, :, u_cpu].to(dev)  # [T,B,k]
        d = (g_sub.reshape(T * B, -1) @ Whh_out.t()).reshape(T, B, Ng)
        drives.append(d)
    return drives


def _drive_scrambled(full_g_cpu, pred_idx, Whh, dev, rng):
    """TIME-shuffle g_intact[:,:,pred] per trajectory, then @ Whh[:,pred].T."""
    u_cpu = torch.as_tensor(np.asarray(pred_idx, dtype=np.int64))
    u_t = u_cpu.to(dev)
    Whh_out = Whh[:, u_t]
    Ng = Whh.shape[0]
    drives = []
    for g in full_g_cpu:
        T, B, _ = g.shape
        g_sub = g[:, :, u_cpu].to(dev).clone()  # [T,B,k]
        # independent time permutation per trajectory b
        for b in range(B):
            perm = torch.as_tensor(rng.permutation(T), device=dev)
            g_sub[:, b, :] = g_sub[perm, b, :]
        d = (g_sub.reshape(T * B, -1) @ Whh_out.t()).reshape(T, B, Ng)
        drives.append(d)
    return drives


def _rollout_decode(model, batches, drive_list):
    err = []
    with torch.no_grad():
        for bi, ((v, p0), pos) in enumerate(batches):
            drive = None if drive_list is None else drive_list[bi]
            g = PI.manual_rollout(model, v, p0, drive=drive)
            err.append(PI.decode_error_cm(model, g, pos))
            del g
    return _mean_err(err)


def _drive_rms(drive_list):
    """Mean over (t,b) of ||drive_{t,b}||_2 -- magnitude of the injected current."""
    vals = []
    for d in drive_list:
        T, B, Ng = d.shape
        n = d.reshape(T * B, Ng).norm(dim=1)
        vals.append(n.mean().item())
    return float(np.mean(vals))


def run_seed(checkpoint, device="cuda", n_batches=8, n_shuffle=5, n_random=5,
             rng_seed=0):
    args = C.AnalysisArgs(batch_size=200, sequence_length=40)
    lm = C.load_model(checkpoint, device=device, args=args)
    dev = lm.device
    Ng = lm.Ng
    box_width = float(lm.options.box_width)
    rng = np.random.default_rng(rng_seed)

    analysis_dir = analysis_dir_for_checkpoint(Path(checkpoint))
    rigor = analysis_dir / "pgc_rigor"
    cl = np.load(rigor / "pgc_classification.npz")
    labels = np.asarray(cl["labels"]).reshape(-1)
    Ng_cls = labels.shape[0]
    pred_units = np.where(labels == C.CLASS_LABELS["predictive"])[0].astype(int)
    pred_units = pred_units[pred_units < Ng]
    n_pred = int(pred_units.size)

    cov = np.load(rigor / "pgc_covariates.npz")
    ratemap_Ng = int(min(max(256, Ng_cls), Ng))

    bundle = C.collect_bundle(lm, n_batches=max(2, n_batches // 2 or 2),
                              Ng_use=ratemap_Ng, res=20,
                              collection_seed=1234, with_ratemaps=True)
    basis_units, basis_desc = PI.select_basis_units(
        cov, pred_units, min(ratemap_Ng, Ng), bundle.rate_maps, box_width)

    # fixed eval batches (seeded for reproducibility; shared across all conditions)
    torch.manual_seed(rng_seed)
    np.random.seed(rng_seed)
    raw = C.collect_eval_batches(lm.traj_gen, n_batches)
    batches = [((v.to(dev), p0.to(dev)), pos.to(dev)) for (v, p0), pos in raw]

    Whh = lm.model.RNN.weight_hh_l0.detach()

    # ---- INTACT: cache full hidden states (CPU) for building any subspace drive
    intact_err, full_g_cpu = [], []
    with torch.no_grad():
        for (v, p0), pos in batches:
            g = PI.manual_rollout(lm.model, v, p0, drive=None)
            intact_err.append(PI.decode_error_cm(lm.model, g, pos))
            full_g_cpu.append(g.detach().to("cpu"))
            del g
    err_intact = _mean_err(intact_err)

    # ---- ABLATED (predictive units silenced) ----
    ablated = copy.deepcopy(lm.model)
    C.zero_unit_weights_in_place(ablated, pred_units)
    ablated.eval()
    err_ablated = _rollout_decode(ablated, batches, None)

    gap = err_ablated - err_intact

    def frac(err_cond):
        return float((err_ablated - err_cond) / gap) if abs(gap) > 1e-9 else float("nan")

    # ---- (b) TRUE rescue ----
    true_drives = _drive_from_units(full_g_cpu, pred_units, Whh, dev)
    err_true = _rollout_decode(ablated, batches, true_drives)
    rms_true = _drive_rms(true_drives)

    # ---- (c) SCRAMBLED replay (n_shuffle draws) ----
    scr_errs, scr_rms = [], []
    for _ in range(n_shuffle):
        d = _drive_scrambled(full_g_cpu, pred_units, Whh, dev, rng)
        scr_errs.append(_rollout_decode(ablated, batches, d))
        scr_rms.append(_drive_rms(d))
        del d

    # ---- (d) RANDOM subspace (n_random draws of n_pred OTHER grid units) ----
    is_grid = np.asarray(cov["is_grid"]).reshape(-1).astype(bool)
    grid_all = np.where(is_grid)[0]
    grid_all = grid_all[(grid_all < Ng)]
    pred_set = set(int(x) for x in pred_units.tolist())
    grid_pool = np.array([g for g in grid_all if int(g) not in pred_set], dtype=int)
    rnd_errs, rnd_rms = [], []
    rndm_errs, rndm_rms = [], []  # magnitude-matched to TRUE rescue
    used_random = []
    can_random = grid_pool.size >= n_pred
    if can_random:
        for _ in range(n_random):
            rand_units = rng.choice(grid_pool, size=n_pred, replace=False)
            used_random.append(sorted(int(x) for x in rand_units.tolist()))
            d = _drive_from_units(full_g_cpu, rand_units, Whh, dev)
            rms_d = _drive_rms(d)
            rnd_errs.append(_rollout_decode(ablated, batches, d))
            rnd_rms.append(rms_d)
            # magnitude-matched variant: rescale so mean RMS == TRUE rescue's RMS
            scale = (rms_true / rms_d) if rms_d > 1e-12 else 1.0
            dm = [x * scale for x in d]
            rndm_errs.append(_rollout_decode(ablated, batches, dm))
            rndm_rms.append(_drive_rms(dm))
            del d, dm

    out = {
        "checkpoint": str(checkpoint),
        "seed_id": lm.seed_id,
        "device": dev,
        "n_predictive_units": n_pred,
        "grid_pool_size": int(grid_pool.size),
        "basis_units_desc": basis_desc,
        "n_batches": n_batches,
        "err_intact": err_intact,
        "err_ablated": err_ablated,
        "ablation_gap_cm": gap,
        "true_rescue": {"err": err_true, "frac": frac(err_true), "drive_rms": rms_true},
        "scrambled": {
            "err_mean": float(np.mean(scr_errs)), "err_std": float(np.std(scr_errs)),
            "frac_mean": float(np.mean([frac(e) for e in scr_errs])),
            "errs": [float(e) for e in scr_errs],
            "drive_rms_mean": float(np.mean(scr_rms)),
        },
        "random_subspace": ({
            "err_mean": float(np.mean(rnd_errs)), "err_std": float(np.std(rnd_errs)),
            "frac_mean": float(np.mean([frac(e) for e in rnd_errs])),
            "errs": [float(e) for e in rnd_errs],
            "drive_rms_mean": float(np.mean(rnd_rms)),
        } if can_random else {"error": "grid_pool too small for n_pred draws"}),
        "random_subspace_magmatched": ({
            "err_mean": float(np.mean(rndm_errs)), "err_std": float(np.std(rndm_errs)),
            "frac_mean": float(np.mean([frac(e) for e in rndm_errs])),
            "errs": [float(e) for e in rndm_errs],
            "drive_rms_mean": float(np.mean(rndm_rms)),
        } if can_random else {"error": "grid_pool too small for n_pred draws"}),
    }
    del full_g_cpu
    return out


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, nargs="+", required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--n_batches", type=int, default=8)
    p.add_argument("--n_shuffle", type=int, default=5)
    p.add_argument("--n_random", type=int, default=5)
    p.add_argument("--out", default=None)
    a = p.parse_args()

    run = "steps_20_batch_200_RNN_4096_relu_rf_012_DoG_True_periodic_False_lr_00001_weight_decay_1e-06"
    results = []
    for s in a.seeds:
        ckpt = f"Models/canonical_cohort_v1/seed_{s}/{run}/most_recent_model.pth"
        print(f"\n===== seed_{s} =====", flush=True)
        r = run_seed(ckpt, device=a.device, n_batches=a.n_batches,
                     n_shuffle=a.n_shuffle, n_random=a.n_random)
        results.append(r)
        tr, sc, rd = r["true_rescue"], r["scrambled"], r["random_subspace"]
        print(f"  intact={r['err_intact']:.3f}  ablated={r['err_ablated']:.3f}  "
              f"gap={r['ablation_gap_cm']:.3f} cm  (n_pred={r['n_predictive_units']})", flush=True)
        print(f"  (b) TRUE  rescue : err={tr['err']:.3f}  frac={tr['frac']:.3f}  "
              f"rms={tr['drive_rms']:.3f}", flush=True)
        print(f"  (c) SCRAMBLED    : err={sc['err_mean']:.3f}+/-{sc['err_std']:.3f}  "
              f"frac={sc['frac_mean']:.3f}  rms={sc['drive_rms_mean']:.3f}", flush=True)
        rdm = r["random_subspace_magmatched"]
        if "err_mean" in rd:
            print(f"  (d) RANDOM subsp : err={rd['err_mean']:.3f}+/-{rd['err_std']:.3f}  "
                  f"frac={rd['frac_mean']:.3f}  rms={rd['drive_rms_mean']:.3f}", flush=True)
            print(f"  (d') RANDOM mag= : err={rdm['err_mean']:.3f}+/-{rdm['err_std']:.3f}  "
                  f"frac={rdm['frac_mean']:.3f}  rms={rdm['drive_rms_mean']:.3f}  "
                  f"(matched to TRUE rms={tr['drive_rms']:.3f})", flush=True)
        else:
            print(f"  (d) RANDOM subsp : {rd['error']}", flush=True)

    if a.out:
        with open(a.out, "w") as fh:
            json.dump(results, fh, indent=2)
        print(f"\n[saved] {a.out}")


if __name__ == "__main__":
    main()
