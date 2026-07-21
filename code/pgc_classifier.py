"""Defensible PGC classifier.

Upgrades the old argmax-of-gridness classifier (which piled optima at the +/-20 cm
sweep edge and had no significance test) with four reviewer-grade controls:

  1. WIDER SHIFT WINDOW.  The sweep reaches +/-max_lag steps (~+/-50 cm at the
     ~2 cm/step of these trajectories, vs the old +/-20 cm), so a unit's true
     preferred shift is interior to the sweep; units whose optimum still sits at
     the terminal shift are flagged (``edge_flag``) and excluded.

  2. BLOCK / CIRCULAR SHUFFLE SIGNIFICANCE.  For each candidate unit a null
     distribution of the best-shift gridness is built by circularly rolling its
     activity in time per trajectory (preserving each unit's own temporal
     autocorrelation but destroying the activity<->position pairing) and
     recomputing the max-over-lags gridness. A unit is only called a grid cell
     if its observed best-shift gridness beats the null at ``alpha``.

  3. HELD-OUT DIRECTION CONFIRMATION.  Units are classified (label + preferred
     shift sign) on trajectory set A; the sign must reproduce on a DISJOINT
     held-out set B with directional gridness on B also clearing the floor.

  4. One classifier, replacing the two divergent copies.

Scoring is fanned across CPU cores via pgc_fastscore (the exact GridScorer),
which makes the shuffle null tractable. Output: ``pgc_classification.npz`` +
summary JSON under the checkpoint's analysis dir.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Tuple

import numpy as np

_CODE_DIR = os.path.dirname(os.path.abspath(__file__))
if _CODE_DIR not in sys.path:
    sys.path.insert(0, _CODE_DIR)

from shift_utils import build_shift_values  # noqa: E402
from path_utils import analysis_dir_for_checkpoint  # noqa: E402
import pgc_common as C  # noqa: E402
import pgc_fastscore as F  # noqa: E402


@dataclass
class ClassifierConfig:
    shift_mode: str = "time"          # 'time' (fast) or 'space' (arc-length)
    max_lag: int = 25                 # time mode: +/- steps  (~+/-50 cm)
    lag_step: int = 1
    max_shift_cm: float = 50.0        # space mode window
    shift_step_cm: float = 2.0
    space_projection: str = "path"
    min_shift_cm: float = 5.0
    gridness_floor: float = 0.10      # prescreen: only null-test units above this
    alpha: float = 0.05
    n_shuffles: int = 100
    null_block: bool = False          # False=circular roll, True=block permutation
    res: int = 20
    n_batches: int = 20
    Ng_use: int = 512
    classify_seed: int = 1234
    confirm_seed: int = 9876
    heldout_confirm: bool = True
    n_workers: int = 0                # 0 -> auto


def _build_lags(cfg: ClassifierConfig):
    if cfg.shift_mode == "time":
        return np.asarray(build_shift_values("time", max_lag=cfg.max_lag,
                                             lag_step=cfg.lag_step), dtype=float)
    return np.asarray(build_shift_values("space", max_shift_cm=cfg.max_shift_cm,
                                         shift_step_cm=cfg.shift_step_cm), dtype=float)


def _best_over_lags(scores_60: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    L, Ng = scores_60.shape
    best_idx = np.full(Ng, -1, dtype=int)
    best_val = np.full(Ng, np.nan, dtype=float)
    for u in range(Ng):
        col = scores_60[:, u]
        if np.isfinite(col).any():
            i = int(np.nanargmax(col))
            best_idx[u] = i
            best_val[u] = col[i]
    return best_idx, best_val


def classify_checkpoint(lm: C.LoadedModel, cfg: ClassifierConfig,
                        verbose: bool = True) -> dict:
    t0 = time.time()
    periodic = bool(getattr(lm.options, "periodic", False))
    bw, bh = lm.options.box_width, lm.options.box_height
    nw = cfg.n_workers or None
    lags = _build_lags(cfg)
    L = lags.size

    def score(bundle):
        return F.parallel_predictive_scores(
            bundle.xs, bundle.ys, bundle.activations, lags,
            shift_mode=cfg.shift_mode, space_projection=cfg.space_projection,
            periodic=periodic, box_width=bw, box_height=bh, res=cfg.res, n_workers=nw)

    # ---- Set A: classification ------------------------------------------
    bundle_a = C.collect_bundle(lm, n_batches=cfg.n_batches, Ng_use=cfg.Ng_use,
                                res=cfg.res, collection_seed=cfg.classify_seed,
                                with_ratemaps=False)
    cmps = C.cm_per_step(bundle_a.xs, bundle_a.ys)
    lag_cm = lags * cmps if cfg.shift_mode == "time" else lags.copy()
    s60_a = score(bundle_a)
    Ng = s60_a.shape[1]
    best_idx, best_val = _best_over_lags(s60_a)
    best_cm = np.where(best_idx >= 0, lag_cm[np.clip(best_idx, 0, L - 1)], np.nan)
    edge_flag = (best_idx == 0) | (best_idx == L - 1)
    if verbose:
        print(f"[classify] setA scored Ng={Ng} L={L} cm/step={cmps:.2f} "
              f"window=+/-{lag_cm.max():.0f}cm ({time.time()-t0:.1f}s)", flush=True)

    # ---- Block/circular shuffle null (candidates only) -------------------
    cand = np.where(np.isfinite(best_val) & (best_val >= cfg.gridness_floor))[0]
    pval = np.ones(Ng, dtype=float)
    if cand.size > 0:
        null_max = F.parallel_null_maxscores(
            bundle_a.xs, bundle_a.ys, bundle_a.activations[:, :, cand], lags,
            cfg.n_shuffles, shift_mode=cfg.shift_mode,
            space_projection=cfg.space_projection, periodic=periodic,
            box_width=bw, box_height=bh, res=cfg.res,
            base_seed=cfg.classify_seed + 7, block=cfg.null_block, n_workers=nw)
        obs = best_val[cand]
        ge = np.sum(null_max >= obs[None, :], axis=0)
        pval[cand] = (1.0 + ge) / (1.0 + cfg.n_shuffles)
    if verbose:
        print(f"[classify] null done {cand.size} cand x {cfg.n_shuffles} "
              f"shuffles ({time.time()-t0:.1f}s)", flush=True)

    # ---- Held-out confirmation (set B) ----------------------------------
    best_cm_b = np.full(Ng, np.nan, dtype=float)
    if cfg.heldout_confirm:
        bundle_b = C.collect_bundle(lm, n_batches=cfg.n_batches, Ng_use=cfg.Ng_use,
                                    res=cfg.res, collection_seed=cfg.confirm_seed,
                                    with_ratemaps=False)
        s60_b = score(bundle_b)
        bidx_b, _ = _best_over_lags(s60_b)
        best_cm_b = np.where(bidx_b >= 0, lag_cm[np.clip(bidx_b, 0, L - 1)], np.nan)
        same_sign = np.sign(best_cm_b) == np.sign(best_cm)
        pos_side, neg_side = lag_cm > 0, lag_cm < 0
        zero_side = np.abs(lag_cm) <= max(cfg.shift_step_cm, cmps)
        dir_ok = np.zeros(Ng, dtype=bool)
        for u in range(Ng):
            if not np.isfinite(best_cm[u]):
                continue
            side = pos_side if best_cm[u] > 0 else (neg_side if best_cm[u] < 0 else zero_side)
            col = s60_b[side, u]
            if np.isfinite(col).any() and np.nanmax(col) >= cfg.gridness_floor:
                dir_ok[u] = True
        heldout_confirmed = same_sign & dir_ok
        if verbose:
            print(f"[classify] held-out confirm done ({time.time()-t0:.1f}s)", flush=True)
    else:
        heldout_confirmed = np.ones(Ng, dtype=bool)

    # ---- Final labels ----------------------------------------------------
    is_grid = np.isfinite(best_val) & (pval < cfg.alpha) & (~edge_flag)
    labels = np.full(Ng, C.CLASS_LABELS["non_grid"], dtype=int)
    for u in np.where(is_grid)[0]:
        if best_cm[u] >= cfg.min_shift_cm:
            lab = "predictive"
        elif best_cm[u] <= -cfg.min_shift_cm:
            lab = "retrospective"
        else:
            lab = "normal"
        if lab in ("predictive", "retrospective") and not heldout_confirmed[u]:
            lab = "normal"
        labels[u] = C.CLASS_LABELS[lab]

    counts = {name: int(np.sum(labels == val)) for name, val in C.CLASS_LABELS.items()}
    summary = {
        "checkpoint": lm.checkpoint_path, "seed_id": lm.seed_id,
        "Ng_evaluated": int(Ng), "cm_per_step": round(cmps, 3),
        "window_cm": float(lag_cm.max()), "config": asdict(cfg), "counts": counts,
        "n_candidates_null_tested": int(cand.size),
        "n_edge_excluded": int(np.sum(edge_flag & np.isfinite(best_val)
                                      & (best_val >= cfg.gridness_floor))),
        "median_predictive_shift_cm": (float(np.nanmedian(best_cm[labels == C.CLASS_LABELS["predictive"]]))
                                       if counts["predictive"] else None),
        "median_retrospective_shift_cm": (float(np.nanmedian(best_cm[labels == C.CLASS_LABELS["retrospective"]]))
                                          if counts["retrospective"] else None),
        "runtime_s": round(time.time() - t0, 1),
    }
    return {
        "labels": labels, "best_cm": best_cm, "best_gridness": best_val,
        "pval": pval, "edge_flag": edge_flag, "heldout_confirmed": heldout_confirmed,
        "best_cm_heldout": best_cm_b, "lag_cm": lag_cm, "scores_60": s60_a,
        "summary": summary,
    }


def save_classification(result: dict, out_dir: str) -> str:
    os.makedirs(out_dir, exist_ok=True)
    npz_path = os.path.join(out_dir, "pgc_classification.npz")
    np.savez_compressed(
        npz_path, labels=result["labels"], best_cm=result["best_cm"],
        best_gridness=result["best_gridness"], pval=result["pval"],
        edge_flag=result["edge_flag"], heldout_confirmed=result["heldout_confirmed"],
        best_cm_heldout=result["best_cm_heldout"], lag_cm=result["lag_cm"],
        scores_60=result["scores_60"])
    with open(os.path.join(out_dir, "pgc_classification_summary.json"), "w") as f:
        json.dump(result["summary"], f, indent=2)
    return npz_path


def main():
    p = argparse.ArgumentParser(description="Defensible PGC classifier")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--device", default="cpu")
    p.add_argument("--out_subdir", default="pgc_rigor")
    p.add_argument("--shift_mode", default="time", choices=["time", "space"])
    p.add_argument("--max_lag", type=int, default=25)
    p.add_argument("--max_shift_cm", type=float, default=50.0)
    p.add_argument("--shift_step_cm", type=float, default=2.0)
    p.add_argument("--min_shift_cm", type=float, default=5.0)
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--gridness_floor", type=float, default=0.10,
                   help="prescreen: only null-test units with best gridness above this")
    p.add_argument("--n_shuffles", type=int, default=100)
    p.add_argument("--null_block", action="store_true")
    p.add_argument("--Ng_use", type=int, default=512)
    p.add_argument("--n_batches", type=int, default=20)
    p.add_argument("--n_workers", type=int, default=0)
    p.add_argument("--no_heldout", action="store_true")
    args = p.parse_args()

    lm = C.load_model(args.checkpoint, device=args.device)
    cfg = ClassifierConfig(
        shift_mode=args.shift_mode, max_lag=args.max_lag, max_shift_cm=args.max_shift_cm,
        shift_step_cm=args.shift_step_cm, min_shift_cm=args.min_shift_cm, alpha=args.alpha,
        gridness_floor=args.gridness_floor,
        n_shuffles=args.n_shuffles, null_block=args.null_block, Ng_use=args.Ng_use,
        n_batches=args.n_batches, heldout_confirm=not args.no_heldout, n_workers=args.n_workers)
    result = classify_checkpoint(lm, cfg)
    out_dir = str(analysis_dir_for_checkpoint(Path(args.checkpoint)) / args.out_subdir)
    npz = save_classification(result, out_dir)
    print(json.dumps(result["summary"], indent=2))
    print(f"saved -> {npz}")


if __name__ == "__main__":
    main()
