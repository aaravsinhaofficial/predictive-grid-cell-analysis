"""Per-unit covariate assembler for property-matched ablation controls.

Assembles, from ONE seeded activation collection (so every covariate shares the
same trajectories and unit indexing 0..Ng_use-1), the properties on which a
control unit must match an ablated PGC:

    module (grid-spacing cluster), gridness, bandness, firing-rate variance,
    head-direction (movement-direction) tuning, decoder-weight magnitude,
    incoming recurrent strength, outgoing recurrent strength.

Also stores the continuous grid spacing (meters) so a matcher can match on
spacing directly rather than a discretized module label. Output:
``pgc_covariates.npz`` (keys = pgc_common.COVARIATE_KEYS + spacing_m).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

_CODE_DIR = os.path.dirname(os.path.abspath(__file__))
if _CODE_DIR not in sys.path:
    sys.path.insert(0, _CODE_DIR)

from scores import band_scores  # noqa: E402
from path_utils import analysis_dir_for_checkpoint  # noqa: E402
import pgc_common as C  # noqa: E402
import pgc_fastscore as F  # noqa: E402


def _hd_rayleigh_all(xs: np.ndarray, ys: np.ndarray, activations: np.ndarray) -> np.ndarray:
    """Vectorized head-direction (movement-direction) Rayleigh r for all units.

    Matches plotting_functional_classes.head_direction_tuning's r but computed
    for every unit at once. r_u = |sum_t a_{t,u} e^{i theta_t}| / sum_t |a_{t,u}|.
    """
    dx = np.diff(xs, axis=0)
    dy = np.diff(ys, axis=0)
    theta = np.arctan2(dy, dx)              # [T-1, B]
    a = activations[:-1]                    # [T-1, B, Ng]
    e = np.exp(1j * theta)[:, :, None]      # [T-1, B, 1]
    num = np.abs(np.sum(a * e, axis=(0, 1)))       # [Ng]
    den = np.sum(np.abs(a), axis=(0, 1)) + 1e-8    # [Ng]
    return num / den


def _cluster_modules(spacing_m: np.ndarray, is_grid: np.ndarray, max_k: int = 3) -> np.ndarray:
    """1D clustering of grid-unit spacings into modules (label -1 for non-grid)."""
    labels = np.full(spacing_m.shape[0], -1, dtype=int)
    idx = np.where(is_grid & np.isfinite(spacing_m))[0]
    if idx.size < 2:
        labels[idx] = 0
        return labels
    x = np.log(spacing_m[idx]).reshape(-1, 1)
    try:
        from sklearn.cluster import KMeans
        from sklearn.metrics import silhouette_score
        best_k, best_s, best_lab = 1, -1.0, np.zeros(idx.size, dtype=int)
        for k in range(1, min(max_k, idx.size) + 1):
            km = KMeans(n_clusters=k, n_init=5, random_state=0).fit(x)
            if k == 1:
                s = 0.0
            else:
                s = silhouette_score(x, km.labels_)
            if s > best_s:
                best_k, best_s, best_lab = k, s, km.labels_
        # order modules by ascending mean spacing for stable labels
        order = np.argsort([x[best_lab == c].mean() for c in range(best_k)])
        remap = {c: r for r, c in enumerate(order)}
        labels[idx] = np.array([remap[c] for c in best_lab])
    except Exception:
        labels[idx] = 0
    return labels


def assemble_covariates(lm: C.LoadedModel, Ng_use: int = 512, n_batches: int = 20,
                        res: int = 20, collection_seed: int = 4321,
                        grid_floor: float = 0.20, n_workers: int = 0) -> dict:
    nw = n_workers or None
    bundle = C.collect_bundle(lm, n_batches=n_batches, Ng_use=Ng_use, res=res,
                              collection_seed=collection_seed, with_ratemaps=True)
    Ng = bundle.rate_maps.shape[0]
    bw, bh = lm.options.box_width, lm.options.box_height

    # gridness + first-ring SAC radius (parallel)
    gridness, sac_ring = F.parallel_ratemap_grid_scores(
        bundle.rate_maps, box_width=bw, box_height=bh, res=res, n_workers=nw)
    spacing_m = sac_ring * (bw / res)  # SAC bin == rate-map bin == bw/res meters

    # bandness
    bandness, best_kx, best_ky = band_scores(bundle.rate_maps, res, bw)
    bandness = np.asarray(bandness, dtype=float)

    # firing-rate variance (spatial)
    rate_var = np.nanvar(bundle.rate_maps.reshape(Ng, -1), axis=1)

    # head-direction tuning (vectorized Rayleigh r)
    hd_r = _hd_rayleigh_all(bundle.xs, bundle.ys, bundle.activations)

    # weight-based covariates (intact model, before any ablation)
    with_np = lm.model.decoder.weight.detach().cpu().numpy()   # [Np, Ng]
    W_hh = lm.model.RNN.weight_hh_l0.detach().cpu().numpy()     # [Ng, Ng]
    decoder_mag = np.linalg.norm(with_np[:, :Ng], axis=0)
    rec_in = np.linalg.norm(W_hh[:Ng, :], axis=1)
    rec_out = np.linalg.norm(W_hh[:, :Ng], axis=0)

    is_grid = np.isfinite(gridness) & (gridness >= grid_floor)
    module = _cluster_modules(spacing_m, is_grid)

    cov = {
        "module": module.astype(float),
        "gridness": gridness.astype(float),
        "bandness": bandness.astype(float),
        "rate_var": rate_var.astype(float),
        "hd_r": hd_r.astype(float),
        "decoder_mag": decoder_mag.astype(float),
        "rec_in": rec_in.astype(float),
        "rec_out": rec_out.astype(float),
        "spacing_m": spacing_m.astype(float),
        "best_kx": np.asarray(best_kx, dtype=float),
        "best_ky": np.asarray(best_ky, dtype=float),
    }
    summary = {
        "checkpoint": lm.checkpoint_path, "seed_id": lm.seed_id,
        "Ng": int(Ng), "collection_seed": collection_seed,
        "n_grid_units": int(np.sum(is_grid)),
        "n_modules": int(module[module >= 0].max() + 1) if np.any(module >= 0) else 0,
        "median_spacing_cm": float(np.nanmedian(spacing_m[is_grid]) * 100) if np.any(is_grid) else None,
    }
    return {"cov": cov, "summary": summary, "is_grid": is_grid}


def save_covariates(result: dict, out_dir: str) -> str:
    os.makedirs(out_dir, exist_ok=True)
    npz = os.path.join(out_dir, "pgc_covariates.npz")
    np.savez_compressed(npz, **result["cov"], is_grid=result["is_grid"])
    with open(os.path.join(out_dir, "pgc_covariates_summary.json"), "w") as f:
        json.dump(result["summary"], f, indent=2)
    return npz


def load_covariate_matrix(npz_path: str, keys=None):
    """Return (P[Ng,K], key_list, extras) from a saved covariates npz."""
    keys = list(keys or C.COVARIATE_KEYS)
    d = np.load(npz_path)
    P = np.stack([d[k] for k in keys], axis=1)
    extras = {k: d[k] for k in d.files if k not in keys}
    return P, keys, extras


def main():
    p = argparse.ArgumentParser(description="Assemble per-unit covariates")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--device", default="cpu")
    p.add_argument("--out_subdir", default="pgc_rigor")
    p.add_argument("--Ng_use", type=int, default=512)
    p.add_argument("--n_batches", type=int, default=20)
    p.add_argument("--n_workers", type=int, default=0)
    args = p.parse_args()
    lm = C.load_model(args.checkpoint, device=args.device)
    result = assemble_covariates(lm, Ng_use=args.Ng_use, n_batches=args.n_batches,
                                 n_workers=args.n_workers)
    out_dir = str(analysis_dir_for_checkpoint(Path(args.checkpoint)) / args.out_subdir)
    npz = save_covariates(result, out_dir)
    print(json.dumps(result["summary"], indent=2))
    print(f"saved -> {npz}")


if __name__ == "__main__":
    main()
