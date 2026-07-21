#!/usr/bin/env python3
"""Data-driven toroidal topology of grid-cell population activity.

This module REPLACES the assumed ``+60deg`` hexagonal Fourier projection (used by
``toroidal_structure_analysis`` / ``generate_fig1b_torus``) with a genuinely
data-driven topology test:

  1. Build a population point cloud from grid-cell activity (per-position rate-map
     columns of a single grid module, or a subsample of raw states).
  2. BETTI VERIFICATION -- run persistent homology (ripser, maxdim=2) and count the
     persistent H1 bars (a torus has 2) and H2 bars (a torus has 1) using a
     significance threshold estimated from a column-shuffle NULL (destroys the
     manifold).  Repeated over several subsamples for stability, and validated on a
     synthetic torus of matched size so the real-data verdict is trustworthy.
  3. DATA-DRIVEN CIRCULAR COORDINATES -- from the two longest-lived H1 cocycles of
     ``ripser(..., do_cocycles=True)``, lift each to a circular coordinate theta in
     [-pi, pi) by a harmonic (least-squares) smoothing of the integer cocycle over
     the Rips graph (scipy.sparse.linalg.lsqr on the coboundary).  Sanity-checked by
     the circular correlation between the two coordinates (should be ~independent)
     and by how well (cos, sin) of the coordinates linearly predict physical
     position.  Flagged ``circular_coords_ok=false`` (with a reason) if unstable.
  4. Comparison against the ASSUMED +60deg basis (imported from
     ``toroidal_structure_analysis``) -- reports the agreement between the
     data-driven thetas and the assumed-basis thetas, WITHOUT the data-driven test
     depending on the assumption.

Outputs (under ``analysis_dir_for_checkpoint(checkpoint)/<out_subdir>``):
  - ``pgc_torus_topology.npz``       : arrays (lifetimes, thetas, positions, ...)
  - ``pgc_torus_topology_summary.json``
  - ``pgc_torus_persistence.png``    : H1/H2 persistence diagram (data + null line)

CLI:
  python code/pgc_torus_topology.py --checkpoint <ckpt.pth> --device cpu \
      --out_subdir pgc_rigor [--n_workers N]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# BLAS single-threaded for stable parallel scoring; caches writable in-repo.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
os.environ.setdefault("NUMBA_CACHE_DIR", str(Path(".numba_cache").resolve()))
os.environ.setdefault("MPLCONFIGDIR", str(Path(".mplconfig").resolve()))
Path(os.environ["NUMBA_CACHE_DIR"]).mkdir(parents=True, exist_ok=True)
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import lsqr
from ripser import ripser

_CODE_DIR = os.path.dirname(os.path.abspath(__file__))
if _CODE_DIR not in sys.path:
    sys.path.insert(0, _CODE_DIR)

import pgc_common as C  # noqa: E402
import pgc_fastscore as F  # noqa: E402
from path_utils import analysis_dir_for_checkpoint  # noqa: E402


# ---------------------------------------------------------------------------
# Small circular-statistics helpers
# ---------------------------------------------------------------------------
def _circ_mean(theta: np.ndarray) -> float:
    return float(np.arctan2(np.mean(np.sin(theta)), np.mean(np.cos(theta))))


def circ_corr(a: np.ndarray, b: np.ndarray) -> float:
    """Jammalamadaka-Sarma circular correlation coefficient in [-1, 1]."""
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    da = np.sin(a - _circ_mean(a))
    db = np.sin(b - _circ_mean(b))
    den = np.sqrt(np.sum(da ** 2) * np.sum(db ** 2))
    if den < 1e-12:
        return 0.0
    return float(np.sum(da * db) / den)


# ---------------------------------------------------------------------------
# Grid-unit / module selection (data-driven, NO 60deg assumption)
# ---------------------------------------------------------------------------
def select_grid_module(rate_maps: np.ndarray, box_width: float, box_height: float,
                       res: int, grid_floor: float, module_select: str,
                       n_workers) -> dict:
    """Pick grid units and (optionally) a single grid module by spacing clustering.

    Returns dict with ``units`` (indices), ``gridness``, ``spacing_m`` and module
    metadata.  A single coherent module is used by default because the population
    code of *one* module is the object that should be a torus (a mixture of modules
    is a product of tori with higher Betti numbers).
    """
    gridness, sac_ring = F.parallel_ratemap_grid_scores(
        rate_maps, box_width=box_width, box_height=box_height, res=res,
        n_workers=n_workers or None)
    spacing_m = sac_ring * (box_width / res)
    is_grid = np.isfinite(gridness) & (gridness >= grid_floor) & np.isfinite(spacing_m)
    grid_idx = np.where(is_grid)[0]
    info = {"n_grid_units": int(grid_idx.size), "grid_floor": float(grid_floor),
            "module_select": module_select, "n_modules": 1}
    if grid_idx.size < 3:
        info["units"] = grid_idx.astype(int)
        info["gridness"] = gridness
        info["spacing_m"] = spacing_m
        info["module_labels"] = np.zeros(grid_idx.size, int)
        info["selected_module"] = 0
        info["module_median_spacing_cm"] = (
            float(np.nanmedian(spacing_m[grid_idx]) * 100) if grid_idx.size else None)
        return info

    labels = np.zeros(grid_idx.size, int)
    n_modules = 1
    if module_select != "all":
        try:
            from sklearn.cluster import KMeans
            from sklearn.metrics import silhouette_score
            x = np.log(spacing_m[grid_idx]).reshape(-1, 1)
            best_k, best_s, best_lab = 1, -1.0, np.zeros(grid_idx.size, int)
            for k in range(1, min(4, grid_idx.size)):
                km = KMeans(n_clusters=k, n_init=5, random_state=0).fit(x)
                s = 0.0 if k == 1 else float(silhouette_score(x, km.labels_))
                if s > best_s:
                    best_k, best_s, best_lab = k, s, km.labels_
            # order modules by ascending mean spacing for stable labels
            order = np.argsort([x[best_lab == c].mean() for c in range(best_k)])
            remap = {c: r for r, c in enumerate(order)}
            labels = np.array([remap[c] for c in best_lab])
            n_modules = best_k
        except Exception as exc:  # pragma: no cover - clustering fallback
            info["module_cluster_error"] = str(exc)

    if module_select == "all":
        units = grid_idx
        sel_mod = -1
    elif module_select == "largest":
        counts = np.bincount(labels, minlength=n_modules)
        sel_mod = int(np.argmax(counts))
        units = grid_idx[labels == sel_mod]
    else:  # explicit module index
        sel_mod = int(module_select)
        sel_mod = max(0, min(sel_mod, n_modules - 1))
        units = grid_idx[labels == sel_mod]

    info.update({
        "units": units.astype(int),
        "gridness": gridness,
        "spacing_m": spacing_m,
        "module_labels": labels,
        "selected_module": int(sel_mod),
        "n_modules": int(n_modules),
        "n_module_units": int(units.size),
        "module_median_spacing_cm": float(np.nanmedian(spacing_m[units]) * 100),
    })
    return info


def _bin_centers(res: int, box_width: float, box_height: float) -> np.ndarray:
    """Position (m) of each rate-map bin, C-order flattened to match rate_map.reshape.

    Matches ``visualize.compute_ratemaps``: rate_maps[u][i, j] -> x-bin i, y-bin j.
    """
    xs = (np.arange(res) + 0.5) / res * box_width - box_width / 2.0
    ys = (np.arange(res) + 0.5) / res * box_height - box_height / 2.0
    Xg, Yg = np.meshgrid(xs, ys, indexing="ij")
    return np.stack([Xg.reshape(-1), Yg.reshape(-1)], axis=1)


def build_ratemap_cloud(rate_maps: np.ndarray, units: np.ndarray, box_width: float,
                        box_height: float, res: int, n_pca: int,
                        drop_border: bool, l2_normalize: bool = True) -> dict:
    """Per-position population point cloud from a module's rate maps.

    cloud[p, u] = firing of module-unit u at spatial bin p.  Columns are z-scored per
    unit, optionally PCA-denoised, then row L2-normalised (cosine geometry, as in
    ``single_seed_torus_distance``).  L2-normalisation is ON by default: it puts the
    data cloud and its column-shuffle null on the same scale so the null threshold is
    meaningful (without it the shuffled blob's spurious bars dwarf the real ones).
    Also returns the RAW (un-normalised) module states aligned to the SAME points
    (for the assumed-basis comparison).
    """
    pos = _bin_centers(res, box_width, box_height)          # [res*res, 2]
    raw = rate_maps[units].reshape(units.size, -1).T        # [res*res, nu]
    keep = np.isfinite(raw).all(axis=1)
    if drop_border:
        # drop the outermost ring of bins (border firing breaks periodicity)
        mask = np.ones((res, res), bool)
        mask[0, :] = mask[-1, :] = mask[:, 0] = mask[:, -1] = False
        keep &= mask.reshape(-1)
    raw = raw[keep]
    pos = pos[keep]

    cloud = raw.copy()
    cloud = (cloud - cloud.mean(0, keepdims=True)) / (cloud.std(0, keepdims=True) + 1e-8)
    used_pca = 0
    if n_pca and n_pca > 0 and cloud.shape[1] > n_pca:
        from sklearn.decomposition import PCA
        cloud = PCA(n_components=n_pca, random_state=0).fit_transform(cloud)
        used_pca = int(n_pca)
    if l2_normalize:
        cloud = cloud / (np.linalg.norm(cloud, axis=1, keepdims=True) + 1e-8)
    cloud = np.nan_to_num(cloud)
    return {"cloud": cloud, "pos": pos, "raw": raw, "n_pca": used_pca}


# ---------------------------------------------------------------------------
# Betti verification with a shuffle null
# ---------------------------------------------------------------------------
def _column_shuffle(X: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    Xs = np.empty_like(X)
    n = X.shape[0]
    for c in range(X.shape[1]):
        Xs[:, c] = X[rng.permutation(n), c]
    return Xs


def _lifetimes(dgm: np.ndarray) -> np.ndarray:
    if dgm is None or len(dgm) == 0:
        return np.zeros(0)
    life = dgm[:, 1] - dgm[:, 0]
    return life[np.isfinite(life)]


def _gap_count(life_sorted: np.ndarray, floor: float, top_k: int = 8) -> int:
    """Betti estimate via the largest multiplicative gap in the lifetime spectrum.

    Sorts lifetimes descending and locates the biggest drop within the top-``top_k``
    bars; the number of bars before that drop is the estimated Betti number.  Unlike
    a raw count-above-null, the gap is robust when the true bars sit only modestly
    above the noise floor.  Returns 0 if even the longest bar does not exceed the
    null floor.
    """
    l = np.sort(life_sorted[np.isfinite(life_sorted)])[::-1]
    l = l[l > 1e-9]
    if l.size == 0 or l[0] <= floor:
        return 0
    lk = l[:min(l.size, top_k)]
    if lk.size == 1:
        return 1
    ratios = lk[:-1] / (lk[1:] + 1e-12)
    return int(np.argmax(ratios) + 1)


def betti_single(cloud: np.ndarray, coeff: int, n_null: int,
                 rng: np.random.Generator, thresh: float | None) -> dict:
    """Run ripser(maxdim=2) on ``cloud`` and count H1/H2 bars above a shuffle null."""
    kw = dict(maxdim=2, coeff=coeff)
    if thresh is not None:
        kw["thresh"] = float(thresh)
    dgms = ripser(cloud, **kw)["dgms"]
    out = {}
    for dim in (1, 2):
        obs = np.sort(_lifetimes(dgms[dim] if dim < len(dgms) else None))[::-1]
        null_max = []
        for _ in range(n_null):
            ns = ripser(_column_shuffle(cloud, rng), **kw)["dgms"]
            nl = _lifetimes(ns[dim] if dim < len(ns) else None)
            null_max.append(float(nl.max()) if nl.size else 0.0)
        null_thr = float(np.max(null_max)) if null_max else 0.0
        b_null = int(np.sum(obs > null_thr))
        out[f"H{dim}"] = {
            "n_bars_above_null": b_null,
            "n_bars_gap": _gap_count(obs, null_thr),
            "null_threshold": null_thr,
            "null_per_shuffle": [float(v) for v in null_max],
            "top_lifetimes": [float(v) for v in obs[:6]],
        }
    return out


def verify_torus_betti(cloud: np.ndarray, coeff: int = 47, n_null: int = 3,
                       n_trials: int = 3, n_sub: int = 320, thresh: float | None = None,
                       seed: int = 0) -> dict:
    """Betti verification repeated over random subsamples for stability."""
    rng = np.random.default_rng(seed)
    N = cloud.shape[0]
    trials = []
    for t in range(n_trials):
        if N > n_sub:
            idx = rng.choice(N, size=n_sub, replace=False)
            sub = cloud[idx]
        else:
            sub = cloud
        trials.append(betti_single(sub, coeff, n_null, rng, thresh))
    b1_null = [t["H1"]["n_bars_above_null"] for t in trials]
    b2_null = [t["H2"]["n_bars_above_null"] for t in trials]
    b1_gap = [t["H1"]["n_bars_gap"] for t in trials]
    b2_gap = [t["H2"]["n_bars_gap"] for t in trials]

    def _mode(vals):
        v, c = np.unique(vals, return_counts=True)
        return int(v[np.argmax(c)])

    return {
        "n_points": int(min(N, n_sub)),
        "n_trials": n_trials,
        "trials": trials,
        "b1_null_median": float(np.median(b1_null)),
        "b2_null_median": float(np.median(b2_null)),
        "b1_null_mode": _mode(b1_null),
        "b2_null_mode": _mode(b2_null),
        "b1_gap_mode": _mode(b1_gap),
        "b2_gap_mode": _mode(b2_gap),
        "b1_null_per_trial": b1_null,
        "b2_null_per_trial": b2_null,
    }


def synthetic_torus_cloud(n: int, ambient_dim: int, noise: float = 0.02,
                          seed: int = 0) -> np.ndarray:
    """Clean torus embedded in ``ambient_dim`` dims, matched-size method validation."""
    rng = np.random.default_rng(seed)
    u = rng.uniform(0, 2 * np.pi, n)
    v = rng.uniform(0, 2 * np.pi, n)
    R, r = 1.0, 0.5
    base = np.stack([(R + r * np.cos(v)) * np.cos(u),
                     (R + r * np.cos(v)) * np.sin(u),
                     r * np.sin(v)], axis=1)
    if ambient_dim > 3:
        # random isometry into higher dim + light noise (like a real population code)
        Q = np.linalg.qr(rng.standard_normal((ambient_dim, 3)))[0]
        cloud = base @ Q.T
    else:
        cloud = base
    cloud = cloud + noise * rng.standard_normal(cloud.shape)
    # NB: deliberately NOT L2-normalised -- that would radially collapse a loop and
    # defeat the purpose of validating that the test recovers b1=2, b2=1.
    return cloud


# ---------------------------------------------------------------------------
# Data-driven circular coordinates (hand-rolled cohomological lift)
# ---------------------------------------------------------------------------
def _harmonic_circular_coordinate(D: np.ndarray, cocycle: np.ndarray, prime: int,
                                  birth: float, death: float, perc: float) -> np.ndarray:
    """Lift one integer H1 cocycle to a circular coordinate via harmonic smoothing.

    Solves min_f || d0 f - alpha ||_2 where d0 is the graph coboundary (edges x
    vertices) at scale ``alpha`` in the bar's (birth, death) interval and ``alpha``
    is the lifted integer cocycle on those edges (de Silva-Morozov-Vejdemo-Johansson
    circular coordinates).  Returns theta_i = (f_i mod 1) mapped to [-pi, pi).
    """
    n = D.shape[0]
    coc = np.asarray(cocycle, float)
    val = coc[:, 2].copy()
    val[val > prime / 2] -= prime  # representative in (-p/2, p/2]
    cmap = {}
    for i, j, v in zip(coc[:, 0].astype(int), coc[:, 1].astype(int), val):
        if i < j:
            cmap[(i, j)] = v
        else:
            cmap[(j, i)] = -v
    alpha = birth + perc * (death - birth)
    iu, ju = np.triu_indices(n, 1)
    emask = D[iu, ju] <= alpha
    iu, ju = iu[emask], ju[emask]
    if iu.size == 0:
        return np.zeros(n)
    ne = iu.size
    rows = np.repeat(np.arange(ne), 2)
    cols = np.empty(2 * ne, int)
    cols[0::2] = iu
    cols[1::2] = ju
    data = np.empty(2 * ne)
    data[0::2] = -1.0
    data[1::2] = 1.0
    d0 = sparse.csr_matrix((data, (rows, cols)), shape=(ne, n))
    bvec = np.array([cmap.get((int(i), int(j)), 0.0) for i, j in zip(iu, ju)])
    f = lsqr(d0, bvec, atol=1e-8, btol=1e-8, iter_lim=2000)[0]
    theta = np.mod(f, 1.0) * 2 * np.pi - np.pi
    return theta


def extract_circular_coords(cloud: np.ndarray, pos: np.ndarray, coeff: int = 47,
                            perc: float = 0.5, max_points: int = 600,
                            indep_tol: float = 0.6, seed: int = 0) -> dict:
    """Two data-driven circle coordinates from the two longest H1 cocycles."""
    rng = np.random.default_rng(seed)
    N = cloud.shape[0]
    if N > max_points:
        idx = rng.choice(N, size=max_points, replace=False)
    else:
        idx = np.arange(N)
    sub = cloud[idx]
    pos_sub = pos[idx]
    res = ripser(sub, maxdim=1, coeff=coeff, do_cocycles=True)
    dgm = res["dgms"][1]
    if len(dgm) < 2:
        return {"circular_coords_ok": False,
                "reason": f"only {len(dgm)} H1 bar(s); need >=2 for a torus",
                "idx": idx}
    D = res["dperm2all"]
    life = dgm[:, 1] - dgm[:, 0]
    order = np.argsort(-life)[:2]
    thetas = []
    ok = True
    reason = ""
    for k in order:
        b, d = float(dgm[k, 0]), float(dgm[k, 1])
        try:
            th = _harmonic_circular_coordinate(D, res["cocycles"][1][k], coeff, b, d, perc)
        except Exception as exc:  # pragma: no cover
            ok = False
            reason = f"harmonic lift failed: {exc}"
            th = np.zeros(sub.shape[0])
        thetas.append(th)
    theta1, theta2 = thetas[0], thetas[1]

    cc = circ_corr(theta1, theta2)
    if abs(cc) > indep_tol:
        ok = False
        reason = reason or (f"two circle coords not independent "
                            f"(|circ_corr|={abs(cc):.2f} > {indep_tol})")

    # position prediction: linear map from (cos,sin of thetas) -> position
    Phi = np.column_stack([np.ones_like(theta1), np.cos(theta1), np.sin(theta1),
                           np.cos(theta2), np.sin(theta2)])
    pos_R2 = []
    for dcol in range(pos_sub.shape[1]):
        y = pos_sub[:, dcol]
        beta, *_ = np.linalg.lstsq(Phi, y, rcond=None)
        resid = y - Phi @ beta
        ss_tot = np.sum((y - y.mean()) ** 2)
        pos_R2.append(1.0 - np.sum(resid ** 2) / (ss_tot + 1e-12))
    pos_R2_mean = float(np.mean(pos_R2))

    return {
        "circular_coords_ok": bool(ok),
        "reason": reason,
        "idx": idx,
        "theta1": theta1,
        "theta2": theta2,
        "pos_sub": pos_sub,
        "circ_corr": float(cc),
        "pos_R2": pos_R2_mean,
        "pos_R2_per_dim": [float(v) for v in pos_R2],
        "h1_lifetimes_used": [float(life[k]) for k in order],
    }


# ---------------------------------------------------------------------------
# Assumed +60deg basis comparison (imported; does NOT feed the data-driven test)
# ---------------------------------------------------------------------------
def assumed_basis_thetas(rate_maps: np.ndarray, units: np.ndarray, box_width: float,
                         cloud_idx: np.ndarray, keep_pos_mask_shape) -> dict:
    """Project the same points onto the ASSUMED hexagonal basis for comparison."""
    try:
        from toroidal_structure_analysis import build_torus_basis, project_states_to_torus
    except Exception as exc:  # pragma: no cover
        return {"available": False, "reason": f"import failed: {exc}"}
    try:
        basis = build_torus_basis(rate_maps, np.asarray(units, int), box_width)
        Ng = rate_maps.shape[0]
        states_full = rate_maps.reshape(Ng, -1).T            # [res*res, Ng]
        # align to the border/finite mask used for the cloud, then subsample idx
        states_full = states_full[keep_pos_mask_shape]
        states_full = states_full[cloud_idx]
        Npts = states_full.shape[0]
        proj = project_states_to_torus(states_full.reshape(Npts, 1, Ng), basis,
                                       (1.0, 0.35), Npts, 1)
        return {"available": True,
                "theta1": proj.theta1.reshape(-1),
                "theta2": proj.theta2.reshape(-1),
                "k1": basis.k1, "k2": basis.k2,
                "angle_deg": float(np.degrees(basis.angle_rad))}
    except Exception as exc:  # pragma: no cover
        return {"available": False, "reason": f"projection failed: {exc}"}


def _best_circ_agreement(data_th1, data_th2, ass_th1, ass_th2) -> dict:
    """Best |circ_corr| matching of (data theta1,theta2) to (assumed theta1,theta2)."""
    d = [data_th1, data_th2]
    a = [ass_th1, ass_th2]
    M = np.array([[abs(circ_corr(d[i], a[j])) for j in range(2)] for i in range(2)])
    # two possible assignments
    diag = M[0, 0] + M[1, 1]
    anti = M[0, 1] + M[1, 0]
    if diag >= anti:
        pair = [(0, 0), (1, 1)]
    else:
        pair = [(0, 1), (1, 0)]
    vals = [float(M[i, j]) for i, j in pair]
    return {"matrix": M.tolist(), "matched_abs_circ_corr": vals,
            "mean_agreement": float(np.mean(vals))}


# ---------------------------------------------------------------------------
# Core driver
# ---------------------------------------------------------------------------
def run_torus_topology(lm: C.LoadedModel, Ng_use: int = 256, n_batches: int = 12,
                       res: int = 20, collection_seed: int = 1234,
                       grid_floor: float = 0.20, module_select: str = "largest",
                       n_pca: int = 6, drop_border: bool = True, coeff: int = 47,
                       n_null: int = 3, n_trials: int = 3, n_sub: int = 320,
                       circ_max_points: int = 500, n_workers: int = 0,
                       cloud_source: str = "ratemap",
                       l2_normalize: bool = True) -> dict:
    bw, bh = lm.options.box_width, lm.options.box_height
    bundle = C.collect_bundle(lm, n_batches=n_batches, Ng_use=Ng_use, res=res,
                              collection_seed=collection_seed, with_ratemaps=True)
    sel = select_grid_module(bundle.rate_maps, bw, bh, res, grid_floor,
                             module_select, n_workers)
    units = np.asarray(sel["units"], int)
    if units.size < 3:
        raise RuntimeError(f"too few grid units ({units.size}) to build a torus cloud")

    if cloud_source == "states":
        # raw population states over trajectories, restricted to module units
        g = bundle.g_flat[:, units]
        pos = bundle.pos_flat
        keep = np.ones(g.shape[0], bool)
        cloud = (g - g.mean(0, keepdims=True)) / (g.std(0, keepdims=True) + 1e-8)
        used_pca = 0
        if n_pca and n_pca > 0 and cloud.shape[1] > n_pca:
            from sklearn.decomposition import PCA
            cloud = PCA(n_components=n_pca, random_state=0).fit_transform(cloud)
            used_pca = int(n_pca)
        if l2_normalize:
            cloud = cloud / (np.linalg.norm(cloud, axis=1, keepdims=True) + 1e-8)
        cloud = np.nan_to_num(cloud)
        cloud_info = {"cloud": cloud, "pos": pos, "raw": g, "n_pca": used_pca}
        keep_mask = keep
    else:
        cloud_info = build_ratemap_cloud(bundle.rate_maps, units, bw, bh, res,
                                         n_pca, drop_border, l2_normalize)
        # recompute the boolean keep mask over the flat rate-map positions
        raw_full = bundle.rate_maps[units].reshape(units.size, -1).T
        keep_mask = np.isfinite(raw_full).all(axis=1)
        if drop_border:
            m = np.ones((res, res), bool)
            m[0, :] = m[-1, :] = m[:, 0] = m[:, -1] = False
            keep_mask &= m.reshape(-1)

    cloud = cloud_info["cloud"]
    pos = cloud_info["pos"]

    # --- (2) Betti verification -------------------------------------------------
    betti = verify_torus_betti(cloud, coeff=coeff, n_null=n_null, n_trials=n_trials,
                               n_sub=n_sub, seed=0)
    # method validation on a synthetic torus (matched ambient dim; enough points to
    # resolve H2, which needs denser sampling than a single H1 loop)
    n_syn = int(min(max(n_sub, 320), 360, cloud.shape[0] if cloud.shape[0] > 320 else 360))
    syn = synthetic_torus_cloud(n_syn, ambient_dim=max(3, cloud.shape[1]), seed=1)
    syn_betti = betti_single(syn, coeff, max(1, n_null), np.random.default_rng(7), None)

    # --- (3) data-driven circular coordinates ----------------------------------
    circ = extract_circular_coords(cloud, pos, coeff=coeff, max_points=circ_max_points,
                                   seed=0)

    # --- (4) assumed-basis comparison ------------------------------------------
    assumed = {"available": False, "reason": "not computed (needs circular coords)"}
    agreement = None
    if circ.get("circular_coords_ok") or circ.get("theta1") is not None:
        if "theta1" in circ and cloud_source == "ratemap":
            assumed = assumed_basis_thetas(bundle.rate_maps, units, bw,
                                           circ["idx"], keep_mask)
            if assumed.get("available") and "theta1" in circ:
                agreement = _best_circ_agreement(
                    circ["theta1"], circ["theta2"], assumed["theta1"], assumed["theta2"])

    # Verdict.  Two estimators are reported: the null-count (bars whose lifetime
    # exceeds the shuffle-null ceiling -- the literal requested statistic) and the
    # lifetime-gap estimator (more robust: a raw null-count tends to over-count H1
    # even on a genuine torus, as the synthetic validation shows).  The headline
    # ``torus_consistent`` uses the gap estimator; the null counts are exposed too.
    b1_null = betti["b1_null_mode"]
    b2_null = betti["b2_null_mode"]
    b1_gap = betti["b1_gap_mode"]
    b2_gap = betti["b2_gap_mode"]
    torus_consistent = bool(b1_gap == 2 and b2_gap == 1)
    torus_consistent_nullcount = bool(b1_null == 2 and b2_null == 1)

    summary = {
        "checkpoint": lm.checkpoint_path,
        "seed_id": lm.seed_id,
        "device": lm.device,
        "cloud_source": cloud_source,
        "n_points_cloud": int(cloud.shape[0]),
        "ambient_dim": int(cloud.shape[1]),
        "n_pca": int(cloud_info["n_pca"]),
        "grid": {k: (int(v) if isinstance(v, (np.integer,)) else v)
                 for k, v in sel.items()
                 if k in ("n_grid_units", "n_modules", "selected_module",
                          "n_module_units", "module_median_spacing_cm",
                          "grid_floor", "module_select")},
        "betti": {
            "b1_gap": int(b1_gap), "b2_gap": int(b2_gap),
            "b1_nullcount": int(b1_null), "b2_nullcount": int(b2_null),
            "b1_null_per_trial": betti["b1_null_per_trial"],
            "b2_null_per_trial": betti["b2_null_per_trial"],
            "H1_null_threshold": betti["trials"][0]["H1"]["null_threshold"],
            "H2_null_threshold": betti["trials"][0]["H2"]["null_threshold"],
            "H1_top_lifetimes": betti["trials"][0]["H1"]["top_lifetimes"],
            "H2_top_lifetimes": betti["trials"][0]["H2"]["top_lifetimes"],
        },
        "method_validation_synthetic_torus": {
            "b1_above_null": syn_betti["H1"]["n_bars_above_null"],
            "b2_above_null": syn_betti["H2"]["n_bars_above_null"],
            "b1_gap": syn_betti["H1"]["n_bars_gap"],
            "b2_gap": syn_betti["H2"]["n_bars_gap"],
            "H1_top_lifetimes": syn_betti["H1"]["top_lifetimes"],
            "H2_top_lifetimes": syn_betti["H2"]["top_lifetimes"],
        },
        "circular_coords_ok": bool(circ.get("circular_coords_ok", False)),
        "circular_coords_reason": circ.get("reason", ""),
        "circ_corr": circ.get("circ_corr"),
        "pos_R2": circ.get("pos_R2"),
        "assumed_basis": {
            "available": assumed.get("available", False),
            "reason": assumed.get("reason", ""),
            "angle_deg": assumed.get("angle_deg"),
            "agreement_with_data_driven": agreement,
        },
        "torus_consistent": torus_consistent,
        "torus_consistent_nullcount": torus_consistent_nullcount,
        "params": {
            "Ng_use": Ng_use, "n_batches": n_batches, "res": res,
            "collection_seed": collection_seed, "grid_floor": grid_floor,
            "module_select": module_select, "n_pca": n_pca,
            "drop_border": drop_border, "coeff": coeff, "n_null": n_null,
            "n_trials": n_trials, "n_sub": n_sub, "l2_normalize": bool(l2_normalize),
        },
    }

    arrays = {
        "cloud": cloud.astype(np.float32),
        "cloud_pos": pos.astype(np.float32),
        "module_units": units.astype(np.int32),
        "gridness": np.asarray(sel.get("gridness", []), np.float32),
        "spacing_m": np.asarray(sel.get("spacing_m", []), np.float32),
        "H1_top_lifetimes": np.asarray(summary["betti"]["H1_top_lifetimes"], np.float32),
        "H2_top_lifetimes": np.asarray(summary["betti"]["H2_top_lifetimes"], np.float32),
    }
    if "theta1" in circ:
        arrays["circ_theta1"] = np.asarray(circ["theta1"], np.float32)
        arrays["circ_theta2"] = np.asarray(circ["theta2"], np.float32)
        arrays["circ_pos"] = np.asarray(circ["pos_sub"], np.float32)
        arrays["circ_idx"] = np.asarray(circ["idx"], np.int32)
    if assumed.get("available"):
        arrays["assumed_theta1"] = np.asarray(assumed["theta1"], np.float32)
        arrays["assumed_theta2"] = np.asarray(assumed["theta2"], np.float32)

    return {"summary": summary, "arrays": arrays, "betti_full": betti,
            "circ_full": circ, "cloud": cloud, "circ_result": circ}


# ---------------------------------------------------------------------------
# Persistence-diagram figure
# ---------------------------------------------------------------------------
def save_persistence_png(cloud: np.ndarray, coeff: int, out_path: str,
                         h1_thr: float, h2_thr: float, max_points: int = 400,
                         seed: int = 0) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from persim import plot_diagrams
    except Exception:
        return
    rng = np.random.default_rng(seed)
    if cloud.shape[0] > max_points:
        cloud = cloud[rng.choice(cloud.shape[0], max_points, replace=False)]
    dgms = ripser(cloud, maxdim=2, coeff=coeff)["dgms"]
    fig, ax = plt.subplots(figsize=(5.2, 5))
    plot_diagrams(dgms, ax=ax, legend=True)
    ax.set_title("Persistence diagram (H0/H1/H2)\n"
                 f"H1 null life={h1_thr:.3f}, H2 null life={h2_thr:.3f}")
    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def save_outputs(result: dict, out_dir: str, coeff: int) -> str:
    os.makedirs(out_dir, exist_ok=True)
    npz = os.path.join(out_dir, "pgc_torus_topology.npz")
    np.savez_compressed(npz, **result["arrays"])
    with open(os.path.join(out_dir, "pgc_torus_topology_summary.json"), "w") as f:
        json.dump(result["summary"], f, indent=2)
    try:
        save_persistence_png(
            result["cloud"], coeff,
            os.path.join(out_dir, "pgc_torus_persistence.png"),
            result["summary"]["betti"]["H1_null_threshold"],
            result["summary"]["betti"]["H2_null_threshold"])
    except Exception:
        pass
    return npz


def main():
    p = argparse.ArgumentParser(description="Data-driven toroidal topology test")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--device", default="cpu")
    p.add_argument("--out_subdir", default="pgc_rigor")
    p.add_argument("--n_workers", type=int, default=0)
    p.add_argument("--Ng_use", type=int, default=256)
    p.add_argument("--n_batches", type=int, default=12)
    p.add_argument("--res", type=int, default=20)
    p.add_argument("--grid_floor", type=float, default=0.20)
    p.add_argument("--module_select", default="largest",
                   help="'largest', 'all', or a module index")
    p.add_argument("--cloud_source", default="ratemap", choices=["ratemap", "states"])
    p.add_argument("--n_pca", type=int, default=6, help="PCA dims (0 = off)")
    p.add_argument("--l2_normalize", action=argparse.BooleanOptionalAction, default=True,
                   help="L2-normalise cloud rows / cosine geometry (on by default; "
                        "keeps data and shuffle-null on the same scale)")
    p.add_argument("--no_drop_border", action="store_true")
    p.add_argument("--coeff", type=int, default=47)
    p.add_argument("--n_null", type=int, default=3)
    p.add_argument("--n_trials", type=int, default=3)
    p.add_argument("--n_sub", type=int, default=320)
    p.add_argument("--circ_max_points", type=int, default=500)
    args = p.parse_args()

    lm = C.load_model(args.checkpoint, device=args.device)
    result = run_torus_topology(
        lm, Ng_use=args.Ng_use, n_batches=args.n_batches, res=args.res,
        grid_floor=args.grid_floor, module_select=args.module_select,
        cloud_source=args.cloud_source, n_pca=args.n_pca,
        drop_border=not args.no_drop_border, coeff=args.coeff, n_null=args.n_null,
        n_trials=args.n_trials, n_sub=args.n_sub, circ_max_points=args.circ_max_points,
        n_workers=args.n_workers, l2_normalize=args.l2_normalize)
    out_dir = str(analysis_dir_for_checkpoint(Path(args.checkpoint)) / args.out_subdir)
    npz = save_outputs(result, out_dir, args.coeff)
    print(json.dumps(result["summary"], indent=2))
    print(f"saved -> {npz}")


if __name__ == "__main__":
    main()
