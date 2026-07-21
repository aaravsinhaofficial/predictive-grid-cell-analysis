"""Parallel wrapper around the EXACT GridScorer.

The per-unit SAC/gridness computation is ~8 ms/cell (time mode); a shuffle-null
classifier needs hundreds of thousands of cells per seed, which is intractable
single-threaded. This module fans the *unchanged* GridScorer.predictive_grid_scores
across CPU cores (fork + copy-on-write shared arrays, so the big activation tensor
is never pickled), giving a ~n_cores speedup with zero change to the gridness
definition. Used by the classifier, covariate assembler, and matched-ablation.
"""
from __future__ import annotations

import os
import sys
from concurrent.futures import ProcessPoolExecutor
from typing import Optional

import numpy as np

_CODE_DIR = os.path.dirname(os.path.abspath(__file__))
if _CODE_DIR not in sys.path:
    sys.path.insert(0, _CODE_DIR)

from scores import GridScorer  # noqa: E402

# Parent sets these before creating the pool; fork children inherit them
# copy-on-write, so no large array is ever pickled across the process boundary.
_SHARED: dict = {}
_WORKER_SCORER: Optional[GridScorer] = None


def _default_workers() -> int:
    return max(1, min(40, (os.cpu_count() or 8) - 4))


def _make_scorer() -> GridScorer:
    s = _SHARED
    starts = [0.2] * 10
    ends = np.linspace(0.4, 1.0, num=10)
    coord_range = ((-s["box_width"] / 2, s["box_width"] / 2),
                   (-s["box_height"] / 2, s["box_height"] / 2))
    return GridScorer(s["res"], coord_range, zip(starts, ends.tolist()))


def _worker_scorer() -> GridScorer:
    global _WORKER_SCORER
    if _WORKER_SCORER is None:
        _WORKER_SCORER = _make_scorer()
    return _WORKER_SCORER


def _ratemap_grid_chunk(bounds):
    u0, u1 = bounds
    sc = _worker_scorer()
    rms = _SHARED["rate_maps"]
    out60 = np.full(u1 - u0, np.nan, dtype=float)
    out_sac_r = np.full(u1 - u0, np.nan, dtype=float)
    for j, u in enumerate(range(u0, u1)):
        rm = rms[u]
        if not np.isfinite(rm).any() or np.nanstd(rm) < 1e-9:
            continue
        s60, s90, _, _, sac, _ = sc.get_scores(rm)
        out60[j] = s60
        out_sac_r[j] = _first_ring_radius(sac)
    return u0, u1, out60, out_sac_r


def _first_ring_radius(sac: np.ndarray) -> float:
    """Radius (in SAC bins) of the first autocorrelation ring beyond the centre.

    Radially averages the SAC and returns the location of the first local
    maximum after the central peak -> grid spacing precursor. NaN if none.
    """
    S = sac.shape[0]
    c = (S - 1) / 2.0
    yy, xx = np.mgrid[0:S, 0:S]
    r = np.sqrt((xx - c) ** 2 + (yy - c) ** 2)
    rbin = np.round(r).astype(int)
    nmax = int(rbin.max())
    prof = np.full(nmax + 1, np.nan)
    for rr in range(nmax + 1):
        m = rbin == rr
        if m.any():
            prof[rr] = np.nanmean(sac[m])
    # first local max after the central descent
    start = 2
    for rr in range(start, nmax - 1):
        if np.isfinite(prof[rr - 1]) and np.isfinite(prof[rr]) and np.isfinite(prof[rr + 1]):
            if prof[rr] >= prof[rr - 1] and prof[rr] >= prof[rr + 1] and prof[rr] > 0:
                return float(rr)
    return float("nan")


def parallel_ratemap_grid_scores(rate_maps, *, box_width=2.2, box_height=2.2,
                                 res=20, n_workers=None):
    """Per-unit zero-shift 60-deg gridness + first-ring SAC radius (bins).

    Returns (gridness[N], sac_ring_radius_bins[N]).
    """
    n_workers = n_workers or _default_workers()
    N = rate_maps.shape[0]
    _prime_shared(np.zeros((2, 2)), np.zeros((2, 2)), [0], "time", "path", False,
                  box_width, box_height, res)
    _SHARED["rate_maps"] = rate_maps
    grid = np.full(N, np.nan, dtype=float)
    ring = np.full(N, np.nan, dtype=float)
    n_workers = max(1, min(n_workers, N))
    step = max(1, N // n_workers + 1)
    bounds = [(i, min(i + step, N)) for i in range(0, N, step)]
    if n_workers == 1:
        for b in bounds:
            u0, u1, g, rr = _ratemap_grid_chunk(b)
            grid[u0:u1] = g; ring[u0:u1] = rr
        return grid, ring
    with ProcessPoolExecutor(max_workers=n_workers,
                             mp_context=__import__("multiprocessing").get_context("fork")) as ex:
        for u0, u1, g, rr in ex.map(_ratemap_grid_chunk, bounds):
            grid[u0:u1] = g; ring[u0:u1] = rr
    return grid, ring


def _score_unit_chunk(bounds):
    u0, u1 = bounds
    sc = _worker_scorer()
    s60, _ = sc.predictive_grid_scores(
        _SHARED["xs"], _SHARED["ys"], _SHARED["act"][:, :, u0:u1], _SHARED["lags"],
        shift_mode=_SHARED["shift_mode"], periodic=_SHARED["periodic"],
        space_projection=_SHARED["space_projection"],
    )
    return u0, u1, s60


def _null_shuffle_chunk(args):
    """Compute null max-over-lags gridness for a set of shuffle seeds.

    Each shuffle circularly rolls (or block-permutes) candidate activity in time
    per trajectory column, decoupling activity from position, then takes the
    max-over-lags gridness. Returns [len(seeds), n_cand].
    """
    seeds, block = args
    sc = _worker_scorer()
    xs, ys = _SHARED["xs"], _SHARED["ys"]
    act = _SHARED["act_cand"]  # [T,B,n_cand]
    lags = _SHARED["lags"]
    T, B, n_cand = act.shape
    t_index = np.arange(T)
    out = np.full((len(seeds), n_cand), np.nan, dtype=float)
    for i, sd in enumerate(seeds):
        rng = np.random.default_rng(sd)
        if block:
            n_blocks = 4
            edges = np.linspace(0, T, n_blocks + 1).astype(int)
            rolled = np.empty_like(act)
            for b in range(B):
                order = rng.permutation(n_blocks)
                rolled[:, b, :] = np.concatenate(
                    [act[edges[k]:edges[k + 1], b, :] for k in order], axis=0)
        else:
            offs = rng.integers(1, T, size=B)
            idx = (t_index[:, None] - offs[None, :]) % T
            rolled = np.take_along_axis(act, idx[:, :, None], axis=0)
        s60, _ = sc.predictive_grid_scores(
            xs, ys, rolled, lags, shift_mode=_SHARED["shift_mode"],
            periodic=_SHARED["periodic"], space_projection=_SHARED["space_projection"],
        )
        with np.errstate(invalid="ignore"):
            out[i] = np.nanmax(s60, axis=0)
    return seeds, out


def _prime_shared(xs, ys, lags, shift_mode, space_projection, periodic,
                  box_width, box_height, res, act=None, act_cand=None):
    _SHARED.clear()
    _SHARED.update(dict(
        xs=xs, ys=ys, lags=np.asarray(lags, dtype=float), shift_mode=shift_mode,
        space_projection=space_projection, periodic=periodic,
        box_width=box_width, box_height=box_height, res=res,
    ))
    if act is not None:
        _SHARED["act"] = act
    if act_cand is not None:
        _SHARED["act_cand"] = act_cand


def parallel_predictive_scores(xs, ys, activations, lags, *, shift_mode="time",
                               space_projection="path", periodic=False,
                               box_width=2.2, box_height=2.2, res=20,
                               n_workers: Optional[int] = None) -> np.ndarray:
    """scores_60 [L, Ng] via the exact scorer, unit-chunks fanned across cores."""
    n_workers = n_workers or _default_workers()
    Ng = activations.shape[2]
    _prime_shared(xs, ys, lags, shift_mode, space_projection, periodic,
                  box_width, box_height, res, act=activations)
    L = len(lags)
    scores = np.full((L, Ng), np.nan, dtype=float)
    n_workers = max(1, min(n_workers, Ng))
    if n_workers == 1:
        _, _, s = _score_unit_chunk((0, Ng))
        return s
    bounds = [(i, min(i + max(1, Ng // n_workers + 1), Ng))
              for i in range(0, Ng, max(1, Ng // n_workers + 1))]
    with ProcessPoolExecutor(max_workers=n_workers,
                             mp_context=__import__("multiprocessing").get_context("fork")) as ex:
        for u0, u1, s60 in ex.map(_score_unit_chunk, bounds):
            scores[:, u0:u1] = s60
    return scores


def parallel_null_maxscores(xs, ys, activations_cand, lags, n_shuffles, *,
                            shift_mode="time", space_projection="path",
                            periodic=False, box_width=2.2, box_height=2.2, res=20,
                            base_seed=0, block=False,
                            n_workers: Optional[int] = None) -> np.ndarray:
    """null_max [n_shuffles, n_cand] for candidate units, shuffles fanned across cores."""
    n_workers = n_workers or _default_workers()
    _prime_shared(xs, ys, lags, shift_mode, space_projection, periodic,
                  box_width, box_height, res, act_cand=activations_cand)
    n_cand = activations_cand.shape[2]
    seeds = [base_seed + i for i in range(n_shuffles)]
    null_max = np.full((n_shuffles, n_cand), np.nan, dtype=float)
    if n_cand == 0:
        return null_max
    n_workers = max(1, min(n_workers, n_shuffles))
    if n_workers == 1:
        _, out = _null_shuffle_chunk((seeds, block))
        return out
    chunks = [seeds[i::n_workers] for i in range(n_workers)]
    chunks = [c for c in chunks if c]
    seed_to_row = {sd: i for i, sd in enumerate(seeds)}
    with ProcessPoolExecutor(max_workers=n_workers,
                             mp_context=__import__("multiprocessing").get_context("fork")) as ex:
        for sd_list, out in ex.map(_null_shuffle_chunk, [(c, block) for c in chunks]):
            for j, sd in enumerate(sd_list):
                null_max[seed_to_row[sd]] = out[j]
    return null_max
