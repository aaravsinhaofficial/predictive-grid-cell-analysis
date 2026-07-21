#!/usr/bin/env python3
"""All-seeds trajectory / path-dependence experiment for predictive grid cells.

Question this tests (same as ``aarav_crossing_population_correlation.py`` and
``future_split_experiment.py``'s ``crossing`` mode, but self-contained and run
across EVERY seed):

    At an IDENTICAL location, does predictive-unit population activity diverge
    MORE as the animal's subsequent HEADING diverges?  If predictive units
    encode the FUTURE trajectory (not just present position), then two visits to
    the same spatial bin that are about to go different directions should show a
    more decorrelated population pattern than for ordinary zero-lag grid units.

Per-seed metric (fully self-contained; reuses only the tested foundation
modules in ``pgc_common`` / ``pgc_fastscore`` / ``pgc_classifier``):

  1. Collect one seeded activation bundle (``C.collect_bundle``) -> xs, ys,
     activations [T, B, Ng].
  2. For each sample (t, b) give it a *subsequent heading* = direction of travel
     over the next ``horizon`` steps (this is the FUTURE the animal is heading
     into; samples whose future displacement is below ``min_disp_cm`` have no
     well-defined heading and are dropped).
  3. Bin samples into a res x res spatial grid.  Within each bin, draw pairs of
     visits to the SAME location.  For each pair compute
        divergence = 1 - population_correlation(class units at visit_i, visit_j)
     and the circular heading separation |dtheta| in [0, pi].
  4. Fit the slope of divergence vs heading separation (per radian).  That slope
     is the per-seed scalar: how strongly the population pattern decorrelates
     with heading, at matched position.  We compute it for PREDICTIVE units and
     for the CONTROL = zero-lag ("normal") grid units, plus retrospective and a
     count-matched random subset for reference.

Driver:
  * Takes ``--checkpoints`` explicitly, or globs ``--models_root`` for
    ``most_recent_model.pth`` files carrying a ``Seed N`` path token.
  * Runs the metric per seed (keyed by ``C.seed_id_from_path``), loading each
    checkpoint's ``pgc_classification.npz`` (generating it with small params if
    missing).
  * Aggregates across seeds: mean +/- SEM of the slope per class, plus a PAIRED
    test (predictive vs control) across seeds.
  * Saves ``pgc_pathdep_allseeds.npz`` + ``pgc_pathdep_allseeds_summary.json`` +
    ``pgc_pathdep_allseeds.png`` under the aggregate output dir, and a per-seed
    JSON under each checkpoint's analysis dir / ``out_subdir``.

Example:
  python code/pgc_pathdep_allseeds.py \
      --models_root "Models/Single agent path integration" \
      --device cuda --class_subdir pgc_rigor
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import warnings
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

_CODE_DIR = os.path.dirname(os.path.abspath(__file__))
if _CODE_DIR not in sys.path:
    sys.path.insert(0, _CODE_DIR)

from path_utils import (  # noqa: E402
    analysis_dir_for_checkpoint,
    analysis_summary_dir,
    model_name_from_checkpoint,
)
import pgc_common as C  # noqa: E402


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
@dataclass
class PathDepConfig:
    Ng_use: int = 256           # capped to model Ng and to the classified count
    n_batches: int = 10
    horizon: int = 5            # steps ahead defining the "subsequent heading"
    bin_res: int = 20           # spatial-bin grid (same-location matching)
    pairs_per_bin: int = 40     # max visit-pairs sampled per spatial bin
    max_pairs: int = 200000     # global cap on pairs
    n_heading_bins: int = 9     # heading-separation bins in [0, pi] for the curve
    min_disp_cm: float = 3.0    # min future displacement for a defined heading
    n_random_draws: int = 5     # draws for the count-matched random control
    collection_seed: int = 1234
    metric_seed: int = 20240  # pairing / random-control rng


# Primary class -> label mapping (labels come from pgc_classification.npz)
CLASS_TO_LABEL = {
    "predictive": C.CLASS_LABELS["predictive"],       # 1
    "control": C.CLASS_LABELS["normal"],              # 0  (zero-lag grid)
    "retrospective": C.CLASS_LABELS["retrospective"],  # 2
}
# Classes fit/plotted; "random_matched" and "grid_all" are derived, not label-based.
FIT_CLASSES = ("predictive", "control", "retrospective", "random_matched")


# ---------------------------------------------------------------------------
# Correlation / fitting helpers
# ---------------------------------------------------------------------------
def rowwise_pearson(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Pearson correlation between matching rows of A and B ([P, U] each) -> [P]."""
    A = np.asarray(A, dtype=np.float64)
    B = np.asarray(B, dtype=np.float64)
    A = A - A.mean(axis=1, keepdims=True)
    B = B - B.mean(axis=1, keepdims=True)
    num = (A * B).sum(axis=1)
    den = np.sqrt((A * A).sum(axis=1) * (B * B).sum(axis=1))
    out = np.full(A.shape[0], np.nan)
    m = den > 1e-12
    out[m] = num[m] / den[m]
    return out


def class_divergence(actA: np.ndarray, actB: np.ndarray, units: np.ndarray) -> np.ndarray:
    """1 - population correlation over ``units`` for each visit-pair -> [P]."""
    units = np.asarray(units, dtype=int)
    if units.size < 2:
        return np.full(actA.shape[0], np.nan)
    return 1.0 - rowwise_pearson(actA[:, units], actB[:, units])


def random_matched_divergence(actA: np.ndarray, actB: np.ndarray, Ng: int, k: int,
                              rng: np.random.Generator, n_draws: int = 5) -> np.ndarray:
    """Mean per-pair divergence over random unit subsets of size k."""
    if k < 2 or Ng < 2:
        return np.full(actA.shape[0], np.nan)
    acc = np.zeros(actA.shape[0])
    cnt = np.zeros(actA.shape[0])
    for _ in range(max(1, int(n_draws))):
        idx = rng.choice(Ng, size=min(k, Ng), replace=False)
        d = 1.0 - rowwise_pearson(actA[:, idx], actB[:, idx])
        good = np.isfinite(d)
        acc[good] += d[good]
        cnt[good] += 1
    out = np.full(actA.shape[0], np.nan)
    out[cnt > 0] = acc[cnt > 0] / cnt[cnt > 0]
    return out


def fit_slope(x: np.ndarray, y: np.ndarray) -> float:
    """Least-squares slope of y vs x over finite points (NaN if < 2 points)."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 2 or np.ptp(x[m]) < 1e-12:
        return float("nan")
    return float(np.polyfit(x[m], y[m], 1)[0])


def binned_curve(sep: np.ndarray, div: np.ndarray, edges: np.ndarray):
    """Mean divergence per heading-separation bin -> (means[nb], counts[nb])."""
    nb = edges.size - 1
    which = np.clip(np.digitize(sep, edges) - 1, 0, nb - 1)
    means = np.full(nb, np.nan)
    counts = np.zeros(nb, dtype=int)
    finite = np.isfinite(div)
    for b in range(nb):
        sel = (which == b) & finite
        counts[b] = int(sel.sum())
        if counts[b] > 0:
            means[b] = float(np.mean(div[sel]))
    return means, counts


# ---------------------------------------------------------------------------
# Class-set loading (with optional generation)
# ---------------------------------------------------------------------------
def load_or_make_classification(ckpt: str, class_subdir: str, device: str,
                                n_workers: int, generate_missing: bool,
                                Ng_use: int, verbose: bool = True) -> Tuple[np.ndarray, str]:
    """Return (labels[Ng_class], npz_path); generate a small classification if missing."""
    class_dir = analysis_dir_for_checkpoint(Path(ckpt)) / class_subdir
    npz_path = class_dir / "pgc_classification.npz"
    if npz_path.exists():
        d = np.load(npz_path)
        return np.asarray(d["labels"], dtype=int), str(npz_path)
    if not generate_missing:
        raise FileNotFoundError(
            f"No classification at {npz_path}. Run pgc_classifier.py first "
            f"or pass --generate_missing_classification.")
    if verbose:
        print(f"  [classify] missing -> generating small classification at {class_dir}", flush=True)
    import pgc_classifier as PC  # local import to avoid cost when unused
    lm = C.load_model(ckpt, device=device)
    cfg = PC.ClassifierConfig(
        n_shuffles=50, Ng_use=int(Ng_use), n_batches=6,
        heldout_confirm=True, n_workers=int(n_workers) or 0)
    result = PC.classify_checkpoint(lm, cfg, verbose=verbose)
    PC.save_classification(result, str(class_dir))
    return np.asarray(result["labels"], dtype=int), str(npz_path)


def build_class_sets(labels: np.ndarray, Ng_use: int) -> Dict[str, np.ndarray]:
    """Map labels -> unit-index sets, restricted to indices < Ng_use."""
    Ng_class = labels.shape[0]
    cap = int(min(Ng_use, Ng_class))
    lab = labels[:cap]
    sets: Dict[str, np.ndarray] = {}
    for name, val in CLASS_TO_LABEL.items():
        sets[name] = np.where(lab == val)[0].astype(int)
    grid_labels = [C.CLASS_LABELS["normal"], C.CLASS_LABELS["predictive"],
                   C.CLASS_LABELS["retrospective"]]
    sets["grid_all"] = np.where(np.isin(lab, grid_labels))[0].astype(int)
    sets["Ng_available"] = cap  # scalar (number of usable columns)
    return sets


# ---------------------------------------------------------------------------
# Per-seed path-dependence metric
# ---------------------------------------------------------------------------
def _subsequent_heading(xs: np.ndarray, ys: np.ndarray, horizon: int):
    """Heading + displacement over the next ``horizon`` steps. Shapes [T, B]."""
    T = xs.shape[0]
    theta = np.full_like(xs, np.nan, dtype=float)
    disp = np.full_like(xs, np.nan, dtype=float)
    h = int(horizon)
    if T > h:
        dx = xs[h:] - xs[:-h]
        dy = ys[h:] - ys[:-h]
        theta[:-h] = np.arctan2(dy, dx)
        disp[:-h] = np.sqrt(dx * dx + dy * dy)
    return theta, disp


def _same_bin_pairs(pos: np.ndarray, box: float, bin_res: int,
                    pairs_per_bin: int, max_pairs: int,
                    rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    """Index pairs (I, J) of samples that land in the same spatial bin."""
    bx = np.clip(((pos[:, 0] + box / 2) / box * bin_res).astype(int), 0, bin_res - 1)
    by = np.clip(((pos[:, 1] + box / 2) / box * bin_res).astype(int), 0, bin_res - 1)
    bin_id = bx * bin_res + by
    order = np.argsort(bin_id, kind="stable")
    bin_id_s = bin_id[order]
    uniq, starts = np.unique(bin_id_s, return_index=True)
    starts = list(starts) + [len(bin_id_s)]
    pair_i, pair_j = [], []
    total = 0
    for gi in range(len(uniq)):
        members = order[starts[gi]:starts[gi + 1]]
        if members.size < 2:
            continue
        npick = int(min(pairs_per_bin, members.size * (members.size - 1) // 2))
        if npick <= 0:
            continue
        a = rng.choice(members, size=npick, replace=True)
        b = rng.choice(members, size=npick, replace=True)
        keep = a != b
        if not np.any(keep):
            continue
        pair_i.append(a[keep])
        pair_j.append(b[keep])
        total += int(keep.sum())
        if total >= max_pairs:
            break
    if not pair_i:
        return np.array([], dtype=int), np.array([], dtype=int)
    I = np.concatenate(pair_i)[:max_pairs]
    J = np.concatenate(pair_j)[:max_pairs]
    return I, J


def run_pathdep_metric(lm: C.LoadedModel, class_sets: Dict[str, np.ndarray],
                       cfg: PathDepConfig, verbose: bool = True) -> dict:
    """Compute the per-seed path-dependence slopes. Returns a JSON-able dict."""
    t0 = time.time()
    Ng_use = int(class_sets["Ng_available"])
    bundle = C.collect_bundle(lm, n_batches=cfg.n_batches, Ng_use=Ng_use,
                              res=cfg.bin_res, collection_seed=cfg.collection_seed,
                              with_ratemaps=False)
    xs, ys, acts = bundle.xs, bundle.ys, bundle.activations  # [T,B], [T,B], [T,B,Ng]
    box = float(lm.options.box_width)
    cmps = C.cm_per_step(xs, ys)

    theta, disp = _subsequent_heading(xs, ys, cfg.horizon)
    valid = np.isfinite(theta) & (disp >= cfg.min_disp_cm / 100.0)
    n_valid = int(valid.sum())

    samp_pos = np.stack([xs[valid], ys[valid]], axis=1)          # [Nsamp, 2]
    samp_theta = theta[valid]                                    # [Nsamp]
    samp_act = acts[valid]                                       # [Nsamp, Ng]

    rng = np.random.default_rng(cfg.metric_seed + max(lm.seed_id, 0))
    I, J = _same_bin_pairs(samp_pos, box, cfg.bin_res,
                           cfg.pairs_per_bin, cfg.max_pairs, rng)
    n_pairs = int(I.size)

    edges = np.linspace(0.0, np.pi, cfg.n_heading_bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])

    result = {
        "seed_id": int(lm.seed_id),
        "checkpoint": lm.checkpoint_path,
        "cm_per_step": round(cmps, 3),
        "n_valid_samples": n_valid,
        "n_pairs": n_pairs,
        "Ng_available": Ng_use,
        "class_counts": {k: int(np.asarray(class_sets[k]).size)
                         for k in ("predictive", "control", "retrospective", "grid_all")},
        "heading_bin_centers_rad": centers.tolist(),
        "slopes": {}, "curves": {}, "curve_counts": {},
        "mean_divergence_overall": {},
    }
    if n_pairs < 2:
        for name in FIT_CLASSES:
            result["slopes"][name] = float("nan")
            result["curves"][name] = [float("nan")] * cfg.n_heading_bins
            result["curve_counts"][name] = [0] * cfg.n_heading_bins
            result["mean_divergence_overall"][name] = float("nan")
        result["runtime_s"] = round(time.time() - t0, 1)
        return result

    actA = samp_act[I]
    actB = samp_act[J]
    sep = np.abs(np.angle(np.exp(1j * (samp_theta[I] - samp_theta[J]))))  # [0, pi]

    k_pred = int(np.asarray(class_sets["predictive"]).size)
    for name in FIT_CLASSES:
        if name == "random_matched":
            div = random_matched_divergence(actA, actB, Ng_use, k_pred, rng,
                                            n_draws=cfg.n_random_draws)
        else:
            div = class_divergence(actA, actB, class_sets[name])
        means, counts = binned_curve(sep, div, edges)
        result["slopes"][name] = fit_slope(centers, means)
        result["curves"][name] = means.tolist()
        result["curve_counts"][name] = counts.tolist()
        finite = div[np.isfinite(div)]
        result["mean_divergence_overall"][name] = (float(np.mean(finite))
                                                   if finite.size else float("nan"))

    # secondary: slope fit directly over all pairs (density-weighted)
    result["slopes_allpairs"] = {}
    for name in FIT_CLASSES:
        if name == "random_matched":
            div = random_matched_divergence(actA, actB, Ng_use, k_pred, rng,
                                            n_draws=cfg.n_random_draws)
        else:
            div = class_divergence(actA, actB, class_sets[name])
        result["slopes_allpairs"][name] = fit_slope(sep, div)

    result["runtime_s"] = round(time.time() - t0, 1)
    if verbose:
        s = result["slopes"]
        print(f"  [pathdep] seed {lm.seed_id}: n_pairs={n_pairs} "
              f"slope pred={s['predictive']:.3f} control={s['control']:.3f} "
              f"retro={s['retrospective']:.3f} rand={s['random_matched']:.3f} "
              f"({result['runtime_s']}s)", flush=True)
    return result


def run_pathdep_checkpoint(ckpt: str, cfg: PathDepConfig, class_subdir: str,
                           out_subdir: str, device: str, n_workers: int,
                           generate_missing: bool, verbose: bool = True) -> dict:
    """Load model + classes, run the metric, save per-seed JSON, return result."""
    labels, class_npz = load_or_make_classification(
        ckpt, class_subdir, device, n_workers, generate_missing, cfg.Ng_use, verbose)
    class_sets = build_class_sets(labels, cfg.Ng_use)
    lm = C.load_model(ckpt, device=device)
    result = run_pathdep_metric(lm, class_sets, cfg, verbose=verbose)
    result["classification_npz"] = class_npz
    out_dir = analysis_dir_for_checkpoint(Path(ckpt)) / out_subdir
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "pgc_pathdep_seed.json", "w") as f:
        json.dump(result, f, indent=2)
    return result


# ---------------------------------------------------------------------------
# Cross-seed aggregation
# ---------------------------------------------------------------------------
def _mean_sem(vals: Sequence[float]) -> Tuple[float, float, int]:
    v = np.asarray(vals, dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return float("nan"), float("nan"), 0
    mean = float(v.mean())
    sem = float(v.std(ddof=1) / np.sqrt(v.size)) if v.size > 1 else 0.0
    return mean, sem, int(v.size)


def paired_test(pred: Sequence[float], ctrl: Sequence[float]) -> dict:
    """Paired predictive-vs-control test across seeds (t-test + Wilcoxon)."""
    p = np.asarray(pred, dtype=float)
    c = np.asarray(ctrl, dtype=float)
    m = np.isfinite(p) & np.isfinite(c)
    p, c = p[m], c[m]
    out = {"n_seeds": int(p.size),
           "mean_diff_pred_minus_control": (float(np.mean(p - c)) if p.size else float("nan")),
           "t_stat": float("nan"), "p_ttest": float("nan"),
           "w_stat": float("nan"), "p_wilcoxon": float("nan")}
    if p.size < 2:
        return out
    try:
        from scipy import stats
        t_stat, p_t = stats.ttest_rel(p, c)
        out["t_stat"] = float(t_stat)
        out["p_ttest"] = float(p_t)
        if np.ptp(p - c) > 0:  # wilcoxon needs some non-zero differences
            w_stat, p_w = stats.wilcoxon(p, c)
            out["w_stat"] = float(w_stat)
            out["p_wilcoxon"] = float(p_w)
    except Exception:
        d = p - c
        se = d.std(ddof=1) / np.sqrt(d.size)
        out["t_stat"] = float(d.mean() / se) if se > 0 else float("nan")
    return out


def aggregate_results(results: List[dict], cfg: PathDepConfig) -> dict:
    """Aggregate per-seed results: mean/sem per class + paired test."""
    results = sorted(results, key=lambda r: r["seed_id"])
    seed_ids = [int(r["seed_id"]) for r in results]
    per_class = {name: [r["slopes"].get(name, float("nan")) for r in results]
                 for name in FIT_CLASSES}
    cross = {}
    for name, vals in per_class.items():
        mean, sem, n = _mean_sem(vals)
        cross[name] = {"mean": mean, "sem": sem, "n_seeds_finite": n,
                       "per_seed": [float(v) for v in vals]}
    ptest = paired_test(per_class["predictive"], per_class["control"])
    return {
        "seed_ids": seed_ids,
        "n_seeds": len(seed_ids),
        "cross_seed_slope": cross,
        "paired_predictive_vs_control": ptest,
        "config": asdict(cfg),
        "per_seed": [
            {"seed_id": r["seed_id"], "checkpoint": r["checkpoint"],
             "n_pairs": r["n_pairs"], "class_counts": r["class_counts"],
             "slopes": r["slopes"], "slopes_allpairs": r.get("slopes_allpairs", {}),
             "mean_divergence_overall": r["mean_divergence_overall"]}
            for r in results
        ],
    }


def save_aggregate(results: List[dict], agg: dict, cfg: PathDepConfig,
                   out_dir: Path) -> dict:
    """Write npz + summary json + png. Returns paths."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    results = sorted(results, key=lambda r: r["seed_id"])
    seed_ids = np.array([r["seed_id"] for r in results], dtype=int)
    centers = np.asarray(results[0]["heading_bin_centers_rad"], dtype=float) if results else \
        np.linspace(0, np.pi, cfg.n_heading_bins + 1)[:-1]

    npz_kwargs = {"seed_ids": seed_ids, "heading_bin_centers_rad": centers}
    for name in FIT_CLASSES:
        npz_kwargs[f"slope_{name}"] = np.array(
            [r["slopes"].get(name, np.nan) for r in results], dtype=float)
        npz_kwargs[f"slope_allpairs_{name}"] = np.array(
            [r.get("slopes_allpairs", {}).get(name, np.nan) for r in results], dtype=float)
        npz_kwargs[f"curve_{name}"] = np.array(
            [r["curves"].get(name, [np.nan] * cfg.n_heading_bins) for r in results], dtype=float)
    npz_path = out_dir / "pgc_pathdep_allseeds.npz"
    np.savez_compressed(npz_path, **npz_kwargs)

    json_path = out_dir / "pgc_pathdep_allseeds_summary.json"
    with open(json_path, "w") as f:
        json.dump(agg, f, indent=2)

    png_path = out_dir / "pgc_pathdep_allseeds.png"
    _make_figure(results, agg, cfg, png_path)
    return {"npz": str(npz_path), "summary": str(json_path), "png": str(png_path)}


def _make_figure(results: List[dict], agg: dict, cfg: PathDepConfig, png_path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    style = {
        "predictive": dict(color="#d62728", label="Predictive"),
        "control": dict(color="#1f77b4", label="Control (zero-lag grid)"),
        "retrospective": dict(color="#9467bd", label="Retrospective"),
        "random_matched": dict(color="#7f7f7f", label="Random (matched n)"),
    }
    centers_deg = np.rad2deg(np.asarray(results[0]["heading_bin_centers_rad"], dtype=float)) \
        if results else np.array([])

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    n_seeds = len(results)
    fig.suptitle(
        f"Path dependence of predictive grid units across {n_seeds} seed(s): "
        f"population decorrelation vs subsequent-heading separation, at matched position",
        fontsize=12, fontweight="bold")

    # Panel A: per-class slope across seeds (mean +/- SEM, with per-seed points)
    ax = axes[0]
    cross = agg["cross_seed_slope"]
    xs_pos = np.arange(len(FIT_CLASSES))
    for i, name in enumerate(FIT_CLASSES):
        mean = cross[name]["mean"]
        sem = cross[name]["sem"]
        vals = np.asarray(cross[name]["per_seed"], dtype=float)
        ax.bar(i, mean, 0.6, color=style[name]["color"], alpha=0.85,
               yerr=(sem if np.isfinite(sem) else 0.0), capsize=4)
        jit = (np.random.default_rng(i).uniform(-0.12, 0.12, size=vals.size))
        ax.plot(i + jit, vals, "o", color="k", ms=4, alpha=0.7, zorder=3)
    ax.axhline(0, color="0.4", lw=0.8)
    ax.set_xticks(xs_pos)
    ax.set_xticklabels([style[n]["label"] for n in FIT_CLASSES], rotation=20, ha="right", fontsize=9)
    ax.set_ylabel("Divergence-vs-heading slope\n(1 - corr per radian)")
    p = agg["paired_predictive_vs_control"]
    ptxt = (f"paired pred vs control: dMean={p['mean_diff_pred_minus_control']:.3f}, "
            f"p(t)={p['p_ttest']:.3g}, n={p['n_seeds']}")
    ax.set_title("A.  Per-class path-dependence slope across seeds\n" + ptxt,
                 loc="left", fontsize=10, fontweight="bold")
    ax.grid(alpha=0.25, axis="y")

    # Panel B: divergence-vs-heading curve, averaged across seeds
    ax = axes[1]
    for name in FIT_CLASSES:
        curves = np.array([r["curves"].get(name, [np.nan] * cfg.n_heading_bins)
                           for r in results], dtype=float)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            m = np.where(np.all(np.isnan(curves), axis=0), np.nan,
                         np.nanmean(curves, axis=0))
        ax.plot(centers_deg[:len(m)], m, marker="o", ms=4,
                color=style[name]["color"], label=style[name]["label"])
    ax.set_xlabel("Subsequent-heading separation at matched location (deg)")
    ax.set_ylabel("Population divergence (1 - corr)")
    ax.set_title("B.  Divergence vs heading (seed-averaged)", loc="left",
                 fontsize=10, fontweight="bold")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(alpha=0.25)

    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(png_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Checkpoint discovery
# ---------------------------------------------------------------------------
def discover_checkpoints(models_root: str, filename: str = "most_recent_model.pth") -> List[str]:
    """Glob checkpoints under models_root that carry a 'Seed N' path token."""
    root = Path(models_root)
    found = sorted(str(p) for p in root.rglob(filename))
    keyed: Dict[int, str] = {}
    order: List[str] = []
    for p in found:
        sid = C.seed_id_from_path(p)
        if sid < 0:
            continue
        if sid in keyed:
            print(f"  [warn] duplicate seed {sid}; keeping {keyed[sid]}, skipping {p}", flush=True)
            continue
        keyed[sid] = p
        order.append(p)
    return order


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args(argv=None):
    p = argparse.ArgumentParser(description="All-seeds path-dependence experiment for PGCs")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--checkpoints", nargs="+", help="explicit checkpoint paths")
    src.add_argument("--models_root", help="glob root for most_recent_model.pth with a 'Seed N' token")
    p.add_argument("--device", default="cpu")
    p.add_argument("--out_subdir", default="pgc_pathdep", help="per-seed output subdir under each analysis dir")
    p.add_argument("--class_subdir", default="pgc_rigor", help="subdir holding pgc_classification.npz")
    p.add_argument("--out_dir", default=None, help="aggregate output dir (default: analysis summary dir)")
    p.add_argument("--generate_missing_classification", action="store_true")
    p.add_argument("--n_workers", type=int, default=0)
    # metric params
    p.add_argument("--Ng_use", type=int, default=256)
    p.add_argument("--n_batches", type=int, default=10)
    p.add_argument("--horizon", type=int, default=5)
    p.add_argument("--bin_res", type=int, default=20)
    p.add_argument("--pairs_per_bin", type=int, default=40)
    p.add_argument("--max_pairs", type=int, default=200000)
    p.add_argument("--n_heading_bins", type=int, default=9)
    p.add_argument("--min_disp_cm", type=float, default=3.0)
    p.add_argument("--collection_seed", type=int, default=1234)
    p.add_argument("--metric_seed", type=int, default=20240)
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    cfg = PathDepConfig(
        Ng_use=args.Ng_use, n_batches=args.n_batches, horizon=args.horizon,
        bin_res=args.bin_res, pairs_per_bin=args.pairs_per_bin, max_pairs=args.max_pairs,
        n_heading_bins=args.n_heading_bins, min_disp_cm=args.min_disp_cm,
        collection_seed=args.collection_seed, metric_seed=args.metric_seed)

    if args.checkpoints:
        checkpoints = list(args.checkpoints)
    else:
        checkpoints = discover_checkpoints(args.models_root)
    if not checkpoints:
        raise SystemExit("No checkpoints found.")
    print(f"Running path-dependence on {len(checkpoints)} checkpoint(s):", flush=True)
    for c in checkpoints:
        print(f"  seed {C.seed_id_from_path(c)}: {c}", flush=True)

    results: List[dict] = []
    for ckpt in checkpoints:
        try:
            r = run_pathdep_checkpoint(
                ckpt, cfg, args.class_subdir, args.out_subdir, args.device,
                args.n_workers, args.generate_missing_classification, verbose=True)
            results.append(r)
        except Exception as exc:  # keep going across seeds
            print(f"  [error] seed {C.seed_id_from_path(ckpt)} ({ckpt}): {exc}", flush=True)

    if not results:
        raise SystemExit("No seeds produced results.")

    agg = aggregate_results(results, cfg)

    if args.out_dir:
        out_dir = Path(args.out_dir)
    else:
        model_name = model_name_from_checkpoint(Path(checkpoints[0]))
        out_dir = analysis_summary_dir(model_name, args.out_subdir)
    paths = save_aggregate(results, agg, cfg, out_dir)

    print("\n=== cross-seed slope (1 - corr per radian) ===", flush=True)
    for name in FIT_CLASSES:
        cs = agg["cross_seed_slope"][name]
        print(f"  {name:16s} mean={cs['mean']:.4f} sem={cs['sem']:.4f} "
              f"(n={cs['n_seeds_finite']})", flush=True)
    pt = agg["paired_predictive_vs_control"]
    print(f"paired predictive vs control: dMean={pt['mean_diff_pred_minus_control']:.4f} "
          f"p(t)={pt['p_ttest']:.4g} p(wilcoxon)={pt['p_wilcoxon']:.4g} n={pt['n_seeds']}", flush=True)
    print(f"\nsaved -> {paths['npz']}\n         {paths['summary']}\n         {paths['png']}", flush=True)
    return agg


if __name__ == "__main__":
    main()
