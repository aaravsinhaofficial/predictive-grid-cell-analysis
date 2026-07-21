"""Property-matched causal ablation controls + full dose-response for PGCs.

The existing ablation driver (``predictive_retrospective_ablation.py``) compares
lesioning a functional class (predictive / retrospective / zero-lag grid cells)
against a COUNT-matched RANDOM control drawn from the grid pool. A random control
answers "are these units more important than an average grid cell?" but it does
NOT control for the *properties* of the ablated units (gridness, module, band
score, firing-rate variance, head-direction tuning, read-out weight, recurrent
connectivity). If predictive cells simply tend to be higher-gridness / stronger
read-out units, a random control would flag them as important for a reason that
has nothing to do with their predictive shift.

This module adds a PROPERTY-MATCHED control: for each ablated unit it picks the
nearest *unused* pool unit in standardized covariate space (Euclidean), so the
control group matches the ablated group's covariate distribution unit-for-unit.
It then runs a FULL dose-response (k = 1, 2, 4, 8, ... up to the class size) for
the target class, the matched control, and the random control, measures decoding
error on a FIXED eval set, estimates the chance ceiling by permutation, and fits
a pre-ceiling slope (cm of error added per unit removed) for each group.

Scientific question this enables (computed, not claimed): at LOW dose, is
predictive ablation *uniquely* damaging relative to a property-matched control?
Read it off each group's pre-ceiling slope.

Everything reuses the tested foundation (``pgc_common`` for loading / lesioning /
decoding-error, ``pgc_covariates`` for the covariate matrix, ``pgc_classifier``
for labels) so unit indexing (0..Ng_eval-1) is shared across the npz files and
the model weight rows/cols.

CLI
    python code/pgc_matched_ablation.py --checkpoint <ckpt> [--device cuda] \
        [--out_subdir pgc_rigor] [--n_workers N] [--ablation_batches 8] \
        [--random_trials 5] [--match_exact_module] [--Ng_use 512]

Outputs (under analysis_dir_for_checkpoint(ckpt)/out_subdir):
    pgc_matched_ablation.npz
    pgc_matched_ablation_summary.json
    pgc_matched_ablation.png
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch

_CODE_DIR = os.path.dirname(os.path.abspath(__file__))
if _CODE_DIR not in sys.path:
    sys.path.insert(0, _CODE_DIR)

import pgc_common as C  # noqa: E402
import pgc_covariates as CV  # noqa: E402
from path_utils import analysis_dir_for_checkpoint  # noqa: E402

# Target functional classes -> label id (pgc_common.CLASS_LABELS).
GROUP_LABELS = {
    "predictive": C.CLASS_LABELS["predictive"],      # 1
    "retrospective": C.CLASS_LABELS["retrospective"],  # 2
    "normal": C.CLASS_LABELS["normal"],              # 0 (zero-lag)
}


# ---------------------------------------------------------------------------
# Inputs: labels + covariate matrix (load, or generate if absent).
# ---------------------------------------------------------------------------
def load_or_generate_inputs(lm: C.LoadedModel, out_dir: str, args) -> dict:
    """Return {labels, best_cm, best_gridness, P, keys, is_grid} for out_dir.

    If ``pgc_classification.npz`` / ``pgc_covariates.npz`` are missing under
    ``out_dir`` they are generated (with small params) via the classifier and
    covariate assembler and written there, then loaded. Unit indexing is shared
    across both files and the model weight rows/cols.
    """
    os.makedirs(out_dir, exist_ok=True)
    class_npz = os.path.join(out_dir, "pgc_classification.npz")
    cov_npz = os.path.join(out_dir, "pgc_covariates.npz")

    if not os.path.exists(class_npz):
        import pgc_classifier as PClf
        print(f"[inputs] {class_npz} absent -> classifying", flush=True)
        cfg = PClf.ClassifierConfig(
            Ng_use=int(args.Ng_use), n_batches=int(args.gen_n_batches),
            n_shuffles=int(args.gen_n_shuffles), n_workers=int(args.n_workers),
            heldout_confirm=True)
        res = PClf.classify_checkpoint(lm, cfg, verbose=True)
        PClf.save_classification(res, out_dir)

    if not os.path.exists(cov_npz):
        print(f"[inputs] {cov_npz} absent -> assembling covariates", flush=True)
        cres = CV.assemble_covariates(
            lm, Ng_use=int(args.Ng_use), n_batches=int(args.gen_n_batches),
            n_workers=int(args.n_workers))
        CV.save_covariates(cres, out_dir)

    d = np.load(class_npz)
    labels = d["labels"].astype(int)
    best_cm = d["best_cm"].astype(float)
    best_gridness = d["best_gridness"].astype(float)
    P, keys, extras = CV.load_covariate_matrix(cov_npz)
    is_grid = extras.get("is_grid")
    if is_grid is None:
        is_grid = labels >= 0
    return {"labels": labels, "best_cm": best_cm, "best_gridness": best_gridness,
            "P": P, "keys": keys, "is_grid": np.asarray(is_grid, dtype=bool)}


# ---------------------------------------------------------------------------
# Ranking within a class: |best_cm| * gridness (predictive/retro have large
# |best_cm|; for the zero-lag class this collapses to ~0 so we fall back to
# gridness so the ordering is still meaningful).
# ---------------------------------------------------------------------------
def rank_within_class(class_idx: np.ndarray, best_cm: np.ndarray,
                      gridness: np.ndarray) -> np.ndarray:
    class_idx = np.asarray(class_idx, dtype=int)
    if class_idx.size == 0:
        return class_idx
    g = np.nan_to_num(gridness[class_idx], nan=0.0)
    score = np.abs(np.nan_to_num(best_cm[class_idx], nan=0.0)) * g
    if not np.isfinite(score).any() or np.nanmax(score) <= 1e-9:
        score = g  # zero-lag fallback
    order = np.argsort(-score, kind="stable")
    return class_idx[order]


# ---------------------------------------------------------------------------
# Property-matched control.
# ---------------------------------------------------------------------------
def _standardize_over_pool(P: np.ndarray, pool_idx: np.ndarray) -> np.ndarray:
    """Z-score every covariate column using the pool's stats, imputing NaN with
    the pool column median. Returns a standardized copy for ALL units."""
    Pf = P.astype(float).copy()
    sub = Pf[pool_idx]
    med = np.nanmedian(sub, axis=0)
    med = np.where(np.isfinite(med), med, 0.0)
    nan_r, nan_c = np.where(~np.isfinite(Pf))
    Pf[nan_r, nan_c] = med[nan_c]
    sub = Pf[pool_idx]
    mu = sub.mean(axis=0)
    sd = sub.std(axis=0)
    sd = np.where(sd > 1e-12, sd, 1.0)
    return (Pf - mu) / sd


def select_matched_control(target_idx: np.ndarray, P: np.ndarray,
                           pool_idx: np.ndarray, rng: np.random.Generator,
                           match_exact_module: bool = False,
                           module_col: int = 0) -> np.ndarray:
    """Greedy nearest-neighbour covariate match, WITHOUT replacement.

    For each target unit (processed in an rng-shuffled order so repeated draws
    differ), pick the nearest still-unused pool unit in standardized covariate
    space. If ``match_exact_module`` the candidate set is first restricted to
    pool units sharing the target's module label (column ``module_col`` of the
    raw ``P``); if none remain, matching falls back to the full unused pool.

    Returns an array aligned with ``target_idx`` (chosen[i] matches target[i]).
    """
    target_idx = np.asarray(target_idx, dtype=int)
    pool_idx = np.asarray(pool_idx, dtype=int)
    n = target_idx.size
    chosen = np.full(n, -1, dtype=int)
    if n == 0 or pool_idx.size == 0:
        return chosen[chosen >= 0]

    Z = _standardize_over_pool(P, pool_idx)
    used = np.zeros(pool_idx.size, dtype=bool)
    order = rng.permutation(n)
    for oi in order:
        t = int(target_idx[oi])
        avail = ~used
        if not np.any(avail):
            break  # pool exhausted: matched group caps at pool size (count-matched to random)
        if match_exact_module:
            same = P[pool_idx, module_col] == P[t, module_col]
            if np.any(avail & same):
                avail = avail & same
        cand_local = np.where(avail)[0]
        cand = pool_idx[cand_local]
        d = np.linalg.norm(Z[cand] - Z[t][None, :], axis=1)
        pick_local = cand_local[int(np.argmin(d))]
        used[pick_local] = True
        chosen[oi] = int(pool_idx[pick_local])
    return chosen[chosen >= 0]


# ---------------------------------------------------------------------------
# Decoding-error helpers (reuse the tested C.mean_decoding_error_cm per batch).
# ---------------------------------------------------------------------------
def per_batch_errors(model, eval_batches) -> np.ndarray:
    """Mean decoding error (cm) for each eval batch -> array [n_batches]."""
    return np.array([C.mean_decoding_error_cm(model, [b]) for b in eval_batches],
                    dtype=float)


def ablate_and_eval(base_model, units: Sequence[int], eval_batches) -> np.ndarray:
    """Deepcopy the base model, lesion ``units``, return per-batch error [n_batch].

    A fresh deepcopy per call makes every dose / control draw independent.
    """
    m = copy.deepcopy(base_model)
    units = [int(u) for u in units]
    if units:
        C.zero_unit_weights_in_place(m, units)
    return per_batch_errors(m, eval_batches)


def collect_decoded_true(model, eval_batches):
    Ds, Ts = [], []
    with torch.no_grad():
        for inputs, pos in eval_batches:
            preds = model.predict(inputs)
            pp = model.place_cells.get_nearest_cell_pos(preds)
            Ds.append(pp.reshape(-1, 2).detach().cpu().numpy())
            Ts.append(pos.reshape(-1, 2).detach().cpu().numpy())
    return np.concatenate(Ds, axis=0), np.concatenate(Ts, axis=0)


def chance_ceiling_cm(model, eval_batches, rng: np.random.Generator,
                      n_perm: int = 30) -> float:
    """Chance decoding error: mean distance between decoded points and randomly
    RE-PAIRED true points (independent decoded vs true), in cm."""
    D, T = collect_decoded_true(model, eval_batches)
    N = D.shape[0]
    if N < 2:
        return float("nan")
    vals = []
    for _ in range(n_perm):
        perm = rng.permutation(N)
        vals.append(np.mean(np.linalg.norm(D - T[perm], axis=1)))
    return float(np.mean(vals)) * 100.0


# ---------------------------------------------------------------------------
# Pre-ceiling slope: line fit of error vs units-removed over the doses that are
# still below ``frac`` * ceiling, with a CI from per-batch refits.
# ---------------------------------------------------------------------------
def preceiling_slope(counts: np.ndarray, mean_err: np.ndarray,
                     per_batch_mat: np.ndarray, ceiling: float,
                     frac: float = 0.9) -> Optional[dict]:
    counts = np.asarray(counts, dtype=float)
    mean_err = np.asarray(mean_err, dtype=float)
    mask = np.isfinite(mean_err) & np.isfinite(counts)
    if np.isfinite(ceiling):
        pre = mask & (mean_err < frac * ceiling)
        if pre.sum() >= 2:
            mask = pre
    if mask.sum() < 2:
        return None
    slope = float(np.polyfit(counts[mask], mean_err[mask], 1)[0])
    slopes = []
    if per_batch_mat is not None and per_batch_mat.ndim == 2:
        for b in range(per_batch_mat.shape[1]):
            eb = per_batch_mat[:, b]
            m2 = mask & np.isfinite(eb)
            if m2.sum() >= 2:
                slopes.append(float(np.polyfit(counts[m2], eb[m2], 1)[0]))
    if slopes:
        lo, hi = np.percentile(slopes, [2.5, 97.5])
    else:
        lo = hi = float("nan")
    return {"slope_cm_per_unit": slope, "ci_lo": float(lo), "ci_hi": float(hi),
            "n_points": int(mask.sum()), "ceiling_frac": frac}


# ---------------------------------------------------------------------------
# Dose grid: geometric 1,2,4,8,... plus the full class size.
# ---------------------------------------------------------------------------
def geometric_doses(n: int) -> List[int]:
    if n <= 0:
        return []
    doses, k = [], 1
    while k < n:
        doses.append(k)
        k *= 2
    doses.append(int(n))
    return sorted(set(int(d) for d in doses if d <= n))


# ---------------------------------------------------------------------------
# Core: dose-response for one target class + matched + random controls.
# ---------------------------------------------------------------------------
def dose_response_for_class(group_name: str, class_idx: np.ndarray,
                            best_cm: np.ndarray, gridness: np.ndarray,
                            P: np.ndarray, pool_idx: np.ndarray,
                            base_model, eval_batches, ceiling: float,
                            rng: np.random.Generator, random_trials: int,
                            match_exact_module: bool) -> dict:
    ranked = rank_within_class(class_idx, best_cm, gridness)
    n = ranked.size
    doses = geometric_doses(n)

    removed, t_mean, t_sem = [], [], []
    r_mean, r_sem = [], []
    m_mean, m_sem = [], []
    ctrl_removed = []
    t_perbatch_rows = []  # [n_doses, n_batches] target per-batch errors

    for k in doses:
        units = ranked[:k]
        pb = ablate_and_eval(base_model, units, eval_batches)
        removed.append(int(k))
        t_mean.append(float(np.nanmean(pb)))
        t_sem.append(float(np.nanstd(pb) / np.sqrt(max(1, pb.size))))
        t_perbatch_rows.append(pb)

        rand_size = int(min(k, pool_idx.size))
        ctrl_removed.append(rand_size)

        # count-matched RANDOM control
        rv = []
        for _ in range(max(1, random_trials)):
            ru = rng.choice(pool_idx, size=rand_size, replace=False)
            rv.append(float(np.nanmean(ablate_and_eval(base_model, ru, eval_batches))))
        rv = np.asarray(rv, dtype=float)
        r_mean.append(float(np.nanmean(rv)))
        r_sem.append(float(np.nanstd(rv, ddof=1) / np.sqrt(rv.size)) if rv.size > 1 else 0.0)

        # PROPERTY-matched control
        mv = []
        for _ in range(max(1, random_trials)):
            mu = select_matched_control(units, P, pool_idx, rng,
                                        match_exact_module=match_exact_module)
            if mu.size == 0:
                continue
            mv.append(float(np.nanmean(ablate_and_eval(base_model, mu, eval_batches))))
        mv = np.asarray(mv, dtype=float)
        if mv.size:
            m_mean.append(float(np.nanmean(mv)))
            m_sem.append(float(np.nanstd(mv, ddof=1) / np.sqrt(mv.size)) if mv.size > 1 else 0.0)
        else:
            m_mean.append(float("nan"))
            m_sem.append(0.0)

    t_perbatch = np.asarray(t_perbatch_rows, dtype=float) if t_perbatch_rows else None
    slope_target = preceiling_slope(removed, t_mean, t_perbatch, ceiling)
    slope_matched = preceiling_slope(ctrl_removed, m_mean, None, ceiling)
    slope_random = preceiling_slope(ctrl_removed, r_mean, None, ceiling)

    return {
        "group": group_name,
        "class_size": int(n),
        "removed_counts": removed,
        "control_removed_counts": ctrl_removed,
        "error_cm": t_mean,
        "error_sem": t_sem,
        "random_mean": r_mean,
        "random_sem": r_sem,
        "matched_mean": m_mean,
        "matched_sem": m_sem,
        "preceiling_slope_target": slope_target,
        "preceiling_slope_matched": slope_matched,
        "preceiling_slope_random": slope_random,
        "ranked_units": ranked.tolist(),
    }


def run_matched_ablation(lm: C.LoadedModel, args, out_dir: str) -> dict:
    t0 = time.time()
    os.makedirs(out_dir, exist_ok=True)
    inp = load_or_generate_inputs(lm, out_dir, args)
    labels, best_cm = inp["labels"], inp["best_cm"]
    gridness, P = inp["best_gridness"], inp["P"]
    is_grid = inp["is_grid"]

    Ng_data = labels.shape[0]
    Ng_eval = min(int(args.Ng_use) if int(args.Ng_use) > 0 else Ng_data,
                  Ng_data, int(lm.Ng))
    valid = np.arange(Ng_eval, dtype=int)

    # class index sets (clipped to Ng_eval)
    class_sets = {}
    for gname, lab in GROUP_LABELS.items():
        idx = np.where(labels[:Ng_eval] == lab)[0]
        class_sets[gname] = idx.astype(int)

    # grid pool = predictive ∪ retrospective ∪ normal (within Ng_eval)
    grid_union = np.unique(np.concatenate(
        [class_sets[g] for g in GROUP_LABELS] + [np.array([], dtype=int)]))

    rng = np.random.default_rng(int(args.random_seed) + max(0, lm.seed_id))

    place_cells = lm.place_cells
    base_model = lm.model  # deepcopied per ablation inside ablate_and_eval
    eval_batches = C.collect_eval_batches(lm.traj_gen, int(args.ablation_batches))
    if not eval_batches:
        raise RuntimeError("No eval batches collected; raise --ablation_batches")

    baseline_pb = per_batch_errors(base_model, eval_batches)
    baseline_err = float(np.nanmean(baseline_pb))
    ceiling = chance_ceiling_cm(base_model, eval_batches, rng,
                                n_perm=int(args.ceiling_perms))
    print(f"[ablation] baseline={baseline_err:.2f}cm chance_ceiling={ceiling:.2f}cm "
          f"grid_pool={grid_union.size} ({time.time()-t0:.1f}s)", flush=True)

    groups = {}
    for gname in GROUP_LABELS:
        cidx = class_sets[gname]
        if cidx.size == 0:
            print(f"[ablation] group '{gname}' empty -> skipped", flush=True)
            continue
        pool = np.setdiff1d(grid_union, cidx, assume_unique=False).astype(int)
        if pool.size == 0:
            print(f"[ablation] group '{gname}' has empty pool -> skipped", flush=True)
            continue
        gres = dose_response_for_class(
            gname, cidx, best_cm, gridness, P, pool, base_model, eval_batches,
            ceiling, rng, int(args.random_trials), bool(args.match_exact_module))
        groups[gname] = gres
        st = gres["preceiling_slope_target"]
        sm = gres["preceiling_slope_matched"]
        print(f"[ablation] {gname}: n={cidx.size} pool={pool.size} "
              f"slope_target={st['slope_cm_per_unit']:.3f} "
              f"slope_matched={(sm or {}).get('slope_cm_per_unit', float('nan')):.3f} "
              f"cm/unit ({time.time()-t0:.1f}s)", flush=True)

    summary = {
        "checkpoint": lm.checkpoint_path,
        "seed_id": lm.seed_id,
        "device": lm.device,
        "Ng_total": int(lm.Ng),
        "Ng_eval": int(Ng_eval),
        "baseline_error_cm": baseline_err,
        "chance_ceiling_cm": ceiling,
        "grid_pool_size": int(grid_union.size),
        "class_counts": {g: int(class_sets[g].size) for g in GROUP_LABELS},
        "random_trials": int(args.random_trials),
        "ablation_batches": int(args.ablation_batches),
        "match_exact_module": bool(args.match_exact_module),
        "covariate_keys": list(inp["keys"]),
        "runtime_s": round(time.time() - t0, 1),
        "preceiling_slopes": {
            g: {
                "target": groups[g]["preceiling_slope_target"],
                "matched": groups[g]["preceiling_slope_matched"],
                "random": groups[g]["preceiling_slope_random"],
            } for g in groups
        },
        "note": ("Pre-ceiling slope = cm decoding error added per unit removed, "
                 "fit over doses below 0.9*chance_ceiling. Compare 'target' vs "
                 "'matched' per group to read off whether the functional class is "
                 "uniquely damaging beyond its covariate-matched control."),
    }
    return {"summary": summary, "groups": groups, "chance_ceiling_cm": ceiling,
            "baseline_error_cm": baseline_err}


# ---------------------------------------------------------------------------
# Persistence + plot.
# ---------------------------------------------------------------------------
def save_results(result: dict, out_dir: str) -> str:
    os.makedirs(out_dir, exist_ok=True)
    npz_path = os.path.join(out_dir, "pgc_matched_ablation.npz")
    flat = {
        "chance_ceiling_cm": np.asarray(result["chance_ceiling_cm"], dtype=float),
        "baseline_error_cm": np.asarray(result["baseline_error_cm"], dtype=float),
    }
    for gname, g in result["groups"].items():
        flat[f"{gname}__removed_counts"] = np.asarray(g["removed_counts"], dtype=float)
        flat[f"{gname}__control_removed_counts"] = np.asarray(g["control_removed_counts"], dtype=float)
        flat[f"{gname}__error_cm"] = np.asarray(g["error_cm"], dtype=float)
        flat[f"{gname}__error_sem"] = np.asarray(g["error_sem"], dtype=float)
        flat[f"{gname}__random_mean"] = np.asarray(g["random_mean"], dtype=float)
        flat[f"{gname}__random_sem"] = np.asarray(g["random_sem"], dtype=float)
        flat[f"{gname}__matched_mean"] = np.asarray(g["matched_mean"], dtype=float)
        flat[f"{gname}__matched_sem"] = np.asarray(g["matched_sem"], dtype=float)
        for slk in ("preceiling_slope_target", "preceiling_slope_matched",
                    "preceiling_slope_random"):
            sl = g.get(slk) or {}
            flat[f"{gname}__{slk}"] = np.asarray(
                [sl.get("slope_cm_per_unit", np.nan), sl.get("ci_lo", np.nan),
                 sl.get("ci_hi", np.nan), sl.get("n_points", np.nan)], dtype=float)
    np.savez_compressed(npz_path, **flat)
    with open(os.path.join(out_dir, "pgc_matched_ablation_summary.json"), "w") as f:
        json.dump(result["summary"], f, indent=2)
    return npz_path


def plot_results(result: dict, out_dir: str) -> Optional[str]:
    groups = result["groups"]
    if not groups:
        return None
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:  # pragma: no cover
        print(f"[plot] matplotlib unavailable: {e}", flush=True)
        return None

    order = [g for g in ("predictive", "retrospective", "normal") if g in groups]
    ceiling = result["chance_ceiling_cm"]
    base = result["baseline_error_cm"]
    fig, axes = plt.subplots(1, len(order), figsize=(5 * len(order), 4.2),
                             squeeze=False)
    for ax, gname in zip(axes[0], order):
        g = groups[gname]
        x = np.asarray(g["removed_counts"], dtype=float)
        xc = np.asarray(g["control_removed_counts"], dtype=float)
        ax.errorbar(x, g["error_cm"], yerr=g["error_sem"], marker="o",
                    color="C3", label=f"{gname} (target)", capsize=2)
        ax.errorbar(xc, g["matched_mean"], yerr=g["matched_sem"], marker="s",
                    color="C0", label="property-matched", capsize=2)
        ax.errorbar(xc, g["random_mean"], yerr=g["random_sem"], marker="^",
                    color="0.5", label="count-matched random", capsize=2)
        if np.isfinite(ceiling):
            ax.axhline(ceiling, ls="--", color="k", lw=1, label="chance ceiling")
        if np.isfinite(base):
            ax.axhline(base, ls=":", color="green", lw=1, label="baseline")
        ax.set_xlabel("units removed")
        ax.set_ylabel("decoding error (cm)")
        st = (g.get("preceiling_slope_target") or {}).get("slope_cm_per_unit", float("nan"))
        sm = (g.get("preceiling_slope_matched") or {}).get("slope_cm_per_unit", float("nan"))
        ax.set_title(f"{gname}: slope tgt={st:.2f} vs match={sm:.2f}")
        ax.legend(fontsize=7, loc="best")
    fig.suptitle("Property-matched ablation dose-response", fontsize=11)
    fig.tight_layout()
    png = os.path.join(out_dir, "pgc_matched_ablation.png")
    fig.savefig(png, dpi=120)
    plt.close(fig)
    return png


# ---------------------------------------------------------------------------
# CLI.
# ---------------------------------------------------------------------------
def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Property-matched PGC ablation + dose-response")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--device", default="cpu")
    p.add_argument("--out_subdir", default="pgc_rigor")
    p.add_argument("--n_workers", type=int, default=0)
    p.add_argument("--Ng_use", type=int, default=512)
    p.add_argument("--ablation_batches", type=int, default=8,
                   help="fixed eval batches for decoding error")
    p.add_argument("--random_trials", type=int, default=5,
                   help="draws for the random AND matched controls per dose")
    p.add_argument("--match_exact_module", action="store_true",
                   help="restrict matched candidates to the same module label")
    p.add_argument("--ceiling_perms", type=int, default=30)
    p.add_argument("--random_seed", type=int, default=12345)
    # only used when the input npz files must be generated
    p.add_argument("--gen_n_batches", type=int, default=20)
    p.add_argument("--gen_n_shuffles", type=int, default=100)
    return p


def main():
    args = build_argparser().parse_args()
    lm = C.load_model(args.checkpoint, device=args.device)
    out_dir = str(analysis_dir_for_checkpoint(Path(args.checkpoint)) / args.out_subdir)
    result = run_matched_ablation(lm, args, out_dir)
    npz = save_results(result, out_dir)
    png = plot_results(result, out_dir)
    print(json.dumps(result["summary"], indent=2))
    print(f"saved -> {npz}")
    if png:
        print(f"saved -> {png}")


if __name__ == "__main__":
    main()
