"""Cross-seed aggregation of the defensible PGC classification.

Reads each seed's ``pgc_classification.npz`` (written by pgc_classifier) and
produces the population-level results the paper needs: prevalence of predictive /
retrospective / zero-lag grid cells across networks (mean +/- SEM over seeds, with
per-seed points), and the pooled preferred-shift distribution. Seeds are keyed by
the path token via ``pgc_common.seed_id_from_path`` so runs without a "Seed N"
token don't silently collapse together.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from pathlib import Path

import numpy as np

_CODE_DIR = os.path.dirname(os.path.abspath(__file__))
if _CODE_DIR not in sys.path:
    sys.path.insert(0, _CODE_DIR)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from path_utils import analysis_dir_for_checkpoint  # noqa: E402
import pgc_common as C  # noqa: E402


def find_classification_npzs(models_root=None, checkpoints=None, out_subdir="pgc_rigor"):
    paths = []
    if checkpoints:
        for ck in checkpoints:
            p = analysis_dir_for_checkpoint(Path(ck)) / out_subdir / "pgc_classification.npz"
            if p.is_file():
                paths.append((C.seed_id_from_path(ck), str(p)))
    if models_root:
        for ck in glob.glob(os.path.join(models_root, "**", "most_recent_model.pth"), recursive=True):
            p = analysis_dir_for_checkpoint(Path(ck)) / out_subdir / "pgc_classification.npz"
            if p.is_file():
                paths.append((C.seed_id_from_path(ck), str(p)))
    # also allow globbing analysis dirs directly
    if not paths and models_root:
        for p in glob.glob(os.path.join(models_root, "**", out_subdir, "pgc_classification.npz"), recursive=True):
            paths.append((C.seed_id_from_path(p), p))
    # dedupe by seed id (keep first)
    seen, uniq = set(), []
    for sid, p in sorted(paths):
        if sid in seen:
            continue
        seen.add(sid)
        uniq.append((sid, p))
    return uniq


def aggregate(npz_paths):
    per_seed = []
    pooled_pred_shift, pooled_retro_shift = [], []
    for sid, p in npz_paths:
        d = np.load(p)
        labels = d["labels"]
        best_cm = d["best_cm"]
        Ng = labels.size
        n = {name: int(np.sum(labels == val)) for name, val in C.CLASS_LABELS.items()}
        grid_total = n["normal"] + n["predictive"] + n["retrospective"]
        rec = {
            "seed": sid, "Ng": Ng, "counts": n,
            "frac_predictive": n["predictive"] / Ng,
            "frac_retrospective": n["retrospective"] / Ng,
            "frac_zero_lag": n["normal"] / Ng,
            "frac_grid": grid_total / Ng,
            "predictive_frac_of_grid": (n["predictive"] / grid_total) if grid_total else np.nan,
            "retrospective_frac_of_grid": (n["retrospective"] / grid_total) if grid_total else np.nan,
            "median_pred_shift_cm": float(np.nanmedian(best_cm[labels == C.CLASS_LABELS["predictive"]])) if n["predictive"] else np.nan,
            "median_retro_shift_cm": float(np.nanmedian(best_cm[labels == C.CLASS_LABELS["retrospective"]])) if n["retrospective"] else np.nan,
        }
        per_seed.append(rec)
        pooled_pred_shift.extend(best_cm[labels == C.CLASS_LABELS["predictive"]].tolist())
        pooled_retro_shift.extend(best_cm[labels == C.CLASS_LABELS["retrospective"]].tolist())

    def msem(key):
        v = np.array([r[key] for r in per_seed], dtype=float)
        v = v[np.isfinite(v)]
        if v.size == 0:
            return (np.nan, np.nan, 0)
        return (float(np.mean(v)), float(np.std(v, ddof=1) / np.sqrt(v.size)) if v.size > 1 else 0.0, int(v.size))

    summary = {
        "n_seeds": len(per_seed),
        "seeds": sorted(r["seed"] for r in per_seed),
        "frac_predictive_mean_sem": msem("frac_predictive"),
        "frac_retrospective_mean_sem": msem("frac_retrospective"),
        "frac_zero_lag_mean_sem": msem("frac_zero_lag"),
        "frac_grid_mean_sem": msem("frac_grid"),
        "predictive_frac_of_grid_mean_sem": msem("predictive_frac_of_grid"),
        "median_pred_shift_cm_mean_sem": msem("median_pred_shift_cm"),
        "median_retro_shift_cm_mean_sem": msem("median_retro_shift_cm"),
        "per_seed": per_seed,
    }
    return summary, np.array(pooled_pred_shift), np.array(pooled_retro_shift)


def plot(summary, pred_shift, retro_shift, out_png):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    # prevalence across seeds
    ax = axes[0]
    cats = ["frac_predictive", "frac_retrospective", "frac_zero_lag"]
    labels = ["Predictive", "Retrospective", "Zero-lag"]
    means = [summary[f"{c}_mean_sem"][0] * 100 for c in cats]
    sems = [summary[f"{c}_mean_sem"][1] * 100 for c in cats]
    x = np.arange(len(cats))
    ax.bar(x, means, yerr=sems, color=["#1f77b4", "#d62728", "#9467bd"], capsize=4, alpha=0.85)
    for i, r in enumerate(summary["per_seed"]):
        ys = [r["frac_predictive"] * 100, r["frac_retrospective"] * 100, r["frac_zero_lag"] * 100]
        ax.scatter(x + np.random.uniform(-0.12, 0.12, 3), ys, color="k", s=14, alpha=0.5, zorder=3)
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel("% of evaluated units")
    ax.set_title(f"Prevalence across {summary['n_seeds']} networks (mean±SEM, dots=seeds)")
    # pooled shift distribution
    ax = axes[1]
    if pred_shift.size:
        ax.hist(pred_shift, bins=20, color="#1f77b4", alpha=0.7, label=f"Predictive (n={pred_shift.size})")
    if retro_shift.size:
        ax.hist(retro_shift, bins=20, color="#d62728", alpha=0.7, label=f"Retrospective (n={retro_shift.size})")
    ax.axvline(0, color="k", lw=1)
    ax.set_xlabel("Preferred shift (cm)")
    ax.set_ylabel("# units (pooled)")
    ax.set_title("Pooled preferred-shift distribution")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_png, dpi=180)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description="Cross-seed PGC classification aggregation")
    p.add_argument("--models_root", default=None, help="root to glob for most_recent_model.pth")
    p.add_argument("--checkpoints", nargs="*", default=None)
    p.add_argument("--out_subdir", default="pgc_rigor")
    p.add_argument("--out_dir", default="analysis_outputs/pgc_rigor_summary")
    args = p.parse_args()

    npzs = find_classification_npzs(args.models_root, args.checkpoints, args.out_subdir)
    if not npzs:
        print("No pgc_classification.npz found. Run pgc_classifier over the seeds first.")
        return
    print(f"Aggregating {len(npzs)} seeds: {[s for s, _ in npzs]}")
    summary, pred_shift, retro_shift = aggregate(npzs)
    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "pgc_classification_across_seeds.json"), "w") as f:
        json.dump(summary, f, indent=2)
    plot(summary, pred_shift, retro_shift, os.path.join(args.out_dir, "pgc_prevalence_across_seeds.png"))
    print(json.dumps({k: v for k, v in summary.items() if k != "per_seed"}, indent=2))
    print(f"saved -> {args.out_dir}")


if __name__ == "__main__":
    main()
