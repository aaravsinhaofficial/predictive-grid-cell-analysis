#!/usr/bin/env python3
"""Figure: grid population activity is relatively UNCHANGING after predictive ablation, but keeps
CHANGING after random ablation (validates the frozen-torus videos with raw activity).

A: population-vector autocorrelation vs time lag (mean +/- sem over 10 seeds). Predictive-ablated
   stays self-similar at long lags (frozen); random-ablated decorrelates like intact (still moving).
B: autocorrelation at lag 20 (bars + per-seed dots + Wilcoxon predictive vs random).
C: per-step population change rate (ablations add jitter; intact smoothest).
"""
from __future__ import annotations
import argparse, json, os
import numpy as np
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONDS = ["Intact", "Predictive-ablated", "Random-ablated (N)", "Structural-ablated (N)"]
COL = {"Intact": "#000000", "Predictive-ablated": "#d62728", "Random-ablated (N)": "#7f7f7f",
       "Structural-ablated (N)": "#9467bd"}
SHORT = {c: c.replace("-ablated", "").replace(" (N)", "\n(N)").replace(" ", "\n") for c in CONDS}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--analysis_root", default="analysis_outputs/Single agent path integration")
    ap.add_argument("--subdir", default="spatial_shift_allunits")
    args = ap.parse_args()
    base = os.path.join(_REPO, args.analysis_root)
    J = [json.load(open(os.path.join(base, f"Seed {s}", args.subdir, "aarav_population_change",
                                     "population_change_stats.json"))) for s in range(10)]
    curves = {c: np.array([j[c]["autocorr_curve"] for j in J]) for c in CONDS}   # [10,L]
    ac20 = {c: np.array([j[c]["autocorr_lag20"] for j in J]) for c in CONDS}
    cr = {c: np.array([j[c]["change_rate"] for j in J]) for c in CONDS}

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # A: autocorr vs lag
    ax = axes[0]
    L = curves["Intact"].shape[1]; lags = np.arange(1, L + 1)
    for c in CONDS:
        m = curves[c].mean(0); sem = curves[c].std(0) / np.sqrt(10)
        ax.plot(lags, m, color=COL[c], lw=2, label=SHORT[c].replace("\n", " "))
        ax.fill_between(lags, m - sem, m + sem, color=COL[c], alpha=0.18)
    ax.set_xlabel("time lag (steps)"); ax.set_ylabel("population-vector autocorrelation (cosine)")
    ax.set_title("A. Does the grid population activity change over time?\n"
                 "high & flat = frozen;  decays = keeps changing", loc="left", fontweight="bold", fontsize=10)
    ax.legend(frameon=False, fontsize=9); ax.grid(alpha=0.25)

    # B: autocorr@20 bars
    ax = axes[1]; x = np.arange(len(CONDS))
    ax.bar(x, [ac20[c].mean() for c in CONDS], yerr=[ac20[c].std() for c in CONDS], capsize=4,
           color=[COL[c] for c in CONDS], alpha=0.85)
    for i, c in enumerate(CONDS):
        ax.scatter(np.full(10, i) + np.random.default_rng(i).uniform(-.12, .12, 10), ac20[c], c="k", s=12, zorder=5)
    p = stats.wilcoxon(ac20["Predictive-ablated"], ac20["Random-ablated (N)"]).pvalue
    ax.set_xticks(x); ax.set_xticklabels([SHORT[c] for c in CONDS], fontsize=8)
    ax.set_ylabel("autocorrelation at lag 20")
    ax.set_title(f"B. Self-similarity over 20 steps\npredictive>random (more static) 9/10, p={p:.3f}",
                 loc="left", fontweight="bold", fontsize=10); ax.grid(alpha=0.25, axis="y")

    # C: per-step change rate
    ax = axes[2]
    ax.bar(x, [cr[c].mean() for c in CONDS], yerr=[cr[c].std() for c in CONDS], capsize=4,
           color=[COL[c] for c in CONDS], alpha=0.85)
    for i, c in enumerate(CONDS):
        ax.scatter(np.full(10, i) + np.random.default_rng(i + 5).uniform(-.12, .12, 10), cr[c], c="k", s=12, zorder=5)
    ax.set_xticks(x); ax.set_xticklabels([SHORT[c] for c in CONDS], fontsize=8)
    ax.set_ylabel("per-step change  (1 - cos(g_t, g_t+1))")
    ax.set_title("C. Per-step change (jitter)\nablations jitter; intact smoothest", loc="left", fontweight="bold", fontsize=10)
    ax.grid(alpha=0.25, axis="y")

    fig.suptitle("Grid population activity: frozen after predictive ablation, still changing after random ablation (10 seeds)",
                 fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    out_dir = os.path.join(base, "summary", "aarav_activity_space_ablation")
    out = os.path.join(out_dir, "population_activity_change.png")
    fig.savefig(out, dpi=190, bbox_inches="tight"); fig.savefig(out.replace(".png", ".svg"), bbox_inches="tight")
    print("wrote", out)


if __name__ == "__main__":
    main()
