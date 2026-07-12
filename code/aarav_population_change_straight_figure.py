#!/usr/bin/env python3
"""Straight-path autocorrelation figure + side-by-side with the random-walk version.

Reuses the SAME layout/colours/conditions as aarav_population_change_figure.py (the random-walk
figure). Produces:
  1. population_activity_change_straight.png            -- identical 3-panel plot, straight paths
  2. population_activity_change_straight_noboundary.png -- same, boundary-free straight (secondary run)
  3. population_activity_change_rw_vs_straight.png       -- random-walk | straight | straight(no-wall)
     side-by-side: autocorr-vs-lag curves (row 1) and autocorr@20 bars (row 2).
"""
from __future__ import annotations
import argparse, json, os
import numpy as np
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE = os.path.join(_REPO, "analysis_outputs/Single agent path integration")
OUT = os.path.join(BASE, "summary", "aarav_activity_space_ablation")
CONDS = ["Intact", "Predictive-ablated", "Random-ablated (N)", "Structural-ablated (N)"]
COL = {"Intact": "#000000", "Predictive-ablated": "#d62728", "Random-ablated (N)": "#7f7f7f",
       "Structural-ablated (N)": "#9467bd"}
SHORT = {c: c.replace("-ablated", "").replace(" (N)", "\n(N)").replace(" ", "\n") for c in CONDS}


def load(subdir, fname):
    return [json.load(open(os.path.join(BASE, f"Seed {s}", "spatial_shift_allunits", subdir, fname)))
            for s in range(10)]


def dots(ax, i, vals, seed=0):
    ax.scatter(np.full(len(vals), i) + np.random.default_rng(seed).uniform(-.12, .12, len(vals)),
               vals, c="k", s=11, zorder=5, alpha=0.7)


def three_panel(J, out, suptitle):
    """The exact 3-panel autocorrelation figure (A curve, B autocorr@20 bars, C per-step change)."""
    x = np.arange(len(CONDS))
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    ax = axes[0]
    for c in CONDS:
        M = np.array([j[c]["autocorr_curve"] for j in J]); lags = np.arange(1, M.shape[1] + 1)
        m = M.mean(0); sem = M.std(0) / np.sqrt(len(J))
        ax.plot(lags, m, color=COL[c], lw=2.2, label=c.replace("-ablated", " ablated").replace(" (N)", ""))
        ax.fill_between(lags, m - sem, m + sem, color=COL[c], alpha=0.18)
    ax.set_xlabel("time lag (steps)"); ax.set_ylabel("population-vector autocorrelation")
    ax.set_title("A. Grid population activity over time\nhigh & flat = frozen; decays = keeps changing",
                 loc="left", fontweight="bold", fontsize=10)
    ax.legend(frameon=False, fontsize=9); ax.grid(alpha=0.25)

    ax = axes[1]
    ac20 = {c: np.array([j[c]["autocorr_lag20"] for j in J]) for c in CONDS}
    ax.bar(x, [ac20[c].mean() for c in CONDS], yerr=[ac20[c].std() for c in CONDS], capsize=4,
           color=[COL[c] for c in CONDS], alpha=0.85)
    for i, c in enumerate(CONDS):
        dots(ax, i, ac20[c], i)
    p = stats.wilcoxon(ac20["Predictive-ablated"], ac20["Random-ablated (N)"]).pvalue
    ax.set_xticks(x); ax.set_xticklabels([SHORT[c] for c in CONDS], fontsize=8)
    ax.set_ylabel("autocorrelation at 20-step lag")
    ax.set_title(f"B. Self-similarity over 20 steps\npredictive>random p={p:.3f}",
                 loc="left", fontweight="bold", fontsize=10); ax.grid(alpha=0.25, axis="y")

    ax = axes[2]
    cr = {c: np.array([j[c]["change_rate"] for j in J]) for c in CONDS}
    ax.bar(x, [cr[c].mean() for c in CONDS], yerr=[cr[c].std() for c in CONDS], capsize=4,
           color=[COL[c] for c in CONDS], alpha=0.85)
    for i, c in enumerate(CONDS):
        dots(ax, i, cr[c], i + 5)
    ax.set_xticks(x); ax.set_xticklabels([SHORT[c] for c in CONDS], fontsize=8)
    ax.set_ylabel("per-step change  (1 - cos(g_t, g_t+1))")
    ax.set_title("C. Per-step change (jitter)", loc="left", fontweight="bold", fontsize=10)
    ax.grid(alpha=0.25, axis="y")

    fig.suptitle(suptitle, fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(out, dpi=190, bbox_inches="tight"); fig.savefig(out.replace(".png", ".svg"), bbox_inches="tight")
    print("wrote", out, f"| pred>random autocorr@20 p={p:.3f}")
    return p


def side_by_side(datasets, out):
    """datasets: list of (label, J). Row1 = autocorr-vs-lag curves; Row2 = autocorr@20 bars."""
    n = len(datasets); x = np.arange(len(CONDS))
    fig, axes = plt.subplots(2, n, figsize=(5.2 * n, 9))
    ymax20 = max(max(np.array([j[c]["autocorr_lag20"] for j in J]).mean() for c in CONDS) for _, J in datasets)
    for col, (label, J) in enumerate(datasets):
        ax = axes[0, col]
        for c in CONDS:
            M = np.array([j[c]["autocorr_curve"] for j in J]); lags = np.arange(1, M.shape[1] + 1)
            m = M.mean(0); sem = M.std(0) / np.sqrt(len(J))
            ax.plot(lags, m, color=COL[c], lw=2.2, label=c.replace("-ablated", " ablated").replace(" (N)", ""))
            ax.fill_between(lags, m - sem, m + sem, color=COL[c], alpha=0.18)
        ax.set_xlabel("time lag (steps)"); ax.set_ylim(0.3, 1.0); ax.grid(alpha=0.25)
        if col == 0:
            ax.set_ylabel("population-vector autocorrelation"); ax.legend(frameon=False, fontsize=8.5)
        ax.set_title(label, fontweight="bold", fontsize=12)

        ax = axes[1, col]
        ac20 = {c: np.array([j[c]["autocorr_lag20"] for j in J]) for c in CONDS}
        ax.bar(x, [ac20[c].mean() for c in CONDS], yerr=[ac20[c].std() for c in CONDS], capsize=4,
               color=[COL[c] for c in CONDS], alpha=0.85)
        for i, c in enumerate(CONDS):
            dots(ax, i, ac20[c], i)
        p = stats.wilcoxon(ac20["Predictive-ablated"], ac20["Random-ablated (N)"]).pvalue
        ax.set_xticks(x); ax.set_xticklabels([SHORT[c] for c in CONDS], fontsize=8)
        ax.set_ylim(0, min(1.0, ymax20 * 1.25))
        if col == 0:
            ax.set_ylabel("autocorrelation at 20-step lag")
        ax.set_title(f"predictive>random p={p:.3f}", fontsize=10)
        ax.grid(alpha=0.25, axis="y")
    fig.suptitle("Grid population autocorrelation: random-walk vs straight evaluation trajectories "
                 "(same networks, 10 seeds)", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out, dpi=190, bbox_inches="tight"); fig.savefig(out.replace(".png", ".svg"), bbox_inches="tight")
    print("wrote", out)


def main():
    rw = load("aarav_population_change", "population_change_stats.json")
    st = load("aarav_population_change_straight", "population_change_stats.json")
    nb = load("aarav_population_change_straight", "population_change_stats_noboundary.json")

    three_panel(st, os.path.join(OUT, "population_activity_change_straight.png"),
                "Grid population activity, STRAIGHT paths (10 seeds)")
    three_panel(nb, os.path.join(OUT, "population_activity_change_straight_noboundary.png"),
                "Grid population activity, STRAIGHT paths without wall contact (10 seeds)")
    side_by_side([("Random-walk", rw), ("Straight (all)", st), ("Straight (no wall contact)", nb)],
                 os.path.join(OUT, "population_activity_change_rw_vs_straight.png"))

    # console summary
    bf = np.mean([j["boundary_contact_frac"] for j in st])
    print(f"\nmean boundary-contact fraction (straight): {bf*100:.0f}%")
    for label, J in [("random-walk", rw), ("straight-all", st), ("straight-noboundary", nb)]:
        P = np.array([j["Predictive-ablated"]["autocorr_lag20"] for j in J])
        R = np.array([j["Random-ablated (N)"]["autocorr_lag20"] for j in J])
        I = np.array([j["Intact"]["autocorr_lag20"] for j in J])
        print(f"  {label:20s} autocorr@20  intact={I.mean():.3f}  predictive={P.mean():.3f}  random={R.mean():.3f} "
              f"| pred>random {int((P>R).sum())}/10 p={stats.wilcoxon(P,R).pvalue:.3f} | pred>intact {int((P>I).sum())}/10")


if __name__ == "__main__":
    main()
