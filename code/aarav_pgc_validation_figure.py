#!/usr/bin/env python3
"""One clean figure for the two PGC-ablation validations (10 seeds), no annotations:
  A. grid population-vector autocorrelation vs time lag (frozen after predictive ablation)
  B. autocorrelation at lag 20 (quantified)
  C. grid-lattice aliasing index (fraction of decoded wander that is grid-field replicas)
  D. physical wander vs lattice-folded spread (folding collapses the predictive-ablated wander)
"""
from __future__ import annotations
import argparse, json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONDS = ["Intact", "Predictive-ablated", "Random-ablated (N)"]
LBL = {"Intact": "Intact", "Predictive-ablated": "Predictive ablated", "Random-ablated (N)": "Random ablated"}
COL = {"Intact": "#000000", "Predictive-ablated": "#d62728", "Random-ablated (N)": "#7f7f7f"}


def load(subdir, fname):
    base = os.path.join(_REPO, "analysis_outputs/Single agent path integration")
    return [json.load(open(os.path.join(base, f"Seed {s}", "spatial_shift_allunits", subdir, fname))) for s in range(10)]


def dots(ax, i, vals, seed=0):
    ax.scatter(np.full(len(vals), i) + np.random.default_rng(seed).uniform(-.12, .12, len(vals)),
               vals, c="k", s=11, zorder=5, alpha=0.7)


def main():
    pop = load("aarav_population_change", "population_change_stats.json")
    ali = load("aarav_torus_aliasing", "aliasing_stats.json")
    x = np.arange(len(CONDS))

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # A: autocorrelation vs lag
    ax = axes[0, 0]
    for c in CONDS:
        M = np.array([j[c]["autocorr_curve"] for j in pop]); lags = np.arange(1, M.shape[1] + 1)
        m = M.mean(0); sem = M.std(0) / np.sqrt(10)
        ax.plot(lags, m, color=COL[c], lw=2.2, label=LBL[c]); ax.fill_between(lags, m - sem, m + sem, color=COL[c], alpha=0.18)
    ax.set_xlabel("time lag (steps)"); ax.set_ylabel("population-vector autocorrelation")
    ax.set_title("A. Grid population activity over time", fontweight="bold", fontsize=11)
    ax.legend(frameon=False, fontsize=10); ax.grid(alpha=0.25)

    # B: autocorrelation at lag 20
    ax = axes[0, 1]
    ac20 = {c: np.array([j[c]["autocorr_lag20"] for j in pop]) for c in CONDS}
    ax.bar(x, [ac20[c].mean() for c in CONDS], yerr=[ac20[c].std() for c in CONDS], capsize=4,
           color=[COL[c] for c in CONDS], alpha=0.85)
    for i, c in enumerate(CONDS):
        dots(ax, i, ac20[c], i)
    ax.set_xticks(x); ax.set_xticklabels([LBL[c] for c in CONDS], fontsize=9)
    ax.set_ylabel("autocorrelation at 20-step lag")
    ax.set_title("B. Self-similarity over 20 steps", fontweight="bold", fontsize=11); ax.grid(alpha=0.25, axis="y")

    # C: aliasing index
    ax = axes[1, 0]
    alidx = {c: np.array([j[c]["traj_aliasing_index"] for j in ali]) for c in CONDS}
    ax.bar(x, [alidx[c].mean() for c in CONDS], yerr=[alidx[c].std() for c in CONDS], capsize=4,
           color=[COL[c] for c in CONDS], alpha=0.85)
    for i, c in enumerate(CONDS):
        dots(ax, i, alidx[c], i + 7)
    ax.set_xticks(x); ax.set_xticklabels([LBL[c] for c in CONDS], fontsize=9)
    ax.set_ylabel("grid-lattice aliasing index")
    ax.set_title("C. Fraction of wander that is grid-field aliasing", fontweight="bold", fontsize=11)
    ax.grid(alpha=0.25, axis="y")

    # D: physical vs lattice-folded spread
    ax = axes[1, 1]; w = 0.38
    raw = {c: np.array([j[c]["traj_raw_spread_m"] for j in ali]) for c in CONDS}
    win = {c: np.array([j[c]["traj_within_cell_m"] for j in ali]) for c in CONDS}
    ax.bar(x - w / 2, [raw[c].mean() for c in CONDS], w, yerr=[raw[c].std() for c in CONDS], capsize=3,
           color="#4c72b0", alpha=0.9, label="physical wander")
    ax.bar(x + w / 2, [win[c].mean() for c in CONDS], w, yerr=[win[c].std() for c in CONDS], capsize=3,
           color="#dd8452", alpha=0.9, label="folded into one tile")
    ax.set_xticks(x); ax.set_xticklabels([LBL[c] for c in CONDS], fontsize=9)
    ax.set_ylabel("decoded path spread (m)")
    ax.set_title("D. Physical vs lattice-folded spread", fontweight="bold", fontsize=11)
    ax.legend(frameon=False, fontsize=10); ax.grid(alpha=0.25, axis="y")

    fig.tight_layout()
    out = os.path.join(_REPO, "analysis_outputs/Single agent path integration/summary/aarav_pgc_validation/pgc_ablation_validation.png")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=190, bbox_inches="tight"); fig.savefig(out.replace(".png", ".svg"), bbox_inches="tight")
    print("wrote", out)


if __name__ == "__main__":
    main()
