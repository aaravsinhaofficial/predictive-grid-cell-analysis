#!/usr/bin/env python3
"""Output-space quantification figure (10 seeds): decoded spread, step sizes, decode error.

Tests Speaker-2's prediction (predictive-ablated decoded path should CLUMP / tiny steps; grid-
ablated should have large erratic steps). What the data shows: BOTH ablations make the decoded
position wander; predictive ablation wanders the MOST (largest spatial spread and step sizes).
"""
from __future__ import annotations
import argparse, glob, json, os
import numpy as np
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONDS = ["true", "Intact", "Predictive_ablated", "Grid_ablated"]
LBL = {"true": "True path", "Intact": "Intact", "Predictive_ablated": "Predictive\nablated", "Grid_ablated": "Grid\nablated"}
COL = {"true": "#2ca02c", "Intact": "#000000", "Predictive_ablated": "#d62728", "Grid_ablated": "#1f77b4"}


def gyr(p):
    c = p.mean(0, keepdims=True)
    return float(np.sqrt(((p - c) ** 2).sum(2).mean(0)).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--analysis_root", default="analysis_outputs/Single agent path integration")
    ap.add_argument("--subdir", default="spatial_shift_allunits")
    args = ap.parse_args()
    base = os.path.join(_REPO, args.analysis_root)
    npzs = [np.load(os.path.join(base, f"Seed {s}", args.subdir, "aarav_output_space", "trajectories.npz")) for s in range(10)]

    spread = {c: np.array([gyr(n[c]) for n in npzs]) for c in CONDS}            # [10] per condition
    steps = {c: np.concatenate([np.linalg.norm(np.diff(n[c], axis=0), axis=2).reshape(-1) * 100 for n in npzs]) for c in CONDS}

    fig, axes = plt.subplots(1, 3, figsize=(16, 5.2))
    fig.suptitle("Output-space: what the network's decoded position does under ablation (10 seeds)",
                 fontsize=13, fontweight="bold")

    # A: decoded spatial spread (clump vs wander)
    ax = axes[0]; order = ["Intact", "Predictive_ablated", "Grid_ablated"]; x = np.arange(len(order))
    ax.bar(x, [spread[c].mean() for c in order], yerr=[spread[c].std() for c in order], capsize=4,
           color=[COL[c] for c in order], alpha=0.85, error_kw=dict(ecolor="k", lw=1.1))
    for i, c in enumerate(order):
        ax.scatter(np.full(10, i) + np.random.default_rng(i).uniform(-0.12, 0.12, 10), spread[c], color="k", s=14, zorder=5)
    ax.axhline(spread["true"].mean(), color="#2ca02c", ls="--", lw=1.5, label=f"true ({spread['true'].mean():.2f} m)")
    ax.set_xticks(x); ax.set_xticklabels([LBL[c] for c in order]); ax.set_ylabel("Decoded path spread / gyration (m)")
    ax.set_title("A.  Clump or wander? (higher = wanders)", loc="left", fontweight="bold")
    ax.legend(frameon=False); ax.grid(alpha=0.25, axis="y")
    p_pi = stats.wilcoxon(spread["Predictive_ablated"], spread["Intact"]).pvalue
    p_pg = stats.wilcoxon(spread["Predictive_ablated"], spread["Grid_ablated"]).pvalue
    ax.annotate(f"pred vs intact p={p_pi:.3f}\npred vs grid p={p_pg:.3f}", xy=(0.04, 0.82),
                xycoords="axes fraction", fontsize=9, fontweight="bold", color="#d62728")

    # B: per-step displacement distribution (violin)
    ax = axes[1]
    data = [np.random.default_rng(0).choice(steps[c], size=min(6000, steps[c].size), replace=False) for c in CONDS]
    vp = ax.violinplot(data, showmedians=True, widths=0.85)
    for b, c in zip(vp["bodies"], CONDS):
        b.set_facecolor(COL[c]); b.set_alpha(0.6)
    ax.set_xticks(range(1, len(CONDS) + 1)); ax.set_xticklabels([LBL[c] for c in CONDS])
    ax.set_ylabel("Per-step decoded displacement (cm)")
    ax.set_title("B.  Step sizes (decoder is discrete -> spiky)", loc="left", fontweight="bold")
    ax.grid(alpha=0.25, axis="y")

    # C: decode error vs true (from stats json)
    ax = axes[2]
    stats_j = [json.load(open(os.path.join(base, f"Seed {s}", args.subdir, "aarav_output_space", "output_space_stats.json"))) for s in range(10)]
    err = {c: np.array([sj[lbl]["decode_error_end_m"] for sj in stats_j]) for c, lbl in
           [("Intact", "Intact"), ("Predictive_ablated", "Predictive-ablated"), ("Grid_ablated", "Grid-ablated")]}
    order = ["Intact", "Predictive_ablated", "Grid_ablated"]; x = np.arange(len(order))
    ax.bar(x, [err[c].mean() for c in order], yerr=[err[c].std() for c in order], capsize=4,
           color=[COL[c] for c in order], alpha=0.85, error_kw=dict(ecolor="k", lw=1.1))
    for i, c in enumerate(order):
        ax.scatter(np.full(10, i) + np.random.default_rng(i + 9).uniform(-0.12, 0.12, 10), err[c], color="k", s=14, zorder=5)
    ax.set_xticks(x); ax.set_xticklabels([LBL[c] for c in order]); ax.set_ylabel("Decoded-position error vs truth (m)")
    ax.set_title("C.  How wrong is the position?", loc="left", fontweight="bold"); ax.grid(alpha=0.25, axis="y")

    fig.tight_layout(rect=[0, 0, 1, 0.93])
    out = os.path.join(base, "summary", "aarav_activity_space_ablation", "output_space_across_seeds.png")
    fig.savefig(out, dpi=200, bbox_inches="tight"); fig.savefig(out.replace(".png", ".svg"), bbox_inches="tight")
    print("wrote", out)


if __name__ == "__main__":
    main()
