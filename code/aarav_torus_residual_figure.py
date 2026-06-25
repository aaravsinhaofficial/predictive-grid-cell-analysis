#!/usr/bin/env python3
"""Figure for the torus/residual decomposition (10 seeds): where does the ablated wandering live?

A: decoded-output spread vs torus-phase-path spread (output wanders, torus frozen).
B: retrained cross-validated decoder error + R2 (info collapsed vs decoder distribution shift).
C: torus traversal (winding number) per condition (predictive specifically stalls traversal).
D: decoder & outgoing-recurrent weight norms by class (PGCs are not specially weighted).
"""
from __future__ import annotations
import argparse, json, os
import numpy as np
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONDS = ["Intact", "Predictive-ablated", "Grid-ablated", "Random-ablated (N)", "Structural-ablated (N)"]
SHORT = {c: c.replace("-ablated", "").replace(" (N)", "\n(N)").replace(" ", "\n") for c in CONDS}
COL = {"Intact": "#000000", "Predictive-ablated": "#d62728", "Grid-ablated": "#1f77b4",
       "Random-ablated (N)": "#7f7f7f", "Structural-ablated (N)": "#9467bd"}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--analysis_root", default="analysis_outputs/Single agent path integration")
    ap.add_argument("--subdir", default="spatial_shift_allunits")
    args = ap.parse_args()
    base = os.path.join(_REPO, args.analysis_root)
    J = [json.load(open(os.path.join(base, f"Seed {s}", args.subdir, "aarav_torus_residual",
                                     "torus_residual_stats.json"))) for s in range(10)]

    def col(cond, key): return np.array([j[cond][key] for j in J])
    fig, axes = plt.subplots(1, 4, figsize=(20, 5.2))

    # A: output spread vs torus-phase-path spread
    ax = axes[0]; x = np.arange(len(CONDS)); w = 0.38
    ax.bar(x - w / 2, [col(c, "gyr_full").mean() for c in CONDS], w, yerr=[col(c, "gyr_full").std() for c in CONDS],
           capsize=3, color="#d62728", alpha=0.85, label="decoded output")
    ax.bar(x + w / 2, [col(c, "gyr_phase").mean() for c in CONDS], w, yerr=[col(c, "gyr_phase").std() for c in CONDS],
           capsize=3, color="#1f77b4", alpha=0.85, label="torus-phase path")
    ax.set_xticks(x); ax.set_xticklabels([SHORT[c] for c in CONDS], fontsize=8); ax.set_ylabel("path spread / gyration (m)")
    ax.legend(frameon=False, fontsize=9); ax.grid(alpha=0.25, axis="y")
    ax.set_title("A. Output wanders, but the torus is frozen\n(wandering is OFF the torus)", loc="left", fontweight="bold", fontsize=10)

    # B: retrained decoder recovery
    ax = axes[1]
    err = {c: np.array([j["decoder_recovery"][c]["cv_pcdecode_err_cm"] for j in J]) for c in CONDS}
    r2 = {c: np.array([j["decoder_recovery"][c]["cv_R2"] for j in J]) for c in CONDS}
    ax.bar(x, [err[c].mean() for c in CONDS], yerr=[err[c].std() for c in CONDS], capsize=3,
           color=[COL[c] for c in CONDS], alpha=0.85)
    for i, c in enumerate(CONDS):
        ax.scatter(np.full(10, i) + np.random.default_rng(i).uniform(-.12, .12, 10), err[c], c="k", s=10, zorder=5)
    for i, c in enumerate(CONDS):
        ax.annotate(f"R²={r2[c].mean():.2f}", (i, err[c].mean()), ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels([SHORT[c] for c in CONDS], fontsize=8)
    ax.set_ylabel("retrained decoder error (cm)")
    ax.set_title("B. Position NOT recoverable after ablation\n(info collapsed, not decoder shift)", loc="left", fontweight="bold", fontsize=10)
    ax.grid(alpha=0.25, axis="y")

    # C: torus traversal (winding)
    ax = axes[2]
    wind = {c: col(c, "winding1_mean_abs") for c in CONDS}
    ax.bar(x, [wind[c].mean() for c in CONDS], yerr=[wind[c].std() for c in CONDS], capsize=3,
           color=[COL[c] for c in CONDS], alpha=0.85)
    for i, c in enumerate(CONDS):
        ax.scatter(np.full(10, i) + np.random.default_rng(i + 3).uniform(-.12, .12, 10), wind[c], c="k", s=10, zorder=5)
    ax.set_xticks(x); ax.set_xticklabels([SHORT[c] for c in CONDS], fontsize=8)
    ax.set_ylabel("torus winding |turns| / 40 steps")
    p_pi = stats.wilcoxon(wind["Predictive-ablated"], wind["Random-ablated (N)"]).pvalue
    ax.set_title(f"C. Predictive ablation stalls traversal\n(pred vs random p={p_pi:.3f})", loc="left", fontweight="bold", fontsize=10)
    ax.grid(alpha=0.25, axis="y")

    # D: weight norms by class
    ax = axes[3]
    cls = ["predictive", "structural_grid", "other_nongrid"]
    clab = ["predictive", "structural\ngrid", "other\n(non-grid)"]
    dec = {k: np.array([j["weight_norms"][k]["decoder_norm_mean"] for j in J]) for k in cls}
    out = {k: np.array([j["weight_norms"][k]["outrec_norm_mean"] for j in J]) for k in cls}
    xx = np.arange(len(cls))
    ax.bar(xx - w / 2, [dec[k].mean() for k in cls], w, yerr=[dec[k].std() for k in cls], capsize=3,
           color="#2ca02c", alpha=0.85, label="decoder-col norm")
    ax2 = ax.twinx()
    ax2.bar(xx + w / 2, [out[k].mean() for k in cls], w, yerr=[out[k].std() for k in cls], capsize=3,
            color="#ff7f0e", alpha=0.85, label="out-recurrent norm")
    ax.set_xticks(xx); ax.set_xticklabels(clab, fontsize=8)
    ax.set_ylabel("decoder-column norm", color="#2ca02c"); ax2.set_ylabel("out-recurrent norm", color="#ff7f0e")
    ax.set_title("D. PGCs are NOT specially weighted\n(effect is dynamical, not big-weight)", loc="left", fontweight="bold", fontsize=10)

    fig.suptitle("Torus/residual decomposition of ablated path integration (10 seeds)", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    out_dir = os.path.join(base, "summary", "aarav_activity_space_ablation")
    p = os.path.join(out_dir, "torus_residual_decomposition.png")
    fig.savefig(p, dpi=190, bbox_inches="tight"); fig.savefig(p.replace(".png", ".svg"), bbox_inches="tight")
    print("wrote", p)


if __name__ == "__main__":
    main()
