#!/usr/bin/env python3
"""Figure for the freeze-intervention (causal): freezing predictive units corrupts decoded position."""
from __future__ import annotations
import argparse, json, glob, os
import numpy as np
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONDS = ["Intact", "Freeze predictive (N)", "Freeze random (N)", "Freeze structural (N)"]
COL = {"Intact": "#2ca02c", "Freeze predictive (N)": "#d62728", "Freeze random (N)": "#7f7f7f",
       "Freeze structural (N)": "#1f77b4"}
SHORT = {"Intact": "Intact", "Freeze predictive (N)": "Freeze\npredictive", "Freeze random (N)": "Freeze\nrandom",
         "Freeze structural (N)": "Freeze\nstructural"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--analysis_root", default="analysis_outputs/Single agent path integration")
    ap.add_argument("--subdir", default="spatial_shift_allunits")
    args = ap.parse_args()
    D = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(
        _REPO, args.analysis_root, "Seed *", args.subdir, "aarav_predictive_intervention", "predictive_intervention.json")))]
    print(f"{len(D)} seeds")
    def col(m, c): return np.array([d["conditions"][c][m] for d in D])

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.3))
    fig.suptitle(f"Causal freeze intervention (count-matched, N=#predictive; {len(D)} seeds, dots=seeds)",
                 fontsize=13, fontweight="bold")
    specs = [("decode_error_m", "Decoded-position error (m)", "A.  Freezing predictive units corrupts position most"),
             ("advance_ratio", "Decoded advance ratio (1 = tracks truth)", "B.  …by overshooting, not stalling")]
    x = np.arange(len(CONDS))
    for ax, (metric, ylab, title) in zip(axes, specs):
        for i, c in enumerate(CONDS):
            v = col(metric, c)
            ax.bar(i, v.mean(), yerr=v.std(), capsize=4, color=COL[c], alpha=0.85, error_kw=dict(ecolor="k", lw=1.1))
            ax.scatter(np.full(v.size, i) + np.random.default_rng(i).uniform(-0.12, 0.12, v.size), v,
                       color="k", s=16, zorder=5, alpha=0.75)
        ax.set_xticks(x); ax.set_xticklabels([SHORT[c] for c in CONDS], fontsize=9)
        ax.set_ylabel(ylab); ax.set_title(title, loc="left", fontweight="bold"); ax.grid(alpha=0.25, axis="y")
        p = col(metric, "Freeze predictive (N)"); r = col(metric, "Freeze random (N)")
        pv = stats.wilcoxon(p, r).pvalue
        ax.annotate(f"freeze pred vs freeze rand:\nΔ={np.mean(p-r):+.2f}, {int((p>r).sum())}/10, p={pv:.3f}",
                    xy=(0.04, 0.84), xycoords="axes fraction", fontsize=9, color="#d62728", fontweight="bold")
        if metric == "advance_ratio":
            ax.axhline(1.0, color="k", ls=":", lw=1)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    out = os.path.join(_REPO, args.analysis_root, "summary", "aarav_activity_space_ablation",
                       "predictive_intervention_across_seeds.png")
    fig.savefig(out, dpi=200, bbox_inches="tight"); fig.savefig(out.replace(".png", ".svg"), bbox_inches="tight")
    print("wrote", out)


if __name__ == "__main__":
    main()
