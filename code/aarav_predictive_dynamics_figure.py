#!/usr/bin/env python3
"""Figure for the 'predictive ablation = legal state, wrong place?' deep dive (10 seeds, count-matched).

Three panels across all seeds (per-seed dots + summary), with paired Wilcoxon p (vs Random):
  A. off-manifold distance (xintact, log)  -- is the state still legal?
  B. decoded-position error (m)            -- is it at the wrong place?
  C. effective time (1=advanced, ~0=stuck) -- did the state move along the manifold at all?
"""
from __future__ import annotations
import argparse, json, glob, os
import numpy as np
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONDS = ["Intact", "Predictive (N)", "Random (N)", "Structural grid (N)"]
COL = {"Intact": "#2ca02c", "Predictive (N)": "#d62728", "Random (N)": "#7f7f7f", "Structural grid (N)": "#1f77b4"}
SHORT = {"Intact": "Intact", "Predictive (N)": "Predictive", "Random (N)": "Random", "Structural grid (N)": "Structural\ngrid"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--analysis_root", default="analysis_outputs/Single agent path integration")
    ap.add_argument("--subdir", default="spatial_shift_allunits")
    args = ap.parse_args()
    files = sorted(glob.glob(os.path.join(_REPO, args.analysis_root, "Seed *", args.subdir,
                                          "aarav_predictive_dynamics", "predictive_dynamics.json")))
    D = [json.load(open(f)) for f in files]
    print(f"{len(D)} seeds")

    def col(metric, cond): return np.array([d["conditions"][cond][metric]["mean"] for d in D])

    fig, axes = plt.subplots(1, 3, figsize=(16, 5.2))
    fig.suptitle(f"Does 'predictive ablation = legal state, wrong place' hold? Count-matched, {len(D)} seeds "
                 "(dots = seeds; p = paired Wilcoxon vs Random)", fontsize=12.5, fontweight="bold")
    x = np.arange(len(CONDS))
    specs = [("off_manifold_ratio", "Off-manifold distance (× intact)", True,
              "A.  Legal state?  (low = on manifold)"),
             ("decode_error_m", "Decoded-position error (m)", False,
              "B.  Wrong place?  (high = lost)"),
             ("effective_time", "Effective time (1 = advanced, 0 = stuck at start)", False,
              "C.  Did it move along the manifold?")]
    for ax, (metric, ylab, logy, title) in zip(axes, specs):
        for i, c in enumerate(CONDS):
            v = col(metric, c)
            cen = np.exp(np.mean(np.log(np.clip(v, 1e-9, None)))) if logy else v.mean()
            ax.bar(i, cen, color=COL[c], alpha=0.8)
            ax.scatter(np.full(v.size, i) + np.random.default_rng(i).uniform(-0.13, 0.13, v.size),
                       v, color="k", s=16, zorder=5, alpha=0.75)
        if logy:
            ax.set_yscale("log")
        ax.set_xticks(x); ax.set_xticklabels([SHORT[c] for c in CONDS], fontsize=9)
        ax.set_ylabel(ylab); ax.set_title(title, loc="left", fontweight="bold"); ax.grid(alpha=0.25, axis="y", which="both")
        # paired tests vs Random
        r = col(metric, "Random (N)")
        msgs = []
        for c in ["Predictive (N)", "Structural grid (N)"]:
            v = col(metric, c); p = stats.wilcoxon(v, r).pvalue
            sig = "n.s." if p >= 0.05 else ("*" if p >= 0.01 else "**")
            msgs.append(f"{SHORT[c].splitlines()[0]} vs Random: p={p:.3f} {sig}")
        ax.annotate("\n".join(msgs), xy=(0.04, 0.86), xycoords="axes fraction", fontsize=8.5, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    out = os.path.join(_REPO, args.analysis_root, "summary", "aarav_activity_space_ablation",
                       "predictive_dynamics_across_seeds.png")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=200, bbox_inches="tight"); fig.savefig(out.replace(".png", ".svg"), bbox_inches="tight")
    print("wrote", out)


if __name__ == "__main__":
    main()
