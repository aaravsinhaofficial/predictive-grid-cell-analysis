#!/usr/bin/env python3
"""Aggregate the count-matched torus-ablation results across seeds (with between-seed variance).

Figure 1 (summary): torus-degradation metrics per condition across all seeds, bars = mean,
  error bars = between-seed std, per-seed dots. Conditions are count-matched (N = #predictive).
    - theta1_clumping : traversal disruption (predictive ablation should stand out)
    - ring_spread     : shape degradation (grid ablation should stand out; predictive ~ random)
Figure 2 (visual): top-down tori for a few representative seeds, columns Intact / Predictive (N)
  / Random (N) / All grid.
"""
from __future__ import annotations
import argparse, json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
COND_ORDER = ["Intact", "Predictive (N)", "Random (N)", "Grid (N)",
              "Predictive off-module (N')", "Random off-module (N')"]
COND_COL = {"Intact": "#2ca02c", "Predictive (N)": "#d62728", "Random (N)": "#7f7f7f",
            "Grid (N)": "#1f77b4", "Predictive off-module (N')": "#ff9896",
            "Random off-module (N')": "#c7c7c7"}
SHORT = {"Intact": "Intact", "Predictive (N)": "Predictive\n(N)", "Random (N)": "Random\n(N)",
         "Grid (N)": "Grid\n(N)", "Predictive off-module (N')": "Predictive\noff-module",
         "Random off-module (N')": "Random\noff-module"}


def seed_dir(root, s, sub):
    return os.path.join(_REPO, root, f"Seed {s}", sub, "aarav_ablation_torus")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", nargs="+", type=int, default=list(range(10)))
    ap.add_argument("--visual_seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--analysis_root", default="analysis_outputs/Single agent path integration")
    ap.add_argument("--subdir", default="spatial_shift_allunits")
    args = ap.parse_args()

    data = {}
    for s in args.seeds:
        p = os.path.join(seed_dir(args.analysis_root, s, args.subdir), "torus_ablation_metrics.json")
        if os.path.exists(p):
            data[s] = json.load(open(p))
    seeds = sorted(data)
    print(f"loaded {len(seeds)} seeds: {seeds}")
    out_base = os.path.join(_REPO, args.analysis_root, "summary", "aarav_activity_space_ablation")
    os.makedirs(out_base, exist_ok=True)

    # ---------- Figure 1: summary metrics across seeds ----------
    def collect(metric, cond):
        return np.array([data[s]["conditions"][cond][metric] for s in seeds], float)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.4))
    fig.suptitle(f"Count-matched ablation of the toroidal manifold (N = #predictive); "
                 f"{len(seeds)} seeds, error bars = between-seed std", fontsize=13, fontweight="bold")
    specs = [("theta1_clumping", "Traversal clumping of theta1\n(0 = flows around ring, 1 = stuck)",
              "A.  Movement along the manifold"),
             ("ring_spread", "Ring radius spread (CV)\n(higher = shape degraded)",
              "B.  Manifold shape")]
    x = np.arange(len(COND_ORDER))
    for ax, (metric, ylab, title) in zip(axes, specs):
        means = [np.nanmean(collect(metric, c)) for c in COND_ORDER]
        stds = [np.nanstd(collect(metric, c)) for c in COND_ORDER]
        ax.bar(x, means, yerr=stds, capsize=4, color=[COND_COL[c] for c in COND_ORDER], alpha=0.85,
               error_kw=dict(ecolor="k", lw=1.2))
        for i, c in enumerate(COND_ORDER):
            vals = collect(metric, c)
            ax.scatter(np.full(vals.size, i) + np.random.default_rng(i).uniform(-0.12, 0.12, vals.size),
                       vals, color="k", s=14, zorder=5, alpha=0.7)
        ax.set_xticks(x); ax.set_xticklabels([SHORT[c] for c in COND_ORDER], fontsize=8)
        ax.set_ylabel(ylab); ax.set_title(title, loc="left", fontweight="bold"); ax.grid(alpha=0.25, axis="y")
    # annotate the key contrasts (with vs without the module-membership confound)
    from scipy import stats as _st
    pc = collect("theta1_clumping", "Predictive (N)"); rc = collect("theta1_clumping", "Random (N)")
    po = collect("theta1_clumping", "Predictive off-module (N')"); ro = collect("theta1_clumping", "Random off-module (N')")
    axes[0].annotate(f"Pred(N) vs Rand(N): Δ={np.mean(pc-rc):+.2f}, p={_st.wilcoxon(pc,rc).pvalue:.2f}\n"
                     f"OFF-MODULE control: Δ={np.mean(po-ro):+.2f}, p={_st.wilcoxon(po,ro).pvalue:.2f} (n.s.)",
                     xy=(0.30, 0.84), xycoords="axes fraction", fontsize=8.5, color="#d62728", fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    f1 = os.path.join(out_base, "torus_ablation_metrics_across_seeds.png")
    fig.savefig(f1, dpi=200, bbox_inches="tight"); fig.savefig(f1.replace(".png", ".svg"), bbox_inches="tight")
    print("wrote", f1)

    # ---------- Figure 2: torus visuals for representative seeds ----------
    vconds = ["Intact", "Predictive (N)", "Random (N)", "All grid"]
    vseeds = [s for s in args.visual_seeds if s in data]
    nrow, ncol = len(vseeds), len(vconds)
    fig = plt.figure(figsize=(4.0 * ncol, 4.0 * nrow))
    fig.suptitle("Toroidal manifold: count-matched predictive vs random ablation "
                 "(predictive preserves the ring shape but disrupts smooth traversal)",
                 fontsize=13, fontweight="bold")
    for ri, s in enumerate(vseeds):
        npz = np.load(os.path.join(seed_dir(args.analysis_root, s, args.subdir), "torus_coords.npz"))
        for ci, cond in enumerate(vconds):
            ax = fig.add_subplot(nrow, ncol, ri * ncol + ci + 1, projection="3d")
            ck, tk = f"{cond}|coords", f"{cond}|theta1"
            if ck in npz:
                c = npz[ck]; t = npz[tk]
                ax.scatter(c[:, 0], c[:, 1], c[:, 2], c=t, cmap="hsv", s=3, alpha=0.55, linewidths=0)
            m = data[s]["conditions"][cond]
            ax.set_title(f"{cond}\nclump={m['theta1_clumping']:.2f}  spread={m['ring_spread']:.2f}", fontsize=9)
            ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
            ax.set_box_aspect((1, 1, 0.55)); ax.view_init(elev=78, azim=0)
            if ci == 0:
                ax.text2D(-0.1, 0.5, f"Seed {s}", transform=ax.transAxes, rotation=90, va="center",
                          fontsize=11, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    f2 = os.path.join(out_base, "torus_ablation_visual.png")
    fig.savefig(f2, dpi=200, bbox_inches="tight"); fig.savefig(f2.replace(".png", ".svg"), bbox_inches="tight")
    print("wrote", f2)


if __name__ == "__main__":
    main()
