#!/usr/bin/env python3
"""Gridness distribution: straight-trained vs random-walk-trained networks.

Shows the quantitative gap that gates the PGC experiment: straight-trained networks barely develop
grid cells, so predictive grid cells are not well-defined for them.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(_REPO, "analysis_outputs/Single agent path integration/summary/aarav_straight_trained")

series = [
    ("Straight-trained Seed 0", "gridness60_straight-Seed0-straightsample.npy", "#d62728"),
    ("Straight-trained Seed 1", "gridness60_straight-Seed1-straightsample.npy", "#ff7f0e"),
    ("Random-walk-trained Seed 0", "gridness60_randomwalk-Seed0.npy", "#1f77b4"),
]

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
ax = axes[0]
for label, fn, col in series:
    g = np.load(os.path.join(OUT, fn)); g = g[np.isfinite(g)]
    ax.hist(g, bins=60, range=(-0.6, 1.5), histtype="step", lw=2.2, color=col, label=label, density=True)
ax.axvline(0.2, ls="--", c="0.4", lw=1.2)
ax.text(0.21, ax.get_ylim()[1]*0.9, "grid-cell\nthreshold 0.2", fontsize=8, color="0.3")
ax.set_xlabel("gridness (60°)"); ax.set_ylabel("density")
ax.set_title("A. Gridness distribution", fontweight="bold", fontsize=11); ax.legend(frameon=False, fontsize=9)

ax = axes[1]
labels, frac02, frac037, means = [], [], [], []
for label, fn, col in series:
    g = np.load(os.path.join(OUT, fn)); g = g[np.isfinite(g)]
    labels.append(label.replace("-trained ", "\n")); frac02.append(100*(g>0.2).mean()); frac037.append(100*(g>0.37).mean()); means.append(g.mean())
x = np.arange(len(labels)); w = 0.38
ax.bar(x-w/2, frac02, w, color="#7f7f7f", label="% units gridness>0.2")
ax.bar(x+w/2, frac037, w, color="#2ca02c", label="% units gridness>0.37 (strong)")
for i,(f2,f37) in enumerate(zip(frac02,frac037)):
    ax.text(i-w/2, f2+0.5, f"{f2:.0f}%", ha="center", fontsize=8)
    ax.text(i+w/2, f37+0.5, f"{f37:.0f}%", ha="center", fontsize=8)
ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8); ax.set_ylabel("% of 4096 units")
ax.set_title("B. Fraction of grid cells", fontweight="bold", fontsize=11); ax.legend(frameon=False, fontsize=9); ax.grid(alpha=0.25, axis="y")

fig.suptitle("Straight-trained networks barely develop grid cells (rate maps sampled with matched trajectories)", fontweight="bold", fontsize=12)
fig.tight_layout(rect=[0,0,1,0.95])
p = os.path.join(OUT, "straight_gridness_distribution.png")
fig.savefig(p, dpi=180, bbox_inches="tight"); fig.savefig(p.replace(".png",".svg"), bbox_inches="tight")
print("wrote", p)
for label, fn, _ in series:
    g = np.load(os.path.join(OUT, fn)); g=g[np.isfinite(g)]
    print(f"  {label:28s} mean={g.mean():.3f} >0.2:{(g>0.2).sum():5d} >0.37:{(g>0.37).sum():5d}")
