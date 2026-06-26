#!/usr/bin/env python3
"""Toroidal-ablation OUTPUT-space figure (minimal): example decoded trajectories + how far the
decoded position travels under each ablation.

Top row: example agents' true path (gray) vs the network's decoded path (coloured) for
Intact / Predictive-ablated / Grid-ablated (static versions of the trajectory videos).
Bottom: total distance travelled by the decoded position, per condition (10 seeds x 40 agents).

Reads each seed's aarav_output_space/trajectories.npz.
"""
from __future__ import annotations
import argparse, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
COND = [("Intact", "Intact"), ("Predictive ablated", "Predictive_ablated"), ("Grid ablated", "Grid_ablated")]


def path_length(x):   # x [T,B,2] -> [B] total distance travelled by the decoded position
    return np.linalg.norm(np.diff(x, axis=0), axis=2).sum(0)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed_example", type=int, default=0)
    ap.add_argument("--n_example", type=int, default=2)
    ap.add_argument("--analysis_root", default="analysis_outputs/Single agent path integration")
    ap.add_argument("--subdir", default="spatial_shift_allunits")
    args = ap.parse_args()
    base = os.path.join(_REPO, args.analysis_root)

    def npz(s):
        return np.load(os.path.join(base, f"Seed {s}", args.subdir, "aarav_output_space", "trajectories.npz"))

    # example agents (one seed): the ones whose TRUE path covers the most ground
    ex = npz(args.seed_example); true = ex["true"]
    cover = np.linalg.norm(true - true.mean(0, keepdims=True), axis=2).mean(0)
    agents = np.argsort(cover)[::-1][:args.n_example]
    acol = plt.get_cmap("tab10")

    fig = plt.figure(figsize=(14, 9))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.3, 1.0], hspace=0.25, wspace=0.12)

    # top row: example trajectories
    for j, (label, key) in enumerate(COND):
        ax = fig.add_subplot(gs[0, j])
        for ai, a in enumerate(agents):
            c = acol(ai)
            ax.plot(true[:, a, 0], true[:, a, 1], color="0.55", lw=1.6)
            ax.plot(true[0, a, 0], true[0, a, 1], "o", color="0.3", ms=7, zorder=6)
            ax.plot(ex[key][:, a, 0], ex[key][:, a, 1], color=c, lw=1.8)
            ax.plot(ex[key][-1, a, 0], ex[key][-1, a, 1], "*", color=c, ms=16, markeredgecolor="k", zorder=6)
        ax.set_xlim(-1.15, 1.15); ax.set_ylim(-1.15, 1.15); ax.set_aspect("equal")
        ax.set_xticks([]); ax.set_yticks([]); ax.set_title(label, fontweight="bold", fontsize=12)
        if j == 0:
            ax.plot([], [], color="0.55", lw=1.6, label="true path")
            ax.plot([], [], color=acol(0), lw=1.8, label="decoded path")
            ax.legend(frameon=False, fontsize=9, loc="upper left")

    # bottom: total distance travelled per condition (10 seeds x 40 agents)
    keys = ["true", "Intact", "Predictive_ablated", "Grid_ablated"]
    klbl = {"true": "True path", "Intact": "Intact", "Predictive_ablated": "Predictive\nablated", "Grid_ablated": "Grid\nablated"}
    kcol = {"true": "#2ca02c", "Intact": "#000000", "Predictive_ablated": "#d62728", "Grid_ablated": "#1f77b4"}
    per_seed = {k: [] for k in keys}
    for s in range(10):
        d = npz(s)
        for k in keys:
            per_seed[k].append(float(path_length(d[k]).mean()))
    vals = {k: np.array(per_seed[k]) for k in keys}; x = np.arange(len(keys))
    ax = fig.add_subplot(gs[1, :])
    ax.bar(x, [vals[k].mean() for k in keys], yerr=[vals[k].std() for k in keys], capsize=4,
           color=[kcol[k] for k in keys], alpha=0.85, width=0.6)
    for i, k in enumerate(keys):
        ax.scatter(np.full(10, i) + np.random.default_rng(i).uniform(-.1, .1, 10), vals[k], c="k", s=14, zorder=5)
    ax.set_xticks(x); ax.set_xticklabels([klbl[k] for k in keys], fontsize=10)
    ax.set_ylabel("total distance travelled (m)"); ax.grid(alpha=0.25, axis="y")
    ax.set_title("How far the decoded position travels", fontweight="bold", fontsize=12)

    fig.suptitle("Toroidal ablation in output space: example decoded trajectories and distance travelled (10 seeds)",
                 fontsize=13, fontweight="bold")
    out = os.path.join(base, "summary", "aarav_activity_space_ablation", "ablation_trajectories.png")
    fig.savefig(out, dpi=190, bbox_inches="tight"); fig.savefig(out.replace(".png", ".svg"), bbox_inches="tight")
    print("wrote", out)
    print("total distance travelled (m): " + "  ".join(f"{klbl[k].replace(chr(10),' ')}={vals[k].mean():.2f}" for k in keys))


if __name__ == "__main__":
    main()
