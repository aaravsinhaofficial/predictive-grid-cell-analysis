#!/usr/bin/env python3
"""Gate check: do STRAIGHT-trained networks develop grid cells at all?

If they do not, there are no predictive grid cells to ablate and the freeze experiment is undefined
(which is itself the finding). Compares gridness distributions of a straight-trained net vs a
random-walk-trained net, and renders the top-gridness rate maps for each.
"""
from __future__ import annotations
import argparse, os, sys
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
_REPO = os.path.dirname(_HERE)
from aarav_crossing_population_correlation import load_model
from visualize import compute_ratemaps
from scores import GridScorer


def gridness_all(model, traj_gen, opt, res=20, n_avg=25, sample_style="random_walk"):
    opt.trajectory_style = sample_style
    rate_maps = compute_ratemaps(model, traj_gen, opt, res=res, n_avg=n_avg, Ng=model.Ng)
    rm = np.asarray(rate_maps[0] if isinstance(rate_maps, tuple) else rate_maps, float)  # [Ng,res,res]
    starts = [0.2] * 10
    ends = np.linspace(0.4, 1.0, num=10)
    cr = ((-opt.box_width / 2, opt.box_width / 2), (-opt.box_height / 2, opt.box_height / 2))
    scorer = GridScorer(res, cr, zip(starts, ends.tolist()))
    g60 = np.full(rm.shape[0], np.nan)
    for u in range(rm.shape[0]):
        try:
            g60[u] = scorer.get_scores(rm[u])[0]
        except Exception:
            g60[u] = np.nan
    return rm, g60


def summarize(tag, g60):
    v = g60[np.isfinite(g60)]
    return (f"{tag:16s} n={v.size} mean={v.mean():.3f} max={v.max():.3f} "
            f"| gridness>0.2: {(v>0.2).sum()} ({100*(v>0.2).mean():.1f}%)  "
            f">0.3: {(v>0.3).sum()}  >0.37: {(v>0.37).sum()}")


def plot_top(rm, g60, ax_row, title):
    order = np.argsort(np.nan_to_num(g60, nan=-9))[::-1][:8]
    for k, u in enumerate(order):
        ax = ax_row[k]
        ax.imshow(rm[u], interpolation="none", cmap="jet"); ax.axis("off")
        ax.set_title(f"g={g60[u]:.2f}", fontsize=8)
    ax_row[0].set_ylabel(title, fontsize=10)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--res", type=int, default=20)
    ap.add_argument("--seq_len", type=int, default=40)
    args = ap.parse_args()
    dev = args.device if torch.cuda.is_available() else "cpu"
    out = os.path.join(_REPO, "analysis_outputs/Single agent path integration/summary/aarav_straight_trained")
    os.makedirs(out, exist_ok=True)

    nets = [
        ("straight-Seed0-straightsample", "Models/straight/steps_40/Seed 0/most_recent_model.pth", "straight"),
        ("straight-Seed1-straightsample", "Models/straight/steps_40/Seed 1/most_recent_model.pth", "straight"),
        ("randomwalk-Seed0", "Models/Single agent path integration/Seed 0/most_recent_model.pth", "random_walk"),
    ]
    fig, axes = plt.subplots(len(nets), 8, figsize=(16, 2.2 * len(nets)))
    results = {}
    for i, (tag, path, style) in enumerate(nets):
        model, pc, tg, opt, Ng, Np = load_model(os.path.join(_REPO, path), dev, args.seq_len)
        rm, g60 = gridness_all(model, tg, opt, res=args.res, sample_style=style)
        results[tag] = g60
        print(summarize(tag, g60), flush=True)
        plot_top(rm, g60, axes[i], tag)
        np.save(os.path.join(out, f"gridness60_{tag}.npy"), g60)
    fig.suptitle("Top-gridness rate maps: straight-trained vs random-walk-trained", fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(os.path.join(out, "straight_grid_gatecheck.png"), dpi=160, bbox_inches="tight")
    print("wrote", os.path.join(out, "straight_grid_gatecheck.png"))


if __name__ == "__main__":
    main()
