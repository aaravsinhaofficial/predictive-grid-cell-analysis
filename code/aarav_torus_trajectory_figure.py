#!/usr/bin/env python3
"""Latent-space (torus) trajectory figure: static versions of the torus video + how far the
activity travels around the torus.

Top row: a few example trajectories projected onto the torus (3D donut), full path shown, for
Intact / Predictive-ablated / Random-ablated (same machinery as aarav_torus_video.py).
Bottom: angular distance travelled around the torus (theta1) per condition, 10 seeds
(reuses the saved torus-residual angular-velocity metric so no extra model runs).
"""
from __future__ import annotations
import argparse, copy, json, os, sys
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
_REPO = os.path.dirname(_HERE)
from aarav_crossing_population_correlation import load_model, load_classes
from aarav_activity_space_ablation import make_random_walk_inputs
from multi_seed_predictive_analysis import zero_unit_weights_in_place
from visualize import compute_ratemaps
from toroidal_structure_analysis import build_torus_basis, project_states_to_torus, identify_toroidal_cells

COND = [("Intact", None), ("Predictive ablated", "pred"), ("Random ablated", "rand")]
PCOL = plt.get_cmap("tab10")
SCRATCH = "/tmp/claude-1000/-home-ec2-user-predictive-grid-cell-analysis/ba8b77e9-dcb6-4bd8-864f-7e091e2bda68/scratchpad"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed_example", type=int, default=1)
    ap.add_argument("--n_particles", type=int, default=4)
    ap.add_argument("--n_traj", type=int, default=60)
    ap.add_argument("--seq_len", type=int, default=40)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    device = args.device if torch.cuda.is_available() else "cpu"
    s = args.seed_example
    base = os.path.join(_REPO, "analysis_outputs/Single agent path integration")
    model, pc, tg, opt, Ng, Np = load_model(f"Models/Single agent path integration/Seed {s}/most_recent_model.pth", device, args.seq_len)
    cl = load_classes(f"analysis_outputs/Single agent path integration/Seed {s}/spatial_shift_allunits/gridness_data.npz", 0.2, 5.0)
    pred, grid = cl["predictive"], cl["grid_all"]; allu = np.arange(Ng); N = pred.size
    rng = np.random.default_rng(7 + s); randN = rng.choice(allu, N, replace=False)

    rate_maps, _, _, _ = compute_ratemaps(model, tg, opt, res=40, n_avg=10, Ng=grid.size, idxs=grid)
    rate_maps = np.asarray(rate_maps, float)
    det = identify_toroidal_cells(rate_maps, grid, opt.box_width, SCRATCH, embed_mode="umap")
    tor = np.asarray(det.units, int)
    loc = np.array([{int(g): i for i, g in enumerate(grid)}[int(u)] for u in tor])
    basis = build_torus_basis(rate_maps[loc], np.arange(loc.size), opt.box_width); basis.units = tor

    inputs, pos = make_random_walk_inputs(tg, args.n_traj, args.seq_len, 777 + s)
    T, B = args.seq_len, args.n_traj

    def project(units):
        m = model if units is None else copy.deepcopy(model)
        if units is not None:
            zero_unit_weights_in_place(m, list(units))
        with torch.no_grad():
            g = m.g(inputs).detach().cpu().numpy()
        p = project_states_to_torus(g, basis, (1.0, 0.35), T, B)
        return p.coords3d.reshape(T, B, 3), p.theta1

    data = {"Intact": project(None), "Predictive ablated": project(pred), "Random ablated": project(randN)}

    # particles that traverse the most in intact (largest unwrapped theta1 span)
    th_i = data["Intact"][1]
    spans = np.array([np.ptp(np.unwrap(th_i[:, b])) for b in range(B)])
    parts = np.argsort(spans)[::-1][:args.n_particles]

    # fixed axis limits from intact donut
    ic = data["Intact"][0].reshape(-1, 3)
    lim_lo = ic.min(0); lim_hi = ic.max(0); pad = 0.15 * (lim_hi - lim_lo + 1e-6)
    lo, hi = lim_lo - pad, lim_hi + pad
    bd = ic[np.random.default_rng(0).choice(ic.shape[0], min(1500, ic.shape[0]), replace=False)]

    fig = plt.figure(figsize=(16, 11))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.7, 1.0], hspace=0.02, wspace=0.02)
    for j, (label, _) in enumerate(COND):
        ax = fig.add_subplot(gs[0, j], projection="3d")
        coords = data[label][0]
        ax.scatter(bd[:, 0], bd[:, 1], bd[:, 2], color="0.6", s=3, alpha=0.06, linewidths=0)
        for pi, b in enumerate(parts):
            path = coords[:, b]
            ax.plot(path[:, 0], path[:, 1], path[:, 2], color=PCOL(pi), lw=1.8, alpha=0.9)
            ax.scatter(path[0, 0], path[0, 1], path[0, 2], color=PCOL(pi), s=45, marker="o", edgecolor="k", zorder=6)
            ax.scatter(path[-1, 0], path[-1, 1], path[-1, 2], color=PCOL(pi), s=130, marker="*", edgecolor="k", zorder=6)
        ax.set_xlim(lo[0], hi[0]); ax.set_ylim(lo[1], hi[1]); ax.set_zlim(lo[2], hi[2])
        ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
        ax.set_box_aspect((1, 1, 0.62)); ax.view_init(elev=38, azim=-62)
        ax.set_title(label, fontweight="bold", fontsize=13, y=0.97)

    # quantification: angular distance travelled around the torus (theta1), 10 seeds
    J = [json.load(open(os.path.join(base, f"Seed {ss}", "spatial_shift_allunits", "aarav_torus_residual", "torus_residual_stats.json"))) for ss in range(10)]
    keys = [("Intact", "#000000"), ("Predictive-ablated", "#d62728"), ("Random-ablated (N)", "#7f7f7f")]
    klbl = {"Intact": "Intact", "Predictive-ablated": "Predictive\nablated", "Random-ablated (N)": "Random\nablated"}
    loops = {k: np.array([j[k]["angvel1_mean"] for j in J]) * (args.seq_len - 1) / (2 * np.pi) for k, _ in keys}
    ax = fig.add_subplot(gs[1, :])
    x = np.arange(len(keys))
    ax.bar(x, [loops[k].mean() for k, _ in keys], yerr=[loops[k].std() for k, _ in keys], capsize=4,
           color=[c for _, c in keys], alpha=0.85, width=0.55)
    for i, (k, _) in enumerate(keys):
        ax.scatter(np.full(10, i) + np.random.default_rng(i).uniform(-.1, .1, 10), loops[k], c="k", s=14, zorder=5)
    ax.set_xticks(x); ax.set_xticklabels([klbl[k] for k, _ in keys], fontsize=10)
    ax.set_ylabel("angular distance travelled\naround the torus (loops)")
    ax.set_title("How far the activity travels around the torus", fontweight="bold", fontsize=12)
    ax.grid(alpha=0.25, axis="y")

    fig.suptitle("Toroidal ablation in latent space: example trajectories on the torus and how far the activity travels (examples: seed %d)" % s,
                 fontsize=13, fontweight="bold")
    out = os.path.join(base, "summary", "aarav_activity_space_ablation", "torus_trajectories.png")
    fig.savefig(out, dpi=180, bbox_inches="tight"); fig.savefig(out.replace(".png", ".svg"), bbox_inches="tight")
    print("wrote", out)
    for k, _ in keys:
        print(f"  {k:22s} angular distance travelled = {loops[k].mean():.2f} loops")


if __name__ == "__main__":
    main()
