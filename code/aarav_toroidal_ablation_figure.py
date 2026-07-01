#!/usr/bin/env python3
"""Composite toroidal-ablation figure.

Row 1  latent (torus) trajectories : Intact / Predictive / Grid, example paths on the 3D donut.
Row 2  output (decoded) trajectories: Intact / Predictive / Grid, SAME agents in the arena (contrast).
Row 3  quantification (10 seeds):
         - population self-similarity / freeze  (autocorr @ lag 20)  [Intact / Predictive / Random]
         - torus occupancy / coverage           (theta1 spread)      [Intact / Predictive / Grid / Random]
         - off-manifold fraction                (kNN to intact cloud)[Intact / Predictive / Random]
Grid is shown in the trajectory rows (it destroys the torus) and only where meaningful in the bars.

Run AFTER aarav_torus_geometry_metrics.py has produced torus_geometry_stats.json for all seeds.
"""
from __future__ import annotations
import argparse, copy, json, os, sys
import numpy as np
import torch
from scipy import stats
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

SCRATCH = "/tmp/claude-1000/-home-ec2-user-predictive-grid-cell-analysis/ba8b77e9-dcb6-4bd8-864f-7e091e2bda68/scratchpad"
BASE = os.path.join(_REPO, "analysis_outputs/Single agent path integration")
TCOL = plt.get_cmap("tab10")
COL = {"Intact": "#000000", "Predictive-ablated": "#d62728", "Grid-ablated": "#1f77b4", "Random-ablated (N)": "#7f7f7f"}
LBL = {"Intact": "Intact", "Predictive-ablated": "Predictive\nablated", "Grid-ablated": "Grid\nablated", "Random-ablated (N)": "Random\nablated"}


def latent_paths(seed, seq_len, n_traj, device):
    """Return {cond: coords3d[T,B,3]} for Intact/Predictive/Grid + theta1 for particle picking."""
    model, pc, tg, opt, Ng, Np = load_model(f"Models/Single agent path integration/Seed {seed}/most_recent_model.pth", device, seq_len)
    cl = load_classes(f"analysis_outputs/Single agent path integration/Seed {seed}/spatial_shift_allunits/gridness_data.npz", 0.2, 5.0)
    pred, grid = cl["predictive"], cl["grid_all"]
    rm, _, _, _ = compute_ratemaps(model, tg, opt, res=40, n_avg=10, Ng=grid.size, idxs=grid)
    rm = np.asarray(rm, float)
    det = identify_toroidal_cells(rm, grid, opt.box_width, SCRATCH, embed_mode="umap")
    tor = np.asarray(det.units, int)
    loc = np.array([{int(g): i for i, g in enumerate(grid)}[int(u)] for u in tor])
    basis = build_torus_basis(rm[loc], np.arange(loc.size), opt.box_width); basis.units = tor
    inputs, pos = make_random_walk_inputs(tg, n_traj, seq_len, 777 + seed)
    T, B = seq_len, n_traj

    def proj(u):
        m = model if u is None else copy.deepcopy(model)
        if u is not None:
            zero_unit_weights_in_place(m, list(u))
        with torch.no_grad():
            g = m.g(inputs).detach().cpu().numpy()
        p = project_states_to_torus(g, basis, (1.0, 0.35), T, B)
        return p.coords3d.reshape(T, B, 3), p.theta1
    out = {}
    out["Intact"] = proj(None); out["Predictive-ablated"] = proj(pred); out["Grid-ablated"] = proj(grid)
    return out


def bars(ax, keys, vals, title, ylab, ref=None):
    x = np.arange(len(keys))
    ax.bar(x, [vals[k].mean() for k in keys], yerr=[vals[k].std() for k in keys], capsize=4,
           color=[COL[k] for k in keys], alpha=0.85, width=0.6)
    for i, k in enumerate(keys):
        ax.scatter(np.full(len(vals[k]), i) + np.random.default_rng(i).uniform(-.1, .1, len(vals[k])), vals[k], c="k", s=11, zorder=5)
    if ref is not None:
        ax.axhline(ref, ls="--", c="0.4", lw=1)
    ax.set_xticks(x); ax.set_xticklabels([LBL[k] for k in keys], fontsize=8.5)
    ax.set_ylabel(ylab); ax.set_title(title, fontweight="bold", fontsize=11); ax.grid(alpha=0.25, axis="y")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed_example", type=int, default=1)
    ap.add_argument("--n_particles", type=int, default=3)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seq_len", type=int, default=40)
    args = ap.parse_args()
    device = args.device if torch.cuda.is_available() else "cpu"
    s = args.seed_example

    # --- example trajectories: latent (recompute) + output (saved npz), SAME agents (same inputs seed)
    lat = latent_paths(s, args.seq_len, 40, device)
    out = np.load(os.path.join(BASE, f"Seed {s}", "spatial_shift_allunits", "aarav_output_space", "trajectories.npz"))
    th_i = lat["Intact"][1]; B = th_i.shape[1]
    spans = np.array([np.ptp(np.unwrap(th_i[:, b])) for b in range(B)])
    parts = np.argsort(spans)[::-1][:args.n_particles]
    ic = lat["Intact"][0].reshape(-1, 3); lo = ic.min(0) - 0.15 * np.ptp(ic, 0); hi = ic.max(0) + 0.15 * np.ptp(ic, 0)
    bd = ic[np.random.default_rng(0).choice(ic.shape[0], min(1200, ic.shape[0]), replace=False)]

    fig = plt.figure(figsize=(15, 14))
    gs = fig.add_gridspec(3, 3, height_ratios=[1.05, 1.0, 0.9], hspace=0.2, wspace=0.18)
    tconds = [("Intact", "Intact"), ("Predictive-ablated", "Predictive_ablated"), ("Grid-ablated", "Grid_ablated")]

    # row 1: latent torus trajectories
    for j, (cond, _) in enumerate(tconds):
        ax = fig.add_subplot(gs[0, j], projection="3d")
        coords = lat[cond][0]
        ax.scatter(bd[:, 0], bd[:, 1], bd[:, 2], color="0.6", s=4, alpha=0.13, linewidths=0)
        for pi, b in enumerate(parts):
            p = coords[:, b]
            ax.plot(p[:, 0], p[:, 1], p[:, 2], color=TCOL(pi), lw=1.7)
            ax.scatter(p[0, 0], p[0, 1], p[0, 2], color=TCOL(pi), s=40, marker="o", edgecolor="k", zorder=6)
            ax.scatter(p[-1, 0], p[-1, 1], p[-1, 2], color=TCOL(pi), s=120, marker="*", edgecolor="k", zorder=6)
        ax.set_xlim(lo[0], hi[0]); ax.set_ylim(lo[1], hi[1]); ax.set_zlim(lo[2], hi[2])
        ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([]); ax.set_box_aspect((1, 1, 0.6)); ax.view_init(elev=38, azim=-62)
        ax.set_title(("Latent (torus): " if j == 0 else "") + LBL[cond].replace("\n", " "), fontweight="bold", fontsize=11)

    # row 2: output decoded trajectories (same agents)
    true = out["true"]
    for j, (cond, okey) in enumerate(tconds):
        ax = fig.add_subplot(gs[1, j])
        dec = out[okey]
        for pi, b in enumerate(parts):
            ax.plot(true[:, b, 0], true[:, b, 1], color="0.6", lw=1.3)
            ax.plot(true[0, b, 0], true[0, b, 1], "o", color="0.35", ms=6, zorder=6)
            ax.plot(dec[:, b, 0], dec[:, b, 1], color=TCOL(pi), lw=1.6)
            ax.plot(dec[-1, b, 0], dec[-1, b, 1], "*", color=TCOL(pi), ms=15, markeredgecolor="k", zorder=6)
        ax.set_xlim(-1.15, 1.15); ax.set_ylim(-1.15, 1.15); ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(("Output (decoded): " if j == 0 else "") + LBL[cond].replace("\n", " "), fontweight="bold", fontsize=11)
        if j == 0:
            ax.plot([], [], color="0.6", lw=1.3, label="true"); ax.plot([], [], color=TCOL(0), lw=1.6, label="decoded")
            ax.legend(frameon=False, fontsize=8, loc="upper left")

    # row 3: quantification
    pop = [json.load(open(os.path.join(BASE, f"Seed {ss}", "spatial_shift_allunits", "aarav_population_change", "population_change_stats.json"))) for ss in range(10)]
    geo = [json.load(open(os.path.join(BASE, f"Seed {ss}", "spatial_shift_allunits", "aarav_torus_geometry", "torus_geometry_stats.json"))) for ss in range(10)]

    frz_keys = ["Intact", "Predictive-ablated", "Random-ablated (N)"]
    frz = {k: np.array([j[k]["autocorr_lag20"] for j in pop]) for k in frz_keys}
    ax = fig.add_subplot(gs[2, 0]); bars(ax, frz_keys, frz, "Population self-similarity (freeze)", "autocorr at 20-step lag")

    occ_keys = ["Intact", "Predictive-ablated", "Grid-ablated", "Random-ablated (N)"]
    occ = {k: np.array([j[k]["occupancy_spread"] for j in geo]) for k in occ_keys}
    ax = fig.add_subplot(gs[2, 1]); bars(ax, occ_keys, occ, "Torus occupancy / coverage", "theta1 spread (1 - resultant)")

    # third panel: OUTPUT-space distance travelled (the latent->output contrast). Off-manifold was
    # dropped (not predictive-specific: pred~random~grid, p=0.85).
    def path_length(x):
        return np.linalg.norm(np.diff(x, axis=0), axis=2).sum(0).mean()
    okeys = ["true", "Intact", "Predictive_ablated", "Grid_ablated"]
    ocol = {"true": "#2ca02c", "Intact": "#000000", "Predictive_ablated": "#d62728", "Grid_ablated": "#1f77b4"}
    olbl = {"true": "True", "Intact": "Intact", "Predictive_ablated": "Predictive\nablated", "Grid_ablated": "Grid\nablated"}
    dist = {k: [] for k in okeys}
    for ss in range(10):
        d = np.load(os.path.join(BASE, f"Seed {ss}", "spatial_shift_allunits", "aarav_output_space", "trajectories.npz"))
        for k in okeys:
            dist[k].append(float(path_length(d[k])))
    dist = {k: np.array(v) for k, v in dist.items()}
    ax = fig.add_subplot(gs[2, 2]); x = np.arange(len(okeys))
    ax.bar(x, [dist[k].mean() for k in okeys], yerr=[dist[k].std() for k in okeys], capsize=4,
           color=[ocol[k] for k in okeys], alpha=0.85, width=0.6)
    for i, k in enumerate(okeys):
        ax.scatter(np.full(10, i) + np.random.default_rng(i).uniform(-.1, .1, 10), dist[k], c="k", s=11, zorder=5)
    ax.set_xticks(x); ax.set_xticklabels([olbl[k] for k in okeys], fontsize=8.5)
    ax.set_ylabel("distance travelled (m)"); ax.set_title("Output: distance travelled (decoded)", fontweight="bold", fontsize=11)
    ax.grid(alpha=0.25, axis="y")

    def sig(a, b):
        return stats.wilcoxon(a, b).pvalue
    print(f"freeze pred>random p={sig(frz['Predictive-ablated'], frz['Random-ablated (N)']):.3f}")
    print(f"occupancy pred<random p={sig(occ['Predictive-ablated'], occ['Random-ablated (N)']):.3f}")

    fig.suptitle("Toroidal ablation: latent-space vs output-space trajectories and quantification (examples seed %d; bars 10 seeds)" % s,
                 fontsize=13, fontweight="bold", y=0.99)
    outp = os.path.join(BASE, "summary", "aarav_activity_space_ablation", "toroidal_ablation_figure.png")
    fig.savefig(outp, dpi=175, bbox_inches="tight"); fig.savefig(outp.replace(".png", ".svg"), bbox_inches="tight")
    print("wrote", outp)


if __name__ == "__main__":
    main()
