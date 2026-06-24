#!/usr/bin/env python3
"""Animate activity moving on the torus over timesteps, under each ablation/freeze condition.

Fits the torus basis (coherent grid module), runs a batch of random-walk trajectories through the
network under each condition, projects the time-resolved activity onto the torus, and renders an
MP4 where each panel shows the activity points flowing around the donut over timesteps (current
points bold, a fading trail behind). You can watch traversal flow (Intact) vs clump/collapse
(ablations). Count-matched to N = #predictive where applicable.

Usage:  python code/aarav_torus_video.py --seed 0 --device cuda
"""
from __future__ import annotations
import argparse, copy, os, sys
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import imageio.v2 as imageio

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
_REPO = os.path.dirname(_HERE)

from aarav_crossing_population_correlation import load_model, load_classes
from visualize import compute_ratemaps
from toroidal_structure_analysis import build_torus_basis, project_states_to_torus, identify_toroidal_cells
from multi_seed_predictive_analysis import zero_unit_weights_in_place
from aarav_predictive_intervention import manual_forward
from aarav_activity_space_ablation import make_random_walk_inputs


def proj_coords(g, basis, T, B):
    p = project_states_to_torus(g, basis, (1.0, 0.35), T, B)
    return p.coords3d.reshape(T, B, 3).astype(np.float32), p.theta1.astype(np.float32)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--model_root", default="Models/Single agent path integration")
    ap.add_argument("--analysis_root", default="analysis_outputs/Single agent path integration")
    ap.add_argument("--analysis_subdir", default="spatial_shift_allunits")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--gridness_threshold", type=float, default=0.2)
    ap.add_argument("--min_shift_cm", type=float, default=5.0)
    ap.add_argument("--seq_len", type=int, default=40)
    ap.add_argument("--res", type=int, default=40)
    ap.add_argument("--n_traj", type=int, default=60)
    ap.add_argument("--trail", type=int, default=6)
    ap.add_argument("--fps", type=int, default=6)
    args = ap.parse_args()
    device = args.device if torch.cuda.is_available() else "cpu"
    s = args.seed
    ckpt = os.path.join(_REPO, args.model_root, f"Seed {s}", "most_recent_model.pth")
    gpath = os.path.join(_REPO, args.analysis_root, f"Seed {s}", args.analysis_subdir, "gridness_data.npz")
    out_dir = os.path.join(_REPO, args.analysis_root, "summary", "aarav_activity_space_ablation")
    os.makedirs(out_dir, exist_ok=True)

    model, place_cells, traj_gen, opt, Ng, Np = load_model(ckpt, device, args.seq_len)
    classes = load_classes(gpath, args.gridness_threshold, args.min_shift_cm)
    grid_all = classes["grid_all"]; predictive = classes["predictive"]
    structural = np.setdiff1d(grid_all, predictive); all_units = np.arange(Ng)
    N = int(predictive.size); rng = np.random.default_rng(123 + s)

    # torus basis on the detected coherent module
    rate_maps, _, _, _ = compute_ratemaps(model, traj_gen, opt, res=args.res, n_avg=12, Ng=grid_all.size, idxs=grid_all)
    rate_maps = np.asarray(rate_maps, float)
    det = identify_toroidal_cells(rate_maps, grid_all, opt.box_width, out_dir, embed_mode="umap")
    tor = np.asarray(det.units, int)
    loc = np.array([{int(g): i for i, g in enumerate(grid_all)}[int(u)] for u in tor])
    basis = build_torus_basis(rate_maps[loc], np.arange(loc.size), opt.box_width)
    basis.units = tor
    print(f"seed {s}: module={tor.size}  N(predictive)={N}", flush=True)

    inputs, pos = make_random_walk_inputs(traj_gen, args.n_traj, args.seq_len, 777 + s)
    v, init_actv = inputs
    T, B = args.seq_len, args.n_traj

    def g_ablate(units):
        if units is None:
            m = model
        else:
            m = copy.deepcopy(model); zero_unit_weights_in_place(m, list(units))
        with torch.no_grad():
            g = m.g(inputs).detach().cpu().numpy()
        return g

    with torch.no_grad():
        g_intact_t = manual_forward(model, v, init_actv)
    frozen_val = g_intact_t[0][:, torch.as_tensor(predictive, device=device)]
    def g_freeze(units):
        ci = torch.as_tensor(np.asarray(units, int), device=device)
        with torch.no_grad():
            g = manual_forward(model, v, init_actv, clamp_idx=ci, clamp_val=g_intact_t[0][:, ci]).detach().cpu().numpy()
        return g

    conds = [
        ("Intact", g_ablate(None)),
        ("Ablate predictive (N)", g_ablate(predictive)),
        ("Ablate random (N)", g_ablate(rng.choice(all_units, N, replace=False))),
        ("Ablate structural grid (N)", g_ablate(rng.choice(structural, min(N, structural.size), replace=False))),
        ("Ablate ALL grid", g_ablate(grid_all)),
        ("Freeze predictive (N)", g_freeze(predictive)),
    ]
    data = []
    for name, g in conds:
        c, th = proj_coords(g, basis, T, B)
        data.append((name, c, th))

    # fixed axis limits from the intact donut
    ic = data[0][1].reshape(-1, 3)
    lim = np.array([ic[:, i].min() for i in range(3)]), np.array([ic[:, i].max() for i in range(3)])
    pad = 0.15 * (lim[1] - lim[0] + 1e-6)
    lo, hi = lim[0] - pad, lim[1] + pad

    out = os.path.join(out_dir, f"torus_traversal_video_seed{s}.mp4")
    frames = []
    fig = plt.figure(figsize=(15, 9))
    axes = [fig.add_subplot(2, 3, i + 1, projection="3d") for i in range(6)]
    for t in range(T):
        for ax, (name, coords, th) in zip(axes, data):
            ax.cla()
            a = max(0, t - args.trail)
            tr = coords[a:t].reshape(-1, 3); trc = th[a:t].reshape(-1)
            if tr.shape[0]:
                ax.scatter(tr[:, 0], tr[:, 1], tr[:, 2], c=trc, cmap="hsv", s=4, alpha=0.18, linewidths=0)
            cur = coords[t]; curc = th[t]
            ax.scatter(cur[:, 0], cur[:, 1], cur[:, 2], c=curc, cmap="hsv", s=26, alpha=0.95, linewidths=0)
            ax.set_xlim(lo[0], hi[0]); ax.set_ylim(lo[1], hi[1]); ax.set_zlim(lo[2], hi[2])
            ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
            ax.set_box_aspect((1, 1, 0.55)); ax.view_init(elev=80, azim=0)
            ax.set_title(name, fontsize=11)
        fig.suptitle(f"Activity on the torus over time — timestep {t+1}/{T} (seed {s})",
                     fontsize=14, fontweight="bold")
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        fig.canvas.draw()
        w, h = fig.canvas.get_width_height()
        buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape(h, w, 4)[:, :, :3]
        # pad to even dims for h264
        H, W = buf.shape[:2]
        buf = buf[:H - (H % 2), :W - (W % 2)]
        frames.append(buf.copy())
        if t % 10 == 0:
            print(f"  rendered frame {t+1}/{T}", flush=True)
    plt.close(fig)
    imageio.mimsave(out, frames, fps=args.fps, codec="libx264", quality=8)
    print("wrote", out, f"({os.path.getsize(out)/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
