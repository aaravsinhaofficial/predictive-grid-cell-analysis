#!/usr/bin/env python3
"""Side-by-side TRUE vs DECODED trajectory video, per ablation condition (output-space validation).

Reads the saved decoded paths and animates, over timesteps, a few example agents' true path
(gray) vs the network's decoded path (colored) in the 2D arena, for Intact / Predictive-ablated /
Grid-ablated. Shows what the network's path-integration output actually does: intact tracks the
truth; ablations break it. (Empirically the decoded path WANDERS under ablation — most under
predictive ablation — rather than clumping; see aarav_output_space_summary.)

Usage:  python code/aarav_output_video.py --seed 0
"""
from __future__ import annotations
import argparse, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import imageio.v2 as imageio

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONDS = [("Intact", "Intact"), ("Predictive-ablated", "Predictive_ablated"), ("Grid-ablated", "Grid_ablated")]


def gyr(p):
    c = p.mean(0, keepdims=True)
    return float(np.sqrt(((p - c) ** 2).sum(2).mean(0)).mean())


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--analysis_root", default="analysis_outputs/Single agent path integration")
    ap.add_argument("--subdir", default="spatial_shift_allunits")
    ap.add_argument("--n_agents", type=int, default=4)
    ap.add_argument("--fps", type=int, default=6)
    args = ap.parse_args()
    s = args.seed
    npz = np.load(os.path.join(_REPO, args.analysis_root, f"Seed {s}", args.subdir, "aarav_output_space", "trajectories.npz"))
    true = npz["true"]                                       # [T,B,2]
    T, B, _ = true.shape
    # pick agents whose true path covers the most ground (clearest)
    spread = np.sqrt(((true - true.mean(0, keepdims=True)) ** 2).sum(2).mean(0))
    agents = np.argsort(spread)[::-1][:args.n_agents]
    acol = plt.get_cmap("tab10")
    out = os.path.join(_REPO, args.analysis_root, "summary", "aarav_activity_space_ablation",
                       f"output_true_vs_decoded_seed{s}.mp4")
    os.makedirs(os.path.dirname(out), exist_ok=True)

    frames = []
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.4))
    for t in range(1, T):
        for ax, (label, key) in zip(axes, CONDS):
            ax.cla()
            dec = npz[key]
            for ai, a in enumerate(agents):
                col = acol(ai % 10)
                ax.plot(true[:t + 1, a, 0], true[:t + 1, a, 1], color="0.6", lw=1.4, alpha=0.7)
                ax.plot(true[t, a, 0], true[t, a, 1], "o", color="0.4", ms=6)
                ax.plot(dec[:t + 1, a, 0], dec[:t + 1, a, 1], color=col, lw=1.6, alpha=0.9)
                ax.plot(dec[t, a, 0], dec[t, a, 1], "*", color=col, ms=15, markeredgecolor="k")
            ax.set_xlim(-1.15, 1.15); ax.set_ylim(-1.15, 1.15); ax.set_xticks([]); ax.set_yticks([])
            ax.set_aspect("equal")
            ax.set_title(f"{label}\ndecoded spread={gyr(dec):.2f} m (true {gyr(true):.2f})", fontsize=11, fontweight="bold")
        fig.suptitle(f"True (gray) vs network-decoded (color) position — timestep {t+1}/{T}, seed {s}",
                     fontsize=13, fontweight="bold")
        fig.tight_layout(rect=[0, 0, 1, 0.94])
        fig.canvas.draw()
        w, h = fig.canvas.get_width_height()
        buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape(h, w, 4)[:, :, :3]
        H, W = buf.shape[:2]
        frames.append(buf[:H - H % 2, :W - W % 2].copy())
    plt.close(fig)
    imageio.mimsave(out, frames, fps=args.fps, codec="libx264", quality=8)
    print("wrote", out, f"({os.path.getsize(out)/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
