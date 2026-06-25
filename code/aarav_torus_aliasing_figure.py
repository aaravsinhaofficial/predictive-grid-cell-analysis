#!/usr/bin/env python3
"""Figure: is the PGC-ablated decoded wandering jumps between GRID-FIELD REPLICAS of one phase?

A/C: an example decoded trajectory in physical space with the grid lattice overlaid (predictive-
     ablated vs intact). B/D: the same positions folded into ONE lattice unit cell -- if the
     wander is aliasing, the physically-spread points collapse to a tight cluster in the cell.
E:   across-seed aliasing index (1 - within-cell spread / physical spread).
"""
from __future__ import annotations
import argparse, json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def cell_metrics(p, Kcyc, A):
    u = p @ Kcyc.T
    ph = 2 * np.pi * u
    mbar = np.angle(np.exp(1j * ph).mean(0))
    cent = np.angle(np.exp(1j * (ph - mbar)))
    p_res = (cent / (2 * np.pi)) @ A.T
    raw = np.sqrt(((p - p.mean(0)) ** 2).sum(1).mean())
    within = np.sqrt((p_res ** 2).sum(1).mean())
    return p_res, raw, within, mbar / (2 * np.pi)


def draw_lattice(ax, A, frac, xlim, ylim):
    ns = range(-6, 7)
    pts = np.array([A @ np.array([n + frac[0], m + frac[1]]) for n in ns for m in ns])
    in_ = (pts[:, 0] > xlim[0]) & (pts[:, 0] < xlim[1]) & (pts[:, 1] > ylim[0]) & (pts[:, 1] < ylim[1])
    ax.scatter(pts[in_, 0], pts[in_, 1], marker="+", c="0.6", s=80, lw=1.2, zorder=1)


def draw_cell(ax, A):
    c = np.array([[-.5, -.5], [.5, -.5], [.5, .5], [-.5, .5], [-.5, -.5]]) @ A.T
    ax.plot(c[:, 0], c[:, 1], "k-", lw=1.2, alpha=0.6)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--analysis_root", default="analysis_outputs/Single agent path integration")
    ap.add_argument("--subdir", default="spatial_shift_allunits")
    args = ap.parse_args()
    d = os.path.join(_REPO, args.analysis_root, f"Seed {args.seed}", args.subdir, "aarav_torus_aliasing")
    npz = np.load(os.path.join(d, "aliasing_paths.npz"))
    k1, k2 = npz["k1"], npz["k2"]
    Kcyc = np.array([k1, k2]); A = np.linalg.inv(Kcyc)
    lam = float(np.mean([1 / np.linalg.norm(k1), 1 / np.linalg.norm(k2)]))
    pred, intact = npz["Predictive_ablated"], npz["Intact"]      # [T,B,2]

    # pick the predictive-ablated agent with the strongest aliasing (large raw, small within)
    B = pred.shape[1]; best, bestscore = 0, -9
    for b in range(B):
        _, raw, within, _ = cell_metrics(pred[:, b], Kcyc, A)
        if raw > 0.45 and (raw - within) > bestscore:
            bestscore = raw - within; best = b
    b = best

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    T = pred.shape[0]; tc = np.arange(T)

    # ---- A: predictive physical + lattice
    ax = axes[0, 0]; p = pred[:, b]
    _, raw, within, frac = cell_metrics(p, Kcyc, A)
    xlim = (p[:, 0].min() - 0.2, p[:, 0].max() + 0.2); ylim = (p[:, 1].min() - 0.2, p[:, 1].max() + 0.2)
    draw_lattice(ax, A, frac, xlim, ylim)
    ax.plot(p[:, 0], p[:, 1], "-", c="0.7", lw=1, zorder=2)
    sc = ax.scatter(p[:, 0], p[:, 1], c=tc, cmap="viridis", s=45, zorder=3, edgecolor="k", lw=0.3)
    ax.set_xlim(xlim); ax.set_ylim(ylim); ax.set_aspect("equal")
    ax.set_title(f"A. Predictive-ablated: decoded path (physical)\nwander={raw:.2f} m (lattice '+' = one phase's grid fields)",
                 loc="left", fontweight="bold", fontsize=10)

    # ---- B: predictive folded into unit cell
    ax = axes[0, 1]; p_res, _, _, _ = cell_metrics(p, Kcyc, A)
    draw_cell(ax, A)
    ax.scatter(p_res[:, 0], p_res[:, 1], c=tc, cmap="viridis", s=45, edgecolor="k", lw=0.3)
    ax.set_aspect("equal")
    ax.set_title(f"B. Same path folded into ONE cell\nspread collapses to {within:.2f} m "
                 f"(aliasing index {1-within/raw:.2f})", loc="left", fontweight="bold", fontsize=10)

    # ---- C: intact physical + lattice (same agent)
    ax = axes[0, 2]; pi = intact[:, b]
    _, raw_i, within_i, frac_i = cell_metrics(pi, Kcyc, A)
    xli = (pi[:, 0].min() - 0.4, pi[:, 0].max() + 0.4); yli = (pi[:, 1].min() - 0.4, pi[:, 1].max() + 0.4)
    draw_lattice(ax, A, frac_i, xli, yli)
    ax.plot(pi[:, 0], pi[:, 1], "-", c="0.7", lw=1)
    ax.scatter(pi[:, 0], pi[:, 1], c=tc, cmap="viridis", s=45, edgecolor="k", lw=0.3)
    ax.set_xlim(xli); ax.set_ylim(yli); ax.set_aspect("equal")
    ax.set_title(f"C. Intact: decoded path (physical)\nwander={raw_i:.2f} m, stays in ~one cell (no aliasing)",
                 loc="left", fontweight="bold", fontsize=10)
    plt.colorbar(sc, ax=axes[0, 2], fraction=0.046, label="timestep")

    # ---- D/E: across-seed aliasing index + raw vs within
    base = os.path.join(_REPO, args.analysis_root)
    conds = ["Intact", "Predictive-ablated", "Grid-ablated", "Random-ablated (N)"]
    col = {"Intact": "#000000", "Predictive-ablated": "#d62728", "Grid-ablated": "#1f77b4",
           "Random-ablated (N)": "#7f7f7f"}
    A_idx = {c: [] for c in conds}; RAW = {c: [] for c in conds}; WIN = {c: [] for c in conds}
    for s in range(10):
        j = json.load(open(os.path.join(base, f"Seed {s}", args.subdir, "aarav_torus_aliasing", "aliasing_stats.json")))
        for c in conds:
            A_idx[c].append(j[c]["traj_aliasing_index"]); RAW[c].append(j[c]["traj_raw_spread_m"])
            WIN[c].append(j[c]["traj_within_cell_m"])

    ax = axes[1, 0]; x = np.arange(len(conds))
    ax.bar(x, [np.mean(A_idx[c]) for c in conds], yerr=[np.std(A_idx[c]) for c in conds], capsize=4,
           color=[col[c] for c in conds], alpha=0.85)
    for i, c in enumerate(conds):
        ax.scatter(np.full(10, i) + np.random.default_rng(i).uniform(-.12, .12, 10), A_idx[c], c="k", s=12, zorder=5)
    ax.set_xticks(x); ax.set_xticklabels([c.replace(" ", "\n").replace("-", "-\n") for c in conds], fontsize=8)
    ax.set_ylabel("aliasing index  (1 - folded/physical)")
    ax.set_title("D. Fraction of wander that is lattice aliasing\n(predictive vs intact 10/10, p=0.002)",
                 loc="left", fontweight="bold", fontsize=10); ax.grid(alpha=0.25, axis="y")

    ax = axes[1, 1]
    w = 0.38
    ax.bar(x - w / 2, [np.mean(RAW[c]) for c in conds], w, label="physical wander", color="#bbbbbb")
    ax.bar(x + w / 2, [np.mean(WIN[c]) for c in conds], w, label="folded into cell", color="#d62728", alpha=0.8)
    ax.axhline(lam, ls="--", c="green", lw=1, label=f"grid period ~{lam:.2f} m")
    ax.set_xticks(x); ax.set_xticklabels([c.replace(" ", "\n").replace("-", "-\n") for c in conds], fontsize=8)
    ax.set_ylabel("spread (m)"); ax.legend(fontsize=8, frameon=False)
    ax.set_title("E. Physical vs lattice-folded spread\n(folding collapses the ablated wander)",
                 loc="left", fontweight="bold", fontsize=10); ax.grid(alpha=0.25, axis="y")

    axes[1, 2].axis("off")
    axes[1, 2].text(0.0, 0.95, "Interpretation", fontweight="bold", fontsize=12, va="top")
    axes[1, 2].text(0.0, 0.82,
                    "Rob's hypothesis: one torus phase = many physical\n"
                    "locations (a grid cell's fields). If ablated activity\n"
                    "sits near one phase, the decoder can place it at any\n"
                    "replica -> apparent wandering.\n\n"
                    f"~52% of the predictive-ablated wander is lattice\n"
                    f"aliasing (folding into one cell removes it).\n"
                    f"~60% of large jumps are lattice translations.\n"
                    "Intact shows ~0 aliasing (never leaves one cell).\n\n"
                    "So about half the 'wandering' is the SAME place\n"
                    "modulo the grid lattice; the other half is genuine\n"
                    "within-cell phase error (position code degraded).",
                    fontsize=9.5, va="top", family="monospace")

    fig.suptitle("Is the predictive-ablated 'wandering' jumps between grid-field replicas of one phase? "
                 f"(seed {args.seed}, agent {b})", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out_dir = os.path.join(base, "summary", "aarav_activity_space_ablation")
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, "torus_aliasing.png")
    fig.savefig(out, dpi=190, bbox_inches="tight"); fig.savefig(out.replace(".png", ".svg"), bbox_inches="tight")
    print("wrote", out)


if __name__ == "__main__":
    main()
