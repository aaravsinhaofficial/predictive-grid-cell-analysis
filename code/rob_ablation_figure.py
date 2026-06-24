#!/usr/bin/env python3
"""Figures for the activity-space ablation (Exp 2). Separate figure per matching mode.

  --mode count    : bars across conditions (same N units) + 2D geometry/dynamics dissociation.
  --mode percent  : metric vs % of each class ablated.

Headline (William's hypothesis):
  * Ablating PREDICTIVE grid cells -> activity stays ON the normal manifold (low off-manifold
    distance) but the network is at the WRONG place (high decoded-position error): predictive
    cells drive movement ALONG the manifold (dynamics).
  * Ablating STRUCTURAL grid cells -> activity goes OFF the manifold: grid cells define the
    manifold (geometry).
"""
from __future__ import annotations
import argparse, json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

COLORS = {
    "predictive": "#d62728", "structural_grid": "#1f77b4", "random": "#7f7f7f",
    "all_grid": "#17becf", "all_grid_matched": "#17becf", "all_grid_full": "#08306b",
    "intact": "#2ca02c",
}
LABELS = {
    "predictive": "Predictive", "structural_grid": "Structural grid\n(non-predictive)",
    "random": "Random", "all_grid_matched": "All-grid\n(matched N)", "all_grid_full": "All grid\n(full set)",
    "all_grid": "All grid", "intact": "Intact",
}


def load(seed, root, sub, mode):
    p = os.path.join(_REPO, root, f"Seed {seed}", sub, "rob_activity_space_ablation",
                     f"activity_space_ablation_{mode}.json")
    return json.load(open(p))


def gmean(xs):
    xs = np.asarray([x for x in xs if x is not None and np.isfinite(x) and x > 0], float)
    return float(np.exp(np.mean(np.log(xs)))) if xs.size else float("nan")


def fig_count(data, seeds, out):
    order = ["predictive", "structural_grid", "random", "all_grid_matched", "all_grid_full"]
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    fig.suptitle(f"Activity-space ablation — count-matched (N = #predictive); seeds {'+'.join(map(str,seeds))}",
                 fontsize=15, fontweight="bold")

    def per_cond(metric, transform=lambda v: v):
        out_m = {}
        for c in order:
            vals = [transform(d["conditions"][c][metric]["mean"]) for d in data]
            out_m[c] = vals
        return out_m

    # A: off-manifold ratio (log), geometric mean across seeds, seed dots
    ax = axes[0, 0]
    off = per_cond("offmanifold_vs_intact")
    x = np.arange(len(order))
    ax.bar(x, [gmean(off[c]) for c in order], color=[COLORS[c] for c in order], alpha=0.85)
    for i, c in enumerate(order):
        ax.scatter([i] * len(off[c]), off[c], color="k", zorder=5, s=22)
    ax.axhline(1.0, color="#2ca02c", ls="--", lw=1.5, label="Intact (on manifold)")
    ax.set_yscale("log"); ax.set_ylabel("Off-manifold distance (x intact)")
    ax.set_xticks(x); ax.set_xticklabels([LABELS[c] for c in order], fontsize=8)
    ax.set_title("A.  Structure: how far OFF the normal manifold", loc="left", fontweight="bold")
    ax.legend(frameon=False, fontsize=8); ax.grid(alpha=0.25, axis="y", which="both")

    # B: decode endpoint error
    ax = axes[0, 1]
    de = per_cond("decoded_endpoint_error_m")
    intact_de = gmean([d["intact"]["decoded_endpoint_error_m"]["mean"] for d in data])
    ax.bar(x, [np.mean(de[c]) for c in order], color=[COLORS[c] for c in order], alpha=0.85)
    for i, c in enumerate(order):
        ax.scatter([i] * len(de[c]), de[c], color="k", zorder=5, s=22)
    ax.axhline(intact_de, color="#2ca02c", ls="--", lw=1.5, label=f"Intact ({intact_de:.2f} m)")
    ax.set_ylabel("Decoded-position error at endpoint (m)")
    ax.set_xticks(x); ax.set_xticklabels([LABELS[c] for c in order], fontsize=8)
    ax.set_title("B.  Dynamics: how LOST about position", loc="left", fontweight="bold")
    ax.legend(frameon=False, fontsize=8); ax.grid(alpha=0.25, axis="y")

    # C: 2D dissociation scatter
    ax = axes[1, 0]
    for c in ["predictive", "structural_grid", "random", "all_grid_full"]:
        xs = [d["conditions"][c]["offmanifold_vs_intact"]["mean"] for d in data]
        ys = [d["conditions"][c]["decoded_endpoint_error_m"]["mean"] for d in data]
        ax.scatter(xs, ys, color=COLORS[c], s=90, edgecolor="k", zorder=5,
                   label=LABELS[c].replace("\n", " "))
        ax.scatter([gmean(xs)], [np.mean(ys)], color=COLORS[c], s=260, marker="*", edgecolor="k", zorder=6)
    ax.scatter([1.0], [intact_de], color="#2ca02c", s=160, marker="D", edgecolor="k", zorder=6, label="Intact")
    ax.set_xscale("log"); ax.set_xlabel("Off-manifold distance (x intact)  ->  structure broken")
    ax.set_ylabel("Decoded-position error (m)  ->  lost")
    ax.set_title("C.  Double dissociation (stars = seed mean)", loc="left", fontweight="bold")
    ax.legend(frameon=True, framealpha=0.9, fontsize=7.5, loc="upper right"); ax.grid(alpha=0.25, which="both")
    ax.annotate("predictive:\nON manifold, but LOST\n(dynamics role)", xy=(0.02, 0.80), xycoords="axes fraction",
                fontsize=8.5, color="#d62728", fontweight="bold")
    ax.annotate("structural grid:\nOFF manifold\n(geometry role)", xy=(0.40, 0.06), xycoords="axes fraction",
                fontsize=8.5, color="#1f77b4", fontweight="bold")

    # D: activity displacement ratio
    ax = axes[1, 1]
    di = per_cond("displacement_ratio_to_intact")
    ax.bar(x, [np.mean(di[c]) for c in order], color=[COLORS[c] for c in order], alpha=0.85)
    for i, c in enumerate(order):
        ax.scatter([i] * len(di[c]), di[c], color="k", zorder=5, s=22)
    ax.axhline(1.0, color="#2ca02c", ls="--", lw=1.5, label="Intact")
    ax.set_ylabel("Activity-space displacement (x intact)")
    ax.set_xticks(x); ax.set_xticklabels([LABELS[c] for c in order], fontsize=8)
    ax.set_title("D.  Activity travelled start->end", loc="left", fontweight="bold")
    ax.legend(frameon=False, fontsize=8); ax.grid(alpha=0.25, axis="y")

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out, dpi=200, bbox_inches="tight"); fig.savefig(out.replace(".png", ".svg"), bbox_inches="tight")
    print("wrote", out)


def fig_percent(data, seeds, out):
    pools = ["predictive", "structural_grid", "all_grid", "random"]
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.2))
    fig.suptitle(f"Activity-space ablation — percent-matched (fraction of each class); seeds {'+'.join(map(str,seeds))}",
                 fontsize=14, fontweight="bold")
    fracs = [r["fraction"] * 100 for r in data[0]["pools"]["predictive"]]

    def curve(pool, metric, geom=False):
        # average across seeds at each fraction
        M = np.array([[d["pools"][pool][i][metric]["mean"] for i in range(len(fracs))] for d in data], float)
        return (np.exp(np.nanmean(np.log(np.clip(M, 1e-9, None)), axis=0)) if geom else np.nanmean(M, axis=0))

    specs = [
        ("offmanifold_vs_intact", "Off-manifold distance (x intact)", True, "A.  Structure (off-manifold)"),
        ("decoded_endpoint_error_m", "Decoded-position error (m)", False, "B.  Dynamics (position error)"),
        ("displacement_ratio_to_intact", "Activity displacement (x intact)", False, "C.  Activity travelled"),
    ]
    for ax, (metric, ylab, geom, title) in zip(axes, specs):
        for pool in pools:
            ax.plot(fracs, curve(pool, metric, geom), marker="o", ms=4, color=COLORS[pool],
                    lw=2.4 if pool == "predictive" else 1.8, label=LABELS[pool].replace("\n", " "))
        if geom:
            ax.set_yscale("log"); ax.axhline(1.0, color="#2ca02c", ls="--", lw=1.2)
        ax.set_xlabel("% of class ablated"); ax.set_ylabel(ylab)
        ax.set_title(title, loc="left", fontweight="bold"); ax.grid(alpha=0.25, which="both")
        ax.legend(frameon=False, fontsize=8)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(out, dpi=200, bbox_inches="tight"); fig.savefig(out.replace(".png", ".svg"), bbox_inches="tight")
    print("wrote", out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["count", "percent"], required=True)
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1])
    ap.add_argument("--analysis_root", default="analysis_outputs/Single agent path integration")
    ap.add_argument("--subdir", default="spatial_shift_allunits")
    args = ap.parse_args()
    data = [load(s, args.analysis_root, args.subdir, args.mode) for s in args.seeds]
    out = os.path.join(_REPO, args.analysis_root, "summary", "rob_activity_space_ablation",
                       f"activity_space_ablation_{args.mode}.png")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    if args.mode == "count":
        fig_count(data, args.seeds, out)
    else:
        fig_percent(data, args.seeds, out)


if __name__ == "__main__":
    main()
