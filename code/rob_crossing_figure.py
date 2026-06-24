#!/usr/bin/env python3
"""Combined figure for the crossing population-correlation analysis (Exp 1), seeds aggregated.

Panels:
  A. Rob's designed X-crossings: population correlation vs heading-separation angle.
  B. William's same-bin random walks: population correlation vs movement (velocity) difference.
  C. Decorrelation (1 - corr) at the largest movement difference, per class, for both methods.
  D. Bug-fix QC: the corrected movement axis (heading/velocity difference, non-zero) vs the
     position difference at the crossing (~0 = the old 1e-8 artifact), log scale.
"""
from __future__ import annotations
import argparse, json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CLASS_STYLE = {
    "predictive":     dict(color="#d62728", lw=2.6, label="Predictive grid"),
    "standard_grid":  dict(color="#1f77b4", lw=2.0, label="Standard grid"),
    "retrospective":  dict(color="#9467bd", lw=1.8, label="Retrospective"),
    "random_matched": dict(color="#7f7f7f", lw=1.8, label="Random (matched n)"),
}


def load_seed(seed, analysis_root, subdir):
    p = os.path.join(_REPO, analysis_root, f"Seed {seed}", subdir,
                     "rob_crossing_population_corr", "crossing_population_corr.json")
    return json.load(open(p))


def agg(per_seed, getter):
    """Stack getter(d)->1D array across seeds; return (mean, lo, hi) over seeds (min/max band)."""
    arrs = [np.asarray(getter(d), dtype=float) for d in per_seed]
    n = min(len(a) for a in arrs)
    M = np.vstack([a[:n] for a in arrs])
    return np.nanmean(M, axis=0), np.nanmin(M, axis=0), np.nanmax(M, axis=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1])
    ap.add_argument("--analysis_root", default="analysis_outputs/Single agent path integration")
    ap.add_argument("--subdir", default="spatial_shift_allunits")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    data = [load_seed(s, args.analysis_root, args.subdir) for s in args.seeds]
    out = args.out or os.path.join(_REPO, args.analysis_root, "summary",
                                   "rob_crossing_population_corr", "crossing_population_corr.png")
    os.makedirs(os.path.dirname(out), exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    seed_str = "+".join(str(s) for s in args.seeds)
    fig.suptitle(f"Path dependence of predictive grid units — two pipelines agree (seeds {seed_str})",
                 fontsize=15, fontweight="bold")

    # ---- Panel A: Rob X-crossings ----
    ax = axes[0, 0]
    ang = np.asarray(data[0]["rob_x_crossings"]["angle_bin_centers_deg"], float)
    for cls, st in CLASS_STYLE.items():
        m, lo, hi = agg(data, lambda d, c=cls: d["rob_x_crossings"]["curves"][c]["mean"])
        ax.plot(ang[:len(m)], m, marker="o", ms=4, **st)
        ax.fill_between(ang[:len(m)], lo, hi, color=st["color"], alpha=0.12)
    ax.set_xlabel("Heading-separation angle at crossing (deg)")
    ax.set_ylabel("Population activity correlation\n(same position, different heading)")
    ax.set_title("A.  Rob: designed X-crossings", loc="left", fontweight="bold")
    ax.legend(frameon=False, fontsize=9); ax.grid(alpha=0.25)

    # ---- Panel B: William same-bin ----
    ax = axes[0, 1]
    xb = np.nanmean(np.vstack([np.asarray(d["william_same_bin"]["move_bin_centers_m"], float) * 100
                               for d in data]), axis=0)  # cm
    for cls, st in CLASS_STYLE.items():
        m, lo, hi = agg(data, lambda d, c=cls: d["william_same_bin"]["curves"][c]["mean"])
        ax.plot(xb[:len(m)], m, marker="s", ms=4, **st)
        ax.fill_between(xb[:len(m)], lo, hi, color=st["color"], alpha=0.12)
    ax.set_xlabel("Movement (velocity-vector) difference at same location (cm)")
    ax.set_ylabel("Population activity correlation")
    ax.set_title("B.  William: random-walk same-bin", loc="left", fontweight="bold")
    ax.legend(frameon=False, fontsize=9); ax.grid(alpha=0.25)

    # ---- Panel C: decorrelation at max difference ----
    ax = axes[1, 0]
    classes = ["predictive", "standard_grid", "retrospective", "random_matched"]
    rob_dec, will_dec = [], []
    for c in classes:
        rm, _, _ = agg(data, lambda d, cc=c: d["rob_x_crossings"]["curves"][cc]["mean"])
        wm, _, _ = agg(data, lambda d, cc=c: d["william_same_bin"]["curves"][cc]["mean"])
        rob_dec.append(1 - rm[-1]); will_dec.append(1 - wm[-1])
    x = np.arange(len(classes)); w = 0.38
    ax.bar(x - w/2, rob_dec, w, color=[CLASS_STYLE[c]["color"] for c in classes], alpha=0.95, label="Rob X-crossing")
    ax.bar(x + w/2, will_dec, w, color=[CLASS_STYLE[c]["color"] for c in classes], alpha=0.5, hatch="//", label="William same-bin")
    ax.set_xticks(x); ax.set_xticklabels([CLASS_STYLE[c]["label"] for c in classes], rotation=20, ha="right", fontsize=9)
    ax.set_ylabel("Decorrelation (1 - corr) at largest\nmovement / heading difference")
    ax.set_title("C.  Predictive decorrelates the most", loc="left", fontweight="bold")
    ax.legend(frameon=False, fontsize=9); ax.grid(alpha=0.25, axis="y")

    # ---- Panel D: bug-fix QC ----
    ax = axes[1, 1]
    qc = data[0]["rob_x_crossings"]["qc"]["vel_diff_vs_angle"]
    cdeg = np.asarray(qc["centers_deg"], float)
    veld = np.asarray(qc["vel_diff_m"], float) * 100  # cm
    posd = np.asarray(qc["pos_diff_m"], float) * 100  # cm
    posd_plot = np.clip(posd, 1e-7, None)
    ax.plot(cdeg, veld, "o-", color="#2ca02c", lw=2.4, label="Heading/velocity difference (correct axis)")
    ax.plot(cdeg, posd_plot, "x--", color="#d62728", lw=1.8, label="Position difference at crossing (~0 artifact)")
    ax.set_yscale("log"); ax.set_ylim(1e-7, 10)
    ax.set_xlabel("Heading-separation angle (deg)")
    ax.set_ylabel("Difference magnitude (cm, log)")
    ax.set_title("D.  Movement axis fixed (was ~1e-8)", loc="left", fontweight="bold")
    ax.legend(frameon=False, fontsize=9); ax.grid(alpha=0.25, which="both")
    ax.annotate("old plots showed THIS (≈0)", xy=(90, 1e-6), fontsize=8, color="#d62728")

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out, dpi=200, bbox_inches="tight")
    fig.savefig(out.replace(".png", ".svg"), bbox_inches="tight")
    print("wrote", out)


if __name__ == "__main__":
    main()
