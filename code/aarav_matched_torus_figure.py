#!/usr/bin/env python3
"""Aggregate aarav_matched_torus.py across seeds: PGC ablation vs property-matched control on the torus
metrics and the dynamics metrics, for both PGC definitions. Writes summary JSON + figures to
analysis_outputs/Single agent path integration/summary/aarav_matched_torus/."""
from __future__ import annotations
import argparse, json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon

_HERE = os.path.dirname(os.path.abspath(__file__)); _REPO = os.path.dirname(_HERE)
COL = {"intact": "#0b0b0b", "pred": "#2a78d6", "matched": "#eb6834", "rand": "#1baf7a", "module": "#e87ba4", "matchedB": "#4a3aa7"}


def val(rec, key):
    if key in rec["torus"]:
        return float(rec["torus"][key])
    if key in rec["late"]:
        return float(rec["late"][key])
    if key in rec["module"]:
        return float(rec["module"][key])
    return float(rec[key])


def cond_value(r, cond, key):
    return float(np.nanmean([val(rec, key) for rec in r["conditions"][cond]]))


def paired(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    m = np.isfinite(a) & np.isfinite(b); a, b = a[m], b[m]
    if a.size < 5 or np.allclose(a, b):
        return {"n": int(a.size), "p": float("nan"), "wins": int((a > b).sum()), "median_diff": float(np.median(a - b)) if a.size else float("nan")}
    return {"n": int(a.size), "p": float(wilcoxon(a, b).pvalue), "wins": int((a > b).sum()), "median_diff": float(np.median(a - b))}


METRICS = [("theta1_clumping", "Torus-phase clumping\n(resultant of θ1; 1 = stuck)"),
           ("ring_spread", "Ring spread (CV of |r1|)"),
           ("autocorr_med_lag20", "Grid-population autocorr @ lag 20"),
           ("norm_ratio", "Activity norm / intact (late)"),
           ("tracking_cos", "cos(ablated, intact state) (late)"),
           ("decoder_err_m_mean", "Decoder error (m)")]
EXTRA = [("decode_rmse_cm", ""), ("radius_pc12_ratio", ""), ("knn_late_err_m", ""), ("step_ratio", ""), ("vel_tangent", ""), ("autocorr_med_lag10", "")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", nargs="+", type=int, default=list(range(10)))
    ap.add_argument("--analysis_root", default="analysis_outputs/Single agent path integration")
    ap.add_argument("--analysis_subdir", default="spatial_shift_allunits")
    args = ap.parse_args()
    root = os.path.join(_REPO, args.analysis_root)
    R = []
    for s in args.seeds:
        p = os.path.join(root, f"Seed {s}", args.analysis_subdir, "aarav_matched_torus", "matched_torus.json")
        if os.path.exists(p):
            with open(p) as f:
                R.append(json.load(f))
    assert R, "no results"
    n = len(R)
    out_dir = os.path.join(root, "summary", "aarav_matched_torus"); os.makedirs(out_dir, exist_ok=True)
    arms = {"original": ("pred_lib", [("matched_lib", "property-matched", COL["matched"]), ("randpool_lib", "random grid cell (n-matched)", COL["rand"]),
                                      ("module_match_lib", "module-count-matched", COL["module"])]),
            "MEC-style": ("pred_con", [("matched_con_A", "property-matched (pool: non-pred. grid)", COL["matched"]),
                                       ("matched_con_B", "property-matched (pool: non-MEC-pred.)", COL["matchedB"]),
                                       ("randpool_con_A", "random grid cell (n-matched)", COL["rand"]),
                                       ("module_match_con", "module-count-matched", COL["module"])])}
    summary = {"n_seeds": n, "seeds": [r["seed"] for r in R], "values": {}, "tests": {}, "quality": {}, "module": {}}
    for key, _ in METRICS + EXTRA + [("n_in_module", ""), ("phase1_resultant_removed", "")]:
        summary["values"][f"intact/{key}"] = [cond_value(r, "intact", key) for r in R]
        for arm, (pred, ctrls) in arms.items():
            a = [cond_value(r, pred, key) for r in R]
            summary["values"][f"{pred}/{key}"] = a
            for c, _, _ in ctrls:
                b = [cond_value(r, c, key) for r in R]
                summary["values"][f"{c}/{key}"] = b
                summary["tests"][f"{key}: {pred} vs {c}"] = paired(a, b)
    for k in ("nn_dist_matched_lib", "nn_dist_randpool_lib", "nn_dist_matched_con_A", "nn_dist_randpool_con_A", "nn_dist_matched_con_B",
              "matched_lib_in_std_grid", "pred_lib_in_std_grid", "matched_con_A_in_std_grid", "matched_con_B_in_std_grid",
              "matched_con_A_frac_retro", "matched_con_B_frac_predlib"):
        summary["quality"][k] = [r["quality"][k] for r in R]
    summary["quality"]["profiles"] = {k: {kk: float(np.median([r["quality"][k][kk] for r in R])) for kk in R[0]["quality"][k]}
                                      for k in R[0]["quality"] if k.endswith("_profile") or k.endswith("profile_A") or k.endswith("profile_B")}
    summary["toroidal_module"] = [r["toroidal_module"] for r in R]

    # ---------------------------------------------------------------- figure: paired dots
    fig, axes = plt.subplots(2, len(METRICS), figsize=(3.4 * len(METRICS), 8.6))
    for row, (arm, (pred, ctrls)) in enumerate(arms.items()):
        for col, (key, ylabel) in enumerate(METRICS):
            ax = axes[row, col]
            groups = [("intact", "intact", COL["intact"]), (pred, f"PGC ({arm})", COL["pred"])] + [(c, l, cc) for c, l, cc in ctrls]
            xs = np.arange(len(groups))
            vals = [np.asarray(summary["values"][f"{g}/{key}"], float) for g, _, _ in groups]
            for i, (v, (g, l, c)) in enumerate(zip(vals, groups)):
                ax.scatter(np.full(v.size, i) + np.random.default_rng(i).uniform(-0.12, 0.12, v.size), v, s=16, color=c, alpha=0.7, lw=0)
                ax.plot([i - 0.25, i + 0.25], [np.nanmedian(v)] * 2, color=c, lw=2.5)
            for s in range(n):  # paired lines PGC -> controls
                for i in range(2, len(groups)):
                    ax.plot([1, i], [vals[1][s], vals[i][s]], color="#c3c2b7", lw=0.6, zorder=0)
            ax.set_xticks(xs); ax.set_xticklabels([l for _, l, _ in groups], rotation=60, ha="right", fontsize=7)
            ax.set_title(ylabel, fontsize=8.5); ax.grid(alpha=0.25, axis="y"); ax.spines[["top", "right"]].set_visible(False)
            t = summary["tests"][f"{key}: {pred} vs {ctrls[0][0]}"]
            ax.text(0.02, 0.98, f"PGC vs matched: p={t['p']:.3g}\n{t['wins']}/{t['n']} seeds PGC>ctrl", transform=ax.transAxes, fontsize=7, va="top",
                    bbox=dict(fc="white", ec="none", alpha=0.85, pad=1))
    fig.suptitle(f"PGC ablation vs property-matched control (8 z-scored covariates, greedy NN, 5 draws) — n = {n} networks, all 4096 units", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(os.path.join(out_dir, "matched_control_torus_dynamics.png"), dpi=160); plt.close(fig)

    # ---------------------------------------------------------------- figure: torus coordinates, seed 0
    s0 = R[0]["seed"]
    cp = os.path.join(root, f"Seed {s0}", args.analysis_subdir, "aarav_matched_torus", "torus_coords.npz")
    if os.path.exists(cp):
        V = np.load(cp)
        panels = [("intact", "Intact"), ("pred_lib", "PGC ablation (original def.)"), ("matched_lib", "Property-matched control (original)"),
                  ("randpool_lib", "Random grid cells (n-matched)"), ("pred_con", "PGC ablation (MEC-style)"), ("matched_con_A", "Property-matched control (MEC-style)")]
        fig = plt.figure(figsize=(4.2 * len(panels), 4.6))
        for i, (name, title) in enumerate(panels):
            if f"{name}|coords" not in V.files:
                continue
            ax = fig.add_subplot(1, len(panels), i + 1, projection="3d")
            c = V[f"{name}|coords"]; th = V[f"{name}|theta1"]
            ax.scatter(c[:, 0], c[:, 1], c[:, 2], c=th, cmap="twilight", s=4, alpha=0.55, lw=0)
            ax.set_title(title, fontsize=9); ax.set_axis_off()
            clump = cond_value(R[0], name, "theta1_clumping")
            ax.text2D(0.5, -0.02, f"θ1 clumping {clump:.2f}", transform=ax.transAxes, ha="center", fontsize=8)
        fig.suptitle(f"Seed {s0}: population states projected on the fixed torus basis (colour = θ1)", fontsize=10)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "torus_coords_seed0.png"), dpi=150); plt.close(fig)

    with open(os.path.join(out_dir, "matched_torus_summary.json"), "w") as f:
        json.dump(summary, f, indent=1)

    # ---------------------------------------------------------------- digest
    med = lambda k: np.nanmedian(summary["values"][k])
    print(f"n={n} seeds; toroidal module median {int(np.median(summary['toroidal_module']))} units")
    print("matching quality (median NN z-distance): matched_lib %.2f vs random %.2f | matched_con_A %.2f vs random %.2f | matched_con_B %.2f" % tuple(
        np.median(summary["quality"][k]) for k in ("nn_dist_matched_lib", "nn_dist_randpool_lib", "nn_dist_matched_con_A", "nn_dist_randpool_con_A", "nn_dist_matched_con_B")))
    print("std-grid fraction: pred_lib %.2f matched_lib %.2f | matched_con_A %.2f matched_con_B %.2f ; matched_con_A retro frac %.2f ; matched_con_B predlib frac %.2f" % tuple(
        np.median(summary["quality"][k]) for k in ("pred_lib_in_std_grid", "matched_lib_in_std_grid", "matched_con_A_in_std_grid", "matched_con_B_in_std_grid", "matched_con_A_frac_retro", "matched_con_B_frac_predlib")))
    print("\ncovariate profile (z over pool), median across seeds:")
    keys = R[0]["quality"]["keys"]
    for prof in ("pred_lib_profile", "matched_lib_profile", "randpool_lib_profile", "pred_con_profile_A", "matched_con_A_profile", "randpool_con_A_profile"):
        print(f"  {prof:24s} " + " ".join(f"{k[:8]} {summary['quality']['profiles'][prof][k]:+.2f}" for k in keys))
    for arm, (pred, ctrls) in arms.items():
        print(f"\n=== {arm} definition: {pred} (median across seeds) ===")
        for key, _ in METRICS + EXTRA + [("n_in_module", ""), ("phase1_resultant_removed", "")]:
            line = f"  {key:26s} intact {med('intact/'+key):7.3f} | PGC {med(pred+'/'+key):7.3f}"
            for c, l, _ in ctrls:
                t = summary["tests"][f"{key}: {pred} vs {c}"]
                line += f" | {c} {med(c+'/'+key):7.3f} (Δ{t['median_diff']:+.3f} p={t['p']:.3g} {t['wins']}/{t['n']})"
            print(line)
    print(f"\nwrote {out_dir}")


if __name__ == "__main__":
    main()
