#!/usr/bin/env python3
"""Aggregate aarav_definition_dynamics.py outputs across seeds -> stats JSON + figures + README table.

Figures (analysis_outputs/Single agent path integration/summary/aarav_definition_dynamics/):
  autocorr_by_definition.png   Redman-style population-autocorrelation panels, liberal vs conservative PGC definition
  stall_vs_collapse.png        time courses of the collapse/stall metrics + late-window paired comparisons
  drive_decomposition.png      one-step drive of each unit set onto the grid population: propulsion vs radial
  example_pca_trajectories.png seed-0 PC1-2 trajectories (intact vs ablated) for both definitions + random
"""
from __future__ import annotations
import argparse, json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon

_HERE = os.path.dirname(os.path.abspath(__file__)); _REPO = os.path.dirname(_HERE)
LAGS = list(range(1, 30))

# fixed categorical roles (validated default palette, fixed slot order)
COL = {"intact": "#0b0b0b", "pred": "#2a78d6", "rand": "#eb6834", "retro": "#1baf7a", "band": "#eda100",
       "struct": "#e87ba4", "rand_nongrid": "#4a3aa7", "grid": "#52514e"}


def load_all(root, subdir, seeds):
    out = []
    for s in seeds:
        p = os.path.join(root, f"Seed {s}", subdir, "aarav_definition_dynamics", "definition_dynamics.json")
        if os.path.exists(p):
            with open(p) as f:
                out.append(json.load(f))
    return out


def cond_names(res, prefix):
    return sorted(k for k in res["conditions"] if k.startswith(prefix))


def per_seed_curve(res, cond, traj, key):
    """Curve for a condition (random prefixes averaged over draws)."""
    if cond in res["conditions"]:
        return np.asarray(res["conditions"][cond][traj][key], float)
    names = cond_names(res, cond)
    return np.mean([np.asarray(res["conditions"][n][traj][key], float) for n in names], axis=0)


def per_seed_scalar(res, cond, traj, key, sub=None):
    def get(n):
        d = res["conditions"][n][traj]
        if sub == "late":
            return float(d["late"][key])
        if sub == "knn":
            return float(d["knn"]["ablated"][key])
        if sub == "knn_intact":
            return float(d["knn"]["intact"][key])
        return float(d[key])
    if cond in res["conditions"]:
        return get(cond)
    return float(np.mean([get(n) for n in cond_names(res, cond)]))


def paired(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    m = np.isfinite(a) & np.isfinite(b)
    a, b = a[m], b[m]
    if a.size < 5 or np.allclose(a, b):
        return {"n": int(a.size), "p": float("nan"), "wins": int((a > b).sum()), "median_diff": float(np.median(a - b)) if a.size else float("nan")}
    return {"n": int(a.size), "p": float(wilcoxon(a, b).pvalue), "wins": int((a > b).sum()), "median_diff": float(np.median(a - b))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", nargs="+", type=int, default=list(range(10)))
    ap.add_argument("--analysis_root", default="analysis_outputs/Single agent path integration")
    ap.add_argument("--analysis_subdir", default="spatial_shift_allunits")
    args = ap.parse_args()
    root = os.path.join(_REPO, args.analysis_root)
    R = load_all(root, args.analysis_subdir, args.seeds)
    assert R, "no results"
    out_dir = os.path.join(root, "summary", "aarav_definition_dynamics"); os.makedirs(out_dir, exist_ok=True)
    n = len(R)
    seeds = [r["seed"] for r in R]
    summary = {"n_seeds": n, "seeds": seeds, "thresholds": R[0]["thresholds"],
               "class_sizes": {k: [r["class_sizes"][k] for r in R] for k in R[0]["class_sizes"]},
               "n_std_grid": [r["n_std_grid"] for r in R], "n_nondead": [r["n_nondead"] for r in R],
               "overlaps": {k: [r["overlaps"][k] for r in R] for k in R[0]["overlaps"]},
               "pred_lib_gs0_median": [r["pred_lib_gs0_median"] for r in R],
               "pred_con_gs0_median": [r["pred_con_gs0_median"] for r in R]}

    # ------------------------------------------------------------------ 1. autocorrelation panels
    arms = {
        "conservative": [("intact", "intact", COL["intact"], "-"), ("pred_con", "Predictive (MEC-style)", COL["pred"], "-"),
                         ("rand_nongrid_matchcon", "Random non-grid (n-matched, Redman pool)", COL["rand_nongrid"], "-"),
                         ("rand_nd_matchcon", "Random any unit (n-matched)", COL["rand"], "-"),
                         ("struct_matchcon", "Random standard-grid (n-matched)", COL["struct"], "-"),
                         ("retro_con", "Retrospective (MEC-style)", COL["retro"], "-"), ("band", "Band (top 5 %)", COL["band"], "-")],
        "liberal": [("intact", "intact", COL["intact"], "-"), ("pred_lib", "Predictive (original def.)", COL["pred"], "-"),
                    ("rand_nd_matchlib", "Random any unit (n-matched)", COL["rand"], "-"),
                    ("struct_matchlib", "Random standard-grid (n-matched)", COL["struct"], "-"),
                    ("retro_lib", "Retrospective (original def.)", COL["retro"], "-"), ("band90", "Band (top 10 %)", COL["band"], "-")],
    }
    summary["autocorr"] = {}
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.2), sharey=True)
    for row, traj in enumerate(("random_walk", "straight")):
        for col, (arm, items) in enumerate(arms.items()):
            ax = axes[row, col]
            curves = {}
            for cond, label, color, ls in items:
                C = np.stack([per_seed_curve(r, cond, traj, "autocorr_med") for r in R])
                curves[cond] = C
                med = np.nanmedian(C, 0); lo = np.nanpercentile(C, 25, 0); hi = np.nanpercentile(C, 75, 0)
                x = np.array([0] + LAGS); med = np.r_[1.0, med]; lo = np.r_[1.0, lo]; hi = np.r_[1.0, hi]
                ax.fill_between(x, lo, hi, color=color, alpha=0.15, lw=0)
                ax.plot(x, med, color=color, lw=2, ls=ls, label=label)
            ax.set_xlim(0, 29); ax.set_ylim(0.25, 1.0)
            ax.set_xlabel("Lag (steps)"); ax.set_title(f"{arm.capitalize()} definition — {traj.replace('_', ' ')}", fontsize=11)
            ax.grid(alpha=0.25); ax.spines[["top", "right"]].set_visible(False)
            if col == 0:
                ax.set_ylabel("Grid-population autocorrelation\n(median cos(g_t, g_{t+lag}))")
            if row == 0:
                ax.legend(fontsize=7.5, frameon=True, framealpha=0.85, edgecolor="none", loc="lower left")
            # stats per lag: pred vs intact-same-units, pred vs matched random
            pred = "pred_con" if arm == "conservative" else "pred_lib"
            rnd = "rand_nongrid_matchcon" if arm == "conservative" else "rand_nd_matchlib"
            rnd2 = "rand_nd_matchcon" if arm == "conservative" else "struct_matchlib"
            same = np.stack([per_seed_curve(r, pred, traj, "autocorr_med_intact_same_units") for r in R])
            st = {"lags": LAGS,
                  "pred_minus_intact_same_units_median": np.nanmedian(curves[pred] - same, 0).tolist(),
                  "p_pred_vs_intact_same_units": [paired(curves[pred][:, i], same[:, i])["p"] for i in range(len(LAGS))],
                  "wins_pred_gt_intact": [paired(curves[pred][:, i], same[:, i])["wins"] for i in range(len(LAGS))],
                  f"p_pred_vs_{rnd}": [paired(curves[pred][:, i], curves[rnd][:, i])["p"] for i in range(len(LAGS))],
                  f"pred_minus_{rnd}_median": np.nanmedian(curves[pred] - curves[rnd], 0).tolist(),
                  f"p_pred_vs_{rnd2}": [paired(curves[pred][:, i], curves[rnd2][:, i])["p"] for i in range(len(LAGS))],
                  f"pred_minus_{rnd2}_median": np.nanmedian(curves[pred] - curves[rnd2], 0).tolist()}
            if arm == "conservative":
                st["p_band_vs_pred"] = [paired(curves["band"][:, i], curves[pred][:, i])["p"] for i in range(len(LAGS))]
                st["band_minus_pred_median"] = np.nanmedian(curves["band"] - curves[pred], 0).tolist()
                st["p_retro_vs_pred"] = [paired(curves["retro_con"][:, i], curves[pred][:, i])["p"] for i in range(len(LAGS))]
            summary["autocorr"][f"{arm}/{traj}"] = st
            # significance ticks: pred vs intact (same units) and pred vs matched random
            sig1 = [LAGS[i] for i in range(len(LAGS)) if st["p_pred_vs_intact_same_units"][i] < 0.05]
            sig2 = [LAGS[i] for i in range(len(LAGS)) if st[f"p_pred_vs_{rnd}"][i] < 0.05]
            if sig1:
                ax.plot(sig1, [0.985] * len(sig1), "|", color=COL["pred"], ms=6, mew=1.5)
                ax.text(0.4, 0.985, "pred≠intact(same units)", color=COL["pred"], fontsize=7, va="center", ha="left", bbox=dict(fc="white", ec="none", alpha=0.9, pad=0.6))
            if sig2:
                ax.plot(sig2, [0.955] * len(sig2), "|", color=COL["rand_nongrid"] if arm == "conservative" else COL["rand"], ms=6, mew=1.5)
                ax.text(0.4, 0.955, "pred≠random" + (" (non-grid pool)" if arm == "conservative" else ""), color=COL["rand_nongrid"] if arm == "conservative" else COL["rand"], fontsize=7, va="center", ha="left", bbox=dict(fc="white", ec="none", alpha=0.9, pad=0.6))
    fig.suptitle(f"Grid-population autocorrelation after ablation: original vs MEC-style PGC definition (n = {n} networks, median ± IQR)", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(os.path.join(out_dir, "autocorr_by_definition.png"), dpi=170); plt.close(fig)

    # ------------------------------------------------------------------ 2. stall vs collapse
    metrics = [("step_ratio", "Step size (ablated / intact)", "stall index: 1 = moving like intact, 0 = frozen"),
               ("radius_pc12_ratio", "PC1-2 radius (ablated / intact)", "collapse to PCA centre: 1 = on the ring"),
               ("norm_ratio", "Activity norm (ablated / intact)", "bump amplitude of surviving grid units"),
               ("tracking_cos", "cos(ablated state, intact state)", "tracking of the intact state"),
               ("dist_from_start", "Distance from own first state", "return index (intact shown dashed)"),
               ("knn_err", "kNN-decoded position error (m)", "read-out population alone; chance ≈ 1.1 m")]
    conds2 = [("pred_con", "Pred (MEC-style)", COL["pred"], "-"), ("rand_nongrid_matchcon", "Random non-grid (n-matched)", COL["rand_nongrid"], "-"),
              ("rand_nd_matchcon", "Random any (n-matched)", COL["rand"], "-"), ("struct_matchcon", "Random std-grid (n-matched)", COL["struct"], "-"),
              ("pred_lib", "Pred (original def.)", COL["pred"], "--"), ("rand_nd_matchlib", "Random any (n-matched to original)", COL["rand"], "--"),
              ("band", "Band (top 5 %)", COL["band"], "-")]
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    for ax, (key, ylabel, note) in zip(axes.ravel(), metrics):
        for cond, label, color, ls in conds2:
            if key == "knn_err":
                C = np.stack([np.mean([np.asarray(r["conditions"][nn]["random_walk"]["knn"]["ablated"]["err_m"]) for nn in
                                       ([cond] if cond in r["conditions"] else cond_names(r, cond))], 0) for r in R])
            else:
                C = np.stack([per_seed_curve(r, cond, "random_walk", key) for r in R])
            x = np.arange(C.shape[1]) + (1 if key == "step_ratio" else 0)
            ax.plot(x, np.nanmedian(C, 0), color=color, ls=ls, lw=2, label=label)
            ax.fill_between(x, np.nanpercentile(C, 25, 0), np.nanpercentile(C, 75, 0), color=color, alpha=0.12, lw=0)
        if key == "dist_from_start":
            C = np.stack([per_seed_curve(r, "pred_con", "random_walk", "dist_from_start_intact") for r in R])
            ax.plot(np.arange(C.shape[1]), np.nanmedian(C, 0), color=COL["intact"], ls=":", lw=2, label="intact")
        if key == "knn_err":
            C = np.stack([np.asarray(r["conditions"]["intact"]["random_walk"]["knn"]["intact"]["err_m"]) for r in R])
            ax.plot(np.arange(C.shape[1]), np.nanmedian(C, 0), color=COL["intact"], ls=":", lw=2, label="intact")
        ax.set_title(note, fontsize=9.5, color="#52514e"); ax.set_ylabel(ylabel); ax.set_xlabel("Time step")
        ax.grid(alpha=0.25); ax.spines[["top", "right"]].set_visible(False)
    handles, labels = axes[1, 2].get_legend_handles_labels()
    h0, l0 = axes[0, 0].get_legend_handles_labels()
    fig.legend(h0 + [handles[-1]], l0 + [labels[-1]], loc="lower center", ncol=4, fontsize=8.5, frameon=False)
    fig.suptitle(f"Stall vs collapse after ablation (random walk, n = {n} networks, median ± IQR across networks; read-out = standard grid units \\ ablated)", fontsize=11.5)
    fig.tight_layout(rect=(0, 0.07, 1, 0.96))
    fig.savefig(os.path.join(out_dir, "stall_vs_collapse.png"), dpi=170); plt.close(fig)

    # late-window paired comparisons
    late_keys = ["step_ratio", "radius_pc12_ratio", "radius_pc34_ratio", "radius_full_ratio", "norm_ratio", "tracking_cos",
                 "dist_from_start_ratio", "dist_to_intact_init_over_intact_motion", "vel_tangent", "vel_radial_pc6",
                 "autocorr_med_lag20", "autocorr_med_lag10"]
    comps = {"conservative": ("pred_con", ["rand_nongrid_matchcon", "rand_nd_matchcon", "struct_matchcon", "retro_con", "band"]),
             "liberal": ("pred_lib", ["rand_nd_matchlib", "struct_matchlib", "retro_lib", "band90"])}
    summary["late"] = {}
    for traj in ("random_walk", "straight"):
        for arm, (pred, others) in comps.items():
            rec = {"pred": pred, "per_seed": {}, "tests": {}}
            for key in late_keys:
                a = [per_seed_scalar(r, pred, traj, key, "late") for r in R]
                rec["per_seed"][f"{pred}/{key}"] = a
                rec["per_seed"][f"{pred}/{key}_intact_same_units"] = [per_seed_scalar(r, pred, traj, "autocorr_med_lag20_intact_same_units", "late") for r in R] if key == "autocorr_med_lag20" else None
                for o in others:
                    b = [per_seed_scalar(r, o, traj, key, "late") for r in R]
                    rec["per_seed"][f"{o}/{key}"] = b
                    rec["tests"][f"{key}: {pred} vs {o}"] = paired(a, b)
                if key == "autocorr_med_lag20":
                    same = [per_seed_scalar(r, pred, traj, "autocorr_med_lag20_intact_same_units", "late") for r in R]
                    rec["tests"][f"{key}: {pred} vs intact(same units)"] = paired(a, same)
            # knn + decoder
            for key, sub in (("late_err_m", "knn"), ("late_dist_to_start_m", "knn"), ("late_nn_dist", "knn")):
                if traj != "random_walk":
                    continue
                a = [per_seed_scalar(r, pred, traj, key, sub) for r in R]
                rec["per_seed"][f"{pred}/{key}"] = a
                rec["per_seed"][f"intact/{key}"] = [per_seed_scalar(r, "intact", traj, key, "knn_intact") for r in R]
                for o in others:
                    b = [per_seed_scalar(r, o, traj, key, sub) for r in R]
                    rec["per_seed"][f"{o}/{key}"] = b
                    rec["tests"][f"{key}: {pred} vs {o}"] = paired(a, b)
            a = [per_seed_scalar(r, pred, traj, "decoder_err_m_mean") for r in R]
            rec["per_seed"][f"{pred}/decoder_err_m_mean"] = a
            for o in others:
                b = [per_seed_scalar(r, o, traj, "decoder_err_m_mean") for r in R]
                rec["per_seed"][f"{o}/decoder_err_m_mean"] = b
                rec["tests"][f"decoder_err_m_mean: {pred} vs {o}"] = paired(a, b)
            if traj == "random_walk":
                rec["per_seed"][f"{pred}/decoder_err_recurrent_only"] = [r["conditions"][pred].get("decoder_err_m_mean_recurrent_only_lesion", float("nan")) for r in R]
                for o in others:
                    if o in R[0]["conditions"] and "decoder_err_m_mean_recurrent_only_lesion" in R[0]["conditions"][o]:
                        rec["per_seed"][f"{o}/decoder_err_recurrent_only"] = [r["conditions"][o]["decoder_err_m_mean_recurrent_only_lesion"] for r in R]
                    elif o in ("rand_nongrid_matchcon", "rand_nd_matchcon", "struct_matchcon"):
                        rec["per_seed"][f"{o}/decoder_err_recurrent_only"] = [float(np.mean([r["conditions"][nn]["decoder_err_m_mean_recurrent_only_lesion"] for nn in cond_names(r, o)])) for r in R]
            summary["late"][f"{arm}/{traj}"] = rec

    # ------------------------------------------------------------------ 3. drive decomposition
    dsets = [("pred_con", "Pred (MEC-style)", COL["pred"]), ("rand_nd_matchcon", "Random any (n-matched)", COL["rand"]),
             ("struct_matchcon", "Random std-grid (n-matched)", COL["struct"]), ("retro_con", "Retro (MEC-style)", COL["retro"]),
             ("band", "Band (top 5 %)", COL["band"]),
             ("pred_lib", "Pred (original)", COL["pred"]), ("rand_nd_matchlib", "Random any (n-matched)", COL["rand"]),
             ("struct_matchlib", "Random std-grid (n-matched)", COL["struct"]), ("retro_lib", "Retro (original)", COL["retro"]),
             ("normal_lib", "Standard grid (all non-shifted)", COL["grid"])]

    def drive_val(r, name, key):
        if name in r["drive"]:
            return float(r["drive"][name][key])
        names = sorted(k for k in r["drive"] if k.startswith(name))
        return float(np.mean([r["drive"][k][key] for k in names]))

    summary["drive"] = {"sets": {}, "tests": {}}
    for name, _, _ in dsets + [("velocity", "", ""), ("all_recurrent", "", "")]:
        summary["drive"]["sets"][name] = {k: [drive_val(r, name, k) for r in R] for k in ("tangent", "radial", "radial_pc6", "cos_with_motion", "magnitude", "from_start")}
        nset = [r["drive"][name]["n_set"] if name in r["drive"] else int(np.mean([r["drive"][k]["n_set"] for k in r["drive"] if k.startswith(name)])) for r in R]
        summary["drive"]["sets"][name]["n_set"] = nset
        summary["drive"]["sets"][name]["tangent_per_unit_x1000"] = [1000 * t / max(m, 1) for t, m in zip(summary["drive"]["sets"][name]["tangent"], nset)]
        summary["drive"]["sets"][name]["radial_per_unit_x1000"] = [1000 * t / max(m, 1) for t, m in zip(summary["drive"]["sets"][name]["radial"], nset)]
        summary["drive"]["sets"][name]["tangent_over_radial"] = [t / r_ if r_ else float("nan") for t, r_ in zip(summary["drive"]["sets"][name]["tangent"], summary["drive"]["sets"][name]["radial"])]
    for pred, others in (("pred_con", ["rand_nd_matchcon", "struct_matchcon", "retro_con", "band"]), ("pred_lib", ["rand_nd_matchlib", "struct_matchlib", "retro_lib", "normal_lib"])):
        for key in ("tangent", "radial", "cos_with_motion", "tangent_over_radial", "tangent_per_unit_x1000", "radial_per_unit_x1000", "magnitude"):
            for o in others:
                summary["drive"]["tests"][f"{key}: {pred} vs {o}"] = paired(summary["drive"]["sets"][pred][key], summary["drive"]["sets"][o][key])

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.8))
    keys = [("tangent_per_unit_x1000", "Propulsive drive per unit\n(×1e-3, in units of the intact step)"),
            ("radial_per_unit_x1000", "Radial (bump-sustaining) drive per unit\n(×1e-3, in units of the intact step)"),
            ("cos_with_motion", "cos(one-step effect, intact motion)")]
    for ax, (key, ylabel) in zip(axes, keys):
        xs = np.arange(len(dsets))
        for i, (name, label, color) in enumerate(dsets):
            vals = np.asarray(summary["drive"]["sets"][name][key], float)
            ax.scatter(np.full(vals.size, i) + np.random.default_rng(i).uniform(-0.15, 0.15, vals.size), vals, s=14, color=color, alpha=0.6, lw=0)
            ax.plot([i - 0.28, i + 0.28], [np.median(vals)] * 2, color=color, lw=2.5)
        ax.axvline(4.5, color="#c3c2b7", lw=1, ls=":")
        ax.set_xticks(xs); ax.set_xticklabels([d[1] for d in dsets], rotation=55, ha="right", fontsize=7.5)
        ax.set_ylabel(ylabel, fontsize=9); ax.grid(alpha=0.25, axis="y"); ax.spines[["top", "right"]].set_visible(False)
        ax.text(2, ax.get_ylim()[1], "MEC-style sets", ha="center", va="bottom", fontsize=8, color="#52514e")
        ax.text(7, ax.get_ylim()[1], "original-definition sets", ha="center", va="bottom", fontsize=8, color="#52514e")
    fig.suptitle(f"What each unit set's recurrent input does to the grid population (one-step effect, intact network, n = {n} networks; dots = networks, bars = median)", fontsize=10.5)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(os.path.join(out_dir, "drive_decomposition.png"), dpi=170); plt.close(fig)

    # ------------------------------------------------------------------ 4. example trajectories (seed 0 if present)
    ex_seed = seeds[0]
    exp = os.path.join(root, f"Seed {ex_seed}", args.analysis_subdir, "aarav_definition_dynamics", "example_pca_paths.npz")
    if os.path.exists(exp):
        E = np.load(exp)
        panels = [("pred_con", "Predictive ablation (MEC-style def.)", COL["pred"]), ("rand_nd_matchcon_0", "Random ablation (n-matched)", COL["rand"]),
                  ("pred_lib", "Predictive ablation (original def.)", COL["pred"]), ("rand_nd_matchlib_0", "Random ablation (n-matched to original)", COL["rand"])]
        fig, axes = plt.subplots(2, 4, figsize=(15, 7.6))
        for col, (name, title, color) in enumerate(panels):
            cloud = E[f"{name}_cloud"]; I = E[f"{name}_intact"]; A = E[f"{name}_ablated"]
            for row, j in enumerate((0, 1)):
                ax = axes[row, col]
                ax.scatter(cloud[:, 0], cloud[:, 1], s=3, color="#c3c2b7", alpha=0.35, lw=0)
                ax.plot(I[:, j, 0], I[:, j, 1], color=COL["intact"], lw=2, label="intact")
                ax.plot(A[:, j, 0], A[:, j, 1], color=color, lw=2, label="ablated")
                ax.plot(I[0, j, 0], I[0, j, 1], "o", color=COL["intact"], ms=6); ax.plot(A[0, j, 0], A[0, j, 1], "o", color=color, ms=6)
                ax.plot(I[-1, j, 0], I[-1, j, 1], "s", color=COL["intact"], ms=6); ax.plot(A[-1, j, 0], A[-1, j, 1], "s", color=color, ms=6)
                ax.set_xticks([]); ax.set_yticks([]); ax.set_aspect("equal")
                if row == 0:
                    ax.set_title(title, fontsize=10)
                if col == 0:
                    ax.set_ylabel(f"trajectory {j + 1}\nPC2 vs PC1 (std-grid read-out)")
                if row == 0 and col == 0:
                    ax.legend(fontsize=8, frameon=False, loc="lower left")
        fig.suptitle(f"Seed {ex_seed}: grid-population PC1-2 trajectories (gray = intact cloud, t ≥ 10; ● start, ■ end). "
                     "Trajectories START near the PCA centre (encoder-initialised transient) and every ablation drifts back toward it.", fontsize=10)
        fig.tight_layout(rect=(0, 0, 1, 0.95))
        fig.savefig(os.path.join(out_dir, "example_pca_trajectories.png"), dpi=160); plt.close(fig)

    with open(os.path.join(out_dir, "definition_dynamics_summary.json"), "w") as f:
        json.dump(summary, f, indent=1)
    print(f"wrote {out_dir}")

    # ------------------------------------------------------------------ console digest
    def med_iqr(v):
        v = np.asarray(v, float); return f"{np.nanmedian(v):.3f} [{np.nanpercentile(v, 25):.3f}, {np.nanpercentile(v, 75):.3f}]"
    print("\nClass sizes (median across seeds):", {k: int(np.median(v)) for k, v in summary["class_sizes"].items()},
          "std_grid", int(np.median(summary["n_std_grid"])), "nondead", int(np.median(summary["n_nondead"])))
    print("pred_lib in std grid:", med_iqr(np.asarray(summary["overlaps"]["pred_lib_in_std_grid"]) / np.asarray(summary["class_sizes"]["pred_lib"])),
          "| pred_con in pred_lib:", med_iqr(np.asarray(summary["overlaps"]["pred_con_in_pred_lib"]) / np.asarray(summary["class_sizes"]["pred_con"])))
    for k, rec in summary["late"].items():
        print(f"\n=== late-window ({k}) ===")
        for t, v in rec["tests"].items():
            print(f"  {t:55s} n={v['n']} wins={v['wins']:2d} medΔ={v['median_diff']:+.3f} p={v['p']:.3g}")
    print("\n=== autocorr per-lag (pred vs intact same units / vs random), lags 5,10,15,20,25,29 ===")
    for k, st in summary["autocorr"].items():
        idx = [LAGS.index(l) for l in (5, 10, 15, 20, 25, 29)]
        print(f"  {k:26s} Δ(pred-intact) " + " ".join(f"{st['pred_minus_intact_same_units_median'][i]:+.3f}" for i in idx)
              + " | p " + " ".join(f"{st['p_pred_vs_intact_same_units'][i]:.2g}" for i in idx)
              + " | wins " + " ".join(f"{st['wins_pred_gt_intact'][i]:2d}" for i in idx))
        rk = [kk for kk in st if kk.startswith("p_pred_vs_rand") or kk.startswith("p_pred_vs_struct")]
        for kk in rk:
            dk = kk.replace("p_pred_vs_", "pred_minus_") + "_median"
            print(f"  {'':26s} {kk:30s} Δ " + " ".join(f"{st[dk][i]:+.3f}" for i in idx) + " | p " + " ".join(f"{st[kk][i]:.2g}" for i in idx))
        if "p_band_vs_pred" in st:
            print(f"  {'':26s} band-pred Δ " + " ".join(f"{st['band_minus_pred_median'][i]:+.3f}" for i in idx) + " | p " + " ".join(f"{st['p_band_vs_pred'][i]:.2g}" for i in idx))
            print(f"  {'':26s} retro-vs-pred p " + " ".join(f"{st['p_retro_vs_pred'][i]:.2g}" for i in idx))
    print("\n=== drive decomposition (median across seeds) ===")
    for name in summary["drive"]["sets"]:
        s = summary["drive"]["sets"][name]
        print(f"  {name:22s} n={int(np.median(s['n_set'])):5d} tangent {np.median(s['tangent']):+.3f} radial {np.median(s['radial']):+.3f} "
              f"tan/rad {np.median(s['tangent_over_radial']):.3f} cos {np.median(s['cos_with_motion']):+.3f} "
              f"tan/unit {np.median(s['tangent_per_unit_x1000']):.3f} rad/unit {np.median(s['radial_per_unit_x1000']):.3f}")
    for t, v in summary["drive"]["tests"].items():
        print(f"  {t:55s} wins={v['wins']:2d} medΔ={v['median_diff']:+.4f} p={v['p']:.3g}")


if __name__ == "__main__":
    main()
