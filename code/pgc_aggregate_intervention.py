"""Cross-seed aggregation of the PGC-subspace intervention.

Reads each seed's ``pgc_intervention_summary.json`` and reports, across the
cohort: (1) does driving the PGC subspace move toroidal phase predictably
(drive->displacement correlation), and (2) does replaying the predictive
subspace into an ablated network rescue decoding + phase flow.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from pathlib import Path

import numpy as np

_CODE_DIR = os.path.dirname(os.path.abspath(__file__))
if _CODE_DIR not in sys.path:
    sys.path.insert(0, _CODE_DIR)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from path_utils import analysis_dir_for_checkpoint  # noqa: E402
import pgc_common as C  # noqa: E402


def find_summaries(models_root=None, checkpoints=None, out_subdir="pgc_rigor"):
    cands = list(checkpoints or [])
    if models_root:
        cands += glob.glob(os.path.join(models_root, "**", "most_recent_model.pth"), recursive=True)
    out, seen = [], set()
    for ck in cands:
        p = analysis_dir_for_checkpoint(Path(ck)) / out_subdir / "pgc_intervention_summary.json"
        sid = C.seed_id_from_path(ck)
        if p.is_file() and sid not in seen:
            seen.add(sid); out.append((sid, str(p)))
    return sorted(out)


def _msem(a):
    a = np.asarray(a, float); a = a[np.isfinite(a)]
    return [float(np.mean(a)) if a.size else np.nan,
            float(np.std(a, ddof=1) / np.sqrt(a.size)) if a.size > 1 else 0.0, int(a.size)]


def aggregate(summaries):
    rec = {k: [] for k in ("dose_r", "rescue_decode", "rescue_pv", "freeze_ratio",
                           "err_intact", "err_ablated", "err_rescued",
                           "pv_intact", "pv_ablated", "pv_rescued", "rollout_f64")}
    for sid, p in summaries:
        d = json.load(open(p))
        rec["dose_r"].append(d.get("dose_curve", {}).get("pearson_r", np.nan))
        re = d.get("rescue_effect", {})
        rec["rescue_decode"].append(re.get("decode_error_recovered_fraction", np.nan))
        rec["rescue_pv"].append(re.get("phase_velocity_recovered_fraction", np.nan))
        rec["freeze_ratio"].append(re.get("phase_velocity_freeze_ratio", np.nan))
        de = d.get("decode_error_cm", {})
        rec["err_intact"].append(de.get("intact", np.nan))
        rec["err_ablated"].append(de.get("ablated", np.nan))
        rec["err_rescued"].append(de.get("rescued", np.nan))
        pv = d.get("phase_velocity", {})
        rec["pv_intact"].append(pv.get("intact", np.nan))
        rec["pv_ablated"].append(pv.get("ablated", np.nan))
        rec["pv_rescued"].append(pv.get("rescued", np.nan))
        rec["rollout_f64"].append(d.get("rollout_match", {}).get("float64_cpu_max_abs", np.nan))
    from scipy import stats
    # paired test: does ablation raise decode error, and does rescue lower it back?
    ei = np.array(rec["err_intact"], float); ea = np.array(rec["err_ablated"], float); er = np.array(rec["err_rescued"], float)
    def paired(a, b):
        m = np.isfinite(a) & np.isfinite(b)
        if m.sum() < 2:
            return np.nan
        try:
            return float(stats.wilcoxon(a[m], b[m]).pvalue)
        except Exception:
            return np.nan
    result = {
        "n_seeds": len(summaries), "seeds": [s for s, _ in summaries],
        "drive_to_displacement_pearson_r": _msem(rec["dose_r"]),
        "rescue_decode_recovered_fraction": _msem(rec["rescue_decode"]),
        "rescue_phase_velocity_recovered_fraction": _msem(rec["rescue_pv"]),
        "decode_error_intact_cm": _msem(rec["err_intact"]),
        "decode_error_ablated_cm": _msem(rec["err_ablated"]),
        "decode_error_rescued_cm": _msem(rec["err_rescued"]),
        "phase_velocity_intact": _msem(rec["pv_intact"]),
        "phase_velocity_ablated": _msem(rec["pv_ablated"]),
        "phase_velocity_rescued": _msem(rec["pv_rescued"]),
        "rollout_float64_max_abs": _msem(rec["rollout_f64"]),
        "wilcoxon_ablated_vs_intact_error": paired(ea, ei),
        "wilcoxon_rescued_vs_ablated_error": paired(er, ea),
        "per_seed": {k: rec[k] for k in rec},
    }
    return result


def plot(result, out_png):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3))
    ax = axes[0]
    conds = ["intact", "ablated", "rescued"]
    means = [result[f"decode_error_{c}_cm"][0] for c in conds]
    sems = [result[f"decode_error_{c}_cm"][1] for c in conds]
    ax.bar(range(3), means, yerr=sems, capsize=4, color=["#2ca02c", "#d62728", "#1f77b4"])
    ax.set_xticks(range(3)); ax.set_xticklabels(["intact", "ablated", "rescued"])
    ax.set_ylabel("decoding error (cm)")
    ax.set_title(f"PGC ablation & subspace rescue ({result['n_seeds']} seeds)")
    r = result["rescue_decode_recovered_fraction"]
    ax.text(0.5, 0.92, f"rescue recovers {100*r[0]:.0f}±{100*r[1]:.0f}% of the gap",
            transform=ax.transAxes, ha="center", fontsize=9)
    ax = axes[1]
    dr = result["drive_to_displacement_pearson_r"]
    ax.hist([v for v in result["per_seed"]["dose_r"] if np.isfinite(v)], bins=12, color="#1f77b4", alpha=0.8)
    ax.axvline(dr[0], color="k", lw=1.5)
    ax.set_xlabel("drive→phase-displacement Pearson r (per seed)")
    ax.set_ylabel("# seeds")
    ax.set_title(f"PGC drive moves phase: r = {dr[0]:.2f} ± {dr[1]:.2f}")
    fig.tight_layout(); fig.savefig(out_png, dpi=180); plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--models_root", default=None)
    p.add_argument("--checkpoints", nargs="*", default=None)
    p.add_argument("--out_subdir", default="pgc_rigor")
    p.add_argument("--out_dir", default="analysis_outputs/pgc_rigor_summary")
    args = p.parse_args()
    sums = find_summaries(args.models_root, args.checkpoints, args.out_subdir)
    if not sums:
        print("No pgc_intervention_summary.json found."); return
    print(f"Aggregating {len(sums)} seeds: {[s for s, _ in sums]}")
    result = aggregate(sums)
    os.makedirs(args.out_dir, exist_ok=True)
    json.dump(result, open(os.path.join(args.out_dir, "pgc_intervention_across_seeds.json"), "w"), indent=2)
    plot(result, os.path.join(args.out_dir, "pgc_intervention_across_seeds.png"))
    for k in ("drive_to_displacement_pearson_r", "rescue_decode_recovered_fraction",
              "rescue_phase_velocity_recovered_fraction", "decode_error_intact_cm",
              "decode_error_ablated_cm", "decode_error_rescued_cm", "rollout_float64_max_abs"):
        m, s, n = result[k]
        print(f"  {k:44s} {m:.4f} ± {s:.4f} (n={n})")
    print(f"  wilcoxon ablated>intact error p = {result['wilcoxon_ablated_vs_intact_error']}")
    print(f"  wilcoxon rescued<ablated error p = {result['wilcoxon_rescued_vs_ablated_error']}")
    print(f"saved -> {args.out_dir}")


if __name__ == "__main__":
    main()
