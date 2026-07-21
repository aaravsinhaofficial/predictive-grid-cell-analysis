"""Cross-seed aggregation of the property-matched ablation dose-response.

Reads each seed's ``pgc_matched_ablation_summary.json`` and answers the causal
question across the cohort: is ablating a functional class (predictive /
retrospective / zero-lag) more damaging than a property-matched control? For
each class it compares the pre-ceiling slope (cm decoding error added per unit
removed) of the TARGET vs its MATCHED and RANDOM controls, with a paired test
(target vs matched) over seeds.
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

CLASSES = ("predictive", "retrospective", "normal")


def find_summaries(models_root=None, checkpoints=None, out_subdir="pgc_rigor"):
    out = []
    cands = []
    if checkpoints:
        cands += list(checkpoints)
    if models_root:
        cands += glob.glob(os.path.join(models_root, "**", "most_recent_model.pth"), recursive=True)
    for ck in cands:
        p = analysis_dir_for_checkpoint(Path(ck)) / out_subdir / "pgc_matched_ablation_summary.json"
        if p.is_file():
            out.append((C.seed_id_from_path(ck), str(p)))
    seen, uniq = set(), []
    for sid, p in sorted(out):
        if sid in seen:
            continue
        seen.add(sid); uniq.append((sid, p))
    return uniq


def _slope(d, cls, which):
    sl = ((d.get("preceiling_slopes", {}).get(cls) or {}).get(which) or {})
    return sl.get("slope_cm_per_unit", np.nan)


def aggregate(summaries):
    rows = {c: {"target": [], "matched": [], "random": [], "seeds": []} for c in CLASSES}
    baselines, ceilings = [], []
    for sid, p in summaries:
        d = json.load(open(p))
        baselines.append(d.get("baseline_error_cm", np.nan))
        ceilings.append(d.get("chance_ceiling_cm", np.nan))
        for c in CLASSES:
            rows[c]["target"].append(_slope(d, c, "target"))
            rows[c]["matched"].append(_slope(d, c, "matched"))
            rows[c]["random"].append(_slope(d, c, "random"))
            rows[c]["seeds"].append(sid)
    from scipy import stats
    result = {"n_seeds": len(summaries),
              "baseline_error_cm_mean": float(np.nanmean(baselines)),
              "chance_ceiling_cm_mean": float(np.nanmean(ceilings)), "classes": {}}
    for c in CLASSES:
        t = np.array(rows[c]["target"], float)
        m = np.array(rows[c]["matched"], float)
        r = np.array(rows[c]["random"], float)
        both = np.isfinite(t) & np.isfinite(m)
        d_tm = t[both] - m[both]
        pt = pw = np.nan
        if both.sum() >= 2 and np.nanstd(d_tm) > 0:
            pt = float(stats.ttest_rel(t[both], m[both]).pvalue)
            try:
                pw = float(stats.wilcoxon(t[both], m[both]).pvalue)
            except Exception:
                pw = np.nan
        def ms(a):
            a = a[np.isfinite(a)]
            return [float(np.mean(a)) if a.size else np.nan,
                    float(np.std(a, ddof=1) / np.sqrt(a.size)) if a.size > 1 else 0.0, int(a.size)]
        result["classes"][c] = {
            "target_slope_mean_sem": ms(t), "matched_slope_mean_sem": ms(m),
            "random_slope_mean_sem": ms(r),
            "target_minus_matched_mean": float(np.nanmean(d_tm)) if d_tm.size else np.nan,
            "paired_t_p": pt, "wilcoxon_p": pw, "n_paired": int(both.sum()),
            "target_per_seed": t.tolist(), "matched_per_seed": m.tolist(),
        }
    return result


def plot(result, out_png):
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    x = np.arange(len(CLASSES)); w = 0.26
    for i, (which, color, off) in enumerate([("target", "C3", -w), ("matched", "C0", 0), ("random", "0.6", w)]):
        means = [result["classes"][c][f"{which}_slope_mean_sem"][0] for c in CLASSES]
        sems = [result["classes"][c][f"{which}_slope_mean_sem"][1] for c in CLASSES]
        ax.bar(x + off, means, w, yerr=sems, capsize=3, color=color,
               label={"target": "class (target)", "matched": "property-matched", "random": "random"}[which])
    for xi, c in enumerate(CLASSES):
        p = result["classes"][c]["wilcoxon_p"]
        if np.isfinite(p):
            ax.text(xi, ax.get_ylim()[1] * 0.92, f"p={p:.3f}", ha="center", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels([c.capitalize() for c in CLASSES])
    ax.set_ylabel("pre-ceiling slope (cm / unit removed)")
    ax.set_title(f"Matched-ablation dose slope across {result['n_seeds']} seeds (mean±SEM)")
    ax.legend(frameon=False)
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
        print("No pgc_matched_ablation_summary.json found."); return
    print(f"Aggregating {len(sums)} seeds: {[s for s, _ in sums]}")
    result = aggregate(sums)
    os.makedirs(args.out_dir, exist_ok=True)
    json.dump(result, open(os.path.join(args.out_dir, "pgc_matched_ablation_across_seeds.json"), "w"), indent=2)
    plot(result, os.path.join(args.out_dir, "pgc_matched_ablation_across_seeds.png"))
    for c in CLASSES:
        cc = result["classes"][c]
        print(f"  {c:13s} target={cc['target_slope_mean_sem'][0]:.3f} "
              f"matched={cc['matched_slope_mean_sem'][0]:.3f} "
              f"random={cc['random_slope_mean_sem'][0]:.3f} "
              f"| target-matched p(t)={cc['paired_t_p']} p(wil)={cc['wilcoxon_p']}")
    print(f"saved -> {args.out_dir}")


if __name__ == "__main__":
    main()
