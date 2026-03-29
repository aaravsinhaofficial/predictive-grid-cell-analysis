#!/usr/bin/env python3
"""Compute a seed-wise torus+trajectory variation metric from analysis outputs.

Metric definition (per seed):
  1) Manifold variation:
     Mean relative change in torus radius CV from baseline to each ablated condition.
  2) Trajectory variation:
     Average of:
       - style gap: relative spread in off-manifold impact across trajectory styles,
       - within-style variability: average coefficient of variation across ablation %
         settings inside each style.
  3) Combined score:
       alpha * manifold_variation + (1 - alpha) * trajectory_variation

Inputs expected under each seed run's `torus/` directory:
  - torus_metrics.json
  - off_manifold_seed_*_*_pct*_multi.json
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


EPS = 1e-9
SEED_RE = re.compile(r"Seed\s+(\d+)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--analysis-root",
        default="analysis_outputs",
        help="Root analysis folder. If missing, prefix matching is attempted (e.g., analysis_outputs_ -> analysis_outputs).",
    )
    parser.add_argument(
        "--model-subdir",
        default="Single agent path integration",
        help="Optional subdir under analysis root to search first.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.5,
        help="Weight on manifold variation in the combined score.",
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Output directory. Defaults to <analysis-root>/<model-subdir>/summary/torus_trajectory_variation.",
    )
    parser.add_argument(
        "--save-stem",
        default="torus_trajectory_variation",
        help="Filename stem for output plot/csv/json.",
    )
    return parser.parse_args()


def resolve_analysis_root(path_like: str) -> Path:
    """Resolve analysis root; supports typo/prefix input like `analysis_outputs_`."""
    raw = Path(path_like).expanduser()
    if raw.exists():
        return raw.resolve()

    parent = raw.parent if str(raw.parent) else Path(".")
    candidates = sorted(p for p in parent.glob(f"{raw.name}*") if p.is_dir())

    if not candidates and raw.name.endswith("_"):
        trimmed = raw.name.rstrip("_")
        if trimmed:
            candidates = sorted(p for p in parent.glob(f"{trimmed}*") if p.is_dir())

    if not candidates:
        raise FileNotFoundError(f"Could not resolve analysis root from: {path_like}")

    for cand in candidates:
        if cand.name == "analysis_outputs":
            return cand.resolve()
    return candidates[0].resolve()


def extract_seed(path: Path) -> int:
    match = SEED_RE.search(str(path))
    return int(match.group(1)) if match else -1


def _safe_float(x) -> float:
    try:
        val = float(x)
    except (TypeError, ValueError):
        return float("nan")
    return val if math.isfinite(val) else float("nan")


def radius_cv_from_condition(cond: Dict[str, object]) -> float:
    vals: List[float] = []

    major = _safe_float(cond.get("major_radius_cv"))
    minor = _safe_float(cond.get("minor_radius_cv"))
    if math.isfinite(major):
        vals.append(major)
    if math.isfinite(minor):
        vals.append(minor)
    if vals:
        return float(np.mean(vals))

    radius_cv = cond.get("radius_cv")
    if isinstance(radius_cv, dict):
        sub_vals = [_safe_float(v) for v in radius_cv.values()]
        sub_vals = [v for v in sub_vals if math.isfinite(v)]
        if sub_vals:
            return float(np.mean(sub_vals))
    elif isinstance(radius_cv, (int, float)):
        rv = _safe_float(radius_cv)
        if math.isfinite(rv):
            return rv

    return float("nan")


def compute_manifold_variation(metrics: Dict[str, object]) -> Tuple[float, float, int]:
    baseline_key = None
    for key, value in metrics.items():
        if not isinstance(value, dict):
            continue
        if "baseline" in key.lower():
            baseline_key = key
            break
    if baseline_key is None:
        return float("nan"), float("nan"), 0

    baseline_cond = metrics.get(baseline_key, {})
    if not isinstance(baseline_cond, dict):
        return float("nan"), float("nan"), 0

    baseline_cv = radius_cv_from_condition(baseline_cond)
    if not math.isfinite(baseline_cv):
        return float("nan"), float("nan"), 0

    deltas: List[float] = []
    for key, value in metrics.items():
        if key == baseline_key or not isinstance(value, dict):
            continue
        cv = radius_cv_from_condition(value)
        if not math.isfinite(cv):
            continue
        rel_delta = abs(cv - baseline_cv) / (abs(baseline_cv) + EPS)
        deltas.append(float(rel_delta))

    if not deltas:
        return baseline_cv, float("nan"), 0
    return baseline_cv, float(np.mean(deltas)), len(deltas)


def compute_trajectory_variation(torus_dir: Path, seed: int) -> Dict[str, object]:
    style_to_impacts: Dict[str, List[float]] = {}
    n_files = 0

    for off_json in sorted(torus_dir.glob("off_manifold_seed_*_multi.json")):
        with open(off_json, "r") as f:
            payload = json.load(f)
        payload_seed = int(payload.get("seed", seed))
        if seed >= 0 and payload_seed != seed:
            continue

        style = str(payload.get("trajectory_style", "unknown")).strip().lower()
        baseline = _safe_float(payload.get("baseline_mean_distance"))
        pred = _safe_float(payload.get("predictive_mean_distance"))
        rand = _safe_float(payload.get("random_mean_distance"))
        retro = _safe_float(payload.get("retrospective_mean_distance"))
        vals = [pred, rand, retro]
        vals = [v for v in vals if math.isfinite(v)]
        if not math.isfinite(baseline) or abs(baseline) < EPS or not vals:
            continue

        impact = float(np.mean([(v - baseline) / (abs(baseline) + EPS) for v in vals]))
        style_to_impacts.setdefault(style, []).append(impact)
        n_files += 1

    if not style_to_impacts:
        return {
            "trajectory_variation": float("nan"),
            "style_gap": float("nan"),
            "within_style_variation": float("nan"),
            "style_mean_impacts": {},
            "off_file_count": 0,
        }

    style_mean_impacts = {style: float(np.mean(vals)) for style, vals in style_to_impacts.items() if vals}
    mean_arr = np.array(list(style_mean_impacts.values()), dtype=float)
    mean_arr = mean_arr[np.isfinite(mean_arr)]

    if mean_arr.size >= 2:
        style_gap = float(np.ptp(mean_arr) / (np.mean(np.abs(mean_arr)) + EPS))
    else:
        style_gap = float("nan")

    within_vals: List[float] = []
    for vals in style_to_impacts.values():
        arr = np.asarray(vals, dtype=float)
        arr = arr[np.isfinite(arr)]
        if arr.size <= 1:
            continue
        denom = abs(float(np.mean(arr))) + EPS
        within_vals.append(float(np.std(arr) / denom))
    within_style = float(np.mean(within_vals)) if within_vals else 0.0

    if math.isfinite(style_gap):
        trajectory_variation = 0.5 * style_gap + 0.5 * within_style
    else:
        trajectory_variation = within_style

    return {
        "trajectory_variation": float(trajectory_variation),
        "style_gap": float(style_gap),
        "within_style_variation": float(within_style),
        "style_mean_impacts": style_mean_impacts,
        "off_file_count": int(n_files),
    }


def choose_search_root(analysis_root: Path, model_subdir: str) -> Path:
    candidate = analysis_root / model_subdir
    return candidate if model_subdir and candidate.exists() else analysis_root


def collect_records(search_root: Path, alpha: float) -> List[Dict[str, object]]:
    records: List[Dict[str, object]] = []
    for metrics_path in sorted(search_root.rglob("torus_metrics.json")):
        torus_dir = metrics_path.parent
        seed = extract_seed(metrics_path)

        with open(metrics_path, "r") as f:
            metrics = json.load(f)

        baseline_cv, manifold_var, manifold_conds = compute_manifold_variation(metrics)
        traj = compute_trajectory_variation(torus_dir, seed)
        trajectory_var = _safe_float(traj["trajectory_variation"])

        if not (math.isfinite(manifold_var) or math.isfinite(trajectory_var)):
            continue

        if math.isfinite(manifold_var) and math.isfinite(trajectory_var):
            combined = alpha * manifold_var + (1.0 - alpha) * trajectory_var
        elif math.isfinite(manifold_var):
            combined = manifold_var
        else:
            combined = trajectory_var

        style_impacts = traj.get("style_mean_impacts", {})
        rec = {
            "seed": int(seed),
            "torus_dir": str(torus_dir),
            "baseline_radius_cv": baseline_cv,
            "manifold_variation": manifold_var,
            "trajectory_variation": trajectory_var,
            "combined_metric": float(combined),
            "style_gap": _safe_float(traj["style_gap"]),
            "within_style_variation": _safe_float(traj["within_style_variation"]),
            "random_walk_impact": _safe_float(style_impacts.get("random_walk", float("nan"))),
            "straight_impact": _safe_float(style_impacts.get("straight", float("nan"))),
            "off_file_count": int(traj["off_file_count"]),
            "manifold_condition_count": int(manifold_conds),
        }
        records.append(rec)

    # Keep one best record per seed (prefer richer off-manifold coverage).
    by_seed: Dict[int, Dict[str, object]] = {}
    unseeded: List[Dict[str, object]] = []
    for rec in records:
        seed = int(rec["seed"])
        if seed < 0:
            unseeded.append(rec)
            continue
        prev = by_seed.get(seed)
        if prev is None or int(rec["off_file_count"]) > int(prev["off_file_count"]):
            by_seed[seed] = rec

    out = sorted(by_seed.values(), key=lambda r: int(r["seed"])) + unseeded
    return out


def write_csv(records: List[Dict[str, object]], out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "seed",
        "torus_dir",
        "baseline_radius_cv",
        "manifold_variation",
        "trajectory_variation",
        "combined_metric",
        "style_gap",
        "within_style_variation",
        "random_walk_impact",
        "straight_impact",
        "off_file_count",
        "manifold_condition_count",
    ]
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for rec in records:
            writer.writerow(rec)


def write_json(records: List[Dict[str, object]], out_json: Path, alpha: float, search_root: Path) -> None:
    payload = {
        "metric_name": "torus_trajectory_variation",
        "formula": {
            "manifold_variation": "mean_i |cv_i - cv_baseline| / (|cv_baseline| + eps)",
            "trajectory_variation": "0.5 * style_gap + 0.5 * within_style_variation",
            "style_gap": "(max(style_mean_impacts)-min(style_mean_impacts)) / (mean(abs(style_mean_impacts)) + eps)",
            "within_style_variation": "mean_style std(style_impacts) / (|mean(style_impacts)| + eps)",
            "combined_metric": f"{alpha:.3f} * manifold_variation + {1.0 - alpha:.3f} * trajectory_variation",
        },
        "search_root": str(search_root),
        "seed_count": len(records),
        "records": records,
    }
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(payload, f, indent=2)


def plot_records(records: List[Dict[str, object]], out_png: Path, alpha: float) -> None:
    seeds = [int(r["seed"]) for r in records]
    labels = [f"Seed {s}" if s >= 0 else f"Run {i}" for i, s in enumerate(seeds)]
    manifold = np.array([_safe_float(r["manifold_variation"]) for r in records], dtype=float)
    trajectory = np.array([_safe_float(r["trajectory_variation"]) for r in records], dtype=float)
    combined = np.array([_safe_float(r["combined_metric"]) for r in records], dtype=float)

    x = np.arange(len(records))
    width = 0.34
    fig_w = max(8.0, 1.15 * len(records) + 3.0)
    fig, ax = plt.subplots(figsize=(fig_w, 5.2))

    ax.bar(x - width / 2, manifold, width=width, color="#4C72B0", alpha=0.9, label="Manifold variation")
    ax.bar(x + width / 2, trajectory, width=width, color="#55A868", alpha=0.9, label="Trajectory variation")
    ax.plot(x, combined, color="#222222", marker="o", linewidth=2.0, label=f"Combined metric (alpha={alpha:.2f})")

    for xi, yi in zip(x, combined):
        if math.isfinite(float(yi)):
            ax.text(xi, yi + 0.012, f"{yi:.2f}", ha="center", va="bottom", fontsize=8, color="#222222")

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Relative variation (unitless)")
    ax.set_title("Torus-Trajectory Variation Metric Across Seeds")
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, ncol=3, loc="upper left")
    fig.tight_layout()

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=220)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    alpha = float(np.clip(args.alpha, 0.0, 1.0))

    analysis_root = resolve_analysis_root(args.analysis_root)
    search_root = choose_search_root(analysis_root, args.model_subdir)
    records = collect_records(search_root, alpha=alpha)
    if not records:
        raise SystemExit(f"No valid torus/off-manifold records found under: {search_root}")

    if args.out_dir is None:
        if args.model_subdir and (analysis_root / args.model_subdir).exists():
            out_dir = analysis_root / args.model_subdir / "summary" / "torus_trajectory_variation"
        else:
            out_dir = analysis_root / "summary" / "torus_trajectory_variation"
    else:
        out_dir = Path(args.out_dir).expanduser()

    stem = args.save_stem
    out_png = out_dir / f"{stem}.png"
    out_csv = out_dir / f"{stem}.csv"
    out_json = out_dir / f"{stem}.json"

    write_csv(records, out_csv)
    write_json(records, out_json, alpha=alpha, search_root=search_root)
    plot_records(records, out_png, alpha=alpha)

    print(f"[torus_trajectory_variation_metric] search_root: {search_root}")
    print(f"[torus_trajectory_variation_metric] seeds processed: {len(records)}")
    print(f"[torus_trajectory_variation_metric] wrote: {out_png}")
    print(f"[torus_trajectory_variation_metric] wrote: {out_csv}")
    print(f"[torus_trajectory_variation_metric] wrote: {out_json}")


if __name__ == "__main__":
    main()
