#!/usr/bin/env python3
"""Create a publication-ready summary figure for predictive grid analyses.

Panels:
  A) Example units with strong predictive tuning:
     - Ratemaps at shifts (default: 0, -10, -20)
     - Spatial autocorrelograms (SACs) at the same shifts
     - Gridness score (GS) vs shift curve
  B) GS vs shift heatmap for one RNN
  C) Scatter: GS at shift=0 vs GS at best predictive shift
  D) GS ratio distribution for predictive grid cells
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage as ndi
import torch
from matplotlib.transforms import Bbox

from model import RNN
from place_cells import PlaceCells
from trajectory_generator import TrajectoryGenerator
from single_seed_torus_distance import build_options
from plotting_functional_classes import build_scorer, ratemap_and_sac_for_shift
from visualize import collect_sequences
from path_utils import analysis_dir_for_checkpoint, analysis_summary_dir, model_name_from_checkpoint


def parse_shift_list(arg: str, shift_mode: str) -> List[float]:
    caster = int if shift_mode == "time" else float
    return [caster(x) for x in arg.split(",") if x.strip()]


def load_gridness_data(ckpt_path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    grid_path = analysis_dir_for_checkpoint(ckpt_path) / "gridness_data.npz"
    if not grid_path.exists():
        raise FileNotFoundError(f"Missing gridness_data.npz: {grid_path}")
    data = np.load(grid_path)
    scores_60 = np.asarray(data["scores_60"], dtype=float)
    lag_cm = np.asarray(data["lag_cm"], dtype=float)
    if "zero_scores" in data:
        zero_scores = np.asarray(data["zero_scores"], dtype=float)
    else:
        zero_idx = int(np.nanargmin(np.abs(lag_cm)))
        zero_scores = scores_60[zero_idx]
    return scores_60, lag_cm, zero_scores


def discover_checkpoints(root: Path) -> List[Path]:
    ckpts = []
    for ckpt in root.rglob("final_model.pth"):
        if "analysis_outputs" in str(ckpt):
            continue
        ckpts.append(ckpt)
    return sorted(ckpts)


def seed_label(ckpt_path: Path) -> str:
    for part in ckpt_path.parts:
        if part.startswith("Seed "):
            return part
    return ckpt_path.parent.name


def compute_grid_stats(
    scores_60: np.ndarray,
    lag_cm: np.ndarray,
    zero_scores: np.ndarray,
    gridness_threshold: float,
    min_shift_cm: float,
) -> Dict[str, np.ndarray]:
    best_idx = np.nanargmax(scores_60, axis=0)
    best_scores = scores_60[best_idx, np.arange(scores_60.shape[1])]
    best_cm = lag_cm[best_idx]

    grid_mask = np.isfinite(best_scores) & (best_scores >= gridness_threshold)
    pred_mask = grid_mask & (best_cm >= min_shift_cm)

    pred_scores = np.full_like(best_scores, np.nan, dtype=float)
    pos_mask = lag_cm >= min_shift_cm
    if np.any(pos_mask):
        pred_scores = np.nanmax(scores_60[pos_mask], axis=0)

    ratio = pred_scores / (zero_scores + 1e-6)
    return {
        "best_scores": best_scores,
        "best_cm": best_cm,
        "grid_mask": grid_mask,
        "pred_mask": pred_mask,
        "pred_scores": pred_scores,
        "zero_scores": zero_scores,
        "ratio": ratio,
    }


def select_best_checkpoint(
    checkpoints: List[Path],
    gridness_threshold: float,
    min_shift_cm: float,
    min_ratio: float | None,
) -> Tuple[Path, Dict[str, float]]:
    best_ckpt = None
    best_score = -np.inf
    best_meta: Dict[str, float] = {}
    for ckpt in checkpoints:
        try:
            scores_60, lag_cm, zero_scores = load_gridness_data(ckpt)
        except FileNotFoundError:
            continue
        stats = compute_grid_stats(scores_60, lag_cm, zero_scores, gridness_threshold, min_shift_cm)
        candidate_mask = stats["pred_mask"] & np.isfinite(stats["pred_scores"]) & np.isfinite(stats["ratio"])
        if min_ratio is not None:
            candidate_mask &= stats["ratio"] >= min_ratio
        combo = stats["pred_scores"] * stats["ratio"]
        vals = combo[candidate_mask]
        if vals.size == 0:
            continue
        top = np.sort(vals)[-min(10, vals.size) :]
        seed_score = float(np.nanmean(top))
        if seed_score > best_score:
            best_score = seed_score
            best_ckpt = ckpt
            best_meta = {
                "seed_score": seed_score,
                "predictive_units": int(candidate_mask.sum()),
            }

    if best_ckpt is None:
        raise RuntimeError("No checkpoints with predictive gridness data were found.")
    return best_ckpt, best_meta


def pick_predictive_units(
    scores_60: np.ndarray,
    lag_cm: np.ndarray,
    zero_scores: np.ndarray,
    gridness_threshold: float,
    min_shift_cm: float,
    n_units: int,
    rank_mode: str,
    min_ratio: float | None,
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    stats = compute_grid_stats(scores_60, lag_cm, zero_scores, gridness_threshold, min_shift_cm)
    pred_scores = stats["pred_scores"]
    ratio = stats["ratio"]
    candidate_mask = stats["pred_mask"] & np.isfinite(pred_scores) & np.isfinite(ratio)
    if min_ratio is not None:
        candidate_mask &= ratio >= min_ratio

    pred_candidates = np.where(candidate_mask)[0]
    if pred_candidates.size == 0:
        pred_candidates = np.where(stats["pred_mask"] & np.isfinite(pred_scores))[0]
    if pred_candidates.size == 0:
        pred_candidates = np.where(stats["grid_mask"] & np.isfinite(stats["best_scores"]))[0]

    if rank_mode == "ratio":
        rank_vals = ratio[pred_candidates]
    elif rank_mode == "pred_score":
        rank_vals = pred_scores[pred_candidates]
    else:
        rank_vals = pred_scores[pred_candidates] * np.clip(ratio[pred_candidates], 1.0, 10.0)

    order = np.argsort(rank_vals)[::-1]
    selected = pred_candidates[order][:n_units]

    stats = stats
    return selected, stats


def build_seed_data(ckpt_path: Path, gridness_threshold: float, min_shift_cm: float) -> Dict[str, np.ndarray]:
    scores_60, lag_cm, zero_scores = load_gridness_data(ckpt_path)
    stats = compute_grid_stats(scores_60, lag_cm, zero_scores, gridness_threshold, min_shift_cm)
    return {
        "scores_60": scores_60,
        "lag_cm": lag_cm,
        "zero_scores": zero_scores,
        "stats": stats,
    }


def score_panel_b(seed_data: Dict[str, np.ndarray], max_units: int) -> float:
    scores_60 = seed_data["scores_60"]
    lag_cm = seed_data["lag_cm"]
    stats = seed_data["stats"]
    units = np.where(stats["grid_mask"])[0]
    if units.size == 0:
        return -np.inf
    if units.size > max_units:
        top_order = np.argsort(stats["best_scores"][units])[::-1][:max_units]
        units = units[top_order]
    order = np.argsort(stats["best_cm"][units])
    units = units[order]
    heat = scores_60[:, units].T
    p2 = float(np.nanpercentile(heat, 2))
    p98 = float(np.nanpercentile(heat, 98))
    if not np.isfinite(p2) or not np.isfinite(p98):
        return -np.inf
    return (p98 - p2) * math.log1p(heat.shape[0])


def score_panel_c(seed_data: Dict[str, np.ndarray]) -> float:
    stats = seed_data["stats"]
    pred_scores = stats["pred_scores"]
    zero_scores = stats["zero_scores"]
    mask = stats["pred_mask"] & np.isfinite(pred_scores) & np.isfinite(zero_scores)
    if not mask.any():
        return -np.inf
    delta = pred_scores[mask] - zero_scores[mask]
    return float(np.nanmean(delta)) * math.log1p(mask.sum())


def score_panel_d(seed_data: Dict[str, np.ndarray]) -> float:
    stats = seed_data["stats"]
    ratios = stats["ratio"][stats["pred_mask"]]
    ratios = ratios[np.isfinite(ratios)]
    if ratios.size == 0:
        return -np.inf
    return float(np.nanmedian(ratios)) * math.log1p(ratios.size)


def plot_unit_row(
    fig: plt.Figure,
    row_spec,
    scorer,
    xs: np.ndarray,
    ys: np.ndarray,
    activations_u: np.ndarray,
    shifts: List[int],
    gs_curve_x: np.ndarray,
    gs_curve_y: np.ndarray,
    unit_label: str,
    show_headers: bool,
    shift_mode: str,
    periodic: bool,
    space_projection: str,
) -> List[plt.Axes]:
    gs_row = row_spec.subgridspec(2, 4, width_ratios=[1.0, 1.0, 1.0, 1.2], wspace=0.12, hspace=0.08)
    ax_curve = fig.add_subplot(gs_row[:, 3])
    axes = [ax_curve]

    rms: List[np.ndarray] = []
    sacs: List[np.ndarray] = []
    for shift in shifts:
        rm, sac, _, _ = ratemap_and_sac_for_shift(
            scorer,
            xs,
            ys,
            activations_u,
            shift,
            shift_mode=shift_mode,
            periodic=periodic,
            space_projection=space_projection,
        )
        rms.append(rm)
        sacs.append(sac)

    rm_vals = [r for r in rms if r is not None]
    sac_vals = [s for s in sacs if s is not None]
    rm_vmin = np.nanmin([np.nanmin(r) for r in rm_vals]) if rm_vals else 0.0
    rm_vmax = np.nanmax([np.nanmax(r) for r in rm_vals]) if rm_vals else 1.0
    sac_max = np.nanmax([np.nanmax(np.abs(s)) for s in sac_vals]) if sac_vals else 1.0

    for j, shift in enumerate(shifts):
        rm = rms[j]
        sac = sacs[j]
        ax_rm = fig.add_subplot(gs_row[0, j])
        rm_vis = ndi.gaussian_filter(rm, sigma=0.6) if rm is not None else rm
        ax_rm.imshow(rm_vis, cmap="jet", interpolation="nearest", vmin=rm_vmin, vmax=rm_vmax)
        if show_headers:
            units = "cm" if shift_mode == "space" else "steps"
            ax_rm.set_title(f"delta {shift:g} {units}", fontsize=8)
        ax_rm.axis("off")
        axes.append(ax_rm)
        if j == 0:
            ax_rm.text(-0.22, 0.5, "Ratemap", rotation=90, transform=ax_rm.transAxes,
                       ha="center", va="center", fontsize=8)
            ax_rm.text(-0.3, 1.1, unit_label, transform=ax_rm.transAxes, fontsize=9, fontweight="bold")

        ax_sac = fig.add_subplot(gs_row[1, j])
        if sac is not None:
            sac_vis = ndi.gaussian_filter(sac, sigma=0.6)
            ax_sac.imshow(sac_vis, cmap="coolwarm", interpolation="nearest", vmin=-sac_max, vmax=sac_max)
            c = sac.shape[0] // 2
            ax_sac.axhline(c, color="k", lw=0.7, alpha=0.7)
            ax_sac.axvline(c, color="k", lw=0.7, alpha=0.7)
        ax_sac.axis("off")
        axes.append(ax_sac)
        if j == 0:
            ax_sac.text(-0.22, 0.5, "SAC", rotation=90, transform=ax_sac.transAxes,
                        ha="center", va="center", fontsize=8)

    ax_curve.plot(gs_curve_x, gs_curve_y, color="#2C6BB0", lw=2.0)
    ax_curve.axhline(0, color="k", lw=0.6)
    ax_curve.axvline(0, color="k", lw=0.6, ls="--")
    ax_curve.set_xlabel("Position shift (cm)", fontsize=8)
    ax_curve.set_ylabel("GS (60 deg)", fontsize=8)
    ax_curve.set_title("GS vs shift", fontsize=9)
    ax_curve.grid(True, alpha=0.25, lw=0.6)
    ax_curve.tick_params(labelsize=7)
    for spine in ["top", "right"]:
        ax_curve.spines[spine].set_visible(False)
    return axes


def save_panel_svgs(fig: plt.Figure, panels: Dict[str, List[plt.Axes]], out_dir: Path, stem: str) -> None:
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for label, axes in panels.items():
        axes = [ax for ax in axes if ax is not None]
        if not axes:
            continue
        bboxes = [ax.get_tightbbox(renderer) for ax in axes]
        bbox = Bbox.union(bboxes)
        bbox_inches = bbox.transformed(fig.dpi_scale_trans.inverted())
        out_path = out_dir / f"{stem}_panel{label}.svg"
        fig.savefig(out_path, bbox_inches=bbox_inches)


def render_summary(
    ckpt_path: Path,
    args,
    out_path: Path,
    panel_b: Dict[str, np.ndarray] | None = None,
    panel_c: Dict[str, np.ndarray] | None = None,
    panel_d: Dict[str, np.ndarray] | None = None,
) -> None:
    seed_data_a = build_seed_data(ckpt_path, args.gridness_threshold, args.min_shift_cm)
    scores_60 = seed_data_a["scores_60"]
    lag_cm = seed_data_a["lag_cm"]
    zero_scores = seed_data_a["zero_scores"]

    min_ratio = args.min_ratio if args.min_ratio and args.min_ratio > 0 else None
    selected_units, stats = pick_predictive_units(
        scores_60,
        lag_cm,
        zero_scores,
        gridness_threshold=args.gridness_threshold,
        min_shift_cm=args.min_shift_cm,
        n_units=args.num_units,
        rank_mode=args.unit_rank,
        min_ratio=min_ratio,
    )

    shifts = parse_shift_list(args.panel_shifts, args.shift_mode)
    max_shift = max([abs(s) for s in shifts], default=0)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    options = build_options(ckpt_path, device=device)
    options.batch_size = args.batch_size
    if max_shift > 0 and args.shift_mode == "time":
        options.sequence_length = max(int(options.sequence_length), int(max_shift + 5))
    place_cells = PlaceCells(options)
    model = RNN(options, place_cells).to(options.device)
    model.load_state_dict(torch.load(ckpt_path, map_location=options.device))
    model.eval()
    traj_gen = TrajectoryGenerator(options, place_cells)
    periodic = bool(getattr(options, "periodic", False))

    Ng_use = max(args.Ng_use, len(selected_units))
    xs, ys, activations = collect_sequences(
        model,
        traj_gen,
        options,
        n_batches=args.n_batches,
        Ng=Ng_use,
        idxs=selected_units,
    )
    zero_idx = int(np.nanargmin(np.abs(lag_cm)))
    scorer = build_scorer(args.res, options)

    n_units = len(selected_units)

    fig_height = max(10.0, 2.2 * n_units + 5.0)
    fig = plt.figure(figsize=(15.5, fig_height))
    outer = fig.add_gridspec(1, 2, width_ratios=[3.2, 1.6], wspace=0.28)
    left = outer[0].subgridspec(n_units, 1, hspace=0.35)
    right = outer[1].subgridspec(3, 1, hspace=0.45, height_ratios=[1.2, 1.0, 1.0])

    # Panel A: example units
    axes_panel_a: List[plt.Axes] = []
    for i, unit in enumerate(selected_units):
        show_headers = i == 0
        idx = i
        a_u = activations[:, :, idx]
        gs_curve_y = scores_60[:, unit]
        axes_panel_a.extend(plot_unit_row(
            fig,
            left[i],
            scorer,
            xs,
            ys,
            a_u,
            shifts,
            lag_cm,
            gs_curve_y,
            unit_label=f"Unit {unit}",
            show_headers=show_headers,
            shift_mode=args.shift_mode,
            periodic=periodic,
            space_projection=args.space_projection,
        ))

    # Panel B: heatmap (or reuse saved predictive_classes image)
    ax_b = fig.add_subplot(right[0])
    panel_b = panel_b or seed_data_a
    b_scores = panel_b["scores_60"]
    b_lag_cm = panel_b["lag_cm"]
    b_stats = panel_b["stats"]
    b_ckpt = Path(panel_b.get("ckpt_path", ckpt_path))
    b_seed = seed_label(b_ckpt)
    axes_panel_b: List[plt.Axes] = [ax_b]
    if args.panel_b_source == "predictive_classes":
        pred_path = analysis_dir_for_checkpoint(b_ckpt) / "predictive_classes.png"
        if pred_path.exists():
            img = plt.imread(pred_path)
            ax_b.imshow(img)
            ax_b.set_title(f"Predictive classes ({b_seed})", fontsize=10)
            ax_b.axis("off")
        else:
            ax_b.text(0.5, 0.5, "predictive_classes.png not found", ha="center", va="center", fontsize=9)
            ax_b.axis("off")
    else:
        grid_mask = b_stats["grid_mask"]
        best_cm = b_stats["best_cm"]
        units = np.where(grid_mask)[0]
        if units.size == 0:
            ax_b.text(0.5, 0.5, "No units above gridness threshold", ha="center", va="center", fontsize=9)
            ax_b.axis("off")
        else:
            if units.size > args.Ng_use:
                top_order = np.argsort(b_stats["best_scores"][units])[::-1][: args.Ng_use]
                units = units[top_order]
            order = np.argsort(best_cm[units])
            units = units[order]
            heat = b_scores[:, units].T
            vmin = float(np.nanpercentile(heat, 2))
            vmax = float(np.nanpercentile(heat, 98))
            if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
                vmin = np.nanmin(heat)
                vmax = np.nanmax(heat)
            im = ax_b.imshow(
                heat,
                aspect="auto",
                origin="lower",
                extent=[b_lag_cm[0], b_lag_cm[-1], 0, heat.shape[0]],
                cmap="viridis",
                vmin=vmin,
                vmax=vmax,
            )
            ax_b.set_title(f"GS vs shift heatmap ({b_seed})", fontsize=10)
            ax_b.set_xlabel("Shift (cm)", fontsize=9)
            ax_b.set_ylabel("Units (sorted by best shift)", fontsize=9)
            ax_b.tick_params(labelsize=8)
            cbar = fig.colorbar(im, ax=ax_b, fraction=0.046, pad=0.02)
            cbar.set_label("GS (60 deg)", fontsize=8)
            cbar.ax.tick_params(labelsize=7)
            axes_panel_b.append(cbar.ax)

    # Panel C: scatter
    ax_c = fig.add_subplot(right[1])
    axes_panel_c: List[plt.Axes] = [ax_c]
    panel_c = panel_c or seed_data_a
    c_stats = panel_c["stats"]
    c_zero = panel_c["zero_scores"]
    c_pred = c_stats["pred_scores"]
    c_seed = seed_label(Path(panel_c.get("ckpt_path", ckpt_path)))
    scatter_mask = np.isfinite(c_zero) & np.isfinite(c_pred)
    ax_c.scatter(
        c_zero[scatter_mask],
        c_pred[scatter_mask],
        s=8,
        alpha=0.4,
        color="#1f77b4",
        edgecolor="none",
    )
    if scatter_mask.any():
        lo = float(np.nanmin([c_zero[scatter_mask].min(), c_pred[scatter_mask].min()]))
        hi = float(np.nanmax([c_zero[scatter_mask].max(), c_pred[scatter_mask].max()]))
        ax_c.plot([lo, hi], [lo, hi], color="black", lw=0.8, alpha=0.6)
    ax_c.set_title(f"GS at shift=0 vs best predictive ({c_seed})", fontsize=10)
    ax_c.set_xlabel("GS at shift=0", fontsize=9)
    ax_c.set_ylabel("Best GS (predictive)", fontsize=9)
    ax_c.grid(alpha=0.25, lw=0.6)
    ax_c.tick_params(labelsize=8)

    # Panel D: ratio distribution
    ax_d = fig.add_subplot(right[2])
    axes_panel_d: List[plt.Axes] = [ax_d]
    panel_d = panel_d or seed_data_a
    d_stats = panel_d["stats"]
    d_zero = panel_d["zero_scores"]
    d_pred = d_stats["pred_scores"]
    d_seed = seed_label(Path(panel_d.get("ckpt_path", ckpt_path)))
    pred_mask = d_stats["pred_mask"]
    grid_mask = d_stats["grid_mask"]
    ratios_pred = d_pred[pred_mask] / (d_zero[pred_mask] + 1e-6)
    ratios_pred = ratios_pred[np.isfinite(ratios_pred)]
    ratios_all = d_pred[grid_mask] / (d_zero[grid_mask] + 1e-6)
    ratios_all = ratios_all[np.isfinite(ratios_all)]
    if ratios_all.size:
        lo = float(np.nanpercentile(ratios_all, 1))
        hi = float(np.nanpercentile(ratios_all, 99))
        if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
            lo = float(np.nanmin(ratios_all))
            hi = float(np.nanmax(ratios_all))
        bins = 40
        ax_d.hist(ratios_all, bins=bins, range=(lo, hi), color="#bdbdbd", alpha=0.45, label="All grid units")
        if ratios_pred.size:
            ax_d.hist(ratios_pred, bins=bins, range=(lo, hi), color="#ff7f0e", alpha=0.8, label="Predictive units")
        ax_d.axvline(1.0, color="black", lw=0.8, alpha=0.6)
        ax_d.legend(frameon=False, fontsize=8)
    else:
        ax_d.text(0.5, 0.5, "No predictive units", ha="center", va="center", fontsize=9)
        ax_d.set_xticks([])
        ax_d.set_yticks([])
    ax_d.set_title(f"Predictive GS ratio distribution ({d_seed})", fontsize=10)
    ax_d.set_xlabel("Best predictive GS / GS at shift=0", fontsize=9)
    ax_d.set_ylabel("Count", fontsize=9)
    ax_d.grid(alpha=0.2, lw=0.6)
    ax_d.tick_params(labelsize=8)

    # Panel labels
    fig.text(0.02, 0.98, "A", fontsize=14, fontweight="bold")
    fig.text(0.06, 0.98, f"Example units ({seed_label(ckpt_path)})", fontsize=9)
    ax_b.text(-0.1, 1.05, "B", transform=ax_b.transAxes, fontsize=14, fontweight="bold")
    ax_c.text(-0.1, 1.05, "C", transform=ax_c.transAxes, fontsize=14, fontweight="bold")
    ax_d.text(-0.1, 1.05, "D", transform=ax_d.transAxes, fontsize=14, fontweight="bold")

    fig.subplots_adjust(left=0.05, right=0.98, top=0.97, bottom=0.05)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300)
    save_panel_svgs(
        fig,
        {"A": axes_panel_a, "B": axes_panel_b, "C": axes_panel_c, "D": axes_panel_d},
        out_path.parent,
        out_path.stem,
    )
    plt.close(fig)
    print(f"Wrote summary figure to {out_path}")


def resolve_checkpoints(args) -> List[Path]:
    if args.checkpoint_path:
        candidate = Path(args.checkpoint_path)
        if candidate.is_file():
            return [candidate]
        return discover_checkpoints(candidate)
    return discover_checkpoints(Path(args.search_root))


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a publication-ready predictive grid summary figure.")
    parser.add_argument("--checkpoint_path", type=str, default=None)
    parser.add_argument(
        "--search_root",
        type=str,
        default="Models/Single agent path integration",
        help="Root directory to scan when selecting best seed.",
    )
    parser.add_argument(
        "--best_across_seeds",
        action="store_true",
        help="Select the best checkpoint across seeds using predictive GS criteria.",
    )
    parser.add_argument(
        "--best_per_panel",
        action="store_true",
        help="Select the best seed separately for panels A/B/C/D.",
    )
    parser.add_argument(
        "--per_seed",
        action="store_true",
        help="Generate a summary figure per checkpoint in the search root.",
    )
    parser.add_argument("--out_path", type=str, default=None)
    parser.add_argument("--n_batches", type=int, default=25)
    parser.add_argument("--batch_size", type=int, default=200)
    parser.add_argument("--Ng_use", type=int, default=512)
    parser.add_argument("--res", type=int, default=20)
    parser.add_argument("--gridness_threshold", type=float, default=0.5)
    parser.add_argument("--min_shift_cm", type=float, default=5.0)
    parser.add_argument("--num_units", type=int, default=4)
    parser.add_argument("--panel_shifts", type=str, default="0,5,10")
    parser.add_argument("--shift_mode", type=str, default="time", choices=["time", "space"],
                        help="Interpret panel-shift samples as time steps or direct spatial displacement.")
    parser.add_argument("--space_projection", type=str, default="path", choices=["path", "heading"],
                        help="When --shift_mode space, use arc-length along the trajectory or heading-based projection.")
    parser.add_argument(
        "--unit_rank",
        type=str,
        default="combo",
        choices=["combo", "pred_score", "ratio"],
        help="Ranking used to pick example predictive units.",
    )
    parser.add_argument(
        "--panel_b_source",
        type=str,
        default="heatmap",
        choices=["heatmap", "predictive_classes"],
        help="Panel B source: computed heatmap or reuse predictive_classes.png.",
    )
    parser.add_argument(
        "--min_ratio",
        type=float,
        default=1.2,
        help="Minimum predictive/zero GS ratio for example units (set <=0 to disable).",
    )
    args = parser.parse_args()

    checkpoints = resolve_checkpoints(args)
    if not checkpoints:
        raise FileNotFoundError("No checkpoints found to generate summary figures.")

    if args.per_seed:
        for ckpt in checkpoints:
            model_name = model_name_from_checkpoint(ckpt)
            seed_name = seed_label(ckpt).replace(" ", "_")
            out_dir = analysis_summary_dir(model_name, "publication_summary/by_seed")
            out_path = out_dir / f"rnn_publication_summary_{seed_name}.png"
            render_summary(ckpt, args, out_path)

    if args.best_per_panel:
        seed_data_map: Dict[Path, Dict[str, np.ndarray]] = {}
        for ckpt in checkpoints:
            try:
                seed_data = build_seed_data(ckpt, args.gridness_threshold, args.min_shift_cm)
            except FileNotFoundError:
                continue
            seed_data["ckpt_path"] = str(ckpt)
            seed_data_map[ckpt] = seed_data

        if not seed_data_map:
            raise RuntimeError("No checkpoints with gridness_data.npz found for per-panel selection.")

        min_ratio = args.min_ratio if args.min_ratio and args.min_ratio > 0 else None
        panel_a_ckpt, meta = select_best_checkpoint(
            list(seed_data_map.keys()),
            gridness_threshold=args.gridness_threshold,
            min_shift_cm=args.min_shift_cm,
            min_ratio=min_ratio,
        )
        panel_b_ckpt = max(seed_data_map, key=lambda c: score_panel_b(seed_data_map[c], args.Ng_use))
        panel_c_ckpt = max(seed_data_map, key=lambda c: score_panel_c(seed_data_map[c]))
        panel_d_ckpt = max(seed_data_map, key=lambda c: score_panel_d(seed_data_map[c]))

        print(
            "[best panels] "
            f"A={seed_label(panel_a_ckpt)}, "
            f"B={seed_label(panel_b_ckpt)}, "
            f"C={seed_label(panel_c_ckpt)}, "
            f"D={seed_label(panel_d_ckpt)}"
        )

        model_name = model_name_from_checkpoint(panel_a_ckpt)
        out_dir = analysis_summary_dir(model_name, "publication_summary")
        best_path = out_dir / "rnn_publication_summary_best_panels.png"
        render_summary(
            panel_a_ckpt,
            args,
            best_path,
            panel_b=seed_data_map[panel_b_ckpt],
            panel_c=seed_data_map[panel_c_ckpt],
            panel_d=seed_data_map[panel_d_ckpt],
        )

        selection = {
            "panel_a": str(panel_a_ckpt),
            "panel_b": str(panel_b_ckpt),
            "panel_c": str(panel_c_ckpt),
            "panel_d": str(panel_d_ckpt),
            "panel_a_score": float(meta.get("seed_score", float("nan"))),
        }
        selection_path = out_dir / "rnn_publication_summary_best_panels.json"
        selection_path.write_text(json.dumps(selection, indent=2))
        if args.out_path:
            out_path = Path(args.out_path)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(best_path, out_path)
        return

    if args.best_across_seeds:
        min_ratio = args.min_ratio if args.min_ratio and args.min_ratio > 0 else None
        ckpt_path, meta = select_best_checkpoint(
            checkpoints,
            gridness_threshold=args.gridness_threshold,
            min_shift_cm=args.min_shift_cm,
            min_ratio=min_ratio,
        )
        print(
            f"[best seed] {seed_label(ckpt_path)} | "
            f"score={meta.get('seed_score', float('nan')):.3f} | "
            f"predictive_units={meta.get('predictive_units', 0)}"
        )
        model_name = model_name_from_checkpoint(ckpt_path)
        seed_name = seed_label(ckpt_path).replace(" ", "_")
        out_dir = analysis_summary_dir(model_name, "publication_summary")
        best_path = out_dir / f"rnn_publication_summary_best_{seed_name}.png"
        render_summary(ckpt_path, args, best_path)
        if args.out_path:
            out_path = Path(args.out_path)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(best_path, out_path)
        else:
            summary_path = out_dir / "rnn_publication_summary.png"
            shutil.copyfile(best_path, summary_path)
        return

    if not args.per_seed:
        if not args.checkpoint_path:
            raise SystemExit("--checkpoint_path is required unless --best_across_seeds or --per_seed is set.")
        ckpt_path = Path(args.checkpoint_path)
        if args.out_path:
            out_path = Path(args.out_path)
        else:
            model_name = model_name_from_checkpoint(ckpt_path)
            seed_name = seed_label(ckpt_path).replace(" ", "_")
            out_dir = analysis_summary_dir(model_name, "publication_summary")
            out_path = out_dir / f"rnn_publication_summary_{seed_name}.png"
        render_summary(ckpt_path, args, out_path)


if __name__ == "__main__":
    main()
