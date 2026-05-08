#!/usr/bin/env python3
"""Rerender selected shift-mode exemplar figures with SACs underneath.

This script reuses a completed `shift_mode_consistency_analysis.py` output
directory. It does not rescore all units across all shifts. Instead, it:

1. Loads the selected exemplar units from `consistency_summary.json`
2. Regenerates the same sampled trajectories/activations for only those units
3. Recomputes the zero-shift, best-temporal, and best-spatial ratemaps
4. Plots the ratemaps with their SACs underneath

Outputs:
  - rnn_exemplar_ratemaps_shift_modes_with_sac.png
  - rnn_exemplar_pairs_with_sac/
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import gridspec

from model import RNN
from multi_seed_predictive_analysis import build_options, infer_dims_from_state
from place_cells import PlaceCells
from predictive_retrospective_ablation import extract_state
from shift_mode_consistency_analysis import build_grid_scorer, format_duration, log
from trajectory_generator import TrajectoryGenerator
from visualize import collect_sequences


def load_completed_run(output_dir: Path) -> Tuple[dict, dict, np.lib.npyio.NpzFile]:
    summary_path = output_dir / "consistency_summary.json"
    scores_path = output_dir / "shared_shift_scores.npz"
    summary = json.loads(summary_path.read_text())
    scores = np.load(scores_path)
    return summary, summary["settings"], scores


def build_args_from_settings(settings: dict, checkpoint_path: str, device: str) -> SimpleNamespace:
    return SimpleNamespace(
        batch_size=settings["batch_size"],
        sequence_length=settings["sequence_length"],
        place_cell_rf=0.12,
        surround_scale=2.0,
        activation="relu",
        weight_decay=1e-4,
        box_width=2.2,
        box_height=2.2,
        learning_rate=1e-4,
        traj_speed_scale=1.0,
        traj_speed_max=None,
        traj_velocity_smoothing=0.0,
        traj_turn_sigma_scale=1.0,
        traj_border_region=0.03,
        traj_wall_slowdown=0.25,
        traj_wall_turn_scale=1.0,
        trajectory_style=settings["trajectory_style"],
        trajectory_fixed_speed=settings["trajectory_fixed_speed"],
        checkpoint_path=checkpoint_path,
        device=device,
    )


def gather_selected_units(summary: dict) -> Tuple[List[Tuple[str, dict]], np.ndarray]:
    rows: List[Tuple[str, dict]] = []
    unit_ids: List[int] = []
    for class_name in ("predictive", "retrospective"):
        for item in summary["selected_rnn_units"].get(class_name, []):
            rows.append((class_name, item))
            unit_ids.append(int(item["unit_index"]))
    unique_units = np.asarray(sorted(set(unit_ids)), dtype=int)
    return rows, unique_units


def rm_and_sac_for_unit(
    scorer,
    xs: np.ndarray,
    ys: np.ndarray,
    activations: np.ndarray,
    unit_idx: int,
    shift_value: float,
    shift_mode: str,
    space_projection: str,
    periodic: bool,
):
    s60, _, rm, sac = scorer.get_scores_with_shift(
        xs,
        ys,
        activations[:, :, unit_idx],
        shift_value,
        statistic="mean",
        return_maps=True,
        shift_mode=shift_mode,
        periodic=periodic,
        space_projection=space_projection,
    )
    return float(s60) if s60 is not None and np.isfinite(s60) else np.nan, rm, sac


def draw_matrix(ax, mat, title: str, vmin: float | None = None, vmax: float | None = None, cmap: str = "jet") -> None:
    if mat is None or not np.isfinite(np.asarray(mat, dtype=float)).any():
        ax.set_facecolor("#f0f0f0")
        ax.text(0.5, 0.5, "No valid\nsamples", ha="center", va="center", fontsize=9, color="#555555")
    else:
        ax.imshow(mat, cmap=cmap, interpolation="nearest", vmin=vmin, vmax=vmax)
    ax.set_title(title, fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])


def draw_sac(ax, sac, title: str) -> None:
    if sac is None or not np.isfinite(np.asarray(sac, dtype=float)).any():
        ax.set_facecolor("#f0f0f0")
        ax.text(0.5, 0.5, "No valid\nSAC", ha="center", va="center", fontsize=9, color="#555555")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(title, fontsize=10)
        return
    sac_plot = np.asarray(sac, dtype=float).copy()
    if sac_plot.ndim == 2:
        center = sac_plot.shape[0] // 2
        sac_plot[center, center] = np.nan
    finite = np.isfinite(sac_plot)
    vmax = float(np.nanmax(np.abs(sac_plot[finite]))) if np.any(finite) else 1.0
    if vmax <= 0:
        vmax = 1.0
    ax.imshow(sac_plot, cmap="coolwarm", interpolation="nearest", vmin=-vmax, vmax=vmax)
    ax.set_title(title, fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])


def draw_exemplar_with_sac(
    fig: plt.Figure,
    gs: gridspec.GridSpec,
    *,
    class_name: str,
    unit_index: int,
    temporal_cm: np.ndarray,
    spatial_cm: np.ndarray,
    temporal_profile: np.ndarray,
    spatial_profile: np.ndarray,
    temporal_best_cm: float,
    temporal_best_score: float,
    spatial_best_cm: float,
    spatial_best_score: float,
    zero_rm,
    zero_sac,
    temporal_rm,
    temporal_sac,
    spatial_rm,
    spatial_sac,
) -> None:
    ax_curve = fig.add_subplot(gs[:, 0])
    ax_zero = fig.add_subplot(gs[0, 1])
    ax_temp = fig.add_subplot(gs[0, 2])
    ax_space = fig.add_subplot(gs[0, 3])
    ax_zero_sac = fig.add_subplot(gs[1, 1])
    ax_temp_sac = fig.add_subplot(gs[1, 2])
    ax_space_sac = fig.add_subplot(gs[1, 3])

    rm_stack = [
        np.asarray(rm, dtype=float)
        for rm in (zero_rm, temporal_rm, spatial_rm)
        if rm is not None and np.isfinite(np.asarray(rm, dtype=float)).any()
    ]
    if rm_stack:
        vmin = float(min(np.nanmin(rm) for rm in rm_stack))
        vmax = float(max(np.nanmax(rm) for rm in rm_stack))
    else:
        vmin, vmax = 0.0, 1.0

    ax_curve.plot(temporal_cm, temporal_profile, "o-", color="#4c78a8", lw=1.8, ms=4, label="Temporal")
    ax_curve.plot(spatial_cm, spatial_profile, "-", color="#f58518", lw=2.0, label="Spatial")
    if np.isfinite(temporal_best_cm) and np.isfinite(temporal_best_score):
        ax_curve.scatter([temporal_best_cm], [temporal_best_score], color="#4c78a8", s=48, zorder=4)
    if np.isfinite(spatial_best_cm) and np.isfinite(spatial_best_score):
        ax_curve.scatter([spatial_best_cm], [spatial_best_score], color="#f58518", s=48, zorder=4)
    ax_curve.axhline(0.0, color="black", lw=0.8)
    ax_curve.axvline(0.0, color="black", lw=0.8, ls="--")
    ax_curve.set_xlabel("Shift (cm)")
    ax_curve.set_ylabel("Gridness (60°)")
    ax_curve.grid(alpha=0.25)
    ax_curve.set_title(
        f"{class_name.title()} unit {unit_index}\n"
        f"T: {temporal_best_cm:+.1f} cm ({temporal_best_score:.2f}) | "
        f"S: {spatial_best_cm:+.1f} cm ({spatial_best_score:.2f})",
        fontsize=10,
    )
    ax_curve.legend(frameon=False, fontsize=8, loc="upper left")

    draw_matrix(ax_zero, zero_rm, "Zero shift", vmin=vmin, vmax=vmax)
    draw_matrix(ax_temp, temporal_rm, f"Temporal best\n{temporal_best_cm:+.1f} cm", vmin=vmin, vmax=vmax)
    draw_matrix(ax_space, spatial_rm, f"Spatial best\n{spatial_best_cm:+.1f} cm", vmin=vmin, vmax=vmax)
    draw_sac(ax_zero_sac, zero_sac, "Zero SAC")
    draw_sac(ax_temp_sac, temporal_sac, "Temporal SAC")
    draw_sac(ax_space_sac, spatial_sac, "Spatial SAC")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output_dir", required=True, help="Existing shift_mode_consistency output directory.")
    parser.add_argument("--device", default=None)
    parser.add_argument("--combined_name", default="rnn_exemplar_ratemaps_shift_modes_with_sac.png")
    parser.add_argument("--individual_dirname", default="rnn_exemplar_pairs_with_sac")
    args = parser.parse_args()

    run_start = torch.cuda.Event(enable_timing=False) if False else None  # quiet linter about torch import
    del run_start
    output_dir = Path(args.output_dir)
    summary, settings, scores = load_completed_run(output_dir)
    log(f"Loading completed run from {output_dir}")

    rows, selected_units = gather_selected_units(summary)
    if not rows:
        raise ValueError("No selected exemplar units were found in consistency_summary.json.")
    log(f"Rerendering {len(rows)} exemplar panels across {len(selected_units)} unique units.")

    ckpt_path = Path(summary["checkpoint_path"])
    raw = torch.load(ckpt_path, map_location="cpu")
    state = extract_state(raw)
    dims = infer_dims_from_state(state)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    build_args = build_args_from_settings(settings, str(ckpt_path), device)
    options = build_options(build_args, dims, device, str(ckpt_path))
    options.trajectory_style = settings["trajectory_style"]
    options.trajectory_fixed_speed = settings["trajectory_fixed_speed"]

    place_cells = PlaceCells(options)
    np.random.seed(int(settings["seed"]))
    torch.manual_seed(int(settings["seed"]))
    model = RNN(options, place_cells).to(options.device)
    model.load_state_dict(state)
    model.eval()
    traj_gen = TrajectoryGenerator(options, place_cells)
    scorer = build_grid_scorer(20, options)
    periodic = bool(getattr(options, "periodic", False))

    log("Collecting shared trajectories and activations for selected exemplar units only.")
    xs, ys, activations = collect_sequences(
        model,
        traj_gen,
        options,
        n_batches=int(settings["n_batches"]),
        Ng=len(selected_units),
        idxs=selected_units,
        progress=True,
        progress_label="rerender_sequences",
    )
    local_idx = {int(unit): idx for idx, unit in enumerate(selected_units.tolist())}
    temporal_cm = np.asarray(scores["temporal_cm"], dtype=float)
    spatial_cm = np.asarray(scores["spatial_cm"], dtype=float)
    temporal_scores = np.asarray(scores["temporal_scores"], dtype=float)
    spatial_scores = np.asarray(scores["spatial_scores"], dtype=float)

    combined_path = output_dir / args.combined_name
    individual_dir = output_dir / args.individual_dirname
    individual_dir.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(16.0, 5.2 * len(rows)))
    outer = gridspec.GridSpec(len(rows), 1, figure=fig, hspace=0.45)

    for row_idx, (class_name, item) in enumerate(rows):
        unit_index = int(item["unit_index"])
        log(f"Rendering {row_idx + 1}/{len(rows)}: {class_name} unit {unit_index}")
        i = local_idx[unit_index]
        temporal_best_lag = int(item["temporal_best_lag"])
        spatial_best_cm = float(item["spatial_best_cm"])
        zero_score, zero_rm, zero_sac = rm_and_sac_for_unit(
            scorer, xs, ys, activations, i, 0, "time", settings["space_projection"], periodic
        )
        temp_score, temp_rm, temp_sac = rm_and_sac_for_unit(
            scorer, xs, ys, activations, i, temporal_best_lag, "time", settings["space_projection"], periodic
        )
        space_score, space_rm, space_sac = rm_and_sac_for_unit(
            scorer, xs, ys, activations, i, spatial_best_cm, "space", settings["space_projection"], periodic
        )
        del zero_score, temp_score, space_score

        row_gs = gridspec.GridSpecFromSubplotSpec(
            2,
            4,
            subplot_spec=outer[row_idx],
            width_ratios=[1.8, 1.0, 1.0, 1.0],
            height_ratios=[1.0, 1.0],
            wspace=0.18,
            hspace=0.08,
        )
        draw_exemplar_with_sac(
            fig,
            row_gs,
            class_name=class_name,
            unit_index=unit_index,
            temporal_cm=temporal_cm,
            spatial_cm=spatial_cm,
            temporal_profile=temporal_scores[:, unit_index],
            spatial_profile=spatial_scores[:, unit_index],
            temporal_best_cm=float(item["temporal_best_cm"]),
            temporal_best_score=float(item["temporal_best_score"]),
            spatial_best_cm=float(item["spatial_best_cm"]),
            spatial_best_score=float(item["spatial_best_score"]),
            zero_rm=zero_rm,
            zero_sac=zero_sac,
            temporal_rm=temp_rm,
            temporal_sac=temp_sac,
            spatial_rm=space_rm,
            spatial_sac=space_sac,
        )

        fig_single = plt.figure(figsize=(16.0, 5.0))
        gs_single = gridspec.GridSpec(
            2,
            4,
            figure=fig_single,
            width_ratios=[1.8, 1.0, 1.0, 1.0],
            height_ratios=[1.0, 1.0],
            wspace=0.18,
            hspace=0.08,
        )
        draw_exemplar_with_sac(
            fig_single,
            gs_single,
            class_name=class_name,
            unit_index=unit_index,
            temporal_cm=temporal_cm,
            spatial_cm=spatial_cm,
            temporal_profile=temporal_scores[:, unit_index],
            spatial_profile=spatial_scores[:, unit_index],
            temporal_best_cm=float(item["temporal_best_cm"]),
            temporal_best_score=float(item["temporal_best_score"]),
            spatial_best_cm=float(item["spatial_best_cm"]),
            spatial_best_score=float(item["spatial_best_score"]),
            zero_rm=zero_rm,
            zero_sac=zero_sac,
            temporal_rm=temp_rm,
            temporal_sac=temp_sac,
            spatial_rm=space_rm,
            spatial_sac=space_sac,
        )
        single_path = individual_dir / f"{row_idx + 1:02d}_{class_name}_unit_{unit_index}_with_sac.png"
        fig_single.tight_layout()
        fig_single.savefig(single_path, dpi=220, bbox_inches="tight")
        plt.close(fig_single)

    fig.suptitle("Same RNN units under temporal and spatial shifting, with SACs", fontsize=14, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.992])
    fig.savefig(combined_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    log(f"Saved combined figure to {combined_path}")
    log(f"Saved individual figures to {individual_dir}")


if __name__ == "__main__":
    main()
