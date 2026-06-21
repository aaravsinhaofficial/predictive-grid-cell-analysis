#!/usr/bin/env python3
"""Reproduce the RNN predictive-grid result suite with true spatial shifts.

This is an orchestration script: it calls the existing analysis scripts with a
shared spatial-shift configuration and a shared analysis subdirectory.  The
classification axis is centimeters along the trajectory, not timesteps later
converted to centimeters.

Default workflow:
  1. multi_seed_predictive_analysis.py with --shift_mode space
  2. cell_class_summary_across_seeds.py from the same spatial gridness files
  3. predictive_shift_breadth_across_seeds.py from the same spatial files
  4. predictive_retrospective_ablation.py from the same spatial files
  5. band_cell_analysis.py from the same spatial files
  6. toroidal_structure_analysis.py for all checkpoints by default
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Sequence

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.append(str(Path(__file__).resolve().parent))

from path_utils import analysis_dir_for_checkpoint, analysis_summary_dir, model_name_from_checkpoint


def default_single_agent_checkpoints() -> List[str]:
    base = _REPO_ROOT / "Models" / "Single agent path integration"
    suffix = (
        "steps_20_batch_200_RNN_4096_relu_rf_012_DoG_True_"
        "periodic_False_lr_00001_weight_decay_1e-06/final_model.pth"
    )
    paths = []
    for seed in range(5):
        ckpt = base / f"Seed {seed} weight decay 1e-06" / suffix
        if ckpt.exists():
            paths.append(str(ckpt))
    if paths:
        return paths

    candidates = sorted((_REPO_ROOT / "Models").glob("**/*.pth"))
    candidates.extend(sorted((_REPO_ROOT / "extras").glob("**/most_recent_model.pth")))
    candidates.extend(sorted((_REPO_ROOT / "extras").glob("**/final_model.pth")))
    preferred = [p for p in candidates if _is_standard_full_rank_checkpoint(p)]
    return [str(p) for p in preferred]


def _is_standard_full_rank_checkpoint(path: Path) -> bool:
    try:
        import torch

        raw = torch.load(path, map_location="cpu")
    except Exception:
        return False
    if isinstance(raw, dict) and "model_state_dict" in raw:
        state = raw["model_state_dict"]
    elif isinstance(raw, dict) and "state_dict" in raw:
        state = raw["state_dict"]
    elif isinstance(raw, dict):
        state = raw
    else:
        return False
    required = {"encoder.weight", "decoder.weight", "RNN.weight_ih_l0"}
    return required.issubset(set(state.keys()))


def safe_label(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(text)).strip("_") or "spatial_shift"


def script(name: str) -> str:
    return str(Path(__file__).resolve().parent / name)


def spatial_gridness_path(checkpoint: str, analysis_subdir: str) -> Path:
    return analysis_dir_for_checkpoint(Path(checkpoint)) / str(analysis_subdir) / "gridness_data.npz"


def spatial_output_dir(checkpoint: str, analysis_subdir: str, leaf: str) -> Path:
    return analysis_dir_for_checkpoint(Path(checkpoint)) / str(analysis_subdir) / leaf


def append_common_model_args(cmd: List[str], args: argparse.Namespace) -> None:
    cmd.extend(
        [
            "--batch_size",
            str(args.batch_size),
            "--sequence_length",
            str(args.sequence_length),
            "--place_cell_rf",
            str(args.place_cell_rf),
            "--surround_scale",
            str(args.surround_scale),
            "--activation",
            str(args.activation),
            "--weight_decay",
            str(args.weight_decay),
            "--box_width",
            str(args.box_width),
            "--box_height",
            str(args.box_height),
            "--learning_rate",
            str(args.learning_rate),
            "--Ng_use",
            str(args.Ng_use),
            "--gridness_threshold",
            str(args.gridness_threshold),
            "--min_shift_cm",
            str(args.min_shift_cm),
        ]
    )
    if args.device:
        cmd.extend(["--device", str(args.device)])


def run_command(label: str, cmd: Sequence[str], dry_run: bool) -> dict:
    printable = " ".join(subprocess.list2cmdline([str(x)]) for x in cmd)
    print(f"\n[{label}] {printable}", flush=True)
    if dry_run:
        return {"label": label, "cmd": list(cmd), "returncode": None, "elapsed_s": 0.0}
    start = time.time()
    completed = subprocess.run(list(cmd), cwd=str(_REPO_ROOT), check=True)
    elapsed = time.time() - start
    return {"label": label, "cmd": list(cmd), "returncode": completed.returncode, "elapsed_s": elapsed}


def resolve_checkpoints(args: argparse.Namespace) -> List[str]:
    if args.checkpoint_paths:
        return [str(Path(p).expanduser()) for p in args.checkpoint_paths]
    paths = default_single_agent_checkpoints()
    if not paths:
        raise FileNotFoundError(
            "No default single-agent checkpoints were found. Pass --checkpoint_paths explicitly."
        )
    return paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint_paths", nargs="+", default=None)
    parser.add_argument(
        "--analysis_subdir",
        default="spatial_shift",
        help="Subdirectory under each checkpoint analysis folder for spatial outputs.",
    )
    parser.add_argument("--space_projection", default="path", choices=["path", "heading"])
    parser.add_argument("--max_shift_cm", type=float, default=20.0)
    parser.add_argument("--shift_step_cm", type=float, default=1.0)
    parser.add_argument("--min_shift_cm", type=float, default=5.0)
    parser.add_argument("--gridness_threshold", type=float, default=0.2)
    parser.add_argument("--low_grid_threshold", type=float, default=0.2)
    parser.add_argument("--n_batches", type=int, default=25)
    parser.add_argument("--batch_size", type=int, default=100)
    parser.add_argument("--sequence_length", type=int, default=20)
    parser.add_argument("--Ng_use", type=int, default=512)
    parser.add_argument("--res", type=int, default=20)
    parser.add_argument("--place_cell_rf", type=float, default=0.12)
    parser.add_argument("--surround_scale", type=float, default=2.0)
    parser.add_argument("--activation", default="relu")
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--box_width", type=float, default=2.2)
    parser.add_argument("--box_height", type=float, default=2.2)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--device", default=None)
    parser.add_argument("--ablation_batches", type=int, default=8)
    parser.add_argument("--ablation_random_trials", type=int, default=5)
    parser.add_argument("--full_ablation_random_trials", type=int, default=3)
    parser.add_argument("--band_res", type=int, default=40)
    parser.add_argument("--band_n_avg", type=int, default=None)
    parser.add_argument("--band_percentile", type=float, default=90.0)
    parser.add_argument("--band_threshold", type=float, default=None)
    parser.add_argument("--skip_direction_validation", action="store_true")
    parser.add_argument("--direction_validation_batches", type=int, default=8)
    parser.add_argument("--direction_validation_batch_size", type=int, default=128)
    parser.add_argument("--direction_validation_sequence_length", type=int, default=None)
    parser.add_argument("--direction_validation_future_horizon", type=int, default=8)
    parser.add_argument("--direction_validation_min_angle_deg", type=float, default=30.0)
    parser.add_argument("--direction_validation_max_angle_deg", type=float, default=150.0)
    parser.add_argument("--direction_validation_random_repeats", type=int, default=20)
    parser.add_argument("--skip_same_bin_validation", action="store_true")
    parser.add_argument("--same_bin_res", type=int, default=44)
    parser.add_argument("--same_bin_pairs_per_bin", type=int, default=20)
    parser.add_argument("--skip_crossing_step_sweep", action="store_true")
    parser.add_argument("--crossing_steps", default="8 10 12 14 15 16 18 20 22")
    parser.add_argument(
        "--future_split_checkpoint_paths",
        nargs="+",
        default=None,
        help="Optional future_split_full.pth checkpoints for classify/decode/crossing/intervene spatial reruns.",
    )
    parser.add_argument("--future_split_task", default=None, choices=["binary_fork", "multi_route_graph"])
    parser.add_argument("--future_split_batches", type=int, default=8)
    parser.add_argument("--future_split_batch_size", type=int, default=80)
    parser.add_argument("--future_split_sequence_length", type=int, default=None)
    parser.add_argument("--future_split_horizons", default="1 2 4 6 8 10 12")
    parser.add_argument("--future_split_main_horizon", type=int, default=8)
    parser.add_argument("--future_split_history_steps", type=int, default=4)
    parser.add_argument("--future_split_decoder_train_batches", type=int, default=4)
    parser.add_argument("--future_split_intervention_groups", default="predictive standard_grid band random")
    parser.add_argument("--skip_future_split_decode", action="store_true")
    parser.add_argument("--skip_future_split_intervene", action="store_true")
    parser.add_argument(
        "--torus_checkpoints",
        default="all",
        choices=["none", "first", "all"],
        help="Run toroidal_structure_analysis.py for no checkpoints, the first checkpoint, or all checkpoints.",
    )
    parser.add_argument("--torus_batches", type=int, default=12)
    parser.add_argument("--torus_res", type=int, default=40)
    parser.add_argument("--torus_gridness_threshold", type=float, default=0.5)
    parser.add_argument("--skip_multiseed", action="store_true")
    parser.add_argument("--skip_cell_summary", action="store_true")
    parser.add_argument("--skip_breadth", action="store_true")
    parser.add_argument("--skip_full_ablation", action="store_true")
    parser.add_argument("--skip_band", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoints = resolve_checkpoints(args)
    label = safe_label(args.analysis_subdir)
    model_name = model_name_from_checkpoint(Path(checkpoints[0]))
    summary_dir = analysis_summary_dir(model_name, f"spatial_shift_reproduction_{label}")

    runs = []
    python = sys.executable

    if not args.skip_multiseed:
        cmd = [
            python,
            script("multi_seed_predictive_analysis.py"),
            "--checkpoint_paths",
            *checkpoints,
            "--shift_mode",
            "space",
            "--space_projection",
            str(args.space_projection),
            "--max_shift_cm",
            str(args.max_shift_cm),
            "--shift_step_cm",
            str(args.shift_step_cm),
            "--output_subdir",
            str(args.analysis_subdir),
            "--n_batches",
            str(args.n_batches),
            "--res",
            str(args.res),
            "--low_grid_threshold",
            str(args.low_grid_threshold),
            "--ablation_batches",
            str(args.ablation_batches),
            "--ablation_random_trials",
            str(args.ablation_random_trials),
        ]
        append_common_model_args(cmd, args)
        runs.append(run_command("spatial_gridness_and_per_seed_ablation", cmd, args.dry_run))

    if not args.skip_cell_summary:
        cmd = [
            python,
            script("cell_class_summary_across_seeds.py"),
            "--checkpoint_paths",
            *checkpoints,
            "--analysis_subdir",
            str(args.analysis_subdir),
            "--Ng_use",
            str(args.Ng_use),
            "--gridness_threshold",
            str(args.gridness_threshold),
            "--min_shift_cm",
            str(args.min_shift_cm),
        ]
        runs.append(run_command("cross_seed_cell_summary", cmd, args.dry_run))

    if not args.skip_breadth:
        cmd = [
            python,
            script("predictive_shift_breadth_across_seeds.py"),
            "--analysis-root",
            str(_REPO_ROOT / "analysis_outputs"),
            "--model-subdir",
            str(model_name),
            "--analysis-subdir",
            str(args.analysis_subdir),
        ]
        runs.append(run_command("predictive_shift_breadth", cmd, args.dry_run))

    if not args.skip_full_ablation:
        cmd = [
            python,
            script("predictive_retrospective_ablation.py"),
            "--checkpoint_paths",
            *checkpoints,
            "--analysis_subdir",
            str(args.analysis_subdir),
            "--ablation_batches",
            str(args.ablation_batches),
            "--random_trials",
            str(args.full_ablation_random_trials),
        ]
        append_common_model_args(cmd, args)
        runs.append(run_command("full_class_ablation", cmd, args.dry_run))

    if not args.skip_band:
        cmd = [
            python,
            script("band_cell_analysis.py"),
            "--checkpoint_paths",
            *checkpoints,
            "--analysis_subdir",
            str(args.analysis_subdir),
            "--res",
            str(args.band_res),
            "--band_percentile",
            str(args.band_percentile),
        ]
        if args.band_n_avg is not None:
            cmd.extend(["--n_avg", str(args.band_n_avg)])
        if args.band_threshold is not None:
            cmd.extend(["--band_threshold", str(args.band_threshold)])
        append_common_model_args(cmd, args)
        runs.append(run_command("band_cell_analysis", cmd, args.dry_run))

    if not args.skip_direction_validation:
        for idx, ckpt in enumerate(checkpoints):
            gridness_path = spatial_gridness_path(ckpt, args.analysis_subdir)
            cmd = [
                python,
                script("future_split_experiment.py"),
                "crossing",
                "--checkpoint_path",
                ckpt,
                "--gridness_path",
                str(gridness_path),
                "--output_dir",
                str(spatial_output_dir(ckpt, args.analysis_subdir, "matched_direction_x_crossing")),
                "--batch_size",
                str(args.direction_validation_batch_size),
                "--n_batches",
                str(args.direction_validation_batches),
                "--future_horizon",
                str(args.direction_validation_future_horizon),
                "--min_angle_deg",
                str(args.direction_validation_min_angle_deg),
                "--max_angle_deg",
                str(args.direction_validation_max_angle_deg),
                "--matched_random_repeats",
                str(args.direction_validation_random_repeats),
            ]
            if args.direction_validation_sequence_length is not None:
                cmd.extend(["--sequence_length", str(args.direction_validation_sequence_length)])
            if args.device:
                cmd.extend(["--device", str(args.device)])
            runs.append(run_command(f"matched_direction_x_crossing_{idx}", cmd, args.dry_run))

            if not args.skip_same_bin_validation:
                cmd = [
                    python,
                    script("future_split_experiment.py"),
                    "crossing",
                    "--checkpoint_path",
                    ckpt,
                    "--gridness_path",
                    str(gridness_path),
                    "--output_dir",
                    str(spatial_output_dir(ckpt, args.analysis_subdir, "matched_direction_same_bin")),
                    "--pairing_mode",
                    "same_bin",
                    "--batch_size",
                    str(args.direction_validation_batch_size),
                    "--n_batches",
                    str(args.direction_validation_batches),
                    "--future_horizon",
                    str(args.direction_validation_future_horizon),
                    "--min_angle_deg",
                    str(args.direction_validation_min_angle_deg),
                    "--max_angle_deg",
                    str(args.direction_validation_max_angle_deg),
                    "--matched_random_repeats",
                    str(args.direction_validation_random_repeats),
                    "--same_bin_res",
                    str(args.same_bin_res),
                    "--same_bin_pairs_per_bin",
                    str(args.same_bin_pairs_per_bin),
                ]
                if args.direction_validation_sequence_length is not None:
                    cmd.extend(["--sequence_length", str(args.direction_validation_sequence_length)])
                if args.device:
                    cmd.extend(["--device", str(args.device)])
                runs.append(run_command(f"matched_direction_same_bin_{idx}", cmd, args.dry_run))

            if not args.skip_crossing_step_sweep:
                cmd = [
                    python,
                    script("future_split_experiment.py"),
                    "crossing_step_sweep",
                    "--checkpoint_path",
                    ckpt,
                    "--gridness_path",
                    str(gridness_path),
                    "--output_dir",
                    str(spatial_output_dir(ckpt, args.analysis_subdir, "matched_direction_step_sweep")),
                    "--batch_size",
                    str(args.direction_validation_batch_size),
                    "--n_batches",
                    str(args.direction_validation_batches),
                    "--future_horizon",
                    str(args.direction_validation_future_horizon),
                    "--crossing_steps",
                    str(args.crossing_steps),
                    "--min_angle_deg",
                    str(args.direction_validation_min_angle_deg),
                    "--max_angle_deg",
                    str(args.direction_validation_max_angle_deg),
                    "--matched_random_repeats",
                    str(args.direction_validation_random_repeats),
                ]
                if args.direction_validation_sequence_length is not None:
                    cmd.extend(["--sequence_length", str(args.direction_validation_sequence_length)])
                if args.device:
                    cmd.extend(["--device", str(args.device)])
                runs.append(run_command(f"matched_direction_step_sweep_{idx}", cmd, args.dry_run))

    if args.future_split_checkpoint_paths:
        for idx, ckpt in enumerate(args.future_split_checkpoint_paths):
            fs_root = spatial_output_dir(ckpt, args.analysis_subdir, "future_split_spatial_suite")
            fs_classify_dir = fs_root / "classify"
            fs_gridness = fs_classify_dir / "gridness_data.npz"
            cmd = [
                python,
                script("future_split_experiment.py"),
                "classify",
                "--checkpoint_path",
                ckpt,
                "--output_dir",
                str(fs_classify_dir),
                "--batch_size",
                str(args.future_split_batch_size),
                "--n_batches",
                str(args.future_split_batches),
                "--Ng_use",
                str(args.Ng_use),
                "--res",
                str(args.res),
                "--shift_mode",
                "space",
                "--space_projection",
                str(args.space_projection),
                "--max_shift_cm",
                str(args.max_shift_cm),
                "--shift_step_cm",
                str(args.shift_step_cm),
                "--min_shift_cm",
                str(args.min_shift_cm),
                "--gridness_threshold",
                str(args.gridness_threshold),
            ]
            if args.future_split_sequence_length is not None:
                cmd.extend(["--sequence_length", str(args.future_split_sequence_length)])
            if args.device:
                cmd.extend(["--device", str(args.device)])
            runs.append(run_command(f"future_split_classify_spatial_{idx}", cmd, args.dry_run))

            if not args.skip_future_split_decode:
                cmd = [
                    python,
                    script("future_split_experiment.py"),
                    "decode",
                    "--checkpoint_path",
                    ckpt,
                    "--gridness_path",
                    str(fs_gridness),
                    "--output_dir",
                    str(fs_root / "decode"),
                    "--batch_size",
                    str(args.future_split_batch_size),
                    "--n_batches",
                    str(args.future_split_batches),
                    "--horizons",
                    str(args.future_split_horizons),
                    "--main_horizon",
                    str(args.future_split_main_horizon),
                    "--history_steps",
                    str(args.future_split_history_steps),
                    "--route_controls",
                    "--horizon_specificity",
                ]
                if args.future_split_task:
                    cmd.extend(["--task", str(args.future_split_task)])
                if args.device:
                    cmd.extend(["--device", str(args.device)])
                runs.append(run_command(f"future_split_decode_spatial_classes_{idx}", cmd, args.dry_run))

            cmd = [
                python,
                script("future_split_experiment.py"),
                "crossing",
                "--checkpoint_path",
                ckpt,
                "--gridness_path",
                str(fs_gridness),
                "--output_dir",
                str(fs_root / "crossing"),
                "--batch_size",
                str(args.direction_validation_batch_size),
                "--n_batches",
                str(args.direction_validation_batches),
                "--future_horizon",
                str(args.direction_validation_future_horizon),
                "--min_angle_deg",
                str(args.direction_validation_min_angle_deg),
                "--max_angle_deg",
                str(args.direction_validation_max_angle_deg),
                "--matched_random_repeats",
                str(args.direction_validation_random_repeats),
            ]
            if args.device:
                cmd.extend(["--device", str(args.device)])
            runs.append(run_command(f"future_split_crossing_spatial_classes_{idx}", cmd, args.dry_run))

            if not args.skip_same_bin_validation:
                cmd = [
                    python,
                    script("future_split_experiment.py"),
                    "crossing",
                    "--checkpoint_path",
                    ckpt,
                    "--gridness_path",
                    str(fs_gridness),
                    "--output_dir",
                    str(fs_root / "same_bin"),
                    "--pairing_mode",
                    "same_bin",
                    "--batch_size",
                    str(args.direction_validation_batch_size),
                    "--n_batches",
                    str(args.direction_validation_batches),
                    "--future_horizon",
                    str(args.direction_validation_future_horizon),
                    "--min_angle_deg",
                    str(args.direction_validation_min_angle_deg),
                    "--max_angle_deg",
                    str(args.direction_validation_max_angle_deg),
                    "--matched_random_repeats",
                    str(args.direction_validation_random_repeats),
                    "--same_bin_res",
                    str(args.same_bin_res),
                    "--same_bin_pairs_per_bin",
                    str(args.same_bin_pairs_per_bin),
                ]
                if args.device:
                    cmd.extend(["--device", str(args.device)])
                runs.append(run_command(f"future_split_same_bin_spatial_classes_{idx}", cmd, args.dry_run))

            if not args.skip_future_split_intervene:
                cmd = [
                    python,
                    script("future_split_experiment.py"),
                    "intervene",
                    "--checkpoint_path",
                    ckpt,
                    "--gridness_path",
                    str(fs_gridness),
                    "--output_dir",
                    str(fs_root / "intervene"),
                    "--batch_size",
                    str(args.future_split_batch_size),
                    "--main_horizon",
                    str(args.future_split_main_horizon),
                    "--horizons",
                    str(args.future_split_horizons),
                    "--intervention_groups",
                    str(args.future_split_intervention_groups),
                    "--decoder_train_batches",
                    str(args.future_split_decoder_train_batches),
                ]
                if args.future_split_task:
                    cmd.extend(["--task", str(args.future_split_task)])
                if args.device:
                    cmd.extend(["--device", str(args.device)])
                runs.append(run_command(f"future_split_intervene_spatial_classes_{idx}", cmd, args.dry_run))

    if args.torus_checkpoints != "none":
        torus_paths = checkpoints[:1] if args.torus_checkpoints == "first" else checkpoints
        for idx, ckpt in enumerate(torus_paths):
            cmd = [
                python,
                script("toroidal_structure_analysis.py"),
                "--checkpoint_path",
                ckpt,
                "--analysis_subdir",
                str(args.analysis_subdir),
                "--batch_size",
                str(args.batch_size),
                "--sequence_length",
                str(args.sequence_length),
                "--Ng_use",
                str(args.Ng_use),
                "--gridness_threshold",
                str(args.torus_gridness_threshold),
                "--min_shift_cm",
                str(args.min_shift_cm),
                "--res",
                str(args.torus_res),
                "--n_batches",
                str(args.torus_batches),
                "--place_cell_rf",
                str(args.place_cell_rf),
                "--surround_scale",
                str(args.surround_scale),
                "--activation",
                str(args.activation),
                "--weight_decay",
                str(args.weight_decay),
                "--box_width",
                str(args.box_width),
                "--box_height",
                str(args.box_height),
                "--learning_rate",
                str(args.learning_rate),
            ]
            if args.device:
                cmd.extend(["--device", str(args.device)])
            runs.append(run_command(f"torus_spatial_classes_{idx}", cmd, args.dry_run))

    summary = {
        "analysis_axis": "space",
        "analysis_subdir": str(args.analysis_subdir),
        "space_projection": str(args.space_projection),
        "max_shift_cm": float(args.max_shift_cm),
        "shift_step_cm": float(args.shift_step_cm),
        "checkpoints": checkpoints,
        "runs": runs,
    }
    if args.dry_run:
        print("\n[spatial_shift_reproduction] dry run complete; no summary file written.", flush=True)
        return
    summary_dir.mkdir(parents=True, exist_ok=True)
    summary_path = summary_dir / "spatial_shift_reproduction_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[spatial_shift_reproduction] wrote {summary_path}", flush=True)


if __name__ == "__main__":
    main()
