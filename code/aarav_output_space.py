#!/usr/bin/env python3
"""Output-space validation: true vs decoded position under ablation, + decoded step sizes.

For each test trajectory we decode the network's believed position over time (decoder -> nearest
place cells) under three conditions and compare to the TRUE path:
  Intact, Predictive-ablated (all predictive units), Grid-ablated (all grid units).
We quantify the per-step distance of the true path and of each decoded path. Speaker-2 hypothesis:
  intact decoded steps ~ true; predictive-ablated decoded steps SMALL (stuck/clumped);
  grid-ablated decoded steps LARGE/erratic (wandering). We report what actually happens.

We also save a subset of (true, decoded) trajectories per seed for the side-by-side videos.

Count-matched references (Random(N), Structural(N)) are included in the stats so we can tell
whether any predictive effect is specific or just "many cells removed".

Usage:  python code/aarav_output_space.py --seeds 0 1 2 3 4 5 6 7 8 9 --device cuda
"""
from __future__ import annotations
import argparse, copy, json, os, sys
import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
_REPO = os.path.dirname(_HERE)

from aarav_crossing_population_correlation import load_model, load_classes
from aarav_activity_space_ablation import make_random_walk_inputs
from multi_seed_predictive_analysis import zero_unit_weights_in_place


def decode_path(model, units, inputs):
    """Return decoded position path [T,B,2] for the (optionally ablated) network."""
    if units is None or len(units) == 0:
        m = model
    else:
        m = copy.deepcopy(model); zero_unit_weights_in_place(m, list(units))
    with torch.no_grad():
        dec = m.place_cells.get_nearest_cell_pos(m.predict(inputs)).detach().cpu().numpy()
    if units is not None and len(units) > 0:
        del m
    return dec  # [T,B,2]


def steps(path):
    """Per-timestep step magnitudes, pooled: [(T-1)*B] in cm."""
    d = np.linalg.norm(np.diff(path, axis=0), axis=2).reshape(-1) * 100.0
    return d


def stepsum(d):
    return {"median_cm": float(np.median(d)), "mean_cm": float(np.mean(d)),
            "p10_cm": float(np.percentile(d, 10)), "p90_cm": float(np.percentile(d, 90)),
            "std_cm": float(np.std(d))}


def run_seed(seed, args, device):
    ckpt = os.path.join(_REPO, args.model_root, f"Seed {seed}", "most_recent_model.pth")
    gpath = os.path.join(_REPO, args.analysis_root, f"Seed {seed}", args.analysis_subdir, "gridness_data.npz")
    out_dir = os.path.join(_REPO, args.analysis_root, f"Seed {seed}", args.analysis_subdir, "aarav_output_space")
    os.makedirs(out_dir, exist_ok=True)
    model, place_cells, traj_gen, opt, Ng, Np = load_model(ckpt, device, args.seq_len)
    classes = load_classes(gpath, args.gridness_threshold, args.min_shift_cm)
    predictive = classes["predictive"]; grid_all = classes["grid_all"]
    structural = np.setdiff1d(grid_all, predictive); all_units = np.arange(Ng)
    N = int(predictive.size); rng = np.random.default_rng(7 + seed)

    inputs, pos = make_random_walk_inputs(traj_gen, args.test_trajectories, args.seq_len, 777 + seed)
    true = pos.detach().cpu().numpy()                      # [T,B,2]
    true_step = steps(true)

    # main conditions (natural full sets) + count-matched references
    conds = {
        "Intact": None,
        "Predictive-ablated": predictive,
        "Grid-ablated": grid_all,
        "Structural-ablated (N)": rng.choice(structural, min(N, structural.size), replace=False),
        "Random-ablated (N)": rng.choice(all_units, N, replace=False),
    }
    dec = {name: decode_path(model, u, inputs) for name, u in conds.items()}

    # decode error vs true, and step-size stats
    res = {"true_step": stepsum(true_step), "N_predictive": N, "n_grid": int(grid_all.size)}
    for name, d in dec.items():
        err = np.linalg.norm(d - true, axis=2)            # [T,B]
        res[name] = {"n_units": int(0 if conds[name] is None else len(conds[name])),
                     "decoded_step": stepsum(steps(d)),
                     "decode_error_end_m": float(np.mean(err[-1])),
                     "decode_error_mean_m": float(np.mean(err)),
                     # how far decoded endpoint is from decoded start (does it go anywhere?)
                     "decoded_net_disp_m": float(np.mean(np.linalg.norm(d[-1] - d[0], axis=1)))}
    print(f"=== Seed {seed} === true_step_med={res['true_step']['median_cm']:.1f}cm", flush=True)
    for name in dec:
        r = res[name]
        print(f"    {name:24s} dec_step_med={r['decoded_step']['median_cm']:5.1f}cm "
              f"err_end={r['decode_error_end_m']:.2f}m net_disp={r['decoded_net_disp_m']:.2f}m", flush=True)

    with open(os.path.join(out_dir, "output_space_stats.json"), "w") as f:
        json.dump(res, f, indent=2)
    # save a subset of trajectories for videos (true + the 3 main conditions)
    bsave = min(args.save_trajectories, true.shape[1])
    np.savez_compressed(os.path.join(out_dir, "trajectories.npz"),
                        true=true[:, :bsave].astype(np.float32),
                        Intact=dec["Intact"][:, :bsave].astype(np.float32),
                        Predictive_ablated=dec["Predictive-ablated"][:, :bsave].astype(np.float32),
                        Grid_ablated=dec["Grid-ablated"][:, :bsave].astype(np.float32),
                        box_width=np.array(opt.box_width))
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", nargs="+", type=int, default=list(range(10)))
    ap.add_argument("--model_root", default="Models/Single agent path integration")
    ap.add_argument("--analysis_root", default="analysis_outputs/Single agent path integration")
    ap.add_argument("--analysis_subdir", default="spatial_shift_allunits")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--gridness_threshold", type=float, default=0.2)
    ap.add_argument("--min_shift_cm", type=float, default=5.0)
    ap.add_argument("--seq_len", type=int, default=40)
    ap.add_argument("--test_trajectories", type=int, default=128)
    ap.add_argument("--save_trajectories", type=int, default=40)
    args = ap.parse_args()
    device = args.device if torch.cuda.is_available() else "cpu"
    for s in args.seeds:
        run_seed(s, args, device)


if __name__ == "__main__":
    main()
