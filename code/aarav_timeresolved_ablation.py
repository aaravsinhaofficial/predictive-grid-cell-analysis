#!/usr/bin/env python3
"""Time-resolved ablation dynamics: is there a change at the training horizon (20 steps)?

Models were trained on 20-step trajectories; we analyse 40 steps. For each absolute timestep we
measure, per condition: decode error vs truth, instantaneous population change (cosine distance
between consecutive grid population vectors, surviving units), and distance of the population state
from its t=0 value. If the ablated dynamics are well-behaved within the trained horizon and diverge
beyond it, these should show a knee near t=20.

Usage:  python code/aarav_timeresolved_ablation.py --seeds 0 1 2 ... --device cuda
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


def cos_rows(a, b):
    return (a * b).sum(1) / ((np.linalg.norm(a, axis=1) + 1e-9) * (np.linalg.norm(b, axis=1) + 1e-9))


def run_seed(seed, args, device):
    ckpt = os.path.join(_REPO, args.model_root, f"Seed {seed}", "most_recent_model.pth")
    gpath = os.path.join(_REPO, args.analysis_root, f"Seed {seed}", args.analysis_subdir, "gridness_data.npz")
    out_dir = os.path.join(_REPO, args.analysis_root, f"Seed {seed}", args.analysis_subdir, "aarav_timeresolved")
    os.makedirs(out_dir, exist_ok=True)
    model, place_cells, traj_gen, opt, Ng, Np = load_model(ckpt, device, args.seq_len)
    cl = load_classes(gpath, args.gridness_threshold, args.min_shift_cm)
    predictive = cl["predictive"]; grid_all = cl["grid_all"]; all_units = np.arange(Ng)
    N = int(predictive.size); rng = np.random.default_rng(7 + seed)
    Wd = model.decoder.weight.detach().to(device)

    inputs, pos = make_random_walk_inputs(traj_gen, args.test_trajectories, args.seq_len, 777 + seed)
    true = pos.detach().cpu().numpy()                                  # [T,B,2]
    T, B = true.shape[:2]

    def states_and_decode(units):
        m = model if units is None else copy.deepcopy(model)
        if units is not None:
            zero_unit_weights_in_place(m, list(units))
        with torch.no_grad():
            g = m.g(inputs)
            pc = g @ Wd.T
            dpos = place_cells.get_nearest_cell_pos(pc).detach().cpu().numpy()
            g = g.detach().cpu().numpy()
        if units is not None:
            del m
        return g, dpos

    conds = {"Intact": None, "Predictive-ablated": predictive,
             "Random-ablated (N)": rng.choice(all_units, N, replace=False)}
    res = {"seed": seed, "train_horizon": 20, "T": T}
    for name, u in conds.items():
        g, dpos = states_and_decode(u)
        surv = grid_all if u is None else np.setdiff1d(grid_all, u)
        gs = g[:, :, surv]
        err_t = np.linalg.norm(dpos - true, axis=2).mean(1)            # [T] decode error per step
        change_t = np.concatenate([[np.nan], [1 - cos_rows(gs[t - 1], gs[t]).mean() for t in range(1, T)]])
        fromstart_t = np.array([(1 - cos_rows(gs[0], gs[t]).mean()) for t in range(T)])  # decorrelation from t0
        res[name] = {"decode_err_t": err_t.tolist(), "pop_change_t": change_t.tolist(),
                     "decorr_from_start_t": fromstart_t.tolist()}
        print(f"  {name:22s} err[10]={err_t[10]:.2f} err[20]={err_t[20]:.2f} err[39]={err_t[39]:.2f} "
              f"| change[10]={change_t[10]:.3f} change[25]={change_t[25]:.3f}", flush=True)
    with open(os.path.join(out_dir, "timeresolved_stats.json"), "w") as f:
        json.dump(res, f, indent=2)
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
    args = ap.parse_args()
    device = args.device if torch.cuda.is_available() else "cpu"
    for s in args.seeds:
        print(f"=== Seed {s} ===", flush=True)
        run_seed(s, args, device)


if __name__ == "__main__":
    main()
