#!/usr/bin/env python3
"""Frozen-decoder control: is the predictive-ablation decode error a readout artifact or real?

When we ablate units, zero_unit_weights_in_place also zeros their decoder columns, so the
network's decoded-position error could in principle be a *readout* effect. Here we decode the
SAME ablated activity two ways and compare, count-matched (N = #predictive), across seeds:

  ablated decoder : the ablated network's own decoder (zeroed columns for ablated units).
  frozen decoder  : the ORIGINAL intact decoder applied to the ablated activity.

If the two errors match, the decode error is NOT a decoder-weight artifact -- it reflects loss
of position information in the activity/state itself.

Usage:  python code/aarav_frozen_decoder.py --seeds 0 1 2 3 4 5 6 7 8 9 --device cuda
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


def decode_err(intact_model, ablated_model, inputs, true_end):
    with torch.no_grad():
        g = ablated_model.g(inputs)                                   # ablated activity [T,B,Ng]
        dec_abl = ablated_model.place_cells.get_nearest_cell_pos(ablated_model.decoder(g))[-1].cpu().numpy()
        dec_frozen = intact_model.place_cells.get_nearest_cell_pos(intact_model.decoder(g))[-1].cpu().numpy()
    return (np.linalg.norm(dec_abl - true_end, axis=1),
            np.linalg.norm(dec_frozen - true_end, axis=1))


def run_seed(seed, args, device):
    ckpt = os.path.join(_REPO, args.model_root, f"Seed {seed}", "most_recent_model.pth")
    gpath = os.path.join(_REPO, args.analysis_root, f"Seed {seed}", args.analysis_subdir, "gridness_data.npz")
    model, place_cells, traj_gen, opt, Ng, Np = load_model(ckpt, device, args.seq_len)
    classes = load_classes(gpath, args.gridness_threshold, args.min_shift_cm)
    predictive = classes["predictive"]; grid_all = classes["grid_all"]
    structural = np.setdiff1d(grid_all, predictive); all_units = np.arange(Ng)
    N = int(predictive.size); rng = np.random.default_rng(args.seed + seed)
    inputs, pos = make_random_walk_inputs(traj_gen, args.test_trajectories, args.seq_len, args.seed + seed + 777)
    true_end = pos.detach().cpu().numpy()[-1]

    conds = {"Intact": [None], "Predictive (N)": [predictive],
             "Random (N)": [rng.choice(all_units, N, replace=False) for _ in range(args.n_draws)],
             "Structural grid (N)": [rng.choice(structural, min(N, structural.size), replace=False) for _ in range(args.n_draws)]}
    res = {}
    for name, usets in conds.items():
        ab, fr = [], []
        for units in usets:
            if units is None or len(units) == 0:
                m = model
            else:
                m = copy.deepcopy(model); zero_unit_weights_in_place(m, list(units))
            a, f = decode_err(model, m, inputs, true_end)
            if units is not None and len(units) > 0:
                del m
            ab.append(a); fr.append(f)
        ab = np.mean(ab, axis=0); fr = np.mean(fr, axis=0)
        res[name] = {"ablated_decoder_m": float(ab.mean()), "frozen_intact_decoder_m": float(fr.mean()),
                     "max_abs_diff_m": float(np.max(np.abs(ab - fr)))}
        print(f"  {name:20s} ablated={res[name]['ablated_decoder_m']:.3f}m  "
              f"frozen={res[name]['frozen_intact_decoder_m']:.3f}m  maxdiff={res[name]['max_abs_diff_m']:.1e}", flush=True)
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
    ap.add_argument("--seq_len", type=int, default=20)
    ap.add_argument("--test_trajectories", type=int, default=256)
    ap.add_argument("--n_draws", type=int, default=4)
    ap.add_argument("--seed", type=int, default=12345)
    args = ap.parse_args()
    device = args.device if torch.cuda.is_available() else "cpu"
    out = {}
    for s in args.seeds:
        print(f"=== Seed {s} ===", flush=True)
        out[str(s)] = run_seed(s, args, device)
    od = os.path.join(_REPO, args.analysis_root, "summary", "aarav_activity_space_ablation")
    os.makedirs(od, exist_ok=True)
    with open(os.path.join(od, "frozen_decoder_control.json"), "w") as f:
        json.dump(out, f, indent=2)
    print("wrote", os.path.join(od, "frozen_decoder_control.json"))


if __name__ == "__main__":
    main()
