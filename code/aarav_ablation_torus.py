#!/usr/bin/env python3
"""Toroidal manifold under COUNT-MATCHED ablation, across seeds (with between-seed variance).

Fits a torus basis from a detected coherent grid module (FFT reciprocal lattice + per-unit
phases), then projects each ablation condition's population activity onto that fixed basis.

Every ablation removes the SAME number of units N = #predictive, so the comparison is fair:
  Intact            : no ablation
  Predictive (N)    : all predictive units
  Random (N)        : N random units from the whole population (avg over draws)
  Grid (N)          : N random GRID units (avg over draws)
  All grid          : every grid unit (reference: full manifold collapse) -- not count-matched

Torus-degradation metrics per condition (lower = more torus-like):
  ring_spread (r1_cv)  : coefficient of variation of the major-phase magnitude
  theta1_clumping      : resultant length of theta1 (0 = uniform ring traversal, 1 = clumped)
  decode_rmse_cm       : position RMSE decoded from the torus phases

Per-seed JSON + compact coords npz are written; aggregate stats/figures via
aarav_ablation_torus_figure.py.

Usage:  python code/aarav_ablation_torus.py --seeds 0 1 2 3 4 --device cuda
"""
from __future__ import annotations
import argparse, json, os, sys
import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
_REPO = os.path.dirname(_HERE)

from aarav_crossing_population_correlation import load_model, load_classes
from visualize import compute_ratemaps
from multi_seed_predictive_analysis import collect_eval_batches
from toroidal_structure_analysis import build_torus_basis, run_condition, identify_toroidal_cells

VISUAL_CONDS = ["Intact", "Predictive (N)", "Random (N)", "All grid"]


def metrics_from_proj(proj):
    th1 = proj.theta1.reshape(-1); th1 = th1[np.isfinite(th1)]
    resultant = float(np.abs(np.mean(np.exp(1j * th1)))) if th1.size else float("nan")
    return {"ring_spread": float(proj.radius_cv["r1_cv"]),
            "minor_spread": float(proj.radius_cv["r2_cv"]),
            "theta1_clumping": resultant,
            "decode_rmse_cm": float(proj.rmse_cm)}


def subsample_coords(proj, k, seed=0):
    coords = proj.coords3d; th1 = proj.theta1.reshape(-1)
    n = coords.shape[0]
    idx = np.arange(n) if n <= k else np.random.default_rng(seed).choice(n, size=k, replace=False)
    return coords[idx].astype(np.float32), th1[idx].astype(np.float32)


def run_seed(seed, args, device):
    ckpt = os.path.join(_REPO, args.model_root, f"Seed {seed}", "most_recent_model.pth")
    gpath = os.path.join(_REPO, args.analysis_root, f"Seed {seed}", args.analysis_subdir, "gridness_data.npz")
    out_dir = os.path.join(_REPO, args.analysis_root, f"Seed {seed}", args.analysis_subdir, "aarav_ablation_torus")
    os.makedirs(out_dir, exist_ok=True)

    model, place_cells, traj_gen, opt, Ng, Np = load_model(ckpt, device, args.seq_len)
    classes = load_classes(gpath, args.gridness_threshold, args.min_shift_cm)
    grid_all = classes["grid_all"]; predictive = classes["predictive"]
    all_units = np.arange(Ng)
    N = int(predictive.size)
    rng = np.random.default_rng(args.seed + seed)

    rate_maps, _, _, _ = compute_ratemaps(model, traj_gen, opt, res=args.res, n_avg=args.ratemap_batches,
                                          Ng=grid_all.size, idxs=grid_all)
    rate_maps = np.asarray(rate_maps, dtype=float)
    det = identify_toroidal_cells(rate_maps, grid_all, opt.box_width, out_dir, embed_mode="umap")
    tor_global = np.asarray(det.units, dtype=int)
    pos_in_grid = {int(g): i for i, g in enumerate(grid_all)}
    local_tor = np.array([pos_in_grid[int(u)] for u in tor_global], dtype=int)
    basis = build_torus_basis(rate_maps[local_tor], np.arange(local_tor.size), opt.box_width)
    basis.units = tor_global
    n_pred_in_tor = int(np.intersect1d(predictive, tor_global).size)
    print(f"=== Seed {seed} ===  N(predictive)={N}  toroidal_module={tor_global.size}  "
          f"predictive∩toroidal={n_pred_in_tor}", flush=True)

    cached = collect_eval_batches(traj_gen, args.n_batches)
    radii = (1.0, 0.35)
    state = model.state_dict()

    def proj_for(units):
        return run_condition(state, basis, opt, place_cells, traj_gen, cached, torch.device(device),
                             "cond", list(units), radii)

    # Off-module control: predictive cells OUTSIDE the module vs random off-module cells.
    pred_off = np.setdiff1d(predictive, tor_global)          # predictive, not in module
    nontor = np.setdiff1d(all_units, tor_global)             # everything not in module
    Nprime = int(pred_off.size)
    # WITHIN-MODULE control (the decisive test): among torus-module cells, are the PREDICTIVE
    # ones more important for traversal than an equal number of NON-predictive module cells?
    pred_in = np.intersect1d(predictive, tor_global)         # predictive AND in module
    nonpred_in = np.setdiff1d(tor_global, predictive)        # module cells that are not predictive
    Npp = int(pred_in.size)
    print(f"    off-module: pred_off(N')={Nprime}  non_tor_pool={nontor.size}  |  "
          f"in-module: pred_in(N'')={Npp}  nonpred_in_pool={nonpred_in.size}", flush=True)

    # condition -> list of unit-sets (multiple = random draws to average)
    cond_unitsets = {
        "Intact": [np.array([], dtype=int)],
        "Predictive (N)": [predictive],
        "Random (N)": [rng.choice(all_units, size=N, replace=False) for _ in range(args.n_draws)],
        "Grid (N)": [rng.choice(grid_all, size=min(N, grid_all.size), replace=False) for _ in range(args.n_draws)],
        "Predictive off-module (N')": [pred_off if Nprime > 0 else np.array([], dtype=int)],
        "Random off-module (N')": [rng.choice(nontor, size=min(Nprime, nontor.size), replace=False)
                                   for _ in range(args.n_draws)] if Nprime > 0 else [np.array([], dtype=int)],
        "Predictive in-module (N'')": [pred_in if Npp > 0 else np.array([], dtype=int)],
        "Non-pred in-module (N'')": [rng.choice(nonpred_in, size=min(Npp, nonpred_in.size), replace=False)
                                     for _ in range(args.n_draws)] if (Npp > 0 and nonpred_in.size > 0)
                                    else [np.array([], dtype=int)],
        "All grid": [grid_all],
    }

    results, vis = {}, {}
    for cond, unitsets in cond_unitsets.items():
        mlist = []
        for di, units in enumerate(unitsets):
            proj = proj_for(units)
            mlist.append(metrics_from_proj(proj))
            if di == 0 and cond in VISUAL_CONDS:
                c, t = subsample_coords(proj, args.plot_points)
                vis[f"{cond}|coords"] = c; vis[f"{cond}|theta1"] = t
        keys = mlist[0].keys()
        agg = {k: float(np.mean([m[k] for m in mlist])) for k in keys}
        agg.update({f"{k}_std": float(np.std([m[k] for m in mlist])) for k in keys})
        agg["n_units"] = int(len(unitsets[0])); agg["n_draws"] = len(unitsets)
        results[cond] = agg
        print(f"    {cond:16s} n={agg['n_units']:5d}  ring_spread={agg['ring_spread']:.2f}"
              f"  theta1_clump={agg['theta1_clumping']:.2f}  rmse={agg['decode_rmse_cm']:.0f}cm", flush=True)

    out = {"seed": seed, "N_predictive": N, "toroidal_module": int(tor_global.size),
           "predictive_in_toroidal": n_pred_in_tor, "predictive_off_module": Nprime,
           "predictive_in_module": Npp, "nonpred_in_module": int(nonpred_in.size), "conditions": results}
    with open(os.path.join(out_dir, "torus_ablation_metrics.json"), "w") as f:
        json.dump(out, f, indent=2)
    np.savez_compressed(os.path.join(out_dir, "torus_coords.npz"), **vis)
    print(f"  wrote {out_dir}/torus_ablation_metrics.json", flush=True)


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
    ap.add_argument("--res", type=int, default=40)
    ap.add_argument("--ratemap_batches", type=int, default=12)
    ap.add_argument("--n_batches", type=int, default=12)
    ap.add_argument("--n_draws", type=int, default=4)
    ap.add_argument("--plot_points", type=int, default=6000)
    ap.add_argument("--seed", type=int, default=12345)
    args = ap.parse_args()
    device = args.device if torch.cuda.is_available() else "cpu"
    for s in args.seeds:
        run_seed(s, args, device)


if __name__ == "__main__":
    main()
