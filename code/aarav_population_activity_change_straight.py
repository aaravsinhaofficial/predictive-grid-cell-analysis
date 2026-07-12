#!/usr/bin/env python3
"""Straight-path autocorrelation — EXACT copy of aarav_population_activity_change.py with the ONLY
change being the evaluation trajectories: random_walk -> straight.

Controlled comparison, trajectory type is the sole changed variable:
  trained networks       : same existing networks (loaded identically)
  training trajectories  : unchanged (we do NOT retrain)
  evaluation trajectories: random_walk -> straight  (trajectory_style = "straight")
  PGC definition         : spatial-shift-defined predictive cells (load_classes, unchanged)
  seeds                  : same 10 seeds
  conditions             : Intact / Predictive / Grid / Random(N) / Structural(N)  (unchanged)
  random-cell sampling   : identical rng(7+seed) and draw order (unchanged)
  lags / timesteps       : unchanged (seq_len 40, autocorr lags 1..35, and 1/5/10/20)
  statistics             : cos_sim + temporal_metrics IMPORTED from the original script (identical)

The straight generator (trajectory_generator.py, style="straight") keeps heading fixed except for
wall avoidance and draws the SAME Rayleigh speed / RNG sequence as random_walk, so with the same
np_seed the start positions, headings and speeds are identical; only the turning is removed.

We additionally record, from the SAME forward passes (no extra cost), the metrics recomputed after
excluding any trajectory that contacts the arena border during the analysis window (secondary
boundary-filtered robustness run).
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
from multi_seed_predictive_analysis import zero_unit_weights_in_place
# IDENTICAL statistics as the random-walk run -- imported, not re-implemented:
from aarav_population_activity_change import cos_sim, temporal_metrics  # noqa: F401


def make_straight_inputs(traj_gen, batch_size, seq_len, np_seed):
    """EXACT analogue of make_random_walk_inputs, but evaluation trajectory style = 'straight'.

    RNG draws are identical to random_walk (fixed heading, same Rayleigh speed process), so the same
    np_seed yields the same start positions / headings / speeds; only the per-step turning is removed."""
    np.random.seed(int(np_seed))                      # generate_trajectory uses global np.random
    traj_gen.options.sequence_length = int(seq_len)
    traj_gen.options.trajectory_style = "straight"    # <-- THE ONLY CHANGE vs make_random_walk_inputs
    inputs, pos, _ = traj_gen.get_test_batch(batch_size=batch_size)
    return inputs, pos


def boundary_contact_mask(pos_np, box_width, border_region):
    """pos_np [T,B,2] -> bool [B]: True if the trajectory enters the wall border region at any t
    during the analysis window (i.e. where wall-avoidance would bend an otherwise-straight path)."""
    half = box_width / 2.0
    reach = np.abs(pos_np).max(axis=2)                # [T,B] = max(|x|,|y|)
    near = reach > (half - float(border_region))
    return near.any(axis=0)                           # [B]


def run_seed(seed, args, device):
    ckpt = os.path.join(_REPO, args.model_root, f"Seed {seed}", "most_recent_model.pth")
    gpath = os.path.join(_REPO, args.analysis_root, f"Seed {seed}", args.analysis_subdir, "gridness_data.npz")
    out_dir = os.path.join(_REPO, args.analysis_root, f"Seed {seed}", args.analysis_subdir, "aarav_population_change_straight")
    os.makedirs(out_dir, exist_ok=True)
    model, place_cells, traj_gen, opt, Ng, Np = load_model(ckpt, device, args.seq_len)
    classes = load_classes(gpath, args.gridness_threshold, args.min_shift_cm)
    predictive = classes["predictive"]; grid_all = classes["grid_all"]
    structural = np.setdiff1d(grid_all, predictive); all_units = np.arange(Ng)
    N = int(predictive.size); rng = np.random.default_rng(7 + seed)   # SAME rule / seed as random-walk

    # ---- ONLY CHANGE: straight evaluation trajectories (same np_seed as random-walk run) ----
    inputs, pos = make_straight_inputs(traj_gen, args.test_trajectories, args.seq_len, 777 + seed)
    pos_np = pos.detach().cpu().numpy()                               # [T,B,2]
    bcontact = boundary_contact_mask(pos_np, opt.box_width, opt.trajectory_border_region)
    keep = ~bcontact                                                  # boundary-free trajectories
    # straightness diagnostic: max heading deviation from the initial heading, per trajectory
    step = np.diff(pos_np, axis=0)                                    # [T-1,B,2]
    ang = np.arctan2(step[..., 1], step[..., 0])                     # [T-1,B]
    dev = np.abs(np.angle(np.exp(1j * (ang - ang[0:1]))))            # wrap to [-pi,pi], abs
    mean_head_dev_deg = float(np.rad2deg(dev.mean()))

    def states(units):
        m = model if units is None else copy.deepcopy(model)
        if units is not None:
            zero_unit_weights_in_place(m, list(units))
        with torch.no_grad():
            g = m.g(inputs).detach().cpu().numpy()                    # [T,B,Ng]
        if units is not None:
            del m
        return g

    conds = {"Intact": np.array([], int), "Predictive-ablated": predictive, "Grid-ablated": grid_all,
             "Random-ablated (N)": rng.choice(all_units, N, replace=False),
             "Structural-ablated (N)": rng.choice(structural, min(N, structural.size), replace=False)}

    # fixed 2D PCA basis from intact grid activity (for the visual); transform all conditions in it
    g_int = states(None)
    G = grid_all
    flatint = g_int[:, :, G].reshape(-1, G.size)
    mu = flatint.mean(0)
    U, S, Vt = np.linalg.svd(flatint - mu, full_matrices=False)
    PC = Vt[:2].T

    meta = {"seed": seed, "N_predictive": N, "n_grid": int(G.size),
            "n_total": int(pos_np.shape[1]), "n_boundary_free": int(keep.sum()),
            "boundary_contact_frac": float(bcontact.mean()), "mean_head_dev_deg": mean_head_dev_deg,
            "trajectory_style": "straight"}
    res = dict(meta); res_nb = dict(meta)                            # all-straight  /  boundary-free
    save = {}
    for name, u in conds.items():
        g = g_int if u.size == 0 else states(u)
        surv = np.setdiff1d(G, u)                                     # surviving grid units
        tm = temporal_metrics(g[:, :, surv])
        tm_nb = temporal_metrics(g[:, keep][:, :, surv])
        proj = (g[:, :, G].reshape(-1, G.size) - mu) @ PC
        proj = proj.reshape(args.seq_len, args.test_trajectories, 2)
        c = proj.mean(0, keepdims=True)
        pca_spread = float(np.sqrt(((proj - c) ** 2).sum(2).mean(0)).mean())
        res[name] = {"n_ablated": int(u.size), "n_surviving_grid": int(surv.size),
                     "pca_path_spread": pca_spread, **tm}
        res_nb[name] = {"n_ablated": int(u.size), "n_surviving_grid": int(surv.size), **tm_nb}
        save[name.replace("-", "_").replace(" ", "")] = proj[:, :24].astype(np.float32)
        print(f"  seed {seed} {name:24s} change_rate={tm['change_rate']:.3f} "
              f"autocorr@10={tm.get('autocorr_lag10', float('nan')):.3f} "
              f"autocorr@20={tm.get('autocorr_lag20', float('nan')):.3f} pca_spread={pca_spread:.2f}", flush=True)
    print(f"  seed {seed} straightness: mean heading dev={mean_head_dev_deg:.1f} deg | "
          f"boundary contact {bcontact.mean()*100:.0f}% ({keep.sum()}/{len(keep)} kept)", flush=True)

    with open(os.path.join(out_dir, "population_change_stats.json"), "w") as f:
        json.dump(res, f, indent=2)
    with open(os.path.join(out_dir, "population_change_stats_noboundary.json"), "w") as f:
        json.dump(res_nb, f, indent=2)
    np.savez_compressed(os.path.join(out_dir, "pca_population_paths.npz"), **save)
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
        print(f"=== Seed {s} (straight) ===", flush=True)
        run_seed(s, args, device)


if __name__ == "__main__":
    main()
