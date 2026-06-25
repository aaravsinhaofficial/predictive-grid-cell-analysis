#!/usr/bin/env python3
"""Validation: how much does the GRID POPULATION ACTIVITY change over time under each ablation?

Prediction (Rob): after predictive-grid-cell ablation the grid population activity should be
relatively UNCHANGING over the trajectory (traversal stalls -> frozen), whereas random ablation of
the same number of units should leave the activity CHANGING (dynamics preserved, like intact).

This is a raw-activity test -- no torus projection, no decoder. For each condition we run the
(ablated) network and measure the temporal change of the grid population vector, computed ONLY on
that condition's SURVIVING grid units (zeroed units are trivially constant, which would fake
'frozen'), using cosine distance (scale-free) so changes in overall gain don't matter.

Metrics per condition (mean over trajectories):
  change_rate   = mean_t [ 1 - cos(g_t, g_{t+1}) ]              (higher = more changing)
  autocorr(tau) = mean_t cos(g_t, g_{t+tau})                    (higher = more static)
  pca_path_spread = gyration over time of the population state in a fixed 2D PCA space (m-free)
                    (low = frozen, high = sweeps)  -- for the visual

Usage:  python code/aarav_population_activity_change.py --seeds 0 1 2 ... --device cuda
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


def cos_sim(a, b):
    """Row-wise cosine similarity between [N,D] arrays."""
    na = np.linalg.norm(a, axis=1) + 1e-9
    nb = np.linalg.norm(b, axis=1) + 1e-9
    return (a * b).sum(1) / (na * nb)


def temporal_metrics(g_sub):
    """g_sub: [T,B,D] activity of surviving units. Returns change_rate + autocorr at lags."""
    T, B, D = g_sub.shape
    if D == 0:                                                   # no surviving units (e.g. grid-ablated)
        return {"change_rate": float("nan"), **{f"autocorr_lag{t}": float("nan") for t in (1, 5, 10, 20)}}
    flat = lambda x: x.reshape(-1, D)
    change = 1.0 - cos_sim(flat(g_sub[:-1]), flat(g_sub[1:]))     # per (t,b)
    res = {"change_rate": float(change.mean())}
    curve = []
    for tau in range(1, min(T, 36)):
        curve.append(float(cos_sim(flat(g_sub[:-tau]), flat(g_sub[tau:])).mean()))
    res["autocorr_curve"] = curve                                # lags 1..len
    for tau in (1, 5, 10, 20):
        if T > tau:
            res[f"autocorr_lag{tau}"] = curve[tau - 1]
    return res


def run_seed(seed, args, device):
    ckpt = os.path.join(_REPO, args.model_root, f"Seed {seed}", "most_recent_model.pth")
    gpath = os.path.join(_REPO, args.analysis_root, f"Seed {seed}", args.analysis_subdir, "gridness_data.npz")
    out_dir = os.path.join(_REPO, args.analysis_root, f"Seed {seed}", args.analysis_subdir, "aarav_population_change")
    os.makedirs(out_dir, exist_ok=True)
    model, place_cells, traj_gen, opt, Ng, Np = load_model(ckpt, device, args.seq_len)
    classes = load_classes(gpath, args.gridness_threshold, args.min_shift_cm)
    predictive = classes["predictive"]; grid_all = classes["grid_all"]
    structural = np.setdiff1d(grid_all, predictive); all_units = np.arange(Ng)
    N = int(predictive.size); rng = np.random.default_rng(7 + seed)

    inputs, pos = make_random_walk_inputs(traj_gen, args.test_trajectories, args.seq_len, 777 + seed)

    def states(units):
        m = model if units is None else copy.deepcopy(model)
        if units is not None:
            zero_unit_weights_in_place(m, list(units))
        with torch.no_grad():
            g = m.g(inputs).detach().cpu().numpy()                # [T,B,Ng]
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
    PC = Vt[:2].T                                                 # [|G|,2]

    res = {"seed": seed, "N_predictive": N, "n_grid": int(G.size)}
    save = {}
    for name, u in conds.items():
        g = g_int if u.size == 0 else states(u)
        surv = np.setdiff1d(G, u)                                 # surviving grid units
        tm = temporal_metrics(g[:, :, surv])
        # population-state path spread in the fixed intact-PCA space (gyration over time)
        proj = (g[:, :, G].reshape(-1, G.size) - mu) @ PC
        proj = proj.reshape(args.seq_len, args.test_trajectories, 2)
        c = proj.mean(0, keepdims=True)
        pca_spread = float(np.sqrt(((proj - c) ** 2).sum(2).mean(0)).mean())
        res[name] = {"n_ablated": int(u.size), "n_surviving_grid": int(surv.size),
                     "pca_path_spread": pca_spread, **tm}
        save[name.replace("-", "_").replace(" ", "")] = proj[:, :24].astype(np.float32)
        print(f"  {name:24s} change_rate={tm['change_rate']:.3f} "
              f"autocorr@10={tm.get('autocorr_lag10', float('nan')):.3f} pca_spread={pca_spread:.2f}", flush=True)

    with open(os.path.join(out_dir, "population_change_stats.json"), "w") as f:
        json.dump(res, f, indent=2)
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
        print(f"=== Seed {s} ===", flush=True)
        run_seed(s, args, device)


if __name__ == "__main__":
    main()
