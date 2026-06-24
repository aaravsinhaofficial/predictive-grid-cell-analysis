#!/usr/bin/env python3
"""Deep dive on 'predictive ablation = legal state, wrong place' — full population, all seeds.

Count-matched ablations (N = #predictive): Intact, Predictive(N), Random(N), Structural-grid(N).
For each test trajectory we measure, at the endpoint:

  off_manifold      : kNN distance of the ablated endpoint to the cloud of normal intact states
                      (low = still a 'legal' network state).
  decode_error_m    : error of the network's decoded position (high = it's at the wrong place).
  effective_time    : t*/(T-1), where t* = the INTACT timestep on the SAME path whose activity
                      the ablated endpoint best matches (in manifold/PCA space).
                      = 1.0 if the state is where it should be; < 1.0 if the state matches an
                      EARLIER point on the path -> the network under-moved along the manifold.
  along_vs_off      : along-manifold error (how far t* is from the end) vs off-path residual
                      (distance to the closest intact-path point) -> decomposes 'wrong place'
                      (along the manifold) from 'illegal state' (off the manifold).

Mechanism hypothesis: ablating predictive cells keeps the state ON the manifold (low off_manifold)
but at an EARLIER position (effective_time < 1), i.e. predictive cells advance the state to the
correct place along the manifold. Structural-grid ablation instead pushes the state OFF the
manifold.

Usage:  python code/aarav_predictive_dynamics.py --seeds 0 1 2 3 4 --device cuda
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
from aarav_activity_space_ablation import (
    build_normal_cloud, fit_manifold, offmanifold, decoded_endpoint_error,
    collect_activity, make_random_walk_inputs,
)
from multi_seed_predictive_analysis import zero_unit_weights_in_place


def project(states, scaler, pca):
    return pca.transform(scaler.transform(np.asarray(states, dtype=np.float32)))


def effective_time(proj_intact, proj_end):
    """proj_intact [T,B,P], proj_end [B,P] -> (frac in [0,1] per traj, off-path residual per traj)."""
    d = np.linalg.norm(proj_intact - proj_end[None], axis=2)   # [T,B]
    tstar = np.argmin(d, axis=0)                                # [B]
    T = proj_intact.shape[0]
    frac = tstar / max(T - 1, 1)
    resid = d[tstar, np.arange(d.shape[1])]
    return frac, resid, tstar


def summ(x):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    return {"mean": float(x.mean()), "std": float(x.std()),
            "sem": float(x.std(ddof=1) / np.sqrt(x.size)) if x.size > 1 else 0.0, "n": int(x.size)}


def metrics_for_units(model, units, inputs, true_end, g_intact, proj_intact, manifold):
    scaler, pca, nn, k = manifold
    if units is None or len(units) == 0:
        m = model
    else:
        m = copy.deepcopy(model); zero_unit_weights_in_place(m, list(units))
    g = collect_activity(m, inputs)                      # [T,B,Ng]
    derr = decoded_endpoint_error(m, inputs, true_end)
    if units is not None and len(units) > 0:
        del m
    gT = g[-1]
    off = offmanifold(gT, scaler, pca, nn, k)
    proj_end = project(gT, scaler, pca)
    frac, resid, _ = effective_time(proj_intact, proj_end)
    return {"off_manifold": off, "decode_error_m": derr, "effective_time": frac, "off_path_resid": resid}


def eval_condition(model, pool, n_units, inputs, true_end, g_intact, proj_intact, manifold, rng, n_draws):
    pool = np.asarray(pool, dtype=int); n_units = int(n_units)
    if n_units <= 0:
        usets = [None]
    elif n_units >= pool.size:
        usets = [pool]
    else:
        usets = [rng.choice(pool, size=n_units, replace=False) for _ in range(n_draws)]
    acc = None
    for units in usets:
        m = metrics_for_units(model, units, inputs, true_end, g_intact, proj_intact, manifold)
        acc = {k: [v] for k, v in m.items()} if acc is None else {k: acc[k] + [m[k]] for k in acc}
    return {k: np.mean(acc[k], axis=0) for k in acc}      # per-trajectory, averaged over draws


def run_seed(seed, args, device):
    ckpt = os.path.join(_REPO, args.model_root, f"Seed {seed}", "most_recent_model.pth")
    gpath = os.path.join(_REPO, args.analysis_root, f"Seed {seed}", args.analysis_subdir, "gridness_data.npz")
    out_dir = os.path.join(_REPO, args.analysis_root, f"Seed {seed}", args.analysis_subdir, "aarav_predictive_dynamics")
    os.makedirs(out_dir, exist_ok=True)

    model, place_cells, traj_gen, opt, Ng, Np = load_model(ckpt, device, args.seq_len)
    classes = load_classes(gpath, args.gridness_threshold, args.min_shift_cm)
    predictive = classes["predictive"]; grid_all = classes["grid_all"]
    structural = np.setdiff1d(grid_all, predictive); all_units = np.arange(Ng)
    N = int(predictive.size); rng = np.random.default_rng(args.seed + seed)

    cloud = build_normal_cloud(model, traj_gen, args.seq_len, args.cloud_batches, args.cloud_batch_size,
                               args.cloud_max_points, args.seed + seed)
    manifold = fit_manifold(cloud, n_pca=args.n_pca, k=args.knn_k)
    scaler, pca, nn, k = manifold

    test_inputs, test_pos = make_random_walk_inputs(traj_gen, args.test_trajectories, args.seq_len, args.seed + seed + 777)
    g_intact = collect_activity(model, test_inputs)       # [T,B,Ng]
    T, B, _ = g_intact.shape
    proj_intact = project(g_intact.reshape(-1, Ng), scaler, pca).reshape(T, B, -1)
    true_end = test_pos.detach().cpu().numpy()[-1]
    off_intact = offmanifold(g_intact[-1], scaler, pca, nn, k)

    conds = {"Intact": (None, 0), "Predictive (N)": (predictive, N),
             "Random (N)": (all_units, N), "Structural grid (N)": (structural, N)}
    print(f"=== Seed {seed} === N={N} grid={grid_all.size} structural={structural.size} "
          f"off_intact={off_intact.mean():.2f}", flush=True)
    results = {"intact_off_manifold_mean": float(off_intact.mean())}
    for name, (pool, n) in conds.items():
        m = eval_condition(model, pool if pool is not None else all_units, n, test_inputs, true_end,
                           g_intact, proj_intact, manifold, rng, args.n_draws)
        results[name] = {
            "off_manifold": summ(m["off_manifold"]),
            "off_manifold_ratio": summ(m["off_manifold"] / max(off_intact.mean(), 1e-9)),
            "decode_error_m": summ(m["decode_error_m"]),
            "effective_time": summ(m["effective_time"]),
            "off_path_resid": summ(m["off_path_resid"]),
            "n_units": int(n),
        }
        r = results[name]
        print(f"    {name:20s} off={r['off_manifold_ratio']['mean']:5.1f}x  "
              f"decodeErr={r['decode_error_m']['mean']:.2f}m  effTime={r['effective_time']['mean']:.2f}", flush=True)

    out = {"seed": seed, "N_predictive": N, "T": int(T), "conditions": results}
    with open(os.path.join(out_dir, "predictive_dynamics.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(f"  wrote {out_dir}/predictive_dynamics.json", flush=True)


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
    ap.add_argument("--cloud_batches", type=int, default=30)
    ap.add_argument("--cloud_batch_size", type=int, default=200)
    ap.add_argument("--cloud_max_points", type=int, default=40000)
    ap.add_argument("--n_pca", type=int, default=32)
    ap.add_argument("--knn_k", type=int, default=6)
    ap.add_argument("--n_draws", type=int, default=4)
    ap.add_argument("--seed", type=int, default=12345)
    args = ap.parse_args()
    device = args.device if torch.cuda.is_available() else "cpu"
    for s in args.seeds:
        run_seed(s, args, device)


if __name__ == "__main__":
    main()
