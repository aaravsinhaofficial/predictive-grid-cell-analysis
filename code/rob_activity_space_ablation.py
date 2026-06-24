#!/usr/bin/env python3
"""Activity-space ablation: do grid cells define the manifold and predictive cells move along it?

Following William's design. We run a trained path-integrating RNN on a fixed set of test
trajectories under several ablation conditions and ask, in POPULATION-ACTIVITY space:

  (a) Displacement:  how far does the activity travel from start to end of the path?
      (predictive cells drive movement -> ablating them should shrink displacement)
  (b) Deviation:     how far is the ablated endpoint from the INTACT endpoint for the same path?
      (how wrong the response becomes)
  (c) Off-manifold:  how far is the ablated endpoint from the cloud of NORMAL intact states?
      (grid cells define the manifold -> ablating them should push states OFF it; predictive
       ablation should stay ON the manifold but at the wrong point)

Two matching schemes (separate runs / figures), to disentangle "function" from "how many units
were removed":
  --mode count    : every condition silences the SAME absolute number of units N = #predictive.
  --mode percent  : every condition silences the SAME FRACTION of its own class, swept 0..100%.

Conditions: intact, predictive, structural_grid (grid-but-not-predictive), all_grid, random.
Random / structural / all_grid (in count mode) draw random subsets, averaged over draws.

Reuses gridness classes (no recomputation). Usage:
  python code/rob_activity_space_ablation.py --mode count   --seeds 0 1 --device cuda
  python code/rob_activity_space_ablation.py --mode percent --seeds 0 1 --device cuda
"""
from __future__ import annotations
import argparse, copy, json, os, sys, time
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
_REPO = os.path.dirname(_HERE)

from rob_crossing_population_correlation import load_model, load_classes
from multi_seed_predictive_analysis import zero_unit_weights_in_place


# --------------------------------------------------------------------------------------
def collect_activity(model, inputs):
    with torch.no_grad():
        g = model.g(inputs).detach().cpu().numpy()   # [T,B,Ng]
    return g


def make_random_walk_inputs(traj_gen, batch_size, seq_len, np_seed):
    np.random.seed(int(np_seed))                      # generate_trajectory uses global np.random
    traj_gen.options.sequence_length = int(seq_len)
    inputs, pos, _ = traj_gen.get_test_batch(batch_size=batch_size)
    return inputs, pos


def build_normal_cloud(model, traj_gen, seq_len, n_batches, batch_size, max_points, base_seed):
    pts = []
    for i in range(n_batches):
        inputs, _ = make_random_walk_inputs(traj_gen, batch_size, seq_len, base_seed + 1000 + i)
        g = collect_activity(model, inputs)           # [T,B,Ng]
        pts.append(g.reshape(-1, g.shape[-1]))
    cloud = np.concatenate(pts, axis=0)
    if cloud.shape[0] > max_points:
        idx = np.random.default_rng(0).choice(cloud.shape[0], size=max_points, replace=False)
        cloud = cloud[idx]
    return cloud.astype(np.float32)


def fit_manifold(cloud, n_pca=32, k=6):
    scaler = StandardScaler().fit(cloud)
    Xt = scaler.transform(cloud)
    n_pca = int(min(n_pca, Xt.shape[1], Xt.shape[0] - 1))
    pca = PCA(n_components=n_pca, random_state=0).fit(Xt)
    Xp = pca.transform(Xt)
    nn = NearestNeighbors(n_neighbors=k).fit(Xp)
    return scaler, pca, nn, k


def offmanifold(states, scaler, pca, nn, k):
    Xp = pca.transform(scaler.transform(np.asarray(states, dtype=np.float32)))
    d, _ = nn.kneighbors(Xp)
    return d[:, :k].mean(axis=1)                      # mean distance to k nearest normal states


def decoded_endpoint_error(model, inputs, true_end):
    """Per-trajectory L2 error (m) between the network's decoded final position and the true one."""
    with torch.no_grad():
        preds = model.predict(inputs)                 # [T,B,Np]
        dec = model.place_cells.get_nearest_cell_pos(preds).detach().cpu().numpy()  # [T,B,2]
    return np.linalg.norm(dec[-1] - true_end, axis=1)


def ablated_endpoint_metrics(model, units, inputs, g_intact, true_end, scaler, pca, nn, k):
    """Per-trajectory (activity_displacement, deviation_from_intact, offmanifold, decoded_pos_error_m)."""
    if units is None or len(units) == 0:
        m = model
    else:
        m = copy.deepcopy(model)
        zero_unit_weights_in_place(m, list(units))
    g = collect_activity(m, inputs)                   # [T,B,Ng]
    derr = decoded_endpoint_error(m, inputs, true_end)
    if units is not None and len(units) > 0:
        del m
    g0 = g[0]; gT = g[-1]                              # [B,Ng]
    disp = np.linalg.norm(gT - g0, axis=1)
    dev = np.linalg.norm(gT - g_intact[-1], axis=1)
    off = offmanifold(gT, scaler, pca, nn, k)
    return disp, dev, off, derr


def eval_condition(model, pool, n_units, inputs, g_intact, true_end, manifold, rng, n_draws):
    """Evaluate an ablation that removes n_units from `pool`. Averages over draws if random."""
    scaler, pca, nn, k = manifold
    pool = np.asarray(pool, dtype=int)
    n_units = int(n_units)
    if n_units <= 0:
        return ablated_endpoint_metrics(model, None, inputs, g_intact, true_end, scaler, pca, nn, k)
    if n_units >= pool.size:
        units_sets = [pool]                            # whole pool, deterministic
    else:
        units_sets = [rng.choice(pool, size=n_units, replace=False) for _ in range(n_draws)]
    D, V, O, DC = [], [], [], []
    for units in units_sets:
        disp, dev, off, derr = ablated_endpoint_metrics(model, units, inputs, g_intact, true_end, scaler, pca, nn, k)
        D.append(disp); V.append(dev); O.append(off); DC.append(derr)
    return np.mean(D, axis=0), np.mean(V, axis=0), np.mean(O, axis=0), np.mean(DC, axis=0)


def summ(x):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    return {"mean": float(x.mean()), "sem": float(x.std(ddof=1) / np.sqrt(x.size)) if x.size > 1 else 0.0,
            "median": float(np.median(x)), "n": int(x.size)}


# --------------------------------------------------------------------------------------
def run_seed(seed, args, device):
    ckpt = os.path.join(_REPO, args.model_root, f"Seed {seed}", "most_recent_model.pth")
    gpath = os.path.join(_REPO, args.analysis_root, f"Seed {seed}", args.analysis_subdir, "gridness_data.npz")
    out_dir = os.path.join(_REPO, args.analysis_root, f"Seed {seed}", args.analysis_subdir, args.out_subdir)
    os.makedirs(out_dir, exist_ok=True)

    model, place_cells, traj_gen, opt, Ng, Np = load_model(ckpt, device, args.seq_len)
    classes = load_classes(gpath, args.gridness_threshold, args.min_shift_cm)
    predictive = classes["predictive"]; grid_all = classes["grid_all"]
    grid_nonpred = np.setdiff1d(grid_all, predictive)
    all_units = np.arange(Ng)
    rng = np.random.default_rng(args.seed + seed)
    print(f"\n=== Seed {seed} [{args.mode}] === predictive={predictive.size} grid_all={grid_all.size} "
          f"grid_nonpred={grid_nonpred.size} Ng={Ng}", flush=True)

    # normal-activity manifold (intact)
    cloud = build_normal_cloud(model, traj_gen, args.seq_len, args.cloud_batches, args.cloud_batch_size,
                               args.cloud_max_points, args.seed + seed)
    manifold = fit_manifold(cloud, n_pca=args.n_pca, k=args.knn_k)
    print(f"  normal cloud: {cloud.shape[0]} states, PCA={manifold[1].n_components_}, k={manifold[3]}", flush=True)

    # fixed test trajectories (same physical paths across all conditions)
    test_inputs, test_pos = make_random_walk_inputs(traj_gen, args.test_trajectories, args.seq_len, args.seed + seed + 777)
    g_intact = collect_activity(model, test_inputs)
    disp_intact = np.linalg.norm(g_intact[-1] - g_intact[0], axis=1)
    off_intact = offmanifold(g_intact[-1], *manifold[:3], manifold[3])
    pos_np = test_pos.detach().cpu().numpy()
    true_end = pos_np[-1]                                  # [B,2] true final position
    derr_intact = decoded_endpoint_error(model, test_inputs, true_end)
    true_disp = np.linalg.norm(pos_np[-1] - pos_np[0], axis=1)
    print(f"  intact: act_disp={disp_intact.mean():.2f}  offman={off_intact.mean():.3f}  "
          f"decode_err={derr_intact.mean():.3f}m  true_disp={true_disp.mean():.3f}m", flush=True)

    # For percent mode the random baseline is a FIXED random subset the size of the grid set,
    # so "f% of random" is count-comparable to "f% of all_grid" across the whole sweep
    # (rather than degenerately ablating the entire network at f=100%).
    rand_pool = rng.choice(all_units, size=int(min(grid_all.size, Ng)), replace=False)
    pools = {"predictive": predictive, "structural_grid": grid_nonpred, "all_grid": grid_all, "random": rand_pool}
    result = {"seed": seed, "mode": args.mode, "Ng": Ng,
              "class_counts": {"predictive": int(predictive.size), "grid_all": int(grid_all.size),
                               "grid_nonpred": int(grid_nonpred.size)},
              "intact": {"displacement": summ(disp_intact), "offmanifold": summ(off_intact),
                         "decoded_endpoint_error_m": summ(derr_intact), "true_displacement_m": summ(true_disp)}}

    if args.mode == "count":
        N = int(predictive.size)
        result["N_units"] = N
        conds = {
            "predictive":      (predictive, N),
            "structural_grid": (grid_nonpred, N),
            "all_grid_matched": (grid_all, N),
            "all_grid_full":   (grid_all, grid_all.size),     # qualitative William version
            "random":          (all_units, N),
        }
        result["conditions"] = {}
        for name, (pool, n) in conds.items():
            disp, dev, off, derr = eval_condition(model, pool, n, test_inputs, g_intact, true_end, manifold, rng, args.n_draws)
            result["conditions"][name] = {
                "n_units": int(n),
                "displacement": summ(disp),
                "displacement_ratio_to_intact": summ(disp / np.maximum(disp_intact, 1e-9)),
                "deviation_from_intact": summ(dev),
                "deviation_norm": summ(dev / np.maximum(disp_intact, 1e-9)),
                "offmanifold": summ(off),
                "offmanifold_vs_intact": summ(off / np.maximum(off_intact.mean(), 1e-9)),
                "decoded_endpoint_error_m": summ(derr),
            }
            c = result["conditions"][name]
            print(f"    {name:16s} n={n:5d}  dispRatio={c['displacement_ratio_to_intact']['mean']:.2f}"
                  f"  dev={c['deviation_from_intact']['mean']:.2f}"
                  f"  offman={c['offmanifold_vs_intact']['mean']:.1f}x  decErr={c['decoded_endpoint_error_m']['mean']:.2f}m", flush=True)

    else:  # percent
        fracs = args.fractions
        result["fractions"] = fracs
        result["pools"] = {}
        for pname, pool in pools.items():
            rows = []
            for f in fracs:
                n = int(round(f * pool.size))
                disp, dev, off, derr = eval_condition(model, pool, n, test_inputs, g_intact, true_end, manifold, rng, args.n_draws)
                rows.append({
                    "fraction": f, "n_units": n,
                    "displacement_ratio_to_intact": summ(disp / np.maximum(disp_intact, 1e-9)),
                    "deviation_from_intact": summ(dev),
                    "offmanifold": summ(off),
                    "offmanifold_vs_intact": summ(off / np.maximum(off_intact.mean(), 1e-9)),
                    "decoded_endpoint_error_m": summ(derr),
                })
            result["pools"][pname] = rows
            tail = rows[-1]
            print(f"    {pname:16s} @100%: dispRatio={tail['displacement_ratio_to_intact']['mean']:.2f}"
                  f" dev={tail['deviation_from_intact']['mean']:.2f} offman={tail['offmanifold_vs_intact']['mean']:.1f}x"
                  f" decErr={tail['decoded_endpoint_error_m']['mean']:.2f}m", flush=True)

    fn = os.path.join(out_dir, f"activity_space_ablation_{args.mode}.json")
    with open(fn, "w") as f:
        json.dump(result, f, indent=2)
    print(f"  wrote {fn}", flush=True)
    return result


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode", choices=["count", "percent"], required=True)
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1])
    ap.add_argument("--model_root", default="Models/Single agent path integration")
    ap.add_argument("--analysis_root", default="analysis_outputs/Single agent path integration")
    ap.add_argument("--analysis_subdir", default="spatial_shift_allunits")
    ap.add_argument("--out_subdir", default="rob_activity_space_ablation")
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
    ap.add_argument("--fractions", nargs="+", type=float, default=[0.1, 0.25, 0.5, 0.75, 1.0])
    ap.add_argument("--seed", type=int, default=12345)
    return ap.parse_args()


def main():
    args = parse_args()
    device = args.device if torch.cuda.is_available() else "cpu"
    for s in args.seeds:
        t0 = time.time()
        run_seed(s, args, device)
        print(f"  seed {s} done in {time.time()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
