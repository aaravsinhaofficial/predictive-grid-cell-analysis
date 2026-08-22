#!/usr/bin/env python3
"""Does the torus-collapse / freeze phenotype of PGC ablation survive the PROPERTY-MATCHED control?

The control is the one from pgc_matched_ablation.py: every PGC is described by 8 covariates
(module, gridness, bandness, firing-rate variance, head-direction tuning, decoder-weight norm,
incoming and outgoing recurrent weight norm), z-scored over the pool, and — in a random order,
repeated 5x — paired with the closest still-unused NON-predictive grid cell by Euclidean
distance in that 8-D space.  Here the covariates are assembled for ALL 4096 units of the 10
legacy networks (pgc_covariates.assemble_covariates, one seeded collection) and the matched
sets are ablated exactly like the PGC set.  Classes are the cached all-4096 spatial-shift
classes (no re-classification), for both PGC definitions:

  original (ours) : best gridness >= 0.2 at shift >= +5 cm     pool = retro ∪ normal (non-predictive grid cells)
  MEC-style       : gs0 < 0.4 & best >= 0.3 at shift >= +5 cm   pool A = retro ∪ normal (non-predictive under either def.)
                                                               pool B = grid union \ PGC_MEC (non-predictive under the MEC def.)
Count-matched random draws from the same pool are run alongside (5x) so the matched control's
extra value over "random grid cell" is visible.

Read-outs per condition (identical trajectories across conditions):
  torus metrics from aarav_ablation_torus.py  : theta1 clumping (resultant length of the torus phase;
        1 = stuck at one phase), ring spread (CV of the major-phase magnitude), torus-decoded RMSE
  dynamics metrics from aarav_definition_dynamics.py on the surviving standard-grid read-out:
        autocorrelation, step ratio, PC1-2 radius, activity norm, tracking, kNN decode, decoder error

Usage: python code/aarav_matched_torus.py --seeds 0 ... 9 --device cuda:1
"""
from __future__ import annotations
import argparse, copy, json, os, sys, time
import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
_REPO = os.path.dirname(_HERE)

from aarav_crossing_population_correlation import load_model
from aarav_activity_space_ablation import make_random_walk_inputs
from multi_seed_predictive_analysis import zero_unit_weights_in_place, collect_eval_batches
from visualize import compute_ratemaps
from toroidal_structure_analysis import build_torus_basis, run_condition, identify_toroidal_cells
from aarav_ablation_torus import metrics_from_proj, subsample_coords
from aarav_definition_dynamics import derive_classes, dyn_metrics, knn_decode, T_FIT
from pgc_matched_ablation import select_matched_control, _standardize_over_pool
import pgc_common as C
import pgc_covariates as CV


def get_covariates(ckpt, cache_path, device, n_workers):
    if os.path.exists(cache_path):
        P, keys, extras = CV.load_covariate_matrix(cache_path)
        return P, keys
    lm = C.load_model(ckpt, device=device)
    res = CV.assemble_covariates(lm, Ng_use=lm.Ng, n_batches=20, res=20, n_workers=n_workers)
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    np.savez_compressed(cache_path, **res["cov"], is_grid=res["is_grid"])
    with open(cache_path.replace(".npz", "_summary.json"), "w") as f:
        json.dump(res["summary"], f, indent=2)
    del lm; torch.cuda.empty_cache()
    P, keys, extras = CV.load_covariate_matrix(cache_path)
    return P, keys


def match_quality(target, matched, P, pool):
    """Median standardized 8-D distance between each target and its match (pairs aligned by select order
    are not returned, so report distribution of NN distances target->matched set and the covariate means)."""
    Z = _standardize_over_pool(P, pool)
    d = np.linalg.norm(Z[target][:, None, :] - Z[matched][None, :, :], axis=2)
    return float(np.median(d.min(1)))


def run_seed(seed, args, device):
    t0 = time.time()
    ckpt = os.path.join(_REPO, args.model_root, f"Seed {seed}", "most_recent_model.pth")
    base = os.path.join(_REPO, args.analysis_root, f"Seed {seed}", args.analysis_subdir)
    out_dir = os.path.join(base, "aarav_matched_torus"); os.makedirs(out_dir, exist_ok=True)
    cl = derive_classes(os.path.join(base, "gridness_data.npz"), os.path.join(base, "band_cells", "band_scores.npz"),
                        z_thr=args.z_thr, s_thr=args.s_thr, min_shift=args.min_shift_cm, lib_thr=args.lib_thr)
    P, keys = get_covariates(ckpt, os.path.join(out_dir, "pgc_covariates_all4096.npz"), device, args.n_workers)
    Ng = P.shape[0]
    G = cl["std_grid"]
    grid_union = np.unique(np.concatenate([cl["pred_lib"], cl["retro_lib"], cl["normal_lib"]]))
    pool_lib = np.unique(np.concatenate([cl["retro_lib"], cl["normal_lib"]]))
    pool_con_A = pool_lib
    pool_con_B = np.setdiff1d(grid_union, cl["pred_con"])
    rng = np.random.default_rng(2024 + seed)

    sets = {"intact": [np.array([], int)], "pred_lib": [cl["pred_lib"]], "pred_con": [cl["pred_con"]]}
    sets["matched_lib"] = [select_matched_control(cl["pred_lib"], P, pool_lib, rng) for _ in range(args.reps)]
    sets["randpool_lib"] = [rng.choice(pool_lib, min(cl["pred_lib"].size, pool_lib.size), replace=False) for _ in range(args.reps)]
    sets["matched_con_A"] = [select_matched_control(cl["pred_con"], P, pool_con_A, rng) for _ in range(args.reps)]
    sets["randpool_con_A"] = [rng.choice(pool_con_A, min(cl["pred_con"].size, pool_con_A.size), replace=False) for _ in range(args.reps)]
    sets["matched_con_B"] = [select_matched_control(cl["pred_con"], P, pool_con_B, rng) for _ in range(args.reps)]
    sets["randpool_con_B"] = [rng.choice(pool_con_B, min(cl["pred_con"].size, pool_con_B.size), replace=False) for _ in range(args.reps)]

    # matching quality + covariate profile (z-scored over the respective pool)
    def profile(units, pool):
        Z = _standardize_over_pool(P, pool)
        return {k: float(np.mean(Z[units, i])) for i, k in enumerate(keys)}
    quality = {
        "keys": keys,
        "pred_lib_profile": profile(cl["pred_lib"], pool_lib),
        "matched_lib_profile": profile(np.concatenate(sets["matched_lib"]), pool_lib),
        "randpool_lib_profile": profile(np.concatenate(sets["randpool_lib"]), pool_lib),
        "pred_con_profile_A": profile(cl["pred_con"], pool_con_A),
        "matched_con_A_profile": profile(np.concatenate(sets["matched_con_A"]), pool_con_A),
        "randpool_con_A_profile": profile(np.concatenate(sets["randpool_con_A"]), pool_con_A),
        "pred_con_profile_B": profile(cl["pred_con"], pool_con_B),
        "matched_con_B_profile": profile(np.concatenate(sets["matched_con_B"]), pool_con_B),
        "nn_dist_matched_lib": float(np.mean([match_quality(cl["pred_lib"], m, P, pool_lib) for m in sets["matched_lib"]])),
        "nn_dist_randpool_lib": float(np.mean([match_quality(cl["pred_lib"], m, P, pool_lib) for m in sets["randpool_lib"]])),
        "nn_dist_matched_con_A": float(np.mean([match_quality(cl["pred_con"], m, P, pool_con_A) for m in sets["matched_con_A"]])),
        "nn_dist_randpool_con_A": float(np.mean([match_quality(cl["pred_con"], m, P, pool_con_A) for m in sets["randpool_con_A"]])),
        "nn_dist_matched_con_B": float(np.mean([match_quality(cl["pred_con"], m, P, pool_con_B) for m in sets["matched_con_B"]])),
        "pool_sizes": {"pool_lib": int(pool_lib.size), "pool_con_A": int(pool_con_A.size), "pool_con_B": int(pool_con_B.size)},
        "matched_lib_in_std_grid": float(np.mean([np.intersect1d(m, G).size / max(1, m.size) for m in sets["matched_lib"]])),
        "pred_lib_in_std_grid": float(np.intersect1d(cl["pred_lib"], G).size / cl["pred_lib"].size),
        "matched_con_A_in_std_grid": float(np.mean([np.intersect1d(m, G).size / max(1, m.size) for m in sets["matched_con_A"]])),
        "matched_con_B_in_std_grid": float(np.mean([np.intersect1d(m, G).size / max(1, m.size) for m in sets["matched_con_B"]])),
        "matched_con_A_frac_retro": float(np.mean([np.intersect1d(m, cl["retro_lib"]).size / max(1, m.size) for m in sets["matched_con_A"]])),
        "matched_con_B_frac_predlib": float(np.mean([np.intersect1d(m, cl["pred_lib"]).size / max(1, m.size) for m in sets["matched_con_B"]])),
    }
    print(f"[seed {seed}] pred_lib={cl['pred_lib'].size} pool_lib={pool_lib.size} | pred_con={cl['pred_con'].size} "
          f"poolA={pool_con_A.size} poolB={pool_con_B.size} | NN dist matched/random: lib {quality['nn_dist_matched_lib']:.2f}/"
          f"{quality['nn_dist_randpool_lib']:.2f}  conA {quality['nn_dist_matched_con_A']:.2f}/{quality['nn_dist_randpool_con_A']:.2f}  "
          f"conB {quality['nn_dist_matched_con_B']:.2f} | matched_lib in std grid {quality['matched_lib_in_std_grid']:.2f} "
          f"(pred_lib {quality['pred_lib_in_std_grid']:.2f}); matched_con_A retro frac {quality['matched_con_A_frac_retro']:.2f}", flush=True)

    # ------------------------------------------------------------------ model + torus basis (as in aarav_ablation_torus.py)
    model, place_cells, traj_gen, opt, Ng_m, Np = load_model(ckpt, device, args.torus_seq_len)
    rng_t = np.random.default_rng(12345 + seed)
    np.random.seed(4321 + seed)
    rate_maps, _, _, _ = compute_ratemaps(model, traj_gen, opt, res=args.res, n_avg=args.ratemap_batches,
                                          Ng=grid_union.size, idxs=grid_union)
    rate_maps = np.asarray(rate_maps, dtype=float)
    det = identify_toroidal_cells(rate_maps, grid_union, opt.box_width, out_dir, embed_mode="umap")
    tor_global = np.asarray(det.units, dtype=int)
    pos_in_grid = {int(g): i for i, g in enumerate(grid_union)}
    local_tor = np.array([pos_in_grid[int(u)] for u in tor_global], dtype=int)
    basis = build_torus_basis(rate_maps[local_tor], np.arange(local_tor.size), opt.box_width)
    basis.units = tor_global
    np.random.seed(999 + seed)
    cached = collect_eval_batches(traj_gen, args.n_batches)
    radii = (1.0, 0.35)
    state = model.state_dict()
    print(f"[seed {seed}] toroidal module {tor_global.size} units; pred_lib in module {np.intersect1d(cl['pred_lib'], tor_global).size}, "
          f"pred_con in module {np.intersect1d(cl['pred_con'], tor_global).size}", flush=True)
    tor_pos = {int(u): i for i, u in enumerate(tor_global)}

    def module_diag(units):
        inm = np.intersect1d(units, tor_global)
        if inm.size == 0:
            return {"n_in_module": 0, "phase1_resultant_removed": float("nan"), "frac_module_removed": 0.0}
        ph = basis.phase1[[tor_pos[int(u)] for u in inm]]
        return {"n_in_module": int(inm.size), "phase1_resultant_removed": float(np.abs(np.mean(np.exp(1j * ph)))),
                "frac_module_removed": float(inm.size / tor_global.size)}

    # extra control: same number of toroidal-MODULE units as the PGC set removes, drawn from non-predictive module units
    nonpred_mod = np.setdiff1d(tor_global, np.union1d(cl["pred_lib"], cl["pred_con"]))
    n_mod_lib = int(np.intersect1d(cl["pred_lib"], tor_global).size); n_mod_con = int(np.intersect1d(cl["pred_con"], tor_global).size)
    sets["module_match_lib"] = [np.concatenate([rng.choice(nonpred_mod, min(n_mod_lib, nonpred_mod.size), replace=False),
                                                rng.choice(np.setdiff1d(pool_lib, tor_global), max(0, cl["pred_lib"].size - min(n_mod_lib, nonpred_mod.size)), replace=False)])
                               for _ in range(args.reps)]
    sets["module_match_con"] = [np.concatenate([rng.choice(nonpred_mod, min(n_mod_con, nonpred_mod.size), replace=False),
                                                rng.choice(np.setdiff1d(pool_con_A, tor_global), max(0, cl["pred_con"].size - min(n_mod_con, nonpred_mod.size)), replace=False)])
                               for _ in range(args.reps)]

    # ------------------------------------------------------------------ dynamics inputs (same as aarav_definition_dynamics)
    traj_gen.options.sequence_length = args.seq_len
    traj_gen.options.trajectory_style = "random_walk"
    inputs_rw, pos_rw = make_random_walk_inputs(traj_gen, args.test_trajectories, args.seq_len, 777 + seed)
    cloud_in = [make_random_walk_inputs(traj_gen, args.test_trajectories, args.seq_len, 5000 + 100 * seed + i)
                for i in range(args.cloud_batches)]

    def run(m, inputs):
        with torch.no_grad():
            g = m.g(inputs)
            pred = place_cells.get_nearest_cell_pos(m.decoder(g)).cpu().numpy()
        return g.detach().cpu().numpy(), pred

    def ablated_model(units):
        m = copy.deepcopy(model)
        if units.size:
            zero_unit_weights_in_place(m, list(units))
        return m

    g0, dec0 = run(model, inputs_rw)
    pos_np = pos_rw.cpu().numpy()
    cloud_g = np.concatenate([run(model, ci[0])[0][T_FIT // 2:] for ci in cloud_in], 0).reshape(-1, Ng)
    cloud_p = np.concatenate([ci[1].cpu().numpy()[T_FIT // 2:] for ci in cloud_in], 0).reshape(-1, 2)

    results = {"seed": seed, "class_sizes": {k: int(cl[k].size) for k in ("pred_lib", "pred_con", "retro_lib", "normal_lib")},
               "n_std_grid": int(G.size), "toroidal_module": int(tor_global.size), "quality": quality, "conditions": {}}
    vis = {}
    for name, unitsets in sets.items():
        recs = []
        for di, units in enumerate(unitsets):
            # torus metrics
            proj = run_condition(state, basis, opt, place_cells, traj_gen, cached, torch.device(device), name, list(units), radii)
            tm = metrics_from_proj(proj)
            if di == 0 and name in ("intact", "pred_lib", "pred_con", "matched_lib", "matched_con_A", "randpool_lib"):
                c, t = subsample_coords(proj, 4000)
                vis[f"{name}|coords"] = c; vis[f"{name}|theta1"] = t
            # dynamics metrics on the std-grid read-out
            R = np.setdiff1d(G, units)
            if units.size:
                m = ablated_model(units); gc, dec = run(m, inputs_rw); del m
            else:
                gc, dec = g0, dec0
            dm = dyn_metrics(gc[:, :, R], g0[:, :, R])
            kn = knn_decode(gc[:, :, R], g0[:, :, R], cloud_g[:, R], cloud_p, pos_np, device)
            rec = {"n_ablated": int(units.size), "n_readout": int(R.size), "torus": tm, "module": module_diag(units),
                   "late": dm["late"], "autocorr_med": dm["autocorr_med"],
                   "autocorr_med_intact_same_units": dm["autocorr_med_intact_same_units"],
                   "radius_pc12_ratio": dm["radius_pc12_ratio"], "norm_ratio": dm["norm_ratio"],
                   "step_ratio": dm["step_ratio"], "tracking_cos": dm["tracking_cos"],
                   "knn_late_err_m": kn["ablated"]["late_err_m"], "knn_late_nn_dist": kn["ablated"]["late_nn_dist"],
                   "knn_err_m": kn["ablated"]["err_m"],
                   "decoder_err_m_mean": float(np.mean(np.linalg.norm(dec - pos_np, axis=-1)))}
            recs.append(rec)
        results["conditions"][name] = recs
        L = recs[0]["late"]; tm = recs[0]["torus"]
        agg = lambda k: np.mean([r["torus"][k] for r in recs])
        print(f"  {name:16s} n={recs[0]['n_ablated']:4d} x{len(recs)} inmod {np.mean([r['module']['n_in_module'] for r in recs]):5.0f} phres {np.nanmean([r['module']['phase1_resultant_removed'] for r in recs]):.2f} | theta1_clump {agg('theta1_clumping'):.3f} ring_spread {agg('ring_spread'):.3f} "
              f"torus-rmse {agg('decode_rmse_cm'):.0f}cm | ac20 {np.mean([r['late']['autocorr_med_lag20'] for r in recs]):.3f} "
              f"step {np.mean([r['late']['step_ratio'] for r in recs]):.2f} r12 {np.mean([r['late']['radius_pc12_ratio'] for r in recs]):.2f} "
              f"norm {np.mean([r['late']['norm_ratio'] for r in recs]):.2f} track {np.mean([r['late']['tracking_cos'] for r in recs]):.2f} "
              f"knn {np.mean([r['knn_late_err_m'] for r in recs]):.2f}m dec {np.mean([r['decoder_err_m_mean'] for r in recs]):.2f}m", flush=True)

    np.savez_compressed(os.path.join(out_dir, "matched_sets.npz"),
                        **{f"{k}_{i}": u for k, v in sets.items() for i, u in enumerate(v)})
    np.savez_compressed(os.path.join(out_dir, "torus_coords.npz"), **vis)
    with open(os.path.join(out_dir, "matched_torus.json"), "w") as f:
        json.dump(results, f)
    print(f"[seed {seed}] done in {time.time() - t0:.0f}s -> {out_dir}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", nargs="+", type=int, default=list(range(10)))
    ap.add_argument("--model_root", default="Models/Single agent path integration")
    ap.add_argument("--analysis_root", default="analysis_outputs/Single agent path integration")
    ap.add_argument("--analysis_subdir", default="spatial_shift_allunits")
    ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--n_workers", type=int, default=16)
    ap.add_argument("--seq_len", type=int, default=40)
    ap.add_argument("--torus_seq_len", type=int, default=20)
    ap.add_argument("--test_trajectories", type=int, default=256)
    ap.add_argument("--cloud_batches", type=int, default=3)
    ap.add_argument("--res", type=int, default=40)
    ap.add_argument("--ratemap_batches", type=int, default=12)
    ap.add_argument("--n_batches", type=int, default=12)
    ap.add_argument("--z_thr", type=float, default=0.4)
    ap.add_argument("--s_thr", type=float, default=0.3)
    ap.add_argument("--lib_thr", type=float, default=0.2)
    ap.add_argument("--min_shift_cm", type=float, default=5.0)
    args = ap.parse_args()
    device = args.device if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)
    for s in args.seeds:
        run_seed(s, args, device)


if __name__ == "__main__":
    main()
