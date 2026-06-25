#!/usr/bin/env python3
"""Decisive test: decompose the decoded output into TORUS and OFF-MANIFOLD (residual) parts.

For each condition we run the (ablated) network, read off torus angles (theta1, theta2) from the
intact torus basis, reconstruct the *intact* manifold state at those angles  h~ = m(theta1,theta2)
(binned intact-activity lookup), and decode BOTH the real state and its torus reconstruction
through the trained readout D:

    p_full  = D(h)          # what the ablated network actually decodes
    p_torus = D(h~)         # what it would decode if it were exactly on the intact torus
    dp_off  = p_full - p_torus      # off-manifold (residual) contribution

Predictions under test (Speaker 2 / user):
  (1) p_torus is much less dispersed than p_full  -> projecting back onto the torus collapses
      the wandering => off-manifold activity is the mechanism.
  (2) |dp_off| explains most of the PGC-ablated wandering.
  (3) |dp_off| correlates with distance from the intact manifold.

Plus the battery of checks:
  (a) theta1/theta2 winding numbers + angular velocity per condition
  (b) per-timestep correlation of decode error with distance-to-torus
  (c) cross-validated decoder trained ON ablated activity (info present vs collapsed)
  (d) decoder-weight & outgoing-recurrent-weight norms: PGC vs structural grid vs other
  (e) project-back-onto-torus-before-decoding  (== p_torus dispersion, prediction 1)
  (f) systematic drift vs diffusion: mean decoded-velocity error vs cross-trial variance growth

Usage:  python code/aarav_torus_residual_decomposition.py --seeds 0 1 2 ... --device cuda
"""
from __future__ import annotations
import argparse, copy, json, os, sys
import numpy as np
import torch
from scipy.linalg import cho_factor, cho_solve

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
_REPO = os.path.dirname(_HERE)

from aarav_crossing_population_correlation import load_model, load_classes
from aarav_activity_space_ablation import make_random_walk_inputs
from multi_seed_predictive_analysis import zero_unit_weights_in_place
from visualize import compute_ratemaps
from toroidal_structure_analysis import (build_torus_basis, project_states_to_torus,
                                         identify_toroidal_cells)


# ----------------------------------------------------------------------------- helpers
def gyr(p):
    """Radius of gyration of a decoded path set p[T,B,2] (mean over agents of sqrt mean sq dev)."""
    c = p.mean(0, keepdims=True)
    return float(np.sqrt(((p - c) ** 2).sum(2).mean(0)).mean())


def make_decoder(model, place_cells, device):
    """D: full grid activity [.,Ng] -> decoded position [.,2] via trained readout + top-3 cells."""
    Wd = model.decoder.weight.detach().to(device)              # [Np, Ng]

    def D(g_np):
        g = torch.as_tensor(g_np, dtype=torch.float32, device=device)
        flat = g.reshape(-1, g.shape[-1])
        pc = flat @ Wd.T                                       # [N, Np]
        pos = place_cells.get_nearest_cell_pos(pc[None]).squeeze(0)  # expects [.,.,Np]
        return pos.reshape(*g_np.shape[:-1], 2).detach().cpu().numpy()
    return D


def build_manifold_lookup(g_int, th1, th2, nbins):
    """Bin intact activity by (theta1,theta2); return lookup M[nb,nb,Ng] (empty bins -> nearest)."""
    Ng = g_int.shape[-1]
    flat = g_int.reshape(-1, Ng)
    b1 = np.clip(((th1.reshape(-1) + np.pi) / (2 * np.pi) * nbins).astype(int), 0, nbins - 1)
    b2 = np.clip(((th2.reshape(-1) + np.pi) / (2 * np.pi) * nbins).astype(int), 0, nbins - 1)
    M = np.zeros((nbins, nbins, Ng), np.float32)
    cnt = np.zeros((nbins, nbins), np.int64)
    np.add.at(M, (b1, b2), flat)
    np.add.at(cnt, (b1, b2), 1)
    filled = cnt > 0
    M[filled] /= cnt[filled][:, None]
    # fill empty bins from nearest filled bin (toroidal L1 over the angle grid)
    if not filled.all():
        fi = np.argwhere(filled)
        for (i, j) in np.argwhere(~filled):
            d1 = np.minimum(np.abs(fi[:, 0] - i), nbins - np.abs(fi[:, 0] - i))
            d2 = np.minimum(np.abs(fi[:, 1] - j), nbins - np.abs(fi[:, 1] - j))
            k = np.argmin(d1 + d2)
            M[i, j] = M[fi[k, 0], fi[k, 1]]
    return M


def reconstruct(M, th1, th2, nbins):
    """h~ = m(theta1,theta2): nearest-bin intact-manifold activity at given angles. -> [T,B,Ng]."""
    T, B = th1.shape
    b1 = np.clip(((th1 + np.pi) / (2 * np.pi) * nbins).astype(int), 0, nbins - 1)
    b2 = np.clip(((th2 + np.pi) / (2 * np.pi) * nbins).astype(int), 0, nbins - 1)
    return M[b1, b2]                                            # [T,B,Ng]


def winding_and_angvel(th):
    """Per-trajectory winding number (net turns) and mean |angular velocity| (rad/step)."""
    T, B = th.shape
    wind, av = [], []
    for b in range(B):
        u = np.unwrap(th[:, b])
        wind.append((u[-1] - u[0]) / (2 * np.pi))
        av.append(np.mean(np.abs(np.diff(u))))
    return np.array(wind), np.array(av)


def phase_path(theta1, theta2, true, k1, k2):
    """Analytic torus-only position path: integrate dtheta through the lattice, anchored at the
    true start. This is the periodicity-free 'what the torus angles say the path is'. -> [T,B,2]."""
    T, B = theta1.shape
    K = 2 * np.pi * np.array([[k1[0], k1[1]], [k2[0], k2[1]]], float)
    Kinv = np.linalg.inv(K)
    out = np.zeros((T, B, 2))
    for b in range(B):
        t1 = np.unwrap(theta1[:, b]); t2 = np.unwrap(theta2[:, b])
        dpos = np.stack([np.diff(t1), np.diff(t2)], 1) @ Kinv.T
        out[0, b] = true[0, b]
        out[1:, b] = true[0, b] + np.cumsum(dpos, 0)
    return out


def ridge_cv(X, ys, n_folds=3, lam=1e2, seed=0):
    """Cross-validated ridge X->each target in ys (dict name->Y). Shares the Cholesky factor of
    (X'X + lam I) across targets. Returns {name: predictions[N,*]} stacked in original order."""
    N = X.shape[0]
    idx = np.random.default_rng(seed).permutation(N)
    folds = np.array_split(idx, n_folds)
    Xc = X - X.mean(0, keepdims=True)
    preds = {nm: np.zeros_like(Y) for nm, Y in ys.items()}
    for f in range(n_folds):
        te = folds[f]; tr = np.concatenate([folds[g] for g in range(n_folds) if g != f])
        Xtr, Xte = Xc[tr], Xc[te]
        A = Xtr.T @ Xtr + lam * np.eye(Xtr.shape[1], dtype=np.float32)
        cf = cho_factor(A, lower=True)
        for nm, Y in ys.items():
            mu = Y[tr].mean(0, keepdims=True)
            W = cho_solve(cf, Xtr.T @ (Y[tr] - mu))
            preds[nm][te] = Xte @ W + mu
    return preds


def pos_r2(pred, y):
    sse = ((pred - y) ** 2).sum(); sst = ((y - y.mean(0, keepdims=True)) ** 2).sum()
    return float(1 - sse / sst), float(np.median(np.sqrt(((pred - y) ** 2).sum(1))) * 100.0)


# ----------------------------------------------------------------------------- per seed
def run_seed(seed, args, device):
    ckpt = os.path.join(_REPO, args.model_root, f"Seed {seed}", "most_recent_model.pth")
    gpath = os.path.join(_REPO, args.analysis_root, f"Seed {seed}", args.analysis_subdir, "gridness_data.npz")
    out_dir = os.path.join(_REPO, args.analysis_root, f"Seed {seed}", args.analysis_subdir, "aarav_torus_residual")
    os.makedirs(out_dir, exist_ok=True)
    model, place_cells, traj_gen, opt, Ng, Np = load_model(ckpt, device, args.seq_len)
    classes = load_classes(gpath, args.gridness_threshold, args.min_shift_cm)
    predictive = classes["predictive"]; grid_all = classes["grid_all"]
    structural = np.setdiff1d(grid_all, predictive); all_units = np.arange(Ng)
    other = np.setdiff1d(all_units, grid_all)
    N = int(predictive.size); rng = np.random.default_rng(7 + seed)

    # ---- torus basis on the detected coherent module (same pipeline as torus_video)
    rate_maps, _, _, _ = compute_ratemaps(model, traj_gen, opt, res=args.res, n_avg=12,
                                          Ng=grid_all.size, idxs=grid_all)
    rate_maps = np.asarray(rate_maps, float)
    det = identify_toroidal_cells(rate_maps, grid_all, opt.box_width, out_dir, embed_mode="umap")
    tor = np.asarray(det.units, int)
    loc = np.array([{int(g): i for i, g in enumerate(grid_all)}[int(u)] for u in tor])
    basis = build_torus_basis(rate_maps[loc], np.arange(loc.size), opt.box_width)
    basis.units = tor
    mod = tor                                                  # module unit indices (into Ng)
    print(f"seed {seed}: module={tor.size} N(pred)={N} grid={grid_all.size}", flush=True)

    # ---- test trajectories
    inputs, pos = make_random_walk_inputs(traj_gen, args.test_trajectories, args.seq_len, 777 + seed)
    true = pos.detach().cpu().numpy()                          # [T,B,2]
    T, B = true.shape[:2]
    D = make_decoder(model, place_cells, device)

    # ---- intact activity + manifold lookup
    def states(units):
        m = model if units is None else copy.deepcopy(model)
        if units is not None:
            zero_unit_weights_in_place(m, list(units))
        with torch.no_grad():
            g = m.g(inputs).detach().cpu().numpy()             # [T,B,Ng]
        if units is not None:
            del m
        return g

    g_int = states(None)
    pj = project_states_to_torus(g_int, basis, (1.0, 0.35), T, B)
    M = build_manifold_lookup(g_int, pj.theta1, pj.theta2, args.nbins)

    conds = {"Intact": None, "Predictive-ablated": predictive, "Grid-ablated": grid_all,
             "Random-ablated (N)": rng.choice(all_units, N, replace=False),
             "Structural-ablated (N)": rng.choice(structural, min(N, structural.size), replace=False)}

    res = {"seed": seed, "module": int(tor.size), "N_predictive": N, "n_grid": int(grid_all.size)}
    save = {"true": true.astype(np.float32)}
    for name, u in conds.items():
        g = states(u)
        p = project_states_to_torus(g, basis, (1.0, 0.35), T, B)
        h_t = reconstruct(M, p.theta1, p.theta2, args.nbins)   # intact-manifold reconstruction
        p_full = D(g)                                          # [T,B,2]
        p_torus = D(h_t)
        p_phase = phase_path(p.theta1, p.theta2, true, basis.k1, basis.k2)  # periodicity-free torus path
        dp_off = p_full - p_torus
        # distance from intact manifold (module units = where the torus lives; + full pop)
        d_mod = np.linalg.norm((g - h_t)[:, :, mod], axis=2)   # [T,B]
        err_full = np.linalg.norm(p_full - true, axis=2)       # [T,B]
        err_torus = np.linalg.norm(p_torus - true, axis=2)
        w1, av1 = winding_and_angvel(p.theta1)
        w2, av2 = winding_and_angvel(p.theta2)
        # drift vs diffusion
        v_dec = np.diff(p_full, axis=0); v_true = np.diff(true, axis=0)
        drift_bias = float(np.linalg.norm((v_dec - v_true).mean((0, 1))) * 100.0)      # systematic cm/step
        vel_err = float(np.abs(np.linalg.norm(v_dec - v_true, axis=2)).mean() * 100.0)  # total cm/step
        err_var_t = ((p_full - true) - (p_full - true).mean(1, keepdims=True))         # cross-trial
        cross_var = (err_var_t ** 2).sum(2).mean(1)                                    # [T] m^2
        diff_slope = float(np.polyfit(np.arange(T), cross_var, 1)[0])                  # m^2/step
        # correlations
        fin = np.isfinite(d_mod.reshape(-1))
        c_off_dist = float(np.corrcoef(np.linalg.norm(dp_off, axis=2).reshape(-1)[fin],
                                       d_mod.reshape(-1)[fin])[0, 1])
        c_err_dist = float(np.corrcoef(err_full.reshape(-1)[fin], d_mod.reshape(-1)[fin])[0, 1])

        res[name] = {
            "n_units": int(0 if u is None else len(u)),
            "gyr_full": gyr(p_full), "gyr_torus": gyr(p_torus), "gyr_dp_off": gyr(dp_off),
            "gyr_phase": gyr(p_phase),
            "wander_collapse_frac": float(1 - gyr(p_torus) / (gyr(p_full) + 1e-9)),
            "err_full_end_m": float(err_full[-1].mean()), "err_torus_end_m": float(err_torus[-1].mean()),
            "err_full_mean_m": float(err_full.mean()), "err_torus_mean_m": float(err_torus.mean()),
            "manifold_dist_mod_mean": float(np.nanmean(d_mod)),
            "corr_off_vs_manifolddist": c_off_dist,
            "corr_err_vs_manifolddist": c_err_dist,
            "winding1_mean_abs": float(np.mean(np.abs(w1))), "winding2_mean_abs": float(np.mean(np.abs(w2))),
            "angvel1_mean": float(np.mean(av1)), "angvel2_mean": float(np.mean(av2)),
            "drift_bias_cm_step": drift_bias, "vel_err_cm_step": vel_err,
            "diffusion_slope_m2_step": diff_slope,
        }
        if name in ("Intact", "Predictive-ablated", "Grid-ablated"):
            save[name.replace("-", "_").replace(" ", "")] = np.stack(
                [p_full, p_torus]).astype(np.float32)          # [2,T,B,2]
        print(f"  {name:24s} gyr_full={res[name]['gyr_full']:.2f} gyr_phase={res[name]['gyr_phase']:.2f} "
              f"err_full={res[name]['err_full_mean_m']:.2f} drift={res[name]['drift_bias_cm_step']:.1f} "
              f"vel_err={res[name]['vel_err_cm_step']:.1f} diff_slope={res[name]['diffusion_slope_m2_step']:.3f} "
              f"wind1={res[name]['winding1_mean_abs']:.2f} corr(err,d)={c_err_dist:.2f}", flush=True)

    # ---- (c) cross-validated decoder trained ON each activity set (info present vs collapsed)
    # Two readouts: direct g->position (linear decodability) and faithful g->placecells->top3
    # (mirrors the real decoder). If a freshly trained decoder recovers position, the wandering is
    # decoder distribution shift; if it cannot, spatial information has genuinely collapsed.
    res["decoder_recovery"] = {}
    yt = true.reshape(-1, 2).astype(np.float32)
    pc_true = place_cells.get_activation(torch.as_tensor(true, dtype=torch.float32, device=device)
                                         ).detach().cpu().numpy().reshape(-1, Np).astype(np.float32)
    for name in conds:
        g = states(conds[name]); X = g.reshape(-1, Ng).astype(np.float32)
        pr = ridge_cv(X, {"pos": yt, "pc": pc_true}, seed=seed)
        r2, med = pos_r2(pr["pos"], yt)
        pc_pos = place_cells.get_nearest_cell_pos(
            torch.as_tensor(pr["pc"], device=device)[None]).squeeze(0).detach().cpu().numpy()
        pc_med = float(np.median(np.sqrt(((pc_pos - yt) ** 2).sum(1))) * 100.0)
        res["decoder_recovery"][name] = {"cv_R2": r2, "cv_median_err_cm": med, "cv_pcdecode_err_cm": pc_med}
        print(f"  [decoder-recovery] {name:24s} R2={r2:.3f} pos_med={med:5.1f}cm pcdecode_med={pc_med:5.1f}cm",
              flush=True)

    # ---- (d) weight norms: decoder columns + outgoing recurrent columns, by class
    Wd = model.decoder.weight.detach().cpu().numpy()           # [Np,Ng]
    Whh = model.RNN.weight_hh_l0.detach().cpu().numpy()        # [Ng,Ng] (row=to, col=from)
    dec_norm = np.linalg.norm(Wd, axis=0)                      # per unit (incoming readout weight)
    out_norm = np.linalg.norm(Whh, axis=0)                     # per unit (outgoing recurrent)
    res["weight_norms"] = {}
    for cname, idx in [("predictive", predictive), ("structural_grid", structural),
                       ("other_nongrid", other)]:
        res["weight_norms"][cname] = {
            "n": int(idx.size),
            "decoder_norm_mean": float(dec_norm[idx].mean()),
            "outrec_norm_mean": float(out_norm[idx].mean())}
    print(f"  [weights] decoder norm pred={res['weight_norms']['predictive']['decoder_norm_mean']:.3f} "
          f"struct={res['weight_norms']['structural_grid']['decoder_norm_mean']:.3f} | "
          f"outrec pred={res['weight_norms']['predictive']['outrec_norm_mean']:.3f} "
          f"struct={res['weight_norms']['structural_grid']['outrec_norm_mean']:.3f}", flush=True)

    with open(os.path.join(out_dir, "torus_residual_stats.json"), "w") as f:
        json.dump(res, f, indent=2)
    np.savez_compressed(os.path.join(out_dir, "decomp_paths.npz"), **save)
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
    ap.add_argument("--nbins", type=int, default=32)
    ap.add_argument("--res", type=int, default=40)
    args = ap.parse_args()
    device = args.device if torch.cuda.is_available() else "cpu"
    for s in args.seeds:
        run_seed(s, args, device)


if __name__ == "__main__":
    main()
