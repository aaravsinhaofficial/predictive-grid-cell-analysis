#!/usr/bin/env python3
"""Does the PGC-ablation phenotype depend on the PGC *definition*?  Stall vs collapse,
and what the PGC input actually does to the grid population (propulsion vs repulsion).

Motivation (Slack, 2026-08-22): with the MEC-style definition (a PGC may NOT be a standard
grid unit at zero shift) the ablated grid population looks like it "collapses back to the
original state" rather than "stalling", and PGCs look like a "repulsion force".  With our
liberal definition (peak-shift only; most PGCs are also zero-shift grid units) we saw a
"freeze" (population autocorrelation stays high, torus angle stops moving).

NO re-classification: everything is derived from the cached all-4096-unit spatial-shift
score curves in  <seed>/spatial_shift_allunits/gridness_data.npz  (+ band_scores.npz).

Definitions (per network):
  liberal  (ours)      : best gridness >= 0.2  at best shift >= +5 cm            (classify_from_scores)
  conservative (MEC)   : zero-shift gridness < 0.4  (NOT a standard grid unit)
                         AND best gridness >= 0.3 at best shift >= +5 cm
  standard grid (read-out population, = Redman's `grid_ids`) : zero-shift gridness >= 0.4
  band                 : top 5 % band score (paper's 95th percentile)
Retrospective classes are the mirror image (shift <= -5 cm).

Conditions per seed: intact, pred/retro x {liberal, conservative}, band, count-matched random
(all non-dead units; and Redman-style non-grid pool), count-matched structural (standard grid
units that are not predictive/retrospective under either definition).  Lesion = full removal
(encoder row, recurrent row+col, input row, decoder col); for the read-out population's
dynamics this is identical to Redman's recurrent-only lesion (ablated units feed nothing back).

Per condition, on the surviving read-out population R = standard_grid \\ ablated, comparing the
ablated run with the INTACT run on the SAME units and SAME trajectories:
  * population autocorrelation  cos(g_t, g_{t+tau})  (median over (t,b), Redman's statistic; + mean)
  * stall index      : step size ||g_{t+1}-g_t||  relative to intact
  * collapse index   : radius from the intact cloud centre (PC1-2 plane, PC3-4 plane, full space)
                       relative to intact
  * return index     : distance from the trajectory's own first state, relative to intact
  * tracking         : cos(g_c(t), g_0(t))
  * velocity decomposition of the ablated step onto the intact tangent / radial / from-start axes
  * kNN position decode from the read-out population alone (intact-cloud lookup): error vs truth,
    distance to the start position, off-manifold residual
  * place-cell decoder error (full lesion; and Redman's recurrent-only lesion for reference)
One-step drive decomposition (intact network, no ablation): the exact one-step effect of a
unit-set's recurrent input on the read-out population, projected onto the intact motion
direction (propulsion) and the outward radial direction (repulsion / bump sustain).

Usage:  python code/aarav_definition_dynamics.py --seeds 0 1 2 ... --device cuda:1
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
from multi_seed_predictive_analysis import zero_unit_weights_in_place
from predictive_retrospective_ablation import classify_from_scores

LAGS = list(range(1, 30))           # Redman: max_shift 30 -> lags 1..29 on T=40
T_FIT = 10                          # Redman drops the first 10 steps before fitting PCA
LATE = (20, 40)                     # late window for scalar summaries


# ----------------------------------------------------------------------------- classes
def derive_classes(gpath, bpath, z_thr=0.4, s_thr=0.3, min_shift=5.0, lib_thr=0.2, band_pct=95.0):
    d = np.load(gpath)
    S = np.asarray(d["scores_60"], float); lag = np.asarray(d["lag_cm"], float)
    Ng = S.shape[1]
    zi = int(np.argmin(np.abs(lag)))
    gs0 = np.nan_to_num(S[zi], nan=1.0)                          # nan -> treated as grid (excluded from PGC)
    Sf = np.nan_to_num(S, nan=-9.0)
    bi = Sf.argmax(0); best = Sf[bi, np.arange(Ng)]; bcm = lag[bi]
    dead = (np.nan_to_num(S, nan=0).max(0) == 0) & (np.nan_to_num(S, nan=0).min(0) == 0)
    nondead = np.where(~dead)[0]
    lib = classify_from_scores({"lag_cm": lag, "scores_60": S}, min_shift, lib_thr)
    std_grid = np.where((gs0 >= z_thr) & ~dead)[0]
    con_pred = np.where((gs0 < z_thr) & (best >= s_thr) & (bcm >= min_shift) & ~dead)[0]
    con_retro = np.where((gs0 < z_thr) & (best >= s_thr) & (bcm <= -min_shift) & ~dead)[0]
    b = np.load(bpath)
    bs = np.asarray(b["band_scores"], float)
    ok = np.isfinite(bs) & ~dead
    cut = np.percentile(bs[ok], band_pct)
    band = np.where(ok & (bs >= cut))[0]
    band90 = np.asarray(b["band_units"], int) if "band_units" in b.files else band
    return dict(Ng=Ng, nondead=nondead, std_grid=std_grid, band90=band90,
                pred_lib=np.asarray(lib["predictive"], int), retro_lib=np.asarray(lib["retrospective"], int),
                normal_lib=np.asarray(lib["normal"], int),
                pred_con=con_pred, retro_con=con_retro, band=band,
                gs0=gs0, best=best, best_cm=bcm, dead=dead)


# ----------------------------------------------------------------------------- helpers
def _norm(x, axis=-1):
    return np.linalg.norm(x, axis=axis) + 1e-9


def cos_rows(a, b):
    return (a * b).sum(-1) / (_norm(a) * _norm(b))


def autocorr_curves(X):
    """X: [T,B,D] -> median & mean over (t,b) of cos(x_t, x_{t+tau}) for tau in LAGS."""
    T = X.shape[0]
    med, mean = [], []
    for tau in LAGS:
        if tau >= T:
            med.append(np.nan); mean.append(np.nan); continue
        c = cos_rows(X[:-tau], X[tau:]).ravel()
        med.append(float(np.median(c))); mean.append(float(np.mean(c)))
    return med, mean


def pca_basis(X0, n=6):
    D = X0.shape[-1]
    Z = X0[T_FIT:].reshape(-1, D)
    mu = Z.mean(0)
    Zc = Z - mu
    # economical SVD via covariance eigendecomposition (D x D), D <= ~2000
    C = Zc.T @ Zc / Zc.shape[0]
    w, V = np.linalg.eigh(C)
    order = np.argsort(w)[::-1]
    w = w[order]; V = V[:, order]
    return mu, V[:, :n], w[:n] / w.sum()


def dyn_metrics(Xc, X0):
    """Xc, X0: [T,B,D] ablated / intact activity of the SAME read-out units on the SAME trajectories."""
    T, B, D = X0.shape
    out = {}
    ac_med_c, ac_mean_c = autocorr_curves(Xc)
    ac_med_0, ac_mean_0 = autocorr_curves(X0)
    out["autocorr_med"] = ac_med_c; out["autocorr_mean"] = ac_mean_c
    out["autocorr_med_intact_same_units"] = ac_med_0; out["autocorr_mean_intact_same_units"] = ac_mean_0

    mu, PC, ve = pca_basis(X0)
    out["pca_var_explained_6"] = ve.tolist()
    P0 = (X0 - mu) @ PC; Pc = (Xc - mu) @ PC                    # [T,B,6]
    med_t = lambda a: np.median(a, axis=1).tolist()             # per-t median over trajectories
    r0_12, rc_12 = _norm(P0[..., :2]), _norm(Pc[..., :2])
    r0_34, rc_34 = _norm(P0[..., 2:4]), _norm(Pc[..., 2:4])
    r0_full, rc_full = _norm(X0 - mu), _norm(Xc - mu)
    out["radius_pc12_ratio"] = med_t(rc_12 / r0_12)
    out["radius_pc34_ratio"] = med_t(rc_34 / r0_34)
    out["radius_full_ratio"] = med_t(rc_full / r0_full)
    out["norm_ratio"] = med_t(_norm(Xc) / _norm(X0))
    out["tracking_cos"] = med_t(cos_rows(Xc, X0))
    d0_init, dc_init = _norm(X0 - X0[:1]), _norm(Xc - Xc[:1])     # distance from own first state
    out["dist_from_start_intact"] = med_t(d0_init); out["dist_from_start"] = med_t(dc_init)
    out["dist_from_start_ratio"] = med_t((dc_init + 1e-9) / (d0_init + 1e-9))
    out["dist_to_intact_init_over_intact_motion"] = med_t(_norm(Xc - X0[:1]) / (d0_init + 1e-9))   # <1: closer to the start state than the intact state is
    out["dist_to_intact_same_t_over_intact_motion"] = med_t(_norm(Xc - X0) / (d0_init + 1e-9))
    # tracking in PC space: distance between ablated and intact state, normalised by intact cloud radius
    out["pc6_dist_to_intact_over_radius"] = med_t(_norm(Pc - P0) / np.median(_norm(P0[T_FIT:])))
    # steps
    s0 = _norm(X0[1:] - X0[:-1]); sc = _norm(Xc[1:] - Xc[:-1])
    out["step_ratio"] = med_t(sc / s0)
    out["step_intact"] = med_t(s0); out["step_ablated"] = med_t(sc)
    # velocity decomposition of the ablated step (t -> t+1)
    dc = Xc[1:] - Xc[:-1]; d0 = X0[1:] - X0[:-1]
    u_tan = d0 / _norm(d0)[..., None]
    u_rad = (Xc[:-1] - mu) / _norm(Xc[:-1] - mu)[..., None]
    u_ini = (Xc[1:-1] - Xc[:1]); u_ini = u_ini / _norm(u_ini)[..., None]
    n0 = _norm(d0)
    out["vel_tangent"] = med_t((dc * u_tan).sum(-1) / n0)
    out["vel_radial"] = med_t((dc * u_rad).sum(-1) / n0)
    out["vel_from_start"] = med_t((dc[1:] * u_ini).sum(-1) / n0[1:])
    out["vel_cos_with_intact"] = med_t(cos_rows(dc, d0))
    # radial velocity in PC space (bump amplitude change), normalised by intact PC step
    dPc = Pc[1:] - Pc[:-1]; dP0 = P0[1:] - P0[:-1]
    u_rad_pc = Pc[:-1] / _norm(Pc[:-1])[..., None]
    out["vel_radial_pc6"] = med_t((dPc * u_rad_pc).sum(-1) / _norm(dP0))
    out["vel_radial_pc6_intact"] = med_t((dP0 * (P0[:-1] / _norm(P0[:-1])[..., None])).sum(-1) / _norm(dP0))
    # late-window scalars
    a, b = LATE
    late = lambda key: float(np.nanmedian(np.asarray(out[key])[a:b]))
    out["late"] = {k: late(k) for k in ("radius_pc12_ratio", "radius_pc34_ratio", "radius_full_ratio",
                                        "norm_ratio", "tracking_cos", "dist_from_start_ratio",
                                        "dist_to_intact_init_over_intact_motion", "dist_to_intact_same_t_over_intact_motion",
                                        "pc6_dist_to_intact_over_radius", "step_ratio",
                                        "vel_tangent", "vel_radial", "vel_from_start", "vel_cos_with_intact",
                                        "vel_radial_pc6")}
    out["late"]["autocorr_med_lag20"] = ac_med_c[LAGS.index(20)]
    out["late"]["autocorr_med_lag20_intact_same_units"] = ac_med_0[LAGS.index(20)]
    out["late"]["autocorr_med_lag10"] = ac_med_c[LAGS.index(10)]
    out["late"]["autocorr_med_lag10_intact_same_units"] = ac_med_0[LAGS.index(10)]
    return out


def knn_decode(Xc, X0, cloud_X, cloud_pos, pos, device, n_pc=20, k=5):
    """kNN position decode from the read-out population using an intact cloud (activity -> position).
    Xc/X0: [T,B,D]; cloud_X: [N,D]; cloud_pos: [N,2]; pos: [T,B,2] (true positions, metres)."""
    mu = cloud_X.mean(0)
    C = (cloud_X - mu).T @ (cloud_X - mu) / cloud_X.shape[0]
    w, V = np.linalg.eigh(C); V = V[:, np.argsort(w)[::-1][:n_pc]]
    cz = torch.as_tensor((cloud_X - mu) @ V, dtype=torch.float32, device=device)
    cp = torch.as_tensor(cloud_pos, dtype=torch.float32, device=device)
    T, B, D = Xc.shape
    res = {}
    for name, X in (("ablated", Xc), ("intact", X0)):
        q = torch.as_tensor((X.reshape(-1, D) - mu) @ V, dtype=torch.float32, device=device)
        dists, idx = [], []
        for i in range(0, q.shape[0], 2048):
            dd = torch.cdist(q[i:i + 2048], cz)
            v, j = dd.topk(k, dim=1, largest=False)
            dists.append(v); idx.append(j)
        dists = torch.cat(dists); idx = torch.cat(idx)
        phat = cp[idx].mean(1).reshape(T, B, 2).cpu().numpy()
        nnd = dists.mean(1).reshape(T, B).cpu().numpy()
        err = np.linalg.norm(phat - pos, axis=-1)
        d_start = np.linalg.norm(phat - pos[:1], axis=-1)
        true_d_start = np.linalg.norm(pos - pos[:1], axis=-1)
        res[name] = {"err_m": np.median(err, 1).tolist(), "dist_to_start_m": np.median(d_start, 1).tolist(),
                     "true_dist_to_start_m": np.median(true_d_start, 1).tolist(),
                     "nn_dist": np.median(nnd, 1).tolist(),
                     "late_err_m": float(np.median(err[LATE[0]:LATE[1]])),
                     "late_dist_to_start_m": float(np.median(d_start[LATE[0]:LATE[1]])),
                     "late_true_dist_to_start_m": float(np.median(true_d_start[LATE[0]:LATE[1]])),
                     "late_nn_dist": float(np.median(nnd[LATE[0]:LATE[1]]))}
    res["late_nn_dist_ratio"] = res["ablated"]["late_nn_dist"] / (res["intact"]["late_nn_dist"] + 1e-9)
    return res


def drive_decomposition(model, g0, v, h0, P, R, device, mode="set"):
    """Exact one-step effect of unit-set P's recurrent input on read-out units R, in the INTACT network.
    g0: [T,B,Ng] intact activity; v: [T,B,2]; h0: [B,Ng] encoder state feeding step 0."""
    Whh = model.RNN.weight_hh_l0.detach(); Wih = model.RNN.weight_ih_l0.detach()
    g = torch.as_tensor(g0, device=device); vv = torch.as_tensor(v, device=device)
    prev = torch.cat([torch.as_tensor(h0, device=device)[None], g[:-1]], 0)      # [T,B,Ng]
    a = prev @ Whh.T + vv @ Wih.T
    Rt = torch.as_tensor(R, device=device, dtype=torch.long)
    if mode == "velocity":                     # one-step effect of the velocity input
        cP = vv @ Wih.T
    elif mode == "all_recurrent":              # one-step effect of ALL recurrent input
        cP = prev @ Whh.T
    else:
        Pt = torch.as_tensor(P, device=device, dtype=torch.long)
        cP = prev[:, :, Pt] @ Whh[:, Pt].T
    g_wo = torch.relu(a - cP)
    delta = (g - g_wo)[:, :, Rt].cpu().numpy()                                    # [T,B,|R|]
    X0 = g0[:, :, R]
    mu, PC, _ = pca_basis(X0)
    d0 = X0[1:] - X0[:-1]                       # realised motion t-1 -> t  (aligned with delta[1:])
    dl = delta[1:]
    n0 = _norm(d0)
    u_tan = d0 / n0[..., None]
    u_rad = (X0[1:] - mu) / _norm(X0[1:] - mu)[..., None]
    u_ini = X0[1:] - X0[:1]; u_ini = u_ini / _norm(u_ini)[..., None]
    tan = (dl * u_tan).sum(-1) / n0; rad = (dl * u_rad).sum(-1) / n0; ini = (dl * u_ini).sum(-1) / n0
    cosd = cos_rows(dl, d0); mag = _norm(dl) / n0
    # in PC-6 space: radial component of the drive (bump-amplitude direction)
    Pd = dl @ PC; P0 = (X0[1:] - mu) @ PC
    rad_pc = (Pd * (P0 / _norm(P0)[..., None])).sum(-1) / _norm((X0[1:] - X0[:-1]) @ PC)
    sl = slice(max(T_FIT - 1, 0), None)
    med = lambda x: float(np.median(x[sl]))
    return {"n_set": int(len(P)) if mode == "set" else -1, "n_readout": int(len(R)),
            "tangent": med(tan), "radial": med(rad), "from_start": med(ini), "cos_with_motion": med(cosd),
            "magnitude": med(mag), "radial_pc6": med(rad_pc),
            "tangent_curve": np.median(tan, 1).tolist(), "radial_curve": np.median(rad, 1).tolist(),
            "cos_curve": np.median(cosd, 1).tolist()}


# ----------------------------------------------------------------------------- main per seed
def run_seed(seed, args, device):
    t0 = time.time()
    ckpt = os.path.join(_REPO, args.model_root, f"Seed {seed}", "most_recent_model.pth")
    base = os.path.join(_REPO, args.analysis_root, f"Seed {seed}", args.analysis_subdir)
    out_dir = os.path.join(base, "aarav_definition_dynamics"); os.makedirs(out_dir, exist_ok=True)
    model, place_cells, traj_gen, opt, Ng, Np = load_model(ckpt, device, args.seq_len)
    cl = derive_classes(os.path.join(base, "gridness_data.npz"), os.path.join(base, "band_cells", "band_scores.npz"),
                        z_thr=args.z_thr, s_thr=args.s_thr, min_shift=args.min_shift_cm, lib_thr=args.lib_thr)
    G = cl["std_grid"]; nondead = cl["nondead"]
    rng = np.random.default_rng(11 + seed)
    targeted_union = np.unique(np.concatenate([cl["pred_lib"], cl["retro_lib"], cl["pred_con"], cl["retro_con"], cl["band"]]))
    structural_pool = np.setdiff1d(G, np.unique(np.concatenate([cl["pred_lib"], cl["retro_lib"], cl["pred_con"], cl["retro_con"]])))
    nongrid_pool = np.setdiff1d(nondead, G)                               # Redman's random pool
    nL, nC = int(cl["pred_lib"].size), int(cl["pred_con"].size)

    conds = {"intact": np.array([], int),
             "pred_lib": cl["pred_lib"], "retro_lib": cl["retro_lib"],
             "pred_con": cl["pred_con"], "retro_con": cl["retro_con"],
             "band": cl["band"], "band90": cl["band90"]}
    for k in range(args.random_draws):
        conds[f"rand_nd_matchlib_{k}"] = rng.choice(nondead, nL, replace=False)
        conds[f"rand_nd_matchcon_{k}"] = rng.choice(nondead, nC, replace=False)
        conds[f"rand_nongrid_matchcon_{k}"] = rng.choice(nongrid_pool, nC, replace=False)
        conds[f"struct_matchlib_{k}"] = rng.choice(structural_pool, min(nL, structural_pool.size), replace=False)
        conds[f"struct_matchcon_{k}"] = rng.choice(structural_pool, min(nC, structural_pool.size), replace=False)

    # trajectories (identical across conditions)
    traj_gen.options.trajectory_style = "random_walk"
    inputs_rw, pos_rw = make_random_walk_inputs(traj_gen, args.test_trajectories, args.seq_len, 777 + seed)
    traj_gen.options.trajectory_style = "straight"
    inputs_st, pos_st = make_random_walk_inputs(traj_gen, args.test_trajectories, args.seq_len, 777 + seed)
    traj_gen.options.trajectory_style = "random_walk"
    cloud_in = [make_random_walk_inputs(traj_gen, args.test_trajectories, args.seq_len, 5000 + 100 * seed + i)
                for i in range(args.cloud_batches)]

    def run(m, inputs):
        with torch.no_grad():
            g = m.g(inputs)
            pred = place_cells.get_nearest_cell_pos(m.decoder(g)).cpu().numpy()
        return g.detach().cpu().numpy(), pred

    def ablated_model(units, recurrent_only=False):
        m = copy.deepcopy(model)
        if units.size:
            if recurrent_only:
                idx = torch.as_tensor(units, device=device, dtype=torch.long)
                with torch.no_grad():
                    m.RNN.weight_hh_l0[idx, :] = 0; m.RNN.weight_hh_l0[:, idx] = 0
            else:
                zero_unit_weights_in_place(m, list(units))
        return m

    g0_rw, dec0_rw = run(model, inputs_rw)
    g0_st, dec0_st = run(model, inputs_st)
    pos_rw_np = pos_rw.cpu().numpy(); pos_st_np = pos_st.cpu().numpy()
    cloud_g = np.concatenate([run(model, ci[0])[0][T_FIT // 2:] for ci in cloud_in], 0)   # [Tc,B,Ng] (t>=5)
    cloud_p = np.concatenate([ci[1].cpu().numpy()[T_FIT // 2:] for ci in cloud_in], 0)
    cloud_g = cloud_g.reshape(-1, Ng); cloud_p = cloud_p.reshape(-1, 2)
    with torch.no_grad():
        h0_rw = model.encoder(inputs_rw[1]).cpu().numpy()
    v_rw = inputs_rw[0].cpu().numpy()

    results = {"seed": seed, "Ng": Ng, "n_nondead": int(nondead.size), "n_std_grid": int(G.size),
               "thresholds": {"z_thr": args.z_thr, "s_thr": args.s_thr, "min_shift_cm": args.min_shift_cm,
                              "lib_thr": args.lib_thr, "band_pct": 95.0},
               "class_sizes": {k: int(cl[k].size) for k in ("pred_lib", "retro_lib", "normal_lib", "pred_con", "retro_con", "band", "band90")},
               "overlaps": {"pred_con_in_pred_lib": int(np.intersect1d(cl["pred_con"], cl["pred_lib"]).size),
                            "pred_lib_in_std_grid": int(np.intersect1d(cl["pred_lib"], G).size),
                            "pred_con_in_std_grid": int(np.intersect1d(cl["pred_con"], G).size),
                            "band_in_pred_lib": int(np.intersect1d(cl["band"], cl["pred_lib"]).size),
                            "band_in_pred_con": int(np.intersect1d(cl["band"], cl["pred_con"]).size),
                            "band_in_std_grid": int(np.intersect1d(cl["band"], G).size)},
               "pred_lib_gs0_median": float(np.median(cl["gs0"][cl["pred_lib"]])),
               "pred_con_gs0_median": float(np.median(cl["gs0"][cl["pred_con"]])) if nC else float("nan"),
               "pred_lib_bestcm_median": float(np.median(cl["best_cm"][cl["pred_lib"]])),
               "pred_con_bestcm_median": float(np.median(cl["best_cm"][cl["pred_con"]])) if nC else float("nan"),
               "conditions": {}}
    print(f"[seed {seed}] nondead={nondead.size} std_grid={G.size} pred_lib={nL} pred_con={nC} "
          f"retro_lib={cl['retro_lib'].size} retro_con={cl['retro_con'].size} band={cl['band'].size} "
          f"| pred_lib∩std_grid={results['overlaps']['pred_lib_in_std_grid']}", flush=True)

    for name, units in conds.items():
        R = np.setdiff1d(G, units)
        if name == "intact":
            gc_rw, dec_rw = g0_rw, dec0_rw; gc_st, dec_st = g0_st, dec0_st
        else:
            m = ablated_model(units)
            gc_rw, dec_rw = run(m, inputs_rw); gc_st, dec_st = run(m, inputs_st)
            del m
        rec = {"n_ablated": int(units.size), "n_readout": int(R.size)}
        for traj, gc, g0, dec, pos in (("random_walk", gc_rw, g0_rw, dec_rw, pos_rw_np),
                                       ("straight", gc_st, g0_st, dec_st, pos_st_np)):
            dm = dyn_metrics(gc[:, :, R], g0[:, :, R])
            dm["decoder_err_m"] = np.median(np.linalg.norm(dec - pos, axis=-1), 1).tolist()
            dm["decoder_err_m_mean"] = float(np.mean(np.linalg.norm(dec - pos, axis=-1)))
            dm["decoder_dist_to_start_m"] = np.median(np.linalg.norm(dec - pos[:1], axis=-1), 1).tolist()
            if traj == "random_walk":
                dm["knn"] = knn_decode(gc[:, :, R], g0[:, :, R], cloud_g[:, R], cloud_p, pos, device)
            rec[traj] = dm
        if name in ("pred_lib", "pred_con", "retro_lib", "retro_con", "band", "band90") or name.startswith(("rand_nd_matchcon", "rand_nongrid_matchcon", "struct_matchcon")):
            m = ablated_model(units, recurrent_only=True)
            _, dec_rec = run(m, inputs_rw); del m
            rec["decoder_err_m_mean_recurrent_only_lesion"] = float(np.mean(np.linalg.norm(dec_rec - pos_rw_np, axis=-1)))
        results["conditions"][name] = rec
        L = rec["random_walk"]["late"]
        print(f"  {name:24s} abl={units.size:4d} R={R.size:4d} | ac20 {L['autocorr_med_lag20']:.3f} (intact same units "
              f"{L['autocorr_med_lag20_intact_same_units']:.3f}) step {L['step_ratio']:.2f} r12 {L['radius_pc12_ratio']:.2f} "
              f"r34 {L['radius_pc34_ratio']:.2f} rfull {L['radius_full_ratio']:.2f} dstart {L['dist_from_start_ratio']:.2f} "
              f"track {L['tracking_cos']:.2f} vtan {L['vel_tangent']:+.2f} vrad {L['vel_radial']:+.2f} "
              f"| knn err {rec['random_walk']['knn']['ablated']['late_err_m']:.2f} m, to-start "
              f"{rec['random_walk']['knn']['ablated']['late_dist_to_start_m']:.2f} m (true {rec['random_walk']['knn']['ablated']['late_true_dist_to_start_m']:.2f}) "
              f"| dec err {rec['random_walk']['decoder_err_m_mean']*100:.0f} cm", flush=True)

    # one-step drive decomposition (intact network)
    results["drive"] = {}
    drive_sets = {"pred_lib": cl["pred_lib"], "pred_con": cl["pred_con"], "retro_lib": cl["retro_lib"],
                  "retro_con": cl["retro_con"], "band": cl["band"], "band90": cl["band90"], "normal_lib": cl["normal_lib"],
                  "pred_lib_minus_con": np.setdiff1d(cl["pred_lib"], cl["pred_con"]),
                  "pred_lib_in_std_grid": np.intersect1d(cl["pred_lib"], G)}
    for k in range(args.random_draws):
        drive_sets[f"rand_nd_matchlib_{k}"] = conds[f"rand_nd_matchlib_{k}"]
        drive_sets[f"rand_nd_matchcon_{k}"] = conds[f"rand_nd_matchcon_{k}"]
        drive_sets[f"struct_matchlib_{k}"] = conds[f"struct_matchlib_{k}"]
        drive_sets[f"struct_matchcon_{k}"] = conds[f"struct_matchcon_{k}"]
    for ref in ("velocity", "all_recurrent"):
        results["drive"][ref] = drive_decomposition(model, g0_rw, v_rw, h0_rw, np.array([], int), G, device, mode=ref)
        dd = results["drive"][ref]
        print(f"  drive {ref:22s}          | tangent {dd['tangent']:+.3f} radial {dd['radial']:+.3f} cos(motion) {dd['cos_with_motion']:+.3f} |delta|/|d0| {dd['magnitude']:.2f}", flush=True)
    for name, P in drive_sets.items():
        if P.size == 0:
            continue
        R = np.setdiff1d(G, P)
        results["drive"][name] = drive_decomposition(model, g0_rw, v_rw, h0_rw, P, R, device)
        dd = results["drive"][name]
        print(f"  drive {name:22s} n={P.size:4d} | tangent {dd['tangent']:+.3f} radial {dd['radial']:+.3f} "
              f"from-start {dd['from_start']:+.3f} cos(motion) {dd['cos_with_motion']:+.3f} |delta|/|d0| {dd['magnitude']:.2f} "
              f"radial_pc6 {dd['radial_pc6']:+.3f}", flush=True)

    # example trajectories in the intact PCA plane (seed-level visual), read-out = std grid minus each ablation set
    ex = {}
    for name in ("pred_lib", "pred_con", "retro_con", "band", "rand_nd_matchcon_0", "rand_nd_matchlib_0"):
        units = conds[name]; R = np.setdiff1d(G, units)
        mu, PC, _ = pca_basis(g0_rw[:, :, R])
        m = ablated_model(units); gc, _ = run(m, inputs_rw); del m
        ex[f"{name}_intact"] = ((g0_rw[:, :, R] - mu) @ PC[:, :4])[:, :12].astype(np.float32)
        ex[f"{name}_ablated"] = ((gc[:, :, R] - mu) @ PC[:, :4])[:, :12].astype(np.float32)
        ex[f"{name}_cloud"] = ((cloud_g[::7][:, R] - mu) @ PC[:, :4]).astype(np.float32)
    np.savez_compressed(os.path.join(out_dir, "example_pca_paths.npz"), **ex)
    np.savez_compressed(os.path.join(out_dir, "class_indices.npz"),
                        **{k: cl[k] for k in ("std_grid", "pred_lib", "retro_lib", "normal_lib", "pred_con", "retro_con", "band", "band90", "nondead")},
                        **{k: v for k, v in conds.items() if k.startswith(("rand", "struct"))})
    with open(os.path.join(out_dir, "definition_dynamics.json"), "w") as f:
        json.dump(results, f)
    print(f"[seed {seed}] done in {time.time() - t0:.0f}s -> {out_dir}", flush=True)
    return results


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", nargs="+", type=int, default=list(range(10)))
    ap.add_argument("--model_root", default="Models/Single agent path integration")
    ap.add_argument("--analysis_root", default="analysis_outputs/Single agent path integration")
    ap.add_argument("--analysis_subdir", default="spatial_shift_allunits")
    ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--seq_len", type=int, default=40)
    ap.add_argument("--test_trajectories", type=int, default=256)
    ap.add_argument("--cloud_batches", type=int, default=3)
    ap.add_argument("--random_draws", type=int, default=3)
    ap.add_argument("--z_thr", type=float, default=0.4, help="zero-shift gridness threshold for 'standard grid'")
    ap.add_argument("--s_thr", type=float, default=0.3, help="conservative peak-gridness threshold")
    ap.add_argument("--lib_thr", type=float, default=0.2, help="liberal peak-gridness threshold (ours)")
    ap.add_argument("--min_shift_cm", type=float, default=5.0)
    args = ap.parse_args()
    device = args.device if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)
    for s in args.seeds:
        run_seed(s, args, device)


if __name__ == "__main__":
    main()
