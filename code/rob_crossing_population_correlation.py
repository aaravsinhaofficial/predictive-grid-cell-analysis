#!/usr/bin/env python3
"""Crossing-trajectory population-correlation analysis (Rob's X-crossings + William's binning).

Probes PATH / HEADING dependence of predictive grid units in a trained path-integrating RNN.

Two pipelines, designed to tell the same story:

  (A) Rob's designed X-crossings.
      Two straight "arms" are forced through the SAME physical location at the crossing
      step, but approach it with a controlled heading separation Delta-theta. We run the
      RNN on both arms and, AT the crossing, measure the population-activity correlation
      between the two arms for each unit class (predictive / standard-grid / retrospective
      / count-matched-random). Sweeping Delta-theta gives a "path-dependence curve":
      same position, more different heading -> more decorrelation. Predictive units should
      decorrelate the most.

  (B) William's same-bin random-walk method.
      Run many random walks, bin activity by location, and for visit-pairs that land in the
      same spatial bin compute the population correlation as a function of the MOVEMENT
      (velocity-vector) difference between the two visits. Same logic, observational rather
      than designed.

BUG FIX (the "movement difference ~1e-8" artifact):
  By construction the two arms occupy the SAME position at the crossing, so a "movement
  difference" computed as a POSITION difference there is ~0 (1e-8 float noise). The correct,
  non-zero axis is the HEADING / velocity-vector difference. This script computes and reports
  BOTH (heading/velocity diff, which is large; and position diff, which is ~0) so the
  artifact is explicit and corrected. Arms also run at ~2 cm/step to match the training
  speed distribution (the older code used ~7.5 cm/step, ~4x too fast / out of distribution).

Usage:
  python code/rob_crossing_population_correlation.py --seeds 0 1 --device cuda
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from types import SimpleNamespace
from typing import Dict, List, Sequence

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
_REPO = os.path.dirname(_HERE)

from model import RNN
from place_cells import PlaceCells
from trajectory_generator import TrajectoryGenerator
from predictive_retrospective_ablation import classify_from_scores


# --------------------------------------------------------------------------------------
# Model / class loading
# --------------------------------------------------------------------------------------
def build_options(Ng: int, Np: int, device: str, seq_len: int) -> SimpleNamespace:
    return SimpleNamespace(
        batch_size=200, sequence_length=int(seq_len), Ng=int(Ng), Np=int(Np), velocity_dim=2,
        place_cell_rf=0.12, surround_scale=2.0, activation="relu", weight_decay=1e-4,
        DoG=True, periodic=False, box_width=2.2, box_height=2.2, learning_rate=1e-4,
        device=device,
        # trajectory params matching the training distribution
        trajectory_dt=0.02, trajectory_speed_scale=1.0, trajectory_border_region=0.03,
        trajectory_wall_slowdown=0.25, trajectory_wall_turn_scale=1.0, trajectory_style="random_walk",
    )


def load_model(ckpt_path: str, device: str, seq_len: int):
    raw = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if isinstance(raw, dict) and "model_state_dict" in raw:
        state = raw["model_state_dict"]
    elif isinstance(raw, dict) and "state_dict" in raw:
        state = raw["state_dict"]
    else:
        state = raw
    Ng, Np = state["encoder.weight"].shape
    opt = build_options(Ng, Np, device, seq_len)
    place_cells = PlaceCells(opt)            # NOTE: seeds np.random.seed(0) internally (deterministic centers)
    model = RNN(opt, place_cells).to(device)
    model.load_state_dict(state)
    model.eval()
    traj_gen = TrajectoryGenerator(opt, place_cells)
    return model, place_cells, traj_gen, opt, int(Ng), int(Np)


def load_classes(gridness_path: str, gridness_threshold: float, min_shift_cm: float) -> Dict[str, np.ndarray]:
    d = np.load(gridness_path)
    cls = classify_from_scores(
        {"lag_cm": d["lag_cm"], "scores_60": d["scores_60"]},
        min_shift_cm=float(min_shift_cm), gridness_threshold=float(gridness_threshold),
    )
    pred = np.asarray(cls["predictive"], dtype=int)
    retro = np.asarray(cls["retrospective"], dtype=int)
    normal = np.asarray(cls["normal"], dtype=int)
    grid = np.unique(np.concatenate([pred, retro, normal])) if (pred.size + retro.size + normal.size) else np.array([], int)
    return {"predictive": pred, "retrospective": retro, "standard_grid": normal, "grid_all": grid,
            "Ng": int(d["scores_60"].shape[1])}


# --------------------------------------------------------------------------------------
# Correlation helpers
# --------------------------------------------------------------------------------------
def rowwise_pearson(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Pearson correlation between matching rows of A and B ([P, U] each) -> [P]."""
    A = np.asarray(A, dtype=np.float64); B = np.asarray(B, dtype=np.float64)
    A = A - A.mean(axis=1, keepdims=True)
    B = B - B.mean(axis=1, keepdims=True)
    num = (A * B).sum(axis=1)
    den = np.sqrt((A * A).sum(axis=1) * (B * B).sum(axis=1))
    out = np.full(A.shape[0], np.nan)
    m = den > 1e-12
    out[m] = num[m] / den[m]
    return out


def class_corr(gA: np.ndarray, gB: np.ndarray, units: np.ndarray) -> np.ndarray:
    if units.size == 0:
        return np.full(gA.shape[0], np.nan)
    return rowwise_pearson(gA[:, units], gB[:, units])


def random_matched_corr(gA: np.ndarray, gB: np.ndarray, Ng: int, k: int,
                        rng: np.random.Generator, n_draws: int = 5) -> np.ndarray:
    """Mean per-pair correlation over random unit subsets of size k (matched to predictive count)."""
    if k <= 0:
        return np.full(gA.shape[0], np.nan)
    acc = np.zeros(gA.shape[0]); cnt = np.zeros(gA.shape[0])
    for _ in range(n_draws):
        idx = rng.choice(Ng, size=min(k, Ng), replace=False)
        c = rowwise_pearson(gA[:, idx], gB[:, idx])
        good = np.isfinite(c)
        acc[good] += c[good]; cnt[good] += 1
    out = np.full(gA.shape[0], np.nan)
    out[cnt > 0] = acc[cnt > 0] / cnt[cnt > 0]
    return out


def mean_ci(vals: np.ndarray):
    v = np.asarray(vals, dtype=float); v = v[np.isfinite(v)]
    if v.size == 0:
        return float("nan"), float("nan"), float("nan"), 0
    m = float(v.mean())
    sem = float(v.std(ddof=1) / np.sqrt(v.size)) if v.size > 1 else 0.0
    return m, m - 1.96 * sem, m + 1.96 * sem, int(v.size)


# --------------------------------------------------------------------------------------
# (A) Rob's designed X-crossings
# --------------------------------------------------------------------------------------
def build_x_crossings(place_cells, opt, n_pairs: int, seq_len: int, cross_t: int, step_m: float,
                      center_halfspan: float, rng: np.random.Generator, device: str):
    """Build 2*n_pairs straight arms, paired so each pair crosses at the same point.

    Returns dict with torch inputs (v,init), numpy pos [T,B,2], pair arrays, and QC quantities.
    Layout of batch dim: [pair0_A, pair0_B, pair1_A, pair1_B, ...].
    """
    T = int(seq_len)
    cross_t = int(np.clip(cross_t, 1, T - 2))
    centers = rng.uniform(-center_halfspan, center_halfspan, size=(n_pairs, 2)).astype(np.float64)
    theta_a = rng.uniform(-np.pi, np.pi, size=n_pairs)
    # heading separation uniform in [10,170] deg, random sign
    sep = np.deg2rad(rng.uniform(10.0, 170.0, size=n_pairs)) * rng.choice([-1.0, 1.0], size=n_pairs)
    theta_b = theta_a + sep

    headings = np.empty(2 * n_pairs)
    headings[0::2] = theta_a
    headings[1::2] = theta_b
    centers_b = np.repeat(centers, 2, axis=0)                  # [B,2]
    dirs = np.stack([np.cos(headings), np.sin(headings)], axis=1)  # [B,2]
    pair_ids = np.repeat(np.arange(n_pairs), 2)

    # offsets[t] = (t - cross_t) * step_m ; zero at crossing -> matched position
    offsets = (np.arange(T) - cross_t).astype(np.float64) * float(step_m)
    pos = centers_b[None] + offsets[:, None, None] * dirs[None]   # [T,B,2]
    # safety clip (centers_halfspan + arm length kept inside box, but guard anyway)
    lim = opt.box_width / 2 - 0.02
    pos = np.clip(pos, -lim, lim)

    vel = np.zeros_like(pos)
    vel[1:] = pos[1:] - pos[:-1]
    vel[0] = vel[1]

    init_pos = torch.tensor(pos[0], dtype=torch.float32, device=device)[:, None, :]   # [B,1,2]
    init_actv = place_cells.get_activation(init_pos).squeeze(1)                         # [B,Np]
    v_t = torch.tensor(vel, dtype=torch.float32, device=device)                         # [T,B,2]

    sep_deg = np.abs(np.rad2deg(np.arctan2(np.sin(sep), np.cos(sep))))
    # QC quantities at the crossing step, per pair
    a_idx = np.where(pair_ids % 1 == 0)[0]  # all
    iA = np.arange(0, 2 * n_pairs, 2); iB = np.arange(1, 2 * n_pairs, 2)
    pos_diff_cross = np.linalg.norm(pos[cross_t, iA] - pos[cross_t, iB], axis=1)      # ~0 by design
    vel_diff_cross = np.linalg.norm(vel[cross_t, iA] - vel[cross_t, iB], axis=1)      # = 2*step*sin(dtheta/2)
    return {
        "inputs": (v_t, init_actv), "pos": pos.astype(np.float32), "cross_t": cross_t,
        "iA": iA, "iB": iB, "sep_deg": sep_deg,
        "pos_diff_cross": pos_diff_cross, "vel_diff_cross": vel_diff_cross,
        "step_m": float(step_m),
    }


def run_rob_crossings(model, place_cells, opt, classes, device, rng,
                      n_pairs=3000, seq_len=20, step_m=0.02, center_halfspan=0.7,
                      n_angle_bins=9, batch_pairs=2000):
    cross_t = seq_len // 2
    sep_all, corr = [], {k: [] for k in ["predictive", "standard_grid", "retrospective", "grid_all", "random_matched"]}
    pos_diffs, vel_diffs = [], []
    done = 0
    while done < n_pairs:
        npair = min(batch_pairs, n_pairs - done)
        bx = build_x_crossings(place_cells, opt, npair, seq_len, cross_t, step_m, center_halfspan, rng, device)
        with torch.no_grad():
            g = model.g(bx["inputs"]).detach().cpu().numpy()       # [T,B,Ng]
        gt = g[bx["cross_t"]]                                       # [B,Ng]
        gA = gt[bx["iA"]]; gB = gt[bx["iB"]]                        # [npair,Ng]
        sep_all.append(bx["sep_deg"]); pos_diffs.append(bx["pos_diff_cross"]); vel_diffs.append(bx["vel_diff_cross"])
        for name in ["predictive", "standard_grid", "retrospective", "grid_all"]:
            corr[name].append(class_corr(gA, gB, classes[name]))
        corr["random_matched"].append(
            random_matched_corr(gA, gB, classes["Ng"], classes["predictive"].size, rng))
        done += npair
    sep_all = np.concatenate(sep_all)
    pos_diffs = np.concatenate(pos_diffs); vel_diffs = np.concatenate(vel_diffs)
    for k in corr:
        corr[k] = np.concatenate(corr[k])

    # bin by heading separation
    edges = np.linspace(0, 180, n_angle_bins + 1)
    centers_deg = 0.5 * (edges[:-1] + edges[1:])
    which = np.clip(np.digitize(sep_all, edges) - 1, 0, n_angle_bins - 1)
    curves = {}
    for name in corr:
        m = np.full(n_angle_bins, np.nan); lo = np.full(n_angle_bins, np.nan); hi = np.full(n_angle_bins, np.nan)
        for b in range(n_angle_bins):
            mm, ll, hh, _ = mean_ci(corr[name][which == b])
            m[b], lo[b], hi[b] = mm, ll, hh
        curves[name] = {"mean": m.tolist(), "ci_lo": lo.tolist(), "ci_hi": hi.tolist()}
    qc = {
        "angle_bin_centers_deg": centers_deg.tolist(),
        "pos_diff_cross_mean_m": float(np.mean(pos_diffs)), "pos_diff_cross_max_m": float(np.max(pos_diffs)),
        "vel_diff_cross_mean_m": float(np.mean(vel_diffs)), "vel_diff_cross_min_m": float(np.min(vel_diffs)),
        "vel_diff_vs_angle": {  # mean velocity-difference per angle bin (the corrected movement axis)
            "centers_deg": centers_deg.tolist(),
            "vel_diff_m": [float(np.mean(vel_diffs[which == b])) if np.any(which == b) else float("nan")
                           for b in range(n_angle_bins)],
            "pos_diff_m": [float(np.mean(pos_diffs[which == b])) if np.any(which == b) else float("nan")
                           for b in range(n_angle_bins)],
        },
        "n_pairs": int(sep_all.size), "step_m": float(step_m), "seq_len": int(seq_len),
    }
    return {"angle_bin_centers_deg": centers_deg.tolist(), "curves": curves, "qc": qc}


# --------------------------------------------------------------------------------------
# (B) William's same-bin random-walk method
# --------------------------------------------------------------------------------------
def run_william_same_bin(model, place_cells, traj_gen, opt, classes, device, rng,
                         n_batches=40, batch_size=200, seq_len=20, res=44,
                         pairs_per_bin=12, max_pairs=60000, n_move_bins=9):
    box = opt.box_width
    pos_list, vel_list, g_list = [], [], []
    for _ in range(n_batches):
        traj_gen.options.sequence_length = seq_len
        inputs, pos, _ = traj_gen.get_test_batch(batch_size=batch_size)
        with torch.no_grad():
            g = model.g(inputs).detach().cpu().numpy()              # [T,B,Ng]
        v = inputs[0].detach().cpu().numpy()                        # [T,B,2] velocity (displacement/step)
        pos_np = pos.detach().cpu().numpy()                         # [T,B,2]
        # drop t=0 (velocity copied / boundary); use interior timesteps
        pos_list.append(pos_np[1:].reshape(-1, 2))
        vel_list.append(v[1:].reshape(-1, 2))
        g_list.append(g[1:].reshape(-1, g.shape[-1]))
    P = np.concatenate(pos_list); V = np.concatenate(vel_list); G = np.concatenate(g_list)

    # spatial bins
    bx = np.clip(((P[:, 0] + box / 2) / box * res).astype(int), 0, res - 1)
    by = np.clip(((P[:, 1] + box / 2) / box * res).astype(int), 0, res - 1)
    bin_id = bx * res + by
    order = np.argsort(bin_id, kind="stable")
    bin_id_s = bin_id[order]
    # group sample indices by bin
    uniq, starts = np.unique(bin_id_s, return_index=True)
    starts = list(starts) + [len(bin_id_s)]

    pair_i, pair_j = [], []
    for gi in range(len(uniq)):
        members = order[starts[gi]:starts[gi + 1]]
        if members.size < 2:
            continue
        npick = min(pairs_per_bin, members.size * (members.size - 1) // 2)
        a = rng.choice(members, size=npick, replace=True)
        b = rng.choice(members, size=npick, replace=True)
        keep = a != b
        pair_i.append(a[keep]); pair_j.append(b[keep])
        if sum(len(x) for x in pair_i) >= max_pairs:
            break
    if not pair_i:
        return None
    I = np.concatenate(pair_i)[:max_pairs]; J = np.concatenate(pair_j)[:max_pairs]

    # movement (velocity-vector) difference and heading difference
    move_diff = np.linalg.norm(V[I] - V[J], axis=1)
    hi = np.arctan2(V[I, 1], V[I, 0]); hj = np.arctan2(V[J, 1], V[J, 0])
    head_diff = np.abs(np.rad2deg(np.arctan2(np.sin(hi - hj), np.cos(hi - hj))))

    gA = G[I]; gB = G[J]
    corr = {}
    for name in ["predictive", "standard_grid", "retrospective", "grid_all"]:
        corr[name] = class_corr(gA, gB, classes[name])
    corr["random_matched"] = random_matched_corr(gA, gB, classes["Ng"], classes["predictive"].size, rng)

    # bin by movement difference (quantile edges for balanced bins)
    qs = np.linspace(0, 1, n_move_bins + 1)
    edges = np.unique(np.quantile(move_diff, qs))
    if edges.size < 3:
        edges = np.linspace(move_diff.min(), move_diff.max() + 1e-9, n_move_bins + 1)
    nb = edges.size - 1
    which = np.clip(np.digitize(move_diff, edges) - 1, 0, nb - 1)
    centers_move = 0.5 * (edges[:-1] + edges[1:])
    curves = {}
    for name in corr:
        m = np.full(nb, np.nan); lo = np.full(nb, np.nan); hi_ = np.full(nb, np.nan)
        for b in range(nb):
            mm, ll, hh, _ = mean_ci(corr[name][which == b])
            m[b], lo[b], hi_[b] = mm, ll, hh
        curves[name] = {"mean": m.tolist(), "ci_lo": lo.tolist(), "ci_hi": hi_.tolist()}
    return {
        "move_bin_centers_m": centers_move.tolist(),
        "head_diff_per_move_bin_deg": [float(np.mean(head_diff[which == b])) if np.any(which == b) else float("nan")
                                       for b in range(nb)],
        "curves": curves, "n_pairs": int(I.size), "res": int(res),
    }


# --------------------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------------------
def parse_args():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1])
    ap.add_argument("--model_root", default="Models/Single agent path integration")
    ap.add_argument("--analysis_root", default="analysis_outputs/Single agent path integration")
    ap.add_argument("--analysis_subdir", default="spatial_shift_allunits")
    ap.add_argument("--out_subdir", default="rob_crossing_population_corr")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--gridness_threshold", type=float, default=0.2)
    ap.add_argument("--min_shift_cm", type=float, default=5.0)
    ap.add_argument("--seq_len", type=int, default=20)
    ap.add_argument("--step_cm", type=float, default=2.0, help="per-step displacement (cm); ~matches training")
    ap.add_argument("--n_pairs", type=int, default=4000)
    ap.add_argument("--sb_n_batches", type=int, default=40)
    ap.add_argument("--sb_batch_size", type=int, default=200)
    ap.add_argument("--seed", type=int, default=12345)
    return ap.parse_args()


def main():
    args = parse_args()
    device = args.device if torch.cuda.is_available() else "cpu"
    for s in args.seeds:
        t0 = time.time()
        ckpt = os.path.join(_REPO, args.model_root, f"Seed {s}", "most_recent_model.pth")
        gpath = os.path.join(_REPO, args.analysis_root, f"Seed {s}", args.analysis_subdir, "gridness_data.npz")
        out_dir = os.path.join(_REPO, args.analysis_root, f"Seed {s}", args.analysis_subdir, args.out_subdir)
        os.makedirs(out_dir, exist_ok=True)
        print(f"\n=== Seed {s} ===\n  ckpt={ckpt}\n  gridness={gpath}", flush=True)
        model, place_cells, traj_gen, opt, Ng, Np = load_model(ckpt, device, args.seq_len)
        classes = load_classes(gpath, args.gridness_threshold, args.min_shift_cm)
        rng = np.random.default_rng(args.seed + s)
        print(f"  classes: predictive={classes['predictive'].size} standard_grid={classes['standard_grid'].size} "
              f"retro={classes['retrospective'].size} grid_all={classes['grid_all'].size} Ng={classes['Ng']}", flush=True)

        rob = run_rob_crossings(model, place_cells, opt, classes, device, rng,
                                n_pairs=args.n_pairs, seq_len=args.seq_len, step_m=args.step_cm / 100.0)
        print(f"  [Rob X-crossings] n_pairs={rob['qc']['n_pairs']}  "
              f"pos_diff@cross(mean/max)={rob['qc']['pos_diff_cross_mean_m']:.2e}/{rob['qc']['pos_diff_cross_max_m']:.2e} m  "
              f"vel_diff@cross(mean/min)={rob['qc']['vel_diff_cross_mean_m']:.4f}/{rob['qc']['vel_diff_cross_min_m']:.4f} m", flush=True)

        will = run_william_same_bin(model, place_cells, traj_gen, opt, classes, device, rng,
                                    n_batches=args.sb_n_batches, batch_size=args.sb_batch_size, seq_len=args.seq_len)
        if will is not None:
            print(f"  [William same-bin] n_pairs={will['n_pairs']}", flush=True)

        result = {
            "seed": s, "checkpoint": ckpt, "gridness_path": gpath,
            "class_counts": {k: int(classes[k].size) for k in ["predictive", "standard_grid", "retrospective", "grid_all"]},
            "Ng": classes["Ng"], "params": vars(args),
            "rob_x_crossings": rob, "william_same_bin": will,
        }
        with open(os.path.join(out_dir, "crossing_population_corr.json"), "w") as f:
            json.dump(result, f, indent=2)
        print(f"  wrote {out_dir}/crossing_population_corr.json  ({time.time()-t0:.1f}s)", flush=True)


if __name__ == "__main__":
    main()
