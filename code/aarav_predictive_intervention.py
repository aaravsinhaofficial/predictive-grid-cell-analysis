#!/usr/bin/env python3
"""Causal intervention: does FREEZING the predictive subspace stall the decoded position?

Instead of ablating (which zeros units and corrupts the readout), we FREEZE a set of units:
hold their activity at the trajectory's first-step value while the rest of the network runs
normally. Frozen units are still present (decoder reads them), so the network's decoded position
is a clean causal readout. If predictive cells *drive movement along the manifold*, freezing them
should make the decoded position fail to advance (it stalls / lags true position) MORE than
freezing a count-matched random set.

We replicate the RNN forward by hand (h_t = relu(W_ih v_t + W_hh h_{t-1}), no bias) so we can
clamp h_t[:, S] = frozen_value each step. Verified to match model.g() exactly when nothing is clamped.

Conditions (count-matched, N = #predictive): Intact, Freeze predictive, Freeze random,
Freeze structural-grid. Metrics per trajectory:
  decode_error_m         : ||decoded_end - true_end||              (low = tracked correctly)
  advance_ratio          : ||decoded_end - decoded_0|| / ||true_end - true_0||  (1 = advanced, <1 = lagged)

Usage:  python code/aarav_predictive_intervention.py --seeds 0 1 2 3 4 --device cuda
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
from aarav_activity_space_ablation import make_random_walk_inputs


def manual_forward(model, v, init_actv, clamp_idx=None, clamp_val=None):
    """Replicate model.g with optional clamping of units `clamp_idx` to `clamp_val` each step.
    v:[T,B,2], init_actv:[B,Np]. Returns g:[T,B,Ng]."""
    W_ih = model.RNN.weight_ih_l0            # [Ng, 2]
    W_hh = model.RNN.weight_hh_l0            # [Ng, Ng]
    h = model.encoder(init_actv)             # [B, Ng] = h_0
    T = v.shape[0]
    gs = []
    for t in range(T):
        h = torch.relu(v[t] @ W_ih.t() + h @ W_hh.t())
        if clamp_idx is not None:
            h = h.clone()
            h[:, clamp_idx] = clamp_val      # freeze these units
        gs.append(h)
    return torch.stack(gs, dim=0)            # [T, B, Ng]


def decoded_path(model, g):
    with torch.no_grad():
        dec = model.place_cells.get_nearest_cell_pos(model.decoder(g)).detach().cpu().numpy()  # [T,B,2]
    return dec


def run_seed(seed, args, device):
    ckpt = os.path.join(_REPO, args.model_root, f"Seed {seed}", "most_recent_model.pth")
    gpath = os.path.join(_REPO, args.analysis_root, f"Seed {seed}", args.analysis_subdir, "gridness_data.npz")
    out_dir = os.path.join(_REPO, args.analysis_root, f"Seed {seed}", args.analysis_subdir, "aarav_predictive_intervention")
    os.makedirs(out_dir, exist_ok=True)
    model, place_cells, traj_gen, opt, Ng, Np = load_model(ckpt, device, args.seq_len)
    classes = load_classes(gpath, args.gridness_threshold, args.min_shift_cm)
    predictive = classes["predictive"]; grid_all = classes["grid_all"]
    structural = np.setdiff1d(grid_all, predictive); all_units = np.arange(Ng)
    N = int(predictive.size); rng = np.random.default_rng(args.seed + seed)

    inputs, pos = make_random_walk_inputs(traj_gen, args.test_trajectories, args.seq_len, args.seed + seed + 777)
    v, init_actv = inputs
    pos_np = pos.detach().cpu().numpy()                         # [T,B,2]
    true_disp = np.linalg.norm(pos_np[-1] - pos_np[0], axis=1)

    # intact run (also the source of the freeze value = first-step activity)
    with torch.no_grad():
        g_intact = manual_forward(model, v, init_actv)         # [T,B,Ng]
        # sanity: matches model.g
        g_lib = model.g(inputs)
        max_dev = float((g_intact - g_lib).abs().max())
    frozen_full = g_intact[0]                                   # [B,Ng] first-step activity

    def metrics(g):
        dec = decoded_path(model, g)
        derr = np.linalg.norm(dec[-1] - pos_np[-1], axis=1)
        adv = np.linalg.norm(dec[-1] - dec[0], axis=1) / np.maximum(true_disp, 1e-9)
        return derr, adv

    derr_i, adv_i = metrics(g_intact)
    idx = torch.as_tensor(np.arange(Ng), device=device)  # placeholder
    conds = {"Intact": [None], "Freeze predictive (N)": [predictive],
             "Freeze random (N)": [rng.choice(all_units, N, replace=False) for _ in range(args.n_draws)],
             "Freeze structural (N)": [rng.choice(structural, min(N, structural.size), replace=False) for _ in range(args.n_draws)]}
    print(f"=== Seed {seed} === N={N}  manual-vs-lib max dev={max_dev:.1e}  "
          f"intact decodeErr={derr_i.mean():.3f}m advance={adv_i.mean():.2f}", flush=True)

    results = {"manual_match_max_dev": max_dev,
               "Intact": {"decode_error_m": float(derr_i.mean()), "advance_ratio": float(adv_i.mean())}}
    for name, usets in conds.items():
        if name == "Intact":
            continue
        de, ad = [], []
        for units in usets:
            ci = torch.as_tensor(np.asarray(units, dtype=int), device=device)
            cv = frozen_full[:, ci]
            with torch.no_grad():
                g = manual_forward(model, v, init_actv, clamp_idx=ci, clamp_val=cv)
            d, a = metrics(g)
            de.append(d); ad.append(a)
        de = np.mean(de, axis=0); ad = np.mean(ad, axis=0)
        results[name] = {"n_units": int(len(usets[0])), "decode_error_m": float(de.mean()),
                         "decode_error_std": float(de.std()), "advance_ratio": float(ad.mean()),
                         "advance_ratio_std": float(ad.std())}
        print(f"    {name:24s} decodeErr={de.mean():.3f}m  advance={ad.mean():.2f}", flush=True)

    out = {"seed": seed, "N_predictive": N, "conditions": results}
    with open(os.path.join(out_dir, "predictive_intervention.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(f"  wrote {out_dir}/predictive_intervention.json", flush=True)


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
    for s in args.seeds:
        run_seed(s, args, device)


if __name__ == "__main__":
    main()
