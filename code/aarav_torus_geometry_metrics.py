#!/usr/bin/env python3
"""Latent-geometry metrics per ablation condition (10 seeds): torus occupancy/coverage and
off-manifold fraction. For the toroidal-ablation figure quantification panels.

Per seed: build the torus basis on the coherent grid module, project each condition's activity, and
compute, per condition (Intact / Predictive / Grid / Random):
  - occupancy_spread : circular spread of theta1 around the torus, 1 - |mean e^{i theta1}|
                       (per trajectory, then averaged). low = clumped/compact, high = spread.
  - off_manifold     : mean kNN distance of the module activity to the intact module-activity
                       cloud (PCA), normalised so intact = 1. high = activity left the torus.
                       (Grid-ablated is degenerate: the module is fully ablated -> trivially far.)

Usage:  python code/aarav_torus_geometry_metrics.py --seeds 0 1 2 ... --device cuda
"""
from __future__ import annotations
import argparse, copy, json, os, sys
import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
_REPO = os.path.dirname(_HERE)
from aarav_crossing_population_correlation import load_model, load_classes
from aarav_activity_space_ablation import make_random_walk_inputs
from multi_seed_predictive_analysis import zero_unit_weights_in_place
from visualize import compute_ratemaps
from toroidal_structure_analysis import build_torus_basis, project_states_to_torus, identify_toroidal_cells

SCRATCH = "/tmp/claude-1000/-home-ec2-user-predictive-grid-cell-analysis/ba8b77e9-dcb6-4bd8-864f-7e091e2bda68/scratchpad"


def occupancy_spread(theta1):
    """Per-trajectory circular spread of theta1 (1 - resultant length); mean over trajectories."""
    R = np.abs(np.exp(1j * theta1).mean(0))                  # [B] resultant length per trajectory
    return float((1 - R).mean())


def off_manifold(mod_int, mod_c, k=5, npc=10, nref=1500, seed=0):
    """Mean kNN distance of condition module-activity to the intact cloud (PCA), normalised by the
    intact within-cloud kNN distance."""
    rng = np.random.default_rng(seed)
    pca = PCA(n_components=min(npc, mod_int.shape[1])).fit(mod_int)
    Zi = pca.transform(mod_int)
    ref = Zi[rng.choice(Zi.shape[0], min(nref, Zi.shape[0]), replace=False)]
    base = NearestNeighbors(n_neighbors=k + 1).fit(ref).kneighbors(ref)[0][:, 1:].mean()
    Zc = pca.transform(mod_c)
    d = NearestNeighbors(n_neighbors=k).fit(ref).kneighbors(Zc)[0].mean()
    return float(d / (base + 1e-9))


def run_seed(seed, args, device):
    base = os.path.join(_REPO, "analysis_outputs/Single agent path integration")
    out_dir = os.path.join(base, f"Seed {seed}", "spatial_shift_allunits", "aarav_torus_geometry")
    os.makedirs(out_dir, exist_ok=True)
    model, pc, tg, opt, Ng, Np = load_model(f"Models/Single agent path integration/Seed {seed}/most_recent_model.pth", device, args.seq_len)
    cl = load_classes(f"analysis_outputs/Single agent path integration/Seed {seed}/spatial_shift_allunits/gridness_data.npz", 0.2, 5.0)
    pred, grid = cl["predictive"], cl["grid_all"]; allu = np.arange(Ng); N = pred.size
    rng = np.random.default_rng(7 + seed); randN = rng.choice(allu, N, replace=False)

    rate_maps, _, _, _ = compute_ratemaps(model, tg, opt, res=40, n_avg=10, Ng=grid.size, idxs=grid)
    rate_maps = np.asarray(rate_maps, float)
    det = identify_toroidal_cells(rate_maps, grid, opt.box_width, SCRATCH, embed_mode="umap")
    tor = np.asarray(det.units, int)
    loc = np.array([{int(g): i for i, g in enumerate(grid)}[int(u)] for u in tor])
    basis = build_torus_basis(rate_maps[loc], np.arange(loc.size), opt.box_width); basis.units = tor

    inputs, pos = make_random_walk_inputs(tg, args.n_traj, args.seq_len, 777 + seed)
    T, B = args.seq_len, args.n_traj

    def proj(units):
        m = model if units is None else copy.deepcopy(model)
        if units is not None:
            zero_unit_weights_in_place(m, list(units))
        with torch.no_grad():
            g = m.g(inputs).detach().cpu().numpy()
        p = project_states_to_torus(g, basis, (1.0, 0.35), T, B)
        return p.theta1, g[:, :, tor].reshape(-1, tor.size)

    conds = {"Intact": None, "Predictive-ablated": pred, "Grid-ablated": grid, "Random-ablated (N)": randN}
    th = {}; mod = {}
    for name, u in conds.items():
        th[name], mod[name] = proj(u)
    res = {"seed": seed, "module": int(tor.size), "N_predictive": N}
    for name in conds:
        res[name] = {"occupancy_spread": occupancy_spread(th[name]),
                     "off_manifold": off_manifold(mod["Intact"], mod[name], seed=seed)}
        print(f"  seed {seed} {name:22s} occupancy_spread={res[name]['occupancy_spread']:.3f} "
              f"off_manifold={res[name]['off_manifold']:.2f}", flush=True)
    with open(os.path.join(out_dir, "torus_geometry_stats.json"), "w") as f:
        json.dump(res, f, indent=2)
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", nargs="+", type=int, default=list(range(10)))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seq_len", type=int, default=40)
    ap.add_argument("--n_traj", type=int, default=64)
    args = ap.parse_args()
    device = args.device if torch.cuda.is_available() else "cpu"
    for s in args.seeds:
        print(f"=== Seed {s} ===", flush=True)
        run_seed(s, args, device)


if __name__ == "__main__":
    main()
