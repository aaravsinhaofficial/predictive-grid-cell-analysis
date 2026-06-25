#!/usr/bin/env python3
"""Is the PGC-ablated decoded 'wandering' actually JUMPS BETWEEN GRID-FIELD REPLICAS of one phase?

Rob's hypothesis: one location on the torus corresponds to MANY physical locations (the grid
fields of the module, i.e. the lattice of points spaced by the grid period). If the ablated
activity is frozen at ~one torus phase but the decoder reads a periodic code, the decoded position
may jump between the lattice replicas of that single phase -- so the 'different' wandering
locations are all lattice-equivalent (same phase, mod the grid lattice).

Test: fold each decoded position into ONE grid-lattice unit cell.
  u   = K_cyc @ p           (lattice coordinates, in cycles;  K_cyc rows = k1,k2 in cycles/m)
  node= round(u)            (which replica / lattice cell)
  frac= u - node            (within-cell phase, minimal image in [-0.5,0.5))
  p_within = A @ frac       (A = inv(K_cyc) = real-space lattice vectors)
If the wander is pure aliasing: within-cell phase is CONCENTRATED (high circular resultant R,
small within-cell spread) while the raw physical positions are spread over several periods.
  aliasing_index = 1 - within_cell_spread / raw_spread   (->1 = wander is lattice aliasing)

Usage:  python code/aarav_torus_aliasing.py --seeds 0 1 2 ... --device cuda
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
from visualize import compute_ratemaps
from toroidal_structure_analysis import estimate_lattice_vectors


def spread2(P):
    """Total spatial variance of a point set P[...,2] (mean squared distance to mean), m^2."""
    Q = P.reshape(-1, 2)
    return float(((Q - Q.mean(0)) ** 2).sum(1).mean())


def decode(model, place_cells, units, inputs):
    m = model if units is None else copy.deepcopy(model)
    if units is not None:
        zero_unit_weights_in_place(m, list(units))
    with torch.no_grad():
        p = m.place_cells.get_nearest_cell_pos(m.predict(inputs)).detach().cpu().numpy()
    if units is not None:
        del m
    return p                                                   # [T,B,2]


def lattice_fold(p, Kcyc):
    """Return within-cell phase position p_within, lattice-node position p_node, decoded phases."""
    A = np.linalg.inv(Kcyc)                                    # real-space lattice vectors (cols)
    u = p @ Kcyc.T                                             # lattice coords (cycles)
    node = np.round(u)
    frac = u - node                                            # in [-0.5, 0.5)
    p_within = frac @ A.T
    p_node = node @ A.T
    phase = 2 * np.pi * u                                      # radians
    return p_within, p_node, phase, A


def circ_R(phase_axis):
    """Circular resultant length of an angle array (1 = perfectly concentrated)."""
    z = np.exp(1j * phase_axis.reshape(-1))
    return float(np.abs(z.mean()))


def run_seed(seed, args, device):
    ckpt = os.path.join(_REPO, args.model_root, f"Seed {seed}", "most_recent_model.pth")
    gpath = os.path.join(_REPO, args.analysis_root, f"Seed {seed}", args.analysis_subdir, "gridness_data.npz")
    out_dir = os.path.join(_REPO, args.analysis_root, f"Seed {seed}", args.analysis_subdir, "aarav_torus_aliasing")
    os.makedirs(out_dir, exist_ok=True)
    model, place_cells, traj_gen, opt, Ng, Np = load_model(ckpt, device, args.seq_len)
    classes = load_classes(gpath, args.gridness_threshold, args.min_shift_cm)
    predictive = classes["predictive"]; grid_all = classes["grid_all"]
    structural = np.setdiff1d(grid_all, predictive); all_units = np.arange(Ng)
    N = int(predictive.size); rng = np.random.default_rng(7 + seed)

    # dominant grid lattice from the grid-cell population average rate map (FFT peak)
    rate_maps, _, _, _ = compute_ratemaps(model, traj_gen, opt, res=args.res, n_avg=12,
                                          Ng=grid_all.size, idxs=grid_all)
    rate_maps = np.asarray(rate_maps, float)
    k1, k2, angle, power = estimate_lattice_vectors(rate_maps, np.arange(grid_all.size), opt.box_width)
    Kcyc = np.array([k1, k2], float)                           # rows = k1,k2 (cycles/m)
    lam = float(np.mean([1.0 / np.linalg.norm(k1), 1.0 / np.linalg.norm(k2)]))  # grid spacing (m)

    inputs, pos = make_random_walk_inputs(traj_gen, args.test_trajectories, args.seq_len, 777 + seed)
    true = pos.detach().cpu().numpy()

    A = np.linalg.inv(Kcyc)

    def per_traj_alias(p):
        """WITHIN-trajectory test: for each agent, how much of its physical wander is between-cell
        (lattice aliasing of a concentrated phase) vs genuine within-cell phase movement."""
        T, B, _ = p.shape
        u = p @ Kcyc.T                                         # [T,B,2] cycles
        ph = 2 * np.pi * u
        Rs, aliases, raws, withins, jump_frac = [], [], [], [], []
        for b in range(B):
            pb = p[:, b]; phb = ph[:, b]
            raw = np.sqrt(((pb - pb.mean(0)) ** 2).sum(1).mean())
            if raw < 1e-3:
                continue
            mbar = np.angle(np.exp(1j * phb).mean(0))          # circular mean phase per axis
            cent = np.angle(np.exp(1j * (phb - mbar)))         # wrapped residual [-pi,pi)
            p_res = (cent / (2 * np.pi)) @ A.T                 # within-cell residual in meters
            within = np.sqrt((p_res ** 2).sum(1).mean())
            Rb = 0.5 * (np.abs(np.exp(1j * phb[:, 0]).mean()) + np.abs(np.exp(1j * phb[:, 1]).mean()))
            Rs.append(Rb); raws.append(raw); withins.append(within)
            aliases.append(1 - within / raw)
            # large physical steps that are near lattice translations (fold to ~0)
            step = np.diff(pb, axis=0)
            big = np.linalg.norm(step, axis=1) > 0.3
            if big.any():
                du = np.diff(u[:, b], axis=0)[big]
                resid = du - np.round(du)
                folded = np.linalg.norm(resid @ A.T, axis=1)   # residual step after folding (m)
                jump_frac.append(float((folded < 0.5 * lam).mean()))
        return (float(np.mean(Rs)), float(np.mean(aliases)), float(np.mean(raws)),
                float(np.mean(withins)), float(np.mean(jump_frac)) if jump_frac else float("nan"))

    conds = {"Intact": None, "Predictive-ablated": predictive, "Grid-ablated": grid_all,
             "Random-ablated (N)": rng.choice(all_units, N, replace=False)}
    res = {"seed": seed, "grid_spacing_m": lam, "k1": k1.tolist(), "k2": k2.tolist(),
           "N_predictive": N, "n_grid": int(grid_all.size)}
    save = {"true": true.astype(np.float32), "k1": k1, "k2": k2}
    for name, u in conds.items():
        p = decode(model, place_cells, u, inputs)
        # pooled (across-trajectory) reference
        raw_pool = np.sqrt(spread2(p)); within_pool = np.sqrt(spread2(lattice_fold(p, Kcyc)[0]))
        # within-trajectory (the correct aliasing test)
        Rt, alias_t, raw_t, within_t, jfrac = per_traj_alias(p)
        res[name] = {"n_units": int(0 if u is None else len(u)),
                     "pooled_raw_spread_m": raw_pool, "pooled_within_cell_m": within_pool,
                     "traj_phaseR": Rt, "traj_raw_spread_m": raw_t, "traj_within_cell_m": within_t,
                     "traj_aliasing_index": alias_t, "traj_raw_in_periods": raw_t / lam,
                     "frac_big_jumps_are_lattice": jfrac}
        if name in ("Intact", "Predictive-ablated", "Grid-ablated"):
            save[name.replace("-", "_").replace(" ", "")] = p.astype(np.float32)
        print(f"  {name:22s} traj_raw={raw_t:.2f}m ({raw_t/lam:.1f}p) traj_within_cell={within_t:.2f}m "
              f"phaseR={Rt:.2f} aliasing={alias_t:.2f} big_jumps_lattice={jfrac:.2f}", flush=True)
    print(f"seed {seed}: grid spacing lam={lam:.2f}m", flush=True)
    with open(os.path.join(out_dir, "aliasing_stats.json"), "w") as f:
        json.dump(res, f, indent=2)
    np.savez_compressed(os.path.join(out_dir, "aliasing_paths.npz"), **save)
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
    ap.add_argument("--res", type=int, default=40)
    args = ap.parse_args()
    device = args.device if torch.cuda.is_available() else "cpu"
    for s in args.seeds:
        run_seed(s, args, device)


if __name__ == "__main__":
    main()
