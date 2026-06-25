#!/usr/bin/env python3
"""Figure: the 'stop at ~20 steps' is the decode-error metric SATURATING at chance, not a
box-crossing / training-horizon breakdown.

Evidence (rolled out to 80 steps, 2x past the 40 we usually analyse, 4x the 20-step training length):
  A. Decode error saturates at the chance ceiling (~box size); intact stays flat to 80 -> no cliff.
  B. The animal barely moves (true displacement << grid period) while the ablated ESTIMATE flies to
     chance -> the saturation is the estimate drifting to the ceiling, not the animal crossing space.
  C. Random-ablation error grows ~linearly; ceiling / drift-rate ~ 20 steps -> the '20' is arithmetic.
"""
from __future__ import annotations
import argparse, copy, os, sys
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
_REPO = os.path.dirname(_HERE)
from aarav_crossing_population_correlation import load_model, load_classes
from aarav_activity_space_ablation import make_random_walk_inputs
from multi_seed_predictive_analysis import zero_unit_weights_in_place


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", nargs="+", type=int, default=list(range(10)))
    ap.add_argument("--seq_len", type=int, default=80)
    ap.add_argument("--test_trajectories", type=int, default=128)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    device = args.device if torch.cuda.is_available() else "cpu"
    T = args.seq_len
    ERR = {c: [] for c in ("Intact", "Predictive-ablated", "Random-ablated (N)")}
    DISP, CHANCE, LAM = [], [], []
    for s in args.seeds:
        ckpt = f"Models/Single agent path integration/Seed {s}/most_recent_model.pth"
        gpath = f"analysis_outputs/Single agent path integration/Seed {s}/spatial_shift_allunits/gridness_data.npz"
        model, pc, tg, opt, Ng, Np = load_model(ckpt, device, T)
        cl = load_classes(gpath, 0.2, 5.0)
        pred, grid = cl["predictive"], cl["grid_all"]; allu = np.arange(Ng); N = pred.size
        rng = np.random.default_rng(7 + s)
        inp, pos = make_random_walk_inputs(tg, args.test_trajectories, T, 777 + s)
        true = pos.detach().cpu().numpy()
        Wd = model.decoder.weight.detach().to(device)
        conds = {"Intact": None, "Predictive-ablated": pred, "Random-ablated (N)": rng.choice(allu, N, replace=False)}
        for name, u in conds.items():
            m = model if u is None else copy.deepcopy(model)
            if u is not None:
                zero_unit_weights_in_place(m, list(u))
            with torch.no_grad():
                dpos = pc.get_nearest_cell_pos(m.g(inp) @ Wd.T).detach().cpu().numpy()
            if u is not None:
                del m
            ERR[name].append(np.linalg.norm(dpos - true, axis=2).mean(1))     # [T]
        DISP.append(np.linalg.norm(true - true[0:1], axis=2).mean(1))         # [T]
        # empirical chance: mean distance between independent decoded/true points
        flat = true.reshape(-1, 2); idx = rng.permutation(flat.shape[0])
        CHANCE.append(np.linalg.norm(flat - flat[idx], axis=1).mean())
        # grid period
        from visualize import compute_ratemaps
        from toroidal_structure_analysis import estimate_lattice_vectors
        rm, _, _, _ = compute_ratemaps(model, tg, opt, res=40, n_avg=6, Ng=grid.size, idxs=grid)
        k1, k2, _, _ = estimate_lattice_vectors(np.asarray(rm, float), np.arange(grid.size), opt.box_width)
        LAM.append(np.mean([1 / np.linalg.norm(k1), 1 / np.linalg.norm(k2)]))
        print(f"seed {s} done", flush=True)

    err = {c: np.array(ERR[c]) for c in ERR}
    disp = np.array(DISP); chance = float(np.mean(CHANCE)); lam = float(np.mean(LAM))
    t = np.arange(T)
    COL = {"Intact": "#000000", "Predictive-ablated": "#d62728", "Random-ablated (N)": "#7f7f7f"}

    fig, axes = plt.subplots(1, 3, figsize=(17, 5))

    # A: error vs time to 80, chance ceiling, training horizon
    ax = axes[0]
    for c in err:
        m = err[c].mean(0); sem = err[c].std(0) / np.sqrt(len(args.seeds))
        ax.plot(t, m, color=COL[c], lw=2, label=c.replace(" (N)", "")); ax.fill_between(t, m - sem, m + sem, color=COL[c], alpha=0.18)
    ax.axhline(chance, ls=":", c="purple", lw=1.6); ax.text(T * 0.5, chance + 0.02, f"chance ceiling ~{chance:.2f} m (box size)", color="purple", fontsize=8)
    ax.axvline(20, ls="--", c="green", lw=1.4); ax.text(21, 0.1, "training length (20)", color="green", fontsize=8, rotation=90, va="bottom")
    ax.axvline(40, ls="--", c="0.6", lw=1); ax.text(41, 0.1, "usual analysis (40)", color="0.5", fontsize=8, rotation=90, va="bottom")
    ax.set_xlabel("timestep"); ax.set_ylabel("decode error vs truth (m)")
    ax.set_title("A. Error saturates at chance; intact flat to 80\n(no breakdown at 20 -> not a training cliff)",
                 loc="left", fontweight="bold", fontsize=10); ax.legend(frameon=False, fontsize=9); ax.grid(alpha=0.25)

    # B: animal barely moves vs estimate flies off
    ax = axes[1]
    dm = disp.mean(0); dsem = disp.std(0) / np.sqrt(len(args.seeds))
    ax.plot(t, dm, color="#1f77b4", lw=2.2, label="how far the ANIMAL actually moved")
    ax.fill_between(t, dm - dsem, dm + dsem, color="#1f77b4", alpha=0.2)
    pm = err["Predictive-ablated"].mean(0)
    ax.plot(t, pm, color="#d62728", lw=2.2, label="how far the ESTIMATE is off (predictive)")
    ax.axhline(lam, ls=":", c="green", lw=1.4); ax.text(T * 0.4, lam + 0.02, f"one grid period ~{lam:.2f} m", color="green", fontsize=8)
    ax.axhline(chance, ls=":", c="purple", lw=1.4); ax.text(T * 0.4, chance + 0.02, f"chance ~{chance:.2f} m", color="purple", fontsize=8)
    ax.set_xlabel("timestep"); ax.set_ylabel("distance (m)")
    ax.set_title("B. The estimate flies to chance while the animal\nstays local -> '20' isn't the animal crossing space",
                 loc="left", fontweight="bold", fontsize=10); ax.legend(frameon=False, fontsize=8.5); ax.grid(alpha=0.25)

    # C: ceiling / drift-rate ~ 20
    ax = axes[2]
    rm_err = err["Random-ablated (N)"].mean(0)
    fitseg = slice(2, 14)
    slope, intercept = np.polyfit(t[fitseg], rm_err[fitseg], 1)
    tx = np.arange(0, 30)
    ax.plot(t[:45], rm_err[:45], color="#7f7f7f", lw=2.2, label="random-ablation error")
    ax.plot(tx, intercept + slope * tx, "--", color="#ff7f0e", lw=1.8, label=f"linear drift ~{slope*100:.1f} cm/step")
    ax.axhline(chance, ls=":", c="purple", lw=1.4)
    tcross = (chance - intercept) / slope
    ax.axvline(tcross, ls="--", c="purple", lw=1.2)
    ax.text(tcross + 0.5, 0.2, f"ceiling / drift ≈ {tcross:.0f} steps", color="purple", fontsize=9, rotation=90, va="bottom")
    ax.set_xlabel("timestep"); ax.set_ylabel("decode error (m)"); ax.set_ylim(0, chance * 1.15)
    ax.set_title("C. The '20' is arithmetic: chance ceiling /\ndrift-rate (estimate covers the box in ~20 steps)",
                 loc="left", fontweight="bold", fontsize=10); ax.legend(frameon=False, fontsize=8.5); ax.grid(alpha=0.25)

    fig.suptitle("Why the ablated error 'stops' near step 20: a bounded metric saturating at chance, not a breakdown timescale (10 seeds, rolled out to 80)",
                 fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    out = os.path.join(_REPO, "analysis_outputs/Single agent path integration/summary/aarav_activity_space_ablation/why_step20_saturation.png")
    fig.savefig(out, dpi=190, bbox_inches="tight"); fig.savefig(out.replace(".png", ".svg"), bbox_inches="tight")
    print("wrote", out)
    print(f"disp@20={dm[20]:.2f}m disp@40={dm[40]:.2f}m disp@79={dm[-1]:.2f}m | grid_period={lam:.2f} chance={chance:.2f} | drift={slope*100:.1f}cm/step cross={tcross:.0f}")


if __name__ == "__main__":
    main()
