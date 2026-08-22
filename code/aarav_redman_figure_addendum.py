#!/usr/bin/env python3
"""Addendum to Will's RNN_predictive_grid_cell_function_temporal_shift.jpg, built from the networks in which
PGC ablation (original definition) clumps the torus phase.

Selection (stated, post hoc): torus-phase clumping after PGC ablation >= 0.6 AND intact clumping < 0.5 (valid
torus basis), from aarav_matched_torus outputs.  The figure keeps a panel with ALL networks so the selection is
visible.  Style follows Will's figure: grey = intact grid-population cloud (PC1-2, t >= 10), red = base,
cyan = predictive ablation; our additions: orange = property-matched control ablation, grey = random ablation.

Outputs (analysis_outputs/Single agent path integration/summary/aarav_matched_torus/):
  redman_addendum_selected_networks.png   the new panels alone
  redman_figure_plus_addendum.png         Will's figure on top, new panels below
"""
from __future__ import annotations
import argparse, copy, json, os, sys
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
_REPO = os.path.dirname(_HERE)

from aarav_crossing_population_correlation import load_model
from aarav_activity_space_ablation import make_random_walk_inputs
from multi_seed_predictive_analysis import zero_unit_weights_in_place
from aarav_definition_dynamics import derive_classes, pca_basis, autocorr_curves, T_FIT
from visualize import compute_ratemaps
from toroidal_structure_analysis import build_torus_basis, identify_toroidal_cells, project_states_to_torus

C_BASE, C_PRED, C_MATCH, C_RAND, C_CLOUD = "#e8141b", "#12d8e8", "#eb6834", "#7f7f7f", "#b5b5b5"
LAGS = list(range(1, 30))


def select_networks(root, subdir, seeds, thr_pgc=0.6, thr_intact=0.5):
    sel, table = [], {}
    for s in seeds:
        p = os.path.join(root, f"Seed {s}", subdir, "aarav_matched_torus", "matched_torus.json")
        if not os.path.exists(p):
            continue
        r = json.load(open(p))
        cl = lambda c: float(np.mean([rec["torus"]["theta1_clumping"] for rec in r["conditions"][c]]))
        table[s] = {"intact": cl("intact"), "pgc": cl("pred_lib"), "matched": cl("matched_lib"), "random_grid": cl("randpool_lib"),
                    "ac_pgc": [np.asarray(rec["autocorr_med"]) for rec in r["conditions"]["pred_lib"]][0],
                    "ac_base": np.asarray(r["conditions"]["pred_lib"][0]["autocorr_med_intact_same_units"]),
                    "ac_matched": np.mean([np.asarray(rec["autocorr_med"]) for rec in r["conditions"]["matched_lib"]], 0),
                    "ac_randgrid": np.mean([np.asarray(rec["autocorr_med"]) for rec in r["conditions"]["randpool_lib"]], 0)}
        dd = os.path.join(root, f"Seed {s}", subdir, "aarav_definition_dynamics", "definition_dynamics.json")
        if os.path.exists(dd):
            d = json.load(open(dd))
            table[s]["ac_randany"] = np.mean([np.asarray(d["conditions"][k]["random_walk"]["autocorr_med"])
                                              for k in d["conditions"] if k.startswith("rand_nd_matchlib")], 0)
        if table[s]["pgc"] >= thr_pgc and table[s]["intact"] < thr_intact:
            sel.append(s)
    return sel, table


def trajectories_for_seed(seed, args, device, root, subdir):
    ckpt = os.path.join(_REPO, args.model_root, f"Seed {seed}", "most_recent_model.pth")
    base = os.path.join(root, f"Seed {seed}", subdir)
    cl = derive_classes(os.path.join(base, "gridness_data.npz"), os.path.join(base, "band_cells", "band_scores.npz"))
    M = np.load(os.path.join(base, "aarav_matched_torus", "matched_sets.npz"))
    matched = M["matched_lib_0"]
    G = cl["std_grid"]; pred = cl["pred_lib"]
    model, place_cells, traj_gen, opt, Ng, Np = load_model(ckpt, device, args.seq_len)
    traj_gen.options.trajectory_style = "random_walk"
    inputs, pos = make_random_walk_inputs(traj_gen, args.test_trajectories, args.seq_len, 777 + seed)
    cloud_in = [make_random_walk_inputs(traj_gen, args.test_trajectories, args.seq_len, 5000 + 100 * seed + i) for i in range(2)]

    def run(m):
        with torch.no_grad():
            return m.g(inputs).detach().cpu().numpy()

    def ablated(units):
        m = copy.deepcopy(model); zero_unit_weights_in_place(m, list(units)); return m

    g0 = run(model)
    with torch.no_grad():
        cloud = np.concatenate([model.g(ci[0]).detach().cpu().numpy()[T_FIT:] for ci in cloud_in], 0).reshape(-1, Ng)
    out = {}
    for name, units in (("pgc", pred), ("matched", matched)):
        R = np.setdiff1d(G, units)
        gc = run(ablated(units))
        mu, PC, _ = pca_basis(g0[:, :, R])
        out[name] = {"intact": (g0[:, :, R] - mu) @ PC[:, :2], "ablated": (gc[:, :, R] - mu) @ PC[:, :2],
                     "cloud": ((cloud[::5][:, R] - mu) @ PC[:, :2])}
    # choose the trajectory with the largest intact excursion in the PGC read-out plane (clear loop), fixed rule
    exc = np.linalg.norm(out["pgc"]["intact"][-1] - out["pgc"]["intact"][0], axis=1)
    j = int(np.argsort(exc)[-3])            # third-largest: avoids wall-hugging outliers
    out["traj_index"] = j
    return out


def torus_trajectories_for_seed(seed, args, device, root, subdir):
    """Major-circle torus view (r1 cos th1, r1 sin th1) of long rollouts, basis as in aarav_matched_torus.py."""
    ckpt = os.path.join(_REPO, args.model_root, f"Seed {seed}", "most_recent_model.pth")
    base = os.path.join(root, f"Seed {seed}", subdir)
    cl = derive_classes(os.path.join(base, "gridness_data.npz"), os.path.join(base, "band_cells", "band_scores.npz"))
    matched = np.load(os.path.join(base, "aarav_matched_torus", "matched_sets.npz"))["matched_lib_0"]
    pred = cl["pred_lib"]
    grid_union = np.unique(np.concatenate([cl["pred_lib"], cl["retro_lib"], cl["normal_lib"]]))
    model, place_cells, traj_gen, opt, Ng, Np = load_model(ckpt, device, 20)
    np.random.seed(4321 + seed)
    rm, _, _, _ = compute_ratemaps(model, traj_gen, opt, res=40, n_avg=12, Ng=grid_union.size, idxs=grid_union)
    rm = np.asarray(rm, float)
    tmp = os.path.join(base, "aarav_matched_torus"); os.makedirs(tmp, exist_ok=True)
    det = identify_toroidal_cells(rm, grid_union, opt.box_width, tmp, embed_mode="umap")
    tor = np.asarray(det.units, int)
    pos_in = {int(g): i for i, g in enumerate(grid_union)}
    loc = np.array([pos_in[int(u)] for u in tor], dtype=int)
    basis = build_torus_basis(rm[loc], np.arange(loc.size), opt.box_width); basis.units = tor
    T, B = args.seq_len_torus, 64
    traj_gen.options.sequence_length = T
    traj_gen.options.trajectory_style = "random_walk"
    inputs, pos = make_random_walk_inputs(traj_gen, B, T, 777 + seed)

    def run(m):
        with torch.no_grad():
            return m.g(inputs).detach().cpu().numpy()

    def ablated(units):
        m = copy.deepcopy(model); zero_unit_weights_in_place(m, list(units)); return m

    g0 = run(model)
    P0 = project_states_to_torus(g0, basis, (1.0, 0.35), T, B)
    out = {"module_size": int(tor.size), "n_pred_in_module": int(np.intersect1d(pred, tor).size),
           "n_matched_in_module": int(np.intersect1d(matched, tor).size)}
    xy = lambda P: np.stack([P.r1 * np.cos(P.theta1), P.r1 * np.sin(P.theta1)], -1)     # [T,B,2]
    X0 = xy(P0)
    for name, units in (("pgc", pred), ("matched", matched)):
        Pc = project_states_to_torus(run(ablated(units)), basis, (1.0, 0.35), T, B)
        out[name] = {"intact": X0, "ablated": xy(Pc), "cloud": X0[T_FIT:].reshape(-1, 2)}
    # display trajectory: the one whose intact phase th1 travels the most (most revolutions), a fixed rule
    rev = np.abs(np.unwrap(P0.theta1, axis=0)[-1] - np.unwrap(P0.theta1, axis=0)[0]) / (2 * np.pi)
    out["traj_index"] = int(np.argmax(rev)); out["revolutions"] = float(rev.max())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", nargs="+", type=int, default=list(range(10)))
    ap.add_argument("--model_root", default="Models/Single agent path integration")
    ap.add_argument("--analysis_root", default="analysis_outputs/Single agent path integration")
    ap.add_argument("--analysis_subdir", default="spatial_shift_allunits")
    ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--seq_len", type=int, default=40)
    ap.add_argument("--test_trajectories", type=int, default=256)
    ap.add_argument("--redman_jpg", default="RNN_predictive_grid_cell_function_temporal_shift.jpg")
    ap.add_argument("--all_networks", action="store_true", help="show every network (5 per row), not only the selected ones")
    ap.add_argument("--torus_view", action="store_true", help="torus-basis major-circle projection of long rollouts instead of PC1-2")
    ap.add_argument("--seq_len_torus", type=int, default=120)
    args = ap.parse_args()
    device = args.device if torch.cuda.is_available() else "cpu"
    root = os.path.join(_REPO, args.analysis_root)
    out_dir = os.path.join(root, "summary", "aarav_matched_torus")
    sel, table = select_networks(root, args.analysis_subdir, args.seeds)
    print("selected networks:", sel)
    print("clumping per network (intact / PGC / matched / random grid):")
    for s in sorted(table):
        t = table[s]; print(f"  seed {s}: {t['intact']:.2f} / {t['pgc']:.2f} / {t['matched']:.2f} / {t['random_grid']:.2f}" + ("  <- selected" if s in sel else ""))

    show = sorted(table) if args.all_networks else sel
    fn = torus_trajectories_for_seed if args.torus_view else trajectories_for_seed
    traj = {s: fn(s, args, device, root, args.analysis_subdir) for s in show}

    # ------------------------------------------------------------------ addendum figure
    n = len(show)
    ncol = 5 if args.all_networks else n
    nblk = int(np.ceil(n / ncol))
    fig = plt.figure(figsize=(3.1 * ncol, 3.4 * 2 * nblk + 4.2))
    gs = fig.add_gridspec(2 * nblk + 1, ncol, height_ratios=[1] * (2 * nblk) + [1.15], hspace=0.45, wspace=0.25)
    for k, s in enumerate(show):
        blk, col = divmod(k, ncol)
        for row, (key, color, label) in enumerate((("pgc", C_PRED, "Pred. ablat."), ("matched", C_MATCH, "Matched ablat."))):
            ax = fig.add_subplot(gs[2 * blk + row, col])
            d = traj[s][key]; j = traj[s]["traj_index"]
            ax.scatter(d["cloud"][:, 0], d["cloud"][:, 1], s=4, color=C_CLOUD, alpha=0.45, lw=0)
            ax.plot(d["intact"][:, j, 0], d["intact"][:, j, 1], color=C_BASE, lw=2.2, label="Base")
            ax.plot(d["ablated"][:, j, 0], d["ablated"][:, j, 1], color=color, lw=2.2, label=label)
            ax.plot(d["intact"][0, j, 0], d["intact"][0, j, 1], "o", color=C_BASE, ms=5)
            ax.set_xticks([]); ax.set_yticks([]); ax.set_aspect("equal")
            for sp in ax.spines.values():
                sp.set_linewidth(1.2)
            if row == 0:
                tag = " ✓" if (args.all_networks and s in sel) else ""
                extra = f"\nbase: {traj[s]['revolutions']:.1f} revolutions" if args.torus_view else ""
                ax.set_title(f"Network {s}{tag}\nθ1 clumping PGC {table[s]['pgc']:.2f} · matched {table[s]['matched']:.2f}{extra}", fontsize=8,
                             color=(C_PRED if tag else "black"))
            if col == 0:
                ax.set_ylabel("Pred. grid unit ablation" if row == 0 else "Property-matched ablation", fontsize=11)
                if blk == 0:
                    ax.legend(fontsize=8, frameon=False, loc="upper left")
    last = 2 * nblk
    # row 3: autocorrelation (selected networks), autocorrelation (all networks), clumping per network
    ax1 = fig.add_subplot(gs[last, 0:max(1, ncol // 3)])
    ax2 = fig.add_subplot(gs[last, max(1, ncol // 3):max(2, 2 * ncol // 3)])
    ax3 = fig.add_subplot(gs[last, max(2, 2 * ncol // 3):])

    def ac_panel(ax, seeds, title):
        x = np.array([0] + LAGS)
        for key, color, label in (("ac_base", C_BASE, "Base"), ("ac_pgc", C_PRED, "Pred. ablat."), ("ac_matched", C_MATCH, "Matched ablat."),
                                  ("ac_randany", C_RAND, "Random ablat."), ("ac_randgrid", "#c29a00", "Random grid ablat.")):
            C = np.stack([np.r_[1.0, table[s][key]] for s in seeds if key in table[s]])
            ax.fill_between(x, np.percentile(C, 25, 0), np.percentile(C, 75, 0), color=color, alpha=0.25, lw=0)
            ax.plot(x, np.median(C, 0), color=color, lw=2.2, label=label)
        ax.set_xlim(0, 29); ax.set_ylim(0.4, 1.0); ax.set_xlabel("Lag"); ax.set_ylabel("Autocorrelation")
        ax.set_title(title, fontsize=10); ax.spines[["top", "right"]].set_visible(False)
        ax.legend(fontsize=8, frameon=False, loc="lower left")
    ac_panel(ax1, sel, f"Selected networks (n = {len(sel)})")
    ac_panel(ax2, sorted(table), f"All networks (n = {len(table)})")
    seeds_all = sorted(table)
    xs = np.arange(len(seeds_all))
    for off, key, color, label in ((-0.3, "intact", C_BASE, "Base"), (-0.1, "pgc", C_PRED, "Pred. ablat."), (0.1, "matched", C_MATCH, "Matched ablat."), (0.3, "random_grid", "#c29a00", "Random grid ablat.")):
        ax3.bar(xs + off, [table[s][key] for s in seeds_all], width=0.2, color=color, label=label)
    ax3.set_xticks(xs); ax3.set_xticklabels([str(s) + ("\n✓" if s in sel else "") for s in seeds_all]); ax3.set_xlabel("Network")
    for lab, s in zip(ax3.get_xticklabels(), seeds_all):
        if s in sel:
            lab.set_color(C_PRED); lab.set_fontweight("bold")
    ax3.set_ylabel("Torus-phase clumping\n(resultant of θ1; 1 = stuck)"); ax3.set_ylim(0, 1.0)
    ax3.set_title("Selection: PGC clumping ≥ 0.6 & base < 0.5 (✓)", fontsize=10)
    ax3.spines[["top", "right"]].set_visible(False); ax3.legend(fontsize=8, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=2)
    if args.all_networks and args.torus_view:
        fig.suptitle(f"Addendum, all networks, torus view: {args.seq_len_torus}-step rollouts projected on the toroidal module's major circle "
                     "(r1·cos θ1, r1·sin θ1; grey = intact states, ● = start). Per network: top = predictive ablation, bottom = property-matched control ablation\n"
                     f"✓ = networks in which PGC ablation clumps the torus phase ({len(sel)} of {len(table)}); original PGC definition; "
                     "shown trajectory = the one with most revolutions (fixed rule); autocorrelation panels use the 40-step rollouts", fontsize=10.5, y=0.995)
        fname = "redman_addendum_all_networks_torus.png"
    elif args.all_networks:
        fig.suptitle("Addendum, all networks: base vs ablated grid-population PCA trajectories (per network: top = predictive ablation, "
                     "bottom = property-matched control ablation, same number of cells, 8 covariates matched)\n"
                     f"✓ = networks in which PGC ablation clumps the torus phase ({len(sel)} of {len(table)}); original PGC definition; "
                     "grid population = standard grid units minus ablated", fontsize=10.5, y=0.995)
        fname = "redman_addendum_all_networks.png"
    else:
        fig.suptitle("Addendum: networks in which predictive-unit ablation clumps the torus phase — base vs ablated grid-population PCA trajectories "
                     "(top: predictive ablation; middle: property-matched control ablation, same number of cells, 8 covariates matched)\n"
                     f"Selected post hoc ({n} of {len(table)} networks); original PGC definition; grid population = standard grid units minus ablated",
                     fontsize=10.5, y=0.995)
        fname = "redman_addendum_selected_networks.png"
    fig.savefig(os.path.join(out_dir, fname), dpi=170, bbox_inches="tight"); plt.close(fig)

    # ------------------------------------------------------------------ composite with Will's figure
    jp = os.path.join(_REPO, args.redman_jpg)
    if os.path.exists(jp) and not args.all_networks:
        img = mpimg.imread(jp); add = mpimg.imread(os.path.join(out_dir, "redman_addendum_selected_networks.png"))
        h1, w1 = img.shape[:2]; h2, w2 = add.shape[:2]
        W = 20.0
        fig = plt.figure(figsize=(W, W * (h1 / w1) + W * (h2 / w2) + 0.6))
        gs = fig.add_gridspec(2, 1, height_ratios=[h1 / w1, h2 / w2], hspace=0.02)
        a = fig.add_subplot(gs[0]); a.imshow(img); a.axis("off"); a.set_title("Redman — RNN_predictive_grid_cell_function_temporal_shift.jpg", fontsize=12, loc="left")
        b = fig.add_subplot(gs[1]); b.imshow(add); b.axis("off"); b.set_title("Addendum (Sinha) — selected networks, property-matched control", fontsize=12, loc="left")
        fig.savefig(os.path.join(out_dir, "redman_figure_plus_addendum.png"), dpi=130, bbox_inches="tight"); plt.close(fig)
    json.dump({"selected": sel, "criterion": "theta1 clumping after PGC ablation >= 0.6 and intact < 0.5",
               "table": {str(s): {k: (float(v) if not isinstance(v, np.ndarray) else None) for k, v in table[s].items() if not isinstance(v, np.ndarray)} for s in table}},
              open(os.path.join(out_dir, "redman_addendum_selection.json"), "w"), indent=1)
    print(f"wrote {out_dir}/{fname}")


if __name__ == "__main__":
    main()
