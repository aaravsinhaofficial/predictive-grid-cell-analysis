#!/usr/bin/env python3
"""Minimal best-shift gridness check for the 2 straight-trained nets.

Directly tests whether units become grid-like at the predictive-peak spatial shift (~16 cm), i.e.
whether zero-shift gridness undercounts shift-defined (predictive) grid cells. 3 shifts only.
"""
from __future__ import annotations
import argparse, os, sys
import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
_REPO = os.path.dirname(_HERE)
from aarav_crossing_population_correlation import load_model
from visualize import predictive_gridness_analysis


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seq_len", type=int, default=40)
    args = ap.parse_args()
    dev = args.device if torch.cuda.is_available() else "cpu"
    out = os.path.join(_REPO, "analysis_outputs/Single agent path integration/summary/aarav_straight_trained")
    lags = [0.0, 8.0, 16.0]                                   # cm; 0 and around the predictive peak
    nets = [("straight-Seed0", "Models/straight/steps_40/Seed 0/most_recent_model.pth"),
            ("straight-Seed1", "Models/straight/steps_40/Seed 1/most_recent_model.pth")]
    for tag, path in nets:
        model, pc, tg, opt, Ng, Np = load_model(os.path.join(_REPO, path), dev, args.seq_len)
        opt.trajectory_style = "straight"
        s60, s90, xs, ys, acts, scorer = predictive_gridness_analysis(
            model, tg, opt, lags=lags, res=20, n_batches=8, Ng=model.Ng,
            shift_mode="space", space_projection="path")
        s60 = np.asarray(s60, float)                          # [3, Ng]
        best = np.nanmax(s60, axis=0)
        b = best[np.isfinite(best)]
        print(f"{tag}: best-shift gridness over shifts {lags} cm | "
              f">0.37: {(b>=0.37).sum()}  >0.5: {(b>=0.5).sum()}  mean={b.mean():.3f}  max={b.max():.3f}", flush=True)
        np.save(os.path.join(out, f"bestshift_min_{tag}.npy"), best)
    print("MINDONE", flush=True)


if __name__ == "__main__":
    main()
