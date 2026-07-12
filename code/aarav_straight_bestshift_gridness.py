#!/usr/bin/env python3
"""Rigorous best-shift gridness: do straight-trained nets develop grid cells under the SAME
shift-based grid definition used for the predictive/retrospective classification?

My earlier gate used ZERO-shift gridness, which can undercount predictive grid cells (they peak at a
spatial shift, low at zero shift). Here we compute gridness across spatial shifts (-20..+20 cm) and
take the best-shift gridness per unit — the definition used to classify grid/predictive cells — and
compare straight-trained nets to a random-walk-trained net.
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
from shift_utils import build_shift_values


def analyse(path, sample_style, dev, seq_len, res=20, n_batches=10, max_shift_cm=20.0, step=2.5):
    model, pc, tg, opt, Ng, Np = load_model(os.path.join(_REPO, path), dev, seq_len)
    opt.trajectory_style = sample_style
    lags = build_shift_values("space", max_shift_cm=max_shift_cm, shift_step_cm=step)
    lag_cm = np.asarray(lags, float)
    s60, s90, xs, ys, acts, scorer = predictive_gridness_analysis(
        model, tg, opt, lags=lags, res=res, n_batches=n_batches, Ng=model.Ng,
        shift_mode="space", space_projection="path")
    s60 = np.asarray(s60, float)                        # [L, Ng]
    best = np.nanmax(s60, axis=0)                       # best-shift gridness per unit
    best_shift = lag_cm[np.nanargmax(np.nan_to_num(s60, nan=-9), axis=0)]
    zero = s60[int(np.argmin(np.abs(lag_cm)))]          # zero-shift gridness
    return best, best_shift, zero


def report(tag, best, best_shift, zero):
    b = best[np.isfinite(best)]
    for thr in (0.37, 0.5):
        n_grid = int((b >= thr).sum())
        pred = int(((b >= thr) & (best_shift >= 5.0)).sum())
        print(f"  {tag:22s} thr={thr}: grid(best-shift)={n_grid:5d}  predictive(shift>=5cm)={pred:5d}", flush=True)
    print(f"  {tag:22s} mean best-shift gridness={np.nanmean(b):.3f}  max={np.nanmax(b):.3f}  "
          f"mean zero-shift={np.nanmean(zero[np.isfinite(zero)]):.3f}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seq_len", type=int, default=40)
    args = ap.parse_args()
    dev = args.device if torch.cuda.is_available() else "cpu"
    out = os.path.join(_REPO, "analysis_outputs/Single agent path integration/summary/aarav_straight_trained")
    os.makedirs(out, exist_ok=True)
    nets = [
        ("straight-Seed0", "Models/straight/steps_40/Seed 0/most_recent_model.pth", "straight"),
        ("straight-Seed1", "Models/straight/steps_40/Seed 1/most_recent_model.pth", "straight"),
        ("randomwalk-Seed0", "Models/Single agent path integration/Seed 0/most_recent_model.pth", "random_walk"),
    ]
    for tag, path, style in nets:
        best, bshift, zero = analyse(path, style, dev, args.seq_len)
        report(tag, best, bshift, zero)
        np.savez(os.path.join(out, f"bestshift_{tag}.npz"), best=best, best_shift=bshift, zero=zero)
        print(f"[saved {tag}]", flush=True)


if __name__ == "__main__":
    main()
