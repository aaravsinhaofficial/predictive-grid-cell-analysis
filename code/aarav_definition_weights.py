#!/usr/bin/env python3
"""Per-class weight statistics for the definition-dynamics study (no forward passes).

For each seed: velocity-input row norm |W_ih[i,:]|, recurrent out-norm |W_hh[:,i]|, recurrent in-norm
|W_hh[i,:]|, decoder column norm |W_dec[:,i]|, encoder row norm |W_enc[i,:]| — per unit class
(std grid, pred/retro x {original, MEC-style}, band, random draws).  Answers: "do conservative PGCs
receive more velocity input / project more strongly onto the grid population than other units?"
"""
from __future__ import annotations
import argparse, json, os
import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__)); _REPO = os.path.dirname(_HERE)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", nargs="+", type=int, default=list(range(10)))
    ap.add_argument("--model_root", default="Models/Single agent path integration")
    ap.add_argument("--analysis_root", default="analysis_outputs/Single agent path integration")
    ap.add_argument("--analysis_subdir", default="spatial_shift_allunits")
    args = ap.parse_args()
    out = {}
    for s in args.seeds:
        ck = os.path.join(_REPO, args.model_root, f"Seed {s}", "most_recent_model.pth")
        ci = os.path.join(_REPO, args.analysis_root, f"Seed {s}", args.analysis_subdir, "aarav_definition_dynamics", "class_indices.npz")
        if not (os.path.exists(ck) and os.path.exists(ci)):
            continue
        raw = torch.load(ck, map_location="cpu", weights_only=False)
        st = raw.get("model_state_dict", raw.get("state_dict", raw)) if isinstance(raw, dict) else raw
        Whh = st["RNN.weight_hh_l0"].numpy(); Wih = st["RNN.weight_ih_l0"].numpy()
        Wdec = st["decoder.weight"].numpy(); Wenc = st["encoder.weight"].numpy()
        C = np.load(ci)
        G = C["std_grid"]
        feats = {"vel_in": np.linalg.norm(Wih, axis=1), "rec_out": np.linalg.norm(Whh, axis=0),
                 "rec_in": np.linalg.norm(Whh, axis=1), "rec_out_to_stdgrid": np.linalg.norm(Whh[G, :], axis=0),
                 "dec_out": np.linalg.norm(Wdec, axis=0), "enc_in": np.linalg.norm(Wenc, axis=1)}
        rec = {}
        for cls in ("std_grid", "pred_lib", "retro_lib", "normal_lib", "pred_con", "retro_con", "band", "band90",
                    "rand_nd_matchcon_0", "rand_nd_matchlib_0", "struct_matchcon_0", "nondead"):
            idx = C[cls]
            rec[cls] = {k: float(np.median(v[idx])) for k, v in feats.items()}
            rec[cls]["n"] = int(idx.size)
        out[str(s)] = rec
        print(f"seed {s}: " + " | ".join(f"{c}: vel {rec[c]['vel_in']:.3f} recout->grid {rec[c]['rec_out_to_stdgrid']:.3f}"
                                        for c in ("pred_con", "rand_nd_matchcon_0", "struct_matchcon_0", "pred_lib", "normal_lib", "band")))
    od = os.path.join(_REPO, args.analysis_root, "summary", "aarav_definition_dynamics"); os.makedirs(od, exist_ok=True)
    with open(os.path.join(od, "class_weight_stats.json"), "w") as f:
        json.dump(out, f, indent=1)
    # cross-seed medians
    print("\nmedian across seeds:")
    for cls in ("std_grid", "normal_lib", "pred_lib", "pred_con", "retro_con", "band", "rand_nd_matchcon_0", "struct_matchcon_0", "nondead"):
        print(f"  {cls:20s} " + " ".join(f"{k} {np.median([out[s][cls][k] for s in out]):.3f}" for k in ("vel_in", "rec_out", "rec_out_to_stdgrid", "dec_out", "enc_in")))


if __name__ == "__main__":
    main()
