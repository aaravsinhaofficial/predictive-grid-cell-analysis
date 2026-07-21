#!/usr/bin/env bash
# Full reviewer-grade PGC rigor pipeline over a list of checkpoints.
# Per checkpoint (in order): defensible classifier -> per-unit covariates ->
# property-matched ablation dose-response -> data-driven torus topology ->
# causal PGC-subspace intervention. Then cross-seed aggregation + a repro freeze.
#
# Forward runs on GPU; scoring fans across CPU cores (OMP_NUM_THREADS=1).
#
# Usage:
#   bash run_pgc_rigor_pipeline.sh <checkpoint.pth> [<checkpoint.pth> ...]
#   MODELS_ROOT=Models/canonical_cohort_v1 bash run_pgc_rigor_pipeline.sh   # glob all
#
# Env knobs: DEVICE (cuda), NWORK (12), NG (256), OUT_SUBDIR (pgc_rigor),
#            STAGES ("classify covariates ablation topology intervention")
set -uo pipefail
cd "$(dirname "$0")"
PY="./.venv/bin/python"
export MPLCONFIGDIR="$PWD/.mplconfig"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

DEVICE="${DEVICE:-cuda}"
NWORK="${NWORK:-12}"
NG="${NG:-256}"
OUT_SUBDIR="${OUT_SUBDIR:-pgc_rigor}"
STAGES="${STAGES:-classify covariates ablation topology intervention}"

CKPTS=("$@")
if [ ${#CKPTS[@]} -eq 0 ] && [ -n "${MODELS_ROOT:-}" ]; then
  mapfile -t CKPTS < <(ls -1 "$MODELS_ROOT"/**/most_recent_model.pth 2>/dev/null; \
                       find "$MODELS_ROOT" -name most_recent_model.pth 2>/dev/null)
  # dedupe
  mapfile -t CKPTS < <(printf '%s\n' "${CKPTS[@]}" | awk 'NF' | sort -u)
fi
if [ ${#CKPTS[@]} -eq 0 ]; then
  echo "No checkpoints given. Pass paths or set MODELS_ROOT." >&2; exit 1
fi

echo "=== PGC rigor pipeline: ${#CKPTS[@]} checkpoints | stages: $STAGES ==="
run() { echo "  [$(date -u +%H:%M:%S)] $*"; "$@"; }

for ck in "${CKPTS[@]}"; do
  echo "### $ck"
  case " $STAGES " in *" classify "*)
    run "$PY" code/pgc_classifier.py --checkpoint "$ck" --device "$DEVICE" \
      --Ng_use "$NG" --n_batches 16 --n_shuffles 50 --max_lag 25 \
      --n_workers "$NWORK" --out_subdir "$OUT_SUBDIR" || echo "  classify FAILED";; esac
  case " $STAGES " in *" covariates "*)
    run "$PY" code/pgc_covariates.py --checkpoint "$ck" --device "$DEVICE" \
      --Ng_use "$NG" --n_batches 16 --n_workers "$NWORK" --out_subdir "$OUT_SUBDIR" || echo "  covariates FAILED";; esac
  case " $STAGES " in *" ablation "*)
    run "$PY" code/pgc_matched_ablation.py --checkpoint "$ck" --device "$DEVICE" \
      --Ng_use "$NG" --n_workers "$NWORK" --ablation_batches 8 --random_trials 5 \
      --out_subdir "$OUT_SUBDIR" || echo "  ablation FAILED";; esac
  case " $STAGES " in *" topology "*)
    run "$PY" code/pgc_torus_topology.py --checkpoint "$ck" --device "$DEVICE" \
      --n_workers "$NWORK" --out_subdir "$OUT_SUBDIR" || echo "  topology FAILED";; esac
  case " $STAGES " in *" intervention "*)
    run "$PY" code/pgc_intervention.py --checkpoint "$ck" --device "$DEVICE" \
      --n_workers "$NWORK" --out_subdir "$OUT_SUBDIR" || echo "  intervention FAILED";; esac
done

echo "=== cross-seed aggregation ==="
run "$PY" code/pgc_aggregate_seeds.py --checkpoints "${CKPTS[@]}" --out_subdir "$OUT_SUBDIR"

echo "=== pipeline complete ==="
