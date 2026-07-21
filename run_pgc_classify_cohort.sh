#!/usr/bin/env bash
# Run the defensible PGC classifier over every canonical-cohort checkpoint that
# currently exists on disk (safe to re-run as more seeds finish training).
# Sequential (each run already uses many CPU workers); GPU forward, CPU scoring.
set -uo pipefail
cd "$(dirname "$0")"
PY="./.venv/bin/python"
export MPLCONFIGDIR="$PWD/.mplconfig"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
LOG="Models/canonical_cohort_v1/classify.log"
NWORK="${NWORK:-24}"
NG="${NG:-256}"
NSHUF="${NSHUF:-40}"
OUTSUB="${OUTSUB:-pgc_rigor}"

mapfile -t CKPTS < <(ls -1 Models/canonical_cohort_v1/seed_*/*/most_recent_model.pth 2>/dev/null)
echo "[classify-cohort] $(date -u +%H:%M:%S) found ${#CKPTS[@]} checkpoints (workers=$NWORK shuf=$NSHUF)" | tee -a "$LOG"
for ck in "${CKPTS[@]}"; do
  # resolve analysis dir + skip if already classified
  adir="$("$PY" - "$ck" "$OUTSUB" <<'PYX'
import sys; sys.path.insert(0,'code')
from pathlib import Path
from path_utils import analysis_dir_for_checkpoint
print(analysis_dir_for_checkpoint(Path(sys.argv[1]))/sys.argv[2])
PYX
)"
  if [ -f "$adir/pgc_classification.npz" ]; then
    echo "[classify-cohort] $(date -u +%H:%M:%S) SKIP (done) $ck" | tee -a "$LOG"; continue
  fi
  echo "[classify-cohort] $(date -u +%H:%M:%S) START $ck" | tee -a "$LOG"
  "$PY" code/pgc_classifier.py --checkpoint "$ck" --device cuda \
    --Ng_use "$NG" --n_batches 16 --n_shuffles "$NSHUF" --max_lag 25 --gridness_floor 0.15 \
    --n_workers "$NWORK" --out_subdir "$OUTSUB" >> "$LOG" 2>&1
  echo "[classify-cohort] $(date -u +%H:%M:%S) DONE  $ck (rc=$?)" | tee -a "$LOG"
done
echo "[classify-cohort] $(date -u +%H:%M:%S) ALL DONE" | tee -a "$LOG"
