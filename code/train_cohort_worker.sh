#!/usr/bin/env bash
# Worker: train a list of seeds on ONE gpu, sequentially, with the frozen
# canonical recipe. Prunes per-epoch checkpoints (keeps most_recent + a few
# emergence snapshots) so disk stays light. Does NOT use `set -e`: a single
# seed failing must not abort the remaining seeds.
#
# Usage: bash code/train_cohort_worker.sh <gpu_id> <seed> [<seed> ...]
set -uo pipefail

cd "$(dirname "$0")/.."
PY="./.venv/bin/python"
GPU="$1"; shift
SEEDS=("$@")

COHORT_ROOT="Models/canonical_cohort_v1"
LOG_ROOT="$COHORT_ROOT/logs"
mkdir -p "$LOG_ROOT"
export MPLCONFIGDIR="$PWD/.mplconfig"

for s in "${SEEDS[@]}"; do
  SAVE="$COHORT_ROOT/seed_${s}"
  echo "[gpu$GPU $(date -u +%H:%M:%S)] START seed $s"
  CUDA_VISIBLE_DEVICES="$GPU" "$PY" code/main.py \
    --save_dir "$SAVE" --seed "$s" \
    --n_epochs 40 --n_steps 1000 \
    --batch_size 200 --sequence_length 20 \
    --Ng 4096 --Np 512 --RNN_type RNN --activation relu \
    --place_cell_rf 0.12 --surround_scale 2.0 \
    --learning_rate 1e-4 --weight_decay 1e-6 \
    --box_width 2.2 --box_height 2.2 \
    --save_ratemaps_interval 0 --grid_eval_interval 0 \
    --device cuda > "$LOG_ROOT/seed_${s}.log" 2>&1
  rc=$?
  # Prune per-epoch checkpoints; keep emergence snapshots at 10/20/30/39.
  RUNDIR=$(find "$SAVE" -maxdepth 1 -type d -name "steps_*" 2>/dev/null | head -1)
  if [ -n "$RUNDIR" ]; then
    for f in "$RUNDIR"/epoch_*.pth; do
      [ -e "$f" ] || continue
      ep=$(basename "$f" .pth); ep=${ep#epoch_}
      case "$ep" in 10|20|30|39) : ;; *) rm -f "$f" ;; esac
    done
  fi
  FINAL_ERR=$(grep -oE "Err: [0-9.]+cm" "$LOG_ROOT/seed_${s}.log" | tail -1)
  echo "[gpu$GPU $(date -u +%H:%M:%S)] DONE seed $s rc=$rc  ${FINAL_ERR}"
done
echo "[gpu$GPU $(date -u +%H:%M:%S)] ALL DONE"
