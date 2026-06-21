#!/usr/bin/env bash
# FULL spatial-shift reproduction using ALL 4096 RNN units (not the 512 subset),
# across all 10 seeds. Regenerates gridness at 4096 units then runs every downstream
# analysis on the full population. Random-ablation pool defaults to grid cells only.
# Output: analysis_outputs/.../Seed N/spatial_shift_allunits/...
#
# Usage:  bash run_allunits_spatial_shift.sh         (GPU)
set -euo pipefail
cd "$(dirname "$0")"
PY="./.venv/bin/python"
DEVICE="${DEVICE:-cuda}"
MODEL_ROOT="Models/Single agent path integration"

CKPTS=()
for s in 0 1 2 3 4 5 6 7 8 9; do
  CKPTS+=("$MODEL_ROOT/Seed $s/most_recent_model.pth")
done

exec "$PY" code/reproduce_spatial_shift_results.py \
  --analysis_subdir spatial_shift_allunits \
  --space_projection path \
  --max_shift_cm 20 \
  --shift_step_cm 1 \
  --sequence_length 40 \
  --weight_decay 1e-4 \
  --box_width 2.2 --box_height 2.2 \
  --place_cell_rf 0.12 --surround_scale 2.0 \
  --activation relu --learning_rate 1e-4 \
  --Ng_use 4096 \
  --full_ablation_random_trials 10 \
  --device "$DEVICE" \
  --checkpoint_paths "${CKPTS[@]}"
