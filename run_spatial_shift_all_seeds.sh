#!/usr/bin/env bash
# Reproduce the full predictive-grid result suite with TRUE SPATIAL shifts
# (centimeters of arc length along the path), across all 10 trained seeds.
#
# Models: Models/Single agent path integration/Seed {0..9}/most_recent_model.pth
# Env:    ./.venv (Python 3.11, torch cu130, Blackwell-capable)
# Output: analysis_outputs/Single agent path integration/Seed N/spatial_shift/...
#
# Usage:
#   bash run_spatial_shift_all_seeds.sh            # all 10 seeds, GPU
#   DEVICE=cpu bash run_spatial_shift_all_seeds.sh # force CPU
set -euo pipefail

cd "$(dirname "$0")"
PY="./.venv/bin/python"
DEVICE="${DEVICE:-cuda}"
MODEL_ROOT="Models/Single agent path integration"

# Build the 10-seed checkpoint list (each path contains "Seed N" for cross-seed aggregation).
CKPTS=()
for s in 0 1 2 3 4 5 6 7 8 9; do
  CKPTS+=("$MODEL_ROOT/Seed $s/most_recent_model.pth")
done

exec "$PY" code/reproduce_spatial_shift_results.py \
  --analysis_subdir spatial_shift \
  --space_projection path \
  --max_shift_cm 20 \
  --shift_step_cm 1 \
  --sequence_length 40 \
  --weight_decay 1e-4 \
  --box_width 2.2 --box_height 2.2 \
  --place_cell_rf 0.12 --surround_scale 2.0 \
  --activation relu --learning_rate 1e-4 \
  --Ng_use 512 \
  --device "$DEVICE" \
  --checkpoint_paths "${CKPTS[@]}"
  # To also run the future-split spatial suite, append e.g.:
  #   --future_split_checkpoint_paths "Models/.../future_split_full.pth"
