#!/usr/bin/env bash
# Train the canonical 30-seed cohort (seeds 0..29) with ONE frozen, fully
# seed-controlled recipe, split across the two local GPUs. This is the
# reproducible cohort for the rigor-upgrade analysis pipeline.
#
# Frozen recipe (see code/train_cohort_worker.sh):
#   RNN 4096 relu, Np=512 place cells (DoG, rf 0.12), velocity input,
#   batch 200, seq 20, lr 1e-4, weight_decay 1e-6, box 2.2 m, random walk,
#   40 epochs x 1000 steps = 40,000 gradient steps, Adam.
#
# Output: Models/canonical_cohort_v1/seed_<s>/<run_ID>/most_recent_model.pth
#         Models/canonical_cohort_v1/logs/seed_<s>.log
#
# Usage: bash run_train_cohort.sh
set -uo pipefail
cd "$(dirname "$0")"

# GPU 0: seeds 0..14   GPU 1: seeds 15..29
bash code/train_cohort_worker.sh 0 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 &
P0=$!
bash code/train_cohort_worker.sh 1 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 &
P1=$!
wait $P0 $P1
echo "COHORT TRAINING COMPLETE $(date -u +%Y-%m-%dT%H:%M:%SZ)"
