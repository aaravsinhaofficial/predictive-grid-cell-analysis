#!/usr/bin/env bash
# Exact launch commands for the RL navigation experiment (2026-08-15).
# banino repo at commit 508c849 (pirnn agent). Runs are resumable: re-run
# the same command and train_rl picks up <out>/resume.pt.
set -euo pipefail

BANINO=${BANINO_REPO:-$HOME/banino}
PGC=$(cd "$(dirname "$0")/.." && pwd)
RUN="steps_20_batch_200_RNN_4096_relu_rf_012_DoG_True_periodic_False_lr_00001_weight_decay_1e-06"
CKPT() { echo "$PGC/Models/canonical_cohort_v1/seed_$1/$RUN/most_recent_model.pth"; }

cd "$BANINO"

# FakeEnv runs (host python, nohup-detached, 16 envs, 400M frames)
for cfg in "0 pirnn_fake_s0 0" "2 pirnn_fake_s2 0" "20 pirnn_fake_s20 1"; do
  set -- $cfg
  nohup env CUDA_VISIBLE_DEVICES=$3 python3 -m rl.train_rl --fake --agent pirnn \
    --pirnn_ckpt "$(CKPT $1)" --out rl_runs/$2 \
    --n_envs 16 --frames 400000000 --seed 1 --resume \
    > rl_runs/$2.log 2>&1 &
done
nohup env CUDA_VISIBLE_DEVICES=1 python3 -m rl.train_rl --fake --agent a3c \
  --out rl_runs/a3c_fake --n_envs 16 --frames 400000000 --seed 1 --resume \
  > rl_runs/a3c_fake.log 2>&1 &

# DM-Lab flagship (detached container; dmlab-rl:latest has square_arena_goal)
docker run -d --name pirnn-arena-s0 --gpus '"device=1"' --shm-size=8g \
  -v "$BANINO":/workspace -v "$PGC":/pgc -w /workspace dmlab-rl:latest \
  python3 -m rl.train_rl --level square_arena_goal --agent pirnn \
  --pirnn_ckpt "/pgc/Models/canonical_cohort_v1/seed_0/$RUN/most_recent_model.pth" \
  --out rl_runs/pirnn_arena_s0 --n_envs 20 --frames 300000000 \
  --arena_cells 13 --pirnn_env_half 1.625 --seed 1 --resume

# Ablation suites (after checkpoints exist), e.g.:
#   python3 rl_navigation/run_ablation_suite.py \
#       --run "$BANINO/rl_runs/pirnn_fake_s0" --seed 0 --episodes 100
