#!/usr/bin/env bash
# Deadline eval blitz over all doors arms: pull AWS synced states, then run
# the key-condition ablation suite on every arm in parallel.
# Usage: ./eval_blitz.sh [episodes]   (default 60)
set -uo pipefail
EP=${1:-60}
BANINO=${BANINO_REPO:-$HOME/banino}
PGC=$(cd "$(dirname "$0")/.." && pwd)
BUCKET=s3://banino-repro-975050064729
ONLY=chance,all_ro,predictive,matched,randgrid_0
RUN="steps_20_batch_200_RNN_4096_relu_rf_012_DoG_True_periodic_False_lr_00001_weight_decay_1e-06"
cd "$BANINO"

# 1) Pull the latest synced state of each AWS cell into rl_runs/.
for c in pirnn_doors_s2 pirnn_doors_s20 pirnn_doors_s0_lr2 pirnn_doors_s0_aws; do
  if aws s3 cp "$BUCKET/rl_runs/$c/results.tar.gz" "/tmp/$c.tar.gz" --only-show-errors; then
    tar xzf "/tmp/$c.tar.gz" -C rl_runs/ && \
      echo "pulled $c ($(ls rl_runs/$c/ckpt_*.pt 2>/dev/null | wc -l) ckpts)"
  else
    echo "WARN: no synced state for $c"
  fi
done

# 2) Local deep arm (8-action pre-audit container) + AWS arms (6-action),
#    all in parallel; AWS configs point at the instance path /pgc/rnn.pth,
#    so pass the real cohort checkpoint explicitly.
BANINO_ACTION_SET=8 python3 "$PGC/rl_navigation/run_ablation_suite.py" \
  --run "$BANINO/rl_runs/pirnn_doors_s0" --seed 0 --ckpt latest \
  --episodes "$EP" --n_envs 8 --docker_image dmlab-rl:latest \
  --only "$ONLY" --label blitz \
  > "$PGC/rl_navigation/results/blitz_pirnn_doors_s0.log" 2>&1 &

for spec in "pirnn_doors_s2 2" "pirnn_doors_s20 20" \
            "pirnn_doors_s0_lr2 0" "pirnn_doors_s0_aws 0"; do
  set -- $spec
  python3 "$PGC/rl_navigation/run_ablation_suite.py" \
    --run "$BANINO/rl_runs/$1" --seed "$2" --ckpt latest \
    --episodes "$EP" --n_envs 8 --docker_image dmlab-rl:latest \
    --pirnn_ckpt "$PGC/Models/canonical_cohort_v1/seed_$2/$RUN/most_recent_model.pth" \
    --only "$ONLY" --label blitz \
    > "$PGC/rl_navigation/results/blitz_$1.log" 2>&1 &
done
wait
echo "blitz complete"
