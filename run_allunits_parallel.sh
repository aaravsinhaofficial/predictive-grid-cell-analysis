#!/usr/bin/env bash
# Parallel all-units (4096) spatial-shift reproduction.
# Phase 1: 10 seeds run CONCURRENTLY (each its own process, 1 thread, alternating GPUs):
#          multi_seed + band + crossing/same_bin/step_sweep + torus  (per-seed, no clobber)
#          + full grid-random ablation (per-seed output dir).
# Phase 2: cross-seed aggregates (cell summary, breadth) + merge per-seed ablation.
set -uo pipefail
REPO="/home/ec2-user/predictive-grid-cell-analysis"
PY="$REPO/.venv/bin/python"
SUB="spatial_shift_allunits"
LOGD="/home/ec2-user/allunits_parallel_logs"
mkdir -p "$LOGD"
cd "$REPO"
# keep each worker single-threaded so 10 concurrent jobs don't oversubscribe 48 cores
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

MODEL_ROOT="Models/Single agent path integration"
COMMON_MODEL=(--sequence_length 40 --weight_decay 1e-4 --box_width 2.2 --box_height 2.2
              --place_cell_rf 0.12 --surround_scale 2.0 --activation relu --learning_rate 1e-4)

run_seed () {
  local s="$1"; local gpu=$(( s % 2 ))
  local ckpt="$MODEL_ROOT/Seed $s/most_recent_model.pth"
  local log="$LOGD/seed_${s}.log"
  {
    echo "===== seed $s START $(date -u) on GPU $gpu ====="
    # per-seed pipeline: multiseed + band + crossing(+same_bin+step_sweep) + torus
    CUDA_VISIBLE_DEVICES=$gpu "$PY" code/reproduce_spatial_shift_results.py \
      --analysis_subdir "$SUB" --space_projection path --max_shift_cm 20 --shift_step_cm 1 \
      --Ng_use 4096 "${COMMON_MODEL[@]}" \
      --skip_cell_summary --skip_breadth --skip_full_ablation \
      --device cuda --checkpoint_paths "$ckpt"
    local rc1=$?
    echo "----- seed $s pipeline rc=$rc1; starting full ablation $(date -u) -----"
    # per-seed full grid-random ablation, 0..100%, own output dir (no clobber)
    CUDA_VISIBLE_DEVICES=$gpu "$PY" code/predictive_retrospective_ablation.py \
      --checkpoint_paths "$ckpt" --analysis_subdir "$SUB" \
      --ablate_percentiles 0 10 20 25 30 40 50 60 70 75 80 90 100 \
      --random_pool grid --random_trials 10 --ablation_batches 8 --Ng_use 4096 \
      "${COMMON_MODEL[@]}" --gridness_threshold 0.2 --min_shift_cm 5.0 \
      --output_dir "$REPO/analysis_outputs/Single agent path integration/Seed $s/$SUB/full_ablation" \
      --device cuda
    local rc2=$?
    echo "===== seed $s DONE pipeline_rc=$rc1 ablation_rc=$rc2 $(date -u) ====="
  } >> "$log" 2>&1
}

{
  echo "########## PHASE 1: per-seed parallel ($(date -u)) ##########"
  pids=()
  for s in 0 1 2 3 4 5 6 7 8 9; do
    run_seed "$s" &
    pids+=($!)
  done
  echo "launched ${#pids[@]} seed workers: ${pids[*]}"
  fail=0
  for p in "${pids[@]}"; do wait "$p" || fail=$((fail+1)); done
  echo "########## PHASE 1 done; worker failures=$fail ($(date -u)) ##########"

  echo "########## PHASE 2: aggregates ($(date -u)) ##########"
  CK=(); for s in 0 1 2 3 4 5 6 7 8 9; do CK+=("$MODEL_ROOT/Seed $s/most_recent_model.pth"); done
  "$PY" code/cell_class_summary_across_seeds.py --checkpoint_paths "${CK[@]}" \
      --analysis_subdir "$SUB" --Ng_use 4096 --gridness_threshold 0.2 --min_shift_cm 5.0 || echo "cell_summary FAILED"
  "$PY" code/predictive_shift_breadth_across_seeds.py \
      --analysis-root "$REPO/analysis_outputs" --model-subdir "Single agent path integration" \
      --analysis-subdir "$SUB" || echo "breadth FAILED"
  # merge per-seed ablation results.json into one combined file
  "$PY" - <<'PYMERGE' || echo "ablation merge FAILED"
import json, glob, os
REPO="/home/ec2-user/predictive-grid-cell-analysis"
SUB="spatial_shift_allunits"
files=sorted(glob.glob(f"{REPO}/analysis_outputs/Single agent path integration/Seed */{SUB}/full_ablation/results.json"))
if files:
    base=json.load(open(files[0])); seeds=[]
    for f in files:
        d=json.load(open(f)); seeds+=d.get("seeds",[])
    base["seeds"]=seeds
    out=f"{REPO}/analysis_outputs/Single agent path integration/summary/predictive_retrospective_ablation_{SUB}/results.json"
    os.makedirs(os.path.dirname(out),exist_ok=True)
    json.dump(base,open(out,"w"),indent=2)
    print(f"merged {len(files)} seeds -> {out}")
else:
    print("no per-seed ablation results found to merge")
PYMERGE
  echo "########## ALL DONE ($(date -u)) ##########"
  touch /home/ec2-user/allunits_parallel.DONE
} >> "$LOGD/driver.log" 2>&1
