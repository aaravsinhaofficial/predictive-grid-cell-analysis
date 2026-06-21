#!/usr/bin/env bash
# Re-run the ablation with random drawn from the ENTIRE population (--random_pool all),
# all 4096 units, 0..100%, count-matched, 10 trials. Reuses existing spatial_shift_allunits
# gridness. 10 seeds in parallel (GPU-bound, fast). Output -> .../full_ablation_allpool.
set -uo pipefail
REPO="/home/ec2-user/predictive-grid-cell-analysis"
PY="$REPO/.venv/bin/python"
SUB="spatial_shift_allunits"
LOGD="/home/ec2-user/ablation_allpool_logs"; mkdir -p "$LOGD"
cd "$REPO"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
MODEL_ROOT="Models/Single agent path integration"
COMMON=(--sequence_length 40 --weight_decay 1e-4 --box_width 2.2 --box_height 2.2
        --place_cell_rf 0.12 --surround_scale 2.0 --activation relu --learning_rate 1e-4)

run_seed () {
  local s="$1"; local gpu=$(( s % 2 ))
  {
    echo "===== seed $s allpool ablation START $(date -u) GPU $gpu ====="
    CUDA_VISIBLE_DEVICES=$gpu "$PY" code/predictive_retrospective_ablation.py \
      --checkpoint_paths "$MODEL_ROOT/Seed $s/most_recent_model.pth" --analysis_subdir "$SUB" \
      --ablate_percentiles 0 10 20 25 30 40 50 60 70 75 80 90 100 \
      --random_pool all --random_trials 10 --ablation_batches 8 --Ng_use 4096 \
      "${COMMON[@]}" --gridness_threshold 0.2 --min_shift_cm 5.0 \
      --output_dir "$REPO/analysis_outputs/Single agent path integration/Seed $s/$SUB/full_ablation_allpool" \
      --device cuda
    echo "===== seed $s allpool ablation DONE rc=$? $(date -u) ====="
  } >> "$LOGD/seed_${s}.log" 2>&1
}

{
  echo "########## ALLPOOL ablation start $(date -u) ##########"
  pids=()
  for s in 0 1 2 3 4 5 6 7 8 9; do run_seed "$s" & pids+=($!); done
  echo "launched ${#pids[@]} jobs"
  fail=0; for p in "${pids[@]}"; do wait "$p" || fail=$((fail+1)); done
  echo "########## ALLPOOL done; failures=$fail $(date -u) ##########"
  # merge
  "$PY" - <<'PYM'
import json, glob, os
REPO="/home/ec2-user/predictive-grid-cell-analysis"; SUB="spatial_shift_allunits"
files=sorted(glob.glob(f"{REPO}/analysis_outputs/Single agent path integration/Seed */{SUB}/full_ablation_allpool/results.json"))
if files:
    base=json.load(open(files[0])); seeds=[]
    for f in files: seeds+=json.load(open(f)).get("seeds",[])
    base["seeds"]=seeds
    out=f"{REPO}/analysis_outputs/Single agent path integration/summary/predictive_retrospective_ablation_{SUB}_allpool/results.json"
    os.makedirs(os.path.dirname(out),exist_ok=True); json.dump(base,open(out,"w"),indent=2)
    print(f"merged {len(files)} -> {out}")
else: print("no allpool results found")
PYM
  touch /home/ec2-user/ablation_allpool.DONE
  echo "########## ALL DONE $(date -u) ##########"
} >> "$LOGD/driver.log" 2>&1
