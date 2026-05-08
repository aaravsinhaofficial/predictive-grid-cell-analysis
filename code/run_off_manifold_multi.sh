#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

foreground=0
args=()
for arg in "$@"; do
  if [[ "$arg" == "--foreground" ]]; then
    foreground=1
  else
    args+=("$arg")
  fi
done

cmd=(python "$ROOT/single_seed_off_manifold_distance.py" \
  --trajectory-styles random_walk,straight \
  --ablation-percentages 25,75,100)
if ((${#args[@]})); then
  cmd+=("${args[@]}")
fi

if [[ "$foreground" -eq 1 ]]; then
  "${cmd[@]}"
else
  log="$ROOT/off_manifold_run_$(date +%Y%m%d_%H%M%S).log"
  nohup "${cmd[@]}" > "$log" 2>&1 &
  echo "Started: ${cmd[*]}"
  echo "Log: $log"
fi
