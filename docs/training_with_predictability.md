## Training with trajectory predictability + live grid-cell diagnostics

The training script now lets you control how predictable the agent’s motion is and optionally log how predictive/grid-like units emerge during training. No training is run here—use these commands when you’re ready.

### Trajectory predictability knobs
- `--trajectory_style`: `random_walk` (default, smooth turns), `straight` (fixed heading/speed), or `per_step_random` (new random heading + speed every step).
- `--trajectory_fixed_speed`: speed in m/s used when `trajectory_style=straight` (e.g., `0.8` keeps ~2 cm steps with the default `dt=0.02`).
- `--trajectory_turn_sigma_scale`: scale rotational noise (`<1` more predictable, `>1` more erratic).
- `--trajectory_speed_scale` / `--trajectory_speed_max`: scale or cap speeds.
- `--trajectory_velocity_smoothing`: EMA factor in `[0,1)`; >0 smooths speed changes (set to `0` for maximal randomness).
- Wall behavior: `--trajectory_border_region`, `--trajectory_wall_slowdown`, `--trajectory_wall_turn_scale`.

### Grid-cell emergence during training
- Enable with `--grid_eval_interval N` (runs predictive gridness diagnostics every N epochs; `0` disables).
- Other knobs: `--grid_eval_lags` (shifts in steps), `--grid_eval_threshold` (gridness cutoff), `--grid_eval_max_units` (subset for speed), `--grid_eval_batches`, `--grid_eval_res`.
- Outputs per run (under `models_trained_aarav/<run_id>/`):
  - `training_metrics.json`: epoch loss/err plus grid diagnostics (fractions above threshold, cm-per-step, shift stats).
  - `training_curves.png`: loss + position error over epochs, with optional predictive/zero-shift grid fractions overlaid.
  - `epoch_*.pth` + `most_recent_model.pth` checkpoints and per-epoch ratemap PNGs (existing behavior).

### Example commands
Predictable “lazy” regime (straight, fixed speed, low turn noise) with per-epoch grid checks:
```bash
python code/main.py \
  --n_epochs 50 --n_steps 250 \
  --trajectory_style straight --trajectory_fixed_speed 0.8 \
  --trajectory_turn_sigma_scale 0.1 --trajectory_velocity_smoothing 0.2 \
  --grid_eval_interval 1 --grid_eval_lags 0 1 2 3 4 5 6 7 \
  --grid_eval_threshold 0.3 --grid_eval_max_units 256
```

Rich/structure-seeking regime (max unpredictability):
```bash
python code/main.py \
  --n_epochs 50 --n_steps 250 \
  --trajectory_style per_step_random --trajectory_velocity_smoothing 0 \
  --trajectory_turn_sigma_scale 2.0 --trajectory_speed_scale 1.0 \
  --grid_eval_interval 2 --grid_eval_lags 0 1 2 3 4 5 6 7 8 9 \
  --grid_eval_threshold 0.3 --grid_eval_max_units 256
```

Interpretation tips:
- Predictive fraction rising while loss/err drop suggests the richer regime (learning structure); flat predictive fraction with low loss hints at memorization.
- Use higher `trajectory_turn_sigma_scale` and `trajectory_style=per_step_random` to force the network out of a lazy path-integrator; lower noise + `trajectory_style=straight` test how much structure emerges from predictable motion.

All flags can be combined, so you can sweep predictability settings and grid-eval cadence in a single run configuration.
