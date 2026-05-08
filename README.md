## How to recreate all key figures

These instructions assume the repository lives at `/Users/aaravsinha/grid-pattern-formation/predictive-grid-cell-analysis`. Run commands from that directory with dependencies from `requirements.txt` installed (CPU is fine; GPU optional). All scripts live in `code/`, so invoke them as `python code/<script>.py`.

### Training with predictability controls + live diagnostics
To train (or resume) with controllable trajectory predictability and per-epoch predictive-grid monitoring, see `docs/training_with_predictability.md`. It covers the new `--trajectory_style` options, grid-eval flags, and example commands for “lazy” vs “rich” regimes.

### 1) Predictive vs retrospective summary (3-panel figure)
Command (example for Seed 4, 4096-unit model):
```bash
python code/predictive_retrospective_summary.py \
  --checkpoint_path "Models/Single agent path integration/Seed 4 weight decay 1e-06/steps_20_batch_200_RNN_4096_relu_rf_012_DoG_True_periodic_False_lr_00001_weight_decay_1e-06/final_model.pth" \
  --n_batches 10 \
  --Ng_use 512 \
  --gridness_threshold 0.5 \
  --zero_shift_threshold 0.5 \
  --min_shift_cm 5 \
  --shuffle_trials 0
```
Outputs land in `analysis_outputs/<model>/<seed>/<run>/predictive_retrospective/`:
- `*_summary.png` – counts, preferred-shift histogram, and zero-vs-shift gridness box plots.
- `*_summary.json` – counts, preferred-shift stats, mean gridness per class.
- `*_summary_data.npz` – raw lags, gridness matrices, class indices (for custom plots).

Tip: add `--shuffle_trials 100 --shuffle_alpha 0.05` to require shuffle-significant lags.

### 2) Multi-checkpoint predictive analysis suite
Generates the scatter, class heatmaps, shift distribution, low-grid ratemaps, and ablation figure for each checkpoint.
```bash
python code/multi_seed_predictive_analysis.py \
  --checkpoint_paths "Models/Single agent path integration/Seed 4 weight decay 1e-06/steps_20_batch_200_RNN_4096_relu_rf_012_DoG_True_periodic_False_lr_00001_weight_decay_1e-06/final_model.pth" \
  --n_batches 25 \
  --Ng_use 512 \
  --gridness_threshold 0.2 \
  --low_grid_threshold 0.2 \
  --min_shift_cm 5 \
  --ablation_batches 8 \
  --ablation_random_trials 5
```
Outputs (per checkpoint) in `analysis_outputs/<model>/<seed>/<run>/`:
- `gridness_zero_vs_shift.png` – scatter of zero-lag vs best shifted gridness, colored by class.
- `predictive_classes.png` – heatmaps and average gridness curves for predictive / phase-precession / phase-locked units.
- `preferred_shift_distribution.png` – histogram of preferred shifts with ±min-shift overlay.
- `low_grid_ratemaps_lt_*.png` – sample rate maps for low-grid units.
- `predictive_ablation_effects.png` + `predictive_ablation_metrics.json` – decoding error before/after ablating predictive units vs random units.
- `gridness_data.npz` – lags, gridness matrices, class indices, ablation metrics.
- `summary.txt` – text diagnostics (cm-per-step, class counts, preferred shifts, ablation deltas).

Notes:
- `--checkpoint_paths` accepts multiple paths, directories, or globs.
- Increase `--n_batches` or `--Ng_use` for smoother statistics; set `--ablation_batches 0` to skip ablations.
- Trajectory knobs (speed, smoothing, wall behavior) are exposed via CLI flags in the script if you need to match a dataset.

### 3) Predictive/phase-precession/phase-locked heatmaps (single checkpoint)
If you just need the class heatmaps + average curves (no scatter/ablation), run:
```bash
python code/replicate_predictive_grid_figure.py \
  --checkpoint_path "Models/Single agent path integration/Seed 4 weight decay 1e-06/steps_20_batch_200_RNN_4096_relu_rf_012_DoG_True_periodic_False_lr_00001_weight_decay_1e-06/final_model.pth" \
  --save_path "analysis_outputs/Single agent path integration/Seed 4 weight decay 1e-06/steps_20_batch_200_RNN_4096_relu_rf_012_DoG_True_periodic_False_lr_00001_weight_decay_1e-06/predictive_heatmaps.png" \
  --n_batches 40 \
  --Ng_use 512 \
  --gridness_threshold 0.2 \
  --shift_threshold_cm 5
```
The figure is written to `--save_path` (directories auto-created). This script infers Ng/Np/velocity_dim from the checkpoint; override via CLI only if needed.

### 4) Functional-class rate maps and tuning (optional)
`code/plotting_functional_classes.py` provides rate maps plus head-direction tuning and summary for selected units. Example:
```bash
python code/plotting_functional_classes.py \
  --checkpoint_path "Models/Single agent path integration/Seed 4 weight decay 1e-06/steps_20_batch_200_RNN_4096_relu_rf_012_DoG_True_periodic_False_lr_00001_weight_decay_1e-06/final_model.pth" \
  --save_path "analysis_outputs/Single agent path integration/Seed 4 weight decay 1e-06/steps_20_batch_200_RNN_4096_relu_rf_012_DoG_True_periodic_False_lr_00001_weight_decay_1e-06/functional_classes.png" \
  --n_batches 25 \
  --Ng_use 256 \
  --gridness_threshold 0.2
```
Output goes to `--save_path`; tweak `--Ng_use`/`--n_batches` for smoother maps.

### 5) Toroidal structure with class-specific ablations
Requires `gridness_data.npz` beside the checkpoint (run `code/multi_seed_predictive_analysis.py` first).
```bash
python code/toroidal_structure_analysis.py \
  --checkpoint_path "Models/Single agent path integration/Seed 4 weight decay 1e-06/steps_20_batch_200_RNN_4096_relu_rf_012_DoG_True_periodic_False_lr_00001_weight_decay_1e-06/final_model.pth" \
  --n_batches 12 \
  --Ng_use 512 \
  --gridness_threshold 0.5 \
  --min_shift_cm 5 \
  --res 40
```
Outputs in `analysis_outputs/<model>/<seed>/<run>/torus/`:
- `torus_comparison.png` – 3D embeddings for baseline vs predictive / retrospective / normal grid-cell ablations.
- `torus_multiview.png`, `torus_seaborn.png` – alternate 3D/2D views.
- `torus_ring_metrics.png` – major/minor radius histograms.
- `torus_metrics.json` – lattice vectors, counts per class, decoding/radius stats.

### Switching checkpoints
Replace the `--checkpoint_path` in any command with another `.pth` file. Outputs are always placed under `analysis_outputs/<model>/<seed>/<run>/` (or at the provided `--save_path`).
