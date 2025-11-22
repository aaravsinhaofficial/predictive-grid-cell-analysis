## Recreating the predictive vs retrospective summary figure

This walks you through regenerating the three-panel figure (counts, shift histogram, and zero-vs-shift gridness box plots) for predictive vs retrospective coding in the RNN.

### Prerequisites
- Working directory: `/Users/aaravsinha/grid-pattern-formation`
- Dependencies installed from `requirements.txt` (CPU is fine; GPU optional).
- Target checkpoint (example):  
  `Models/Single agent path integration/Seed 4 weight decay 1e-06/steps_20_batch_200_RNN_4096_relu_rf_012_DoG_True_periodic_False_lr_00001_weight_decay_1e-06/final_model.pth`

### Command
Run from the repo root:

```bash
python predictive_retrospective_summary.py \
  --checkpoint_path "Models/Single agent path integration/Seed 4 weight decay 1e-06/steps_20_batch_200_RNN_4096_relu_rf_012_DoG_True_periodic_False_lr_00001_weight_decay_1e-06/final_model.pth" \
  --n_batches 10 \
  --Ng_use 512 \
  --gridness_threshold 0.5 \
  --zero_shift_threshold 0.5 \
  --min_shift_cm 5 \
  --shuffle_trials 0
```

Notes:
- Increase `--n_batches` and/or `--Ng_use` for smoother statistics (at the cost of runtime).
- Add `--shuffle_trials 100 --shuffle_alpha 0.05` to require shuffle-derived significance for the preferred lag.
- Other trajectory options (speed, smoothing, wall behavior) can be changed via the CLI flags in `predictive_retrospective_summary.py` if you need to match a specific dataset.

### Outputs
Files are written to:
```
Models/Single agent path integration/Seed 4 weight decay 1e-06/steps_20_batch_200_RNN_4096_relu_rf_012_DoG_True_periodic_False_lr_00001_weight_decay_1e-06/analysis_outputs/predictive_retrospective/
```

- `final_model.pth_summary.png` — the figure (class counts, preferred-shift histogram, and zero vs shifted gridness box plots).
- `final_model.pth_summary.json` — counts, preferred-shift means/medians, and average gridness values (useful for captions).
- `final_model.pth_summary_data.npz` — raw lag/gridness arrays and class indices for custom plotting.

To process a different checkpoint, change the `--checkpoint_path` and the folder above will be created under that checkpoint’s directory.
