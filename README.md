Instructions to recreate figures:
Recreate Current Predictive vs Retrospective Figure

Work from repo root /Users/aaravsinha/grid-pattern-formation with dependencies from requirements.txt installed (CPU is fine; GPU optional).
The figure targets checkpoint Models/Single agent path integration/Seed 4 weight decay 1e-06/steps_20_batch_200_RNN_4096_relu_rf_012_DoG_True_periodic_False_lr_00001_weight_decay_1e-06/final_model.pth.
Run:
python predictive_retrospective_summary.py \
  --checkpoint_path "Models/Single agent path integration/Seed 4 weight decay 1e-06/steps_20_batch_200_RNN_4096_relu_rf_012_DoG_True_periodic_False_lr_00001_weight_decay_1e-06/final_model.pth" \
  --n_batches 10 \
  --Ng_use 512 \
  --gridness_threshold 0.5 \
  --zero_shift_threshold 0.5 \
  --min_shift_cm 5 \
  --shuffle_trials 0
(Adjust --n_batches or --Ng_use upward for smoother stats, and add --shuffle_trials 100 --shuffle_alpha 0.05 if you need shuffle significance.)
Outputs appear at Models/Single agent path integration/Seed 4 weight decay 1e-06/steps_20_batch_200_RNN_4096_relu_rf_012_DoG_True_periodic_False_lr_00001_weight_decay_1e-06/analysis_outputs/predictive_retrospective/:
final_model.pth_summary.png – three-panel figure (class counts, shift histogram, zero-vs-shift gridness box plots) showing predictive vs retrospective coding in the RNN.
final_model.pth_summary.json – counts, preferred shift stats, average gridness values for captions or further analysis.
final_model.pth_summary_data.npz – raw lag/gridness arrays if you need to build additional plots or probe unit-level data.
