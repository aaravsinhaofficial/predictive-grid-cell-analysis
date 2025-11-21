[![DOI](https://zenodo.org/badge/217773694.svg)](https://zenodo.org/badge/latestdoi/217773694)

# Grid cells in RNNs trained to path integrate

Code to reproduce the trained RNN in [**a unified theory for the origin of grid cells through the lens of pattern formation (NeurIPS '19)**](https://papers.nips.cc/paper/9191-a-unified-theory-for-the-origin-of-grid-cells-through-the-lens-of-pattern-formation) and additional analysis described in this [**preprint**](https://www.biorxiv.org/content/10.1101/2020.12.29.424583v1). 


Quick start:

<img src="./docs/poisson_spiking.gif" width="300" align="right">

* [**inspect_model.ipynb**](inspect_model.ipynb):
  Train a model and visualize its hidden unit ratemaps. 
 
* [**main.py**](main.py):
  or, train a model from the command line.
  
* [**pattern_formation.ipynb**](pattern_formation.ipynb):
  Numerical simulations of pattern-forming dynamics.
  
  
Includes:

* [**trajectory_generator.py**](trajectory_generator.py):
  Generate simulated rat trajectories in a rectangular environment.

* [**place_cells.py**](place_cells.py):
  Tile a set of simulated place cells across the training environment. 
  
* [**model.py**](model.py):
  Contains the vanilla RNN model architecture, as well as an LSTM.
  
* [**trainer.py**](model.py):
  Contains model training loop.
  
* [**models/example_trained_weights.npy**](models/example_trained_weights.npy)
  Contains a set of pre-trained weights.

## Predictive grid-cell analysis

[`multi_seed_predictive_analysis.py`](multi_seed_predictive_analysis.py) now provides a richer view of predictive grid cells and their functional role.

### Controlling movement statistics

The trajectory generator can be steered directly from the CLI so you can probe how speed, smoothness, and wall avoidance affect the predictive vs non-predictive split that appears in the scatter plot saved to `analysis_outputs/gridness_zero_vs_shift.png`.

Key flags:

* `--traj_speed_scale` multiplies the Rayleigh speed scale (use values >1 to raise the max speed).
* `--traj_speed_max` clamps the forward speed in m/s, and `--traj_velocity_smoothing` (0–0.99) adds an EMA for smoother motion.
* `--traj_turn_sigma_scale` reduces or amplifies the turning noise, while `--traj_border_region`, `--traj_wall_slowdown`, and `--traj_wall_turn_scale` control how aggressively trajectories steer away from boundaries.

Example:

```bash
python multi_seed_predictive_analysis.py \
  --checkpoint_paths models_trained_aarav/model.pth \
  --traj_speed_scale 1.4 \
  --traj_velocity_smoothing 0.35 \
  --traj_wall_slowdown 0.35 \
  --traj_turn_sigma_scale 0.6
```

The scatter plot is now color-coded by predictive class (predictive, phase-precession, phase-locked, low-grid, and unclassified), making it easy to see how the distribution of predictive cells shifts as you vary the movement statistics.

### Predictive vs random ablations

Set `--ablation_batches` (>0) to enable unit ablation sweeps that quantify how much predictive cells contribute to decoding accuracy. The script caches the same trajectories for every condition, zeros out the encoder/decoder/recurrent weights of either:

1. All predictive grid cells (`predictive_ablation_effects.png`, blue vs red bars), or
2. Randomly selected non-predictive units (grey swarm, averaged over `--ablation_random_trials` repeats).

Per-checkpoint outputs land in `<checkpoint_dir>/analysis_outputs/`:

* `predictive_ablation_effects.png` – bar + scatter plot comparing baseline, predictive-ablated, and random-ablated decoding errors.
* `predictive_ablation_metrics.json` – exact numbers (cm) plus the number of units removed each time.
* `gridness_data.npz` – now also stores `low_grid_units` and the ablation metrics, alongside the existing class assignments.

Use `--rng_seed` to make random ablations reproducible across runs (each checkpoint increments the seed automatically).

### Predictive vs retrospective summaries

[`predictive_retrospective_summary.py`](predictive_retrospective_summary.py) packages the meeting goals listed in `docs/meeting_notes_2025-11-14.md` into a single figure plus machine-readable summaries. For each checkpoint it:

1. Classifies predictive, retrospective, and zero-shift grid cells at the requested `gridness` thresholds (default 0.5).
2. Tallies counts, plots the preferred-shift distribution, and compares zero-shift vs peak-shift gridness (panels A–C of the exported PNG).
3. Optionally runs shuffle controls (`--shuffle_trials 100 --shuffle_alpha 0.05`) so a unit’s preferred lag must exceed the 95th percentile of its randomized activations.

Example:

```bash
python predictive_retrospective_summary.py \
  --checkpoint_path models_trained_aarav/steps_20_batch_100_RNN_2048_relu_rf_012_DoG_True_periodic_False_lr_00001_weight_decay_00001/most_recent_model.pth \
  --n_batches 25 \
  --Ng_use 512 \
  --gridness_threshold 0.5 \
  --zero_shift_threshold 0.5 \
  --min_shift_cm 5 \
  --shuffle_trials 0
```

Outputs land in `<checkpoint_dir>/analysis_outputs/predictive_retrospective/`:

* `*_summary.png` – the consolidated figure addressing points (1)–(3).
* `*_summary.json` – counts, mean/median preferred shifts, and average gridness per class.
* `*_summary_data.npz` – raw `scores_60/90`, preferred lags, and (if enabled) shuffle thresholds for downstream notebooks.

## Running

We recommend creating a virtual environment:

```shell
$ virtualenv env
$ source env/bin/activate
$ pip install --upgrade pip
```

Then, install the dependencies automatically with `pip install -r requirements.txt`
or manually with:

```shell
$ pip install --upgrade numpy==1.17.2
$ pip install --upgrade tensorflow==2.0.0rc2
$ pip install --upgrade scipy==1.4.1
$ pip install --upgrade matplotlib==3.0.3
$ pip install --upgrade imageio==2.5.0
$ pip install --upgrade opencv-python==4.1.1.26
$ pip install --upgrade tqdm==4.36.0
$ pip install --upgrade opencv-python==4.1.1.26
$ pip install --upgrade torch==1.10.0
```

If you want to train your own models, make sure to properly set the default
save directory in `main.py`! 

## Result

![grid visualization](./docs/RNNgrids.png)
