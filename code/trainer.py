# -*- coding: utf-8 -*-
import json
import os
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch

from visualize import predictive_gridness_analysis, save_ratemaps


class Trainer(object):
    def __init__(self, options, model, trajectory_generator, restore=True):
        self.options = options
        self.model = model
        self.trajectory_generator = trajectory_generator
        lr = self.options.learning_rate
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

        self.loss = []
        self.err = []
        self.history: Dict[str, List[float]] = {"epoch": [], "loss": [], "err": []}
        self.grid_history: List[Dict[str, Any]] = []

        # Set up checkpoints
        self.ckpt_dir = os.path.join(options.save_dir, options.run_ID)
        ckpt_path = os.path.join(self.ckpt_dir, 'most_recent_model.pth')
        if restore and os.path.isdir(self.ckpt_dir) and os.path.isfile(ckpt_path):
            self.model.load_state_dict(torch.load(ckpt_path))
            print("Restored trained model from {}".format(ckpt_path))
        else:
            if not os.path.isdir(self.ckpt_dir):
                os.makedirs(self.ckpt_dir, exist_ok=True)
            print("Initializing new model from scratch.")
            print("Saving to: {}".format(self.ckpt_dir))
        self.metrics_path = os.path.join(self.ckpt_dir, "training_metrics.json")
        self.metrics_plot_path = os.path.join(self.ckpt_dir, "training_curves.png")

    def train_step(self, inputs, pc_outputs, pos):
        ''' 
        Train on one batch of trajectories.

        Args:
            inputs: Batch of 2d velocity inputs with shape [batch_size, sequence_length, 2].
            pc_outputs: Ground truth place cell activations with shape 
                [batch_size, sequence_length, Np].
            pos: Ground truth 2d position with shape [batch_size, sequence_length, 2].

        Returns:
            loss: Avg. loss for this training batch.
            err: Avg. decoded position error in cm.
        '''
        self.model.zero_grad()

        loss, err = self.model.compute_loss(inputs, pc_outputs, pos)

        loss.backward()
        self.optimizer.step()

        return loss.item(), err.item()

    def train(self, n_epochs: int = 1000, n_steps=10, save=True):
        ''' 
        Train model on simulated trajectories.

        Args:
            n_steps: Number of training steps
            save: If true, save a checkpoint after each epoch.
        '''

        # Construct generator
        gen = self.trajectory_generator.get_generator()

        # tbar = tqdm(range(n_steps), leave=False)
        for epoch_idx in range(n_epochs):
            epoch_losses: List[float] = []
            epoch_errs: List[float] = []
            for step_idx in range(n_steps):
                inputs, pc_outputs, pos = next(gen)
                loss, err = self.train_step(inputs, pc_outputs, pos)
                self.loss.append(loss)
                self.err.append(err)
                epoch_losses.append(loss)
                epoch_errs.append(err)

                # Log error rate to progress bar
                # tbar.set_description('Error = ' + str(np.int(100*err)) + 'cm')

                print('Epoch: {}/{}. Step {}/{}. Loss: {}. Err: {}cm'.format(
                    epoch_idx, n_epochs, step_idx, n_steps,
                    np.round(loss, 2), np.round(100 * err, 2)))

            mean_loss = float(np.mean(epoch_losses))
            mean_err = float(np.mean(epoch_errs))
            self.history["epoch"].append(epoch_idx)
            self.history["loss"].append(mean_loss)
            self.history["err"].append(mean_err)
            self._save_metrics_file()
            self._plot_training_curves()

            if save:
                # Save checkpoint
                ckpt_path = os.path.join(self.ckpt_dir, 'epoch_{}.pth'.format(epoch_idx))
                torch.save(self.model.state_dict(), ckpt_path)
                torch.save(self.model.state_dict(), os.path.join(self.ckpt_dir,
                                                                 'most_recent_model.pth'))

                # Save a picture of rate maps
                save_ratemaps(self.model, self.trajectory_generator,
                              self.options, step=epoch_idx)

            # Optional predictive gridness diagnostics
            self._maybe_run_grid_eval(epoch_idx, mean_loss, mean_err)

    def _cm_per_step(self, xs: np.ndarray, ys: np.ndarray) -> float:
        dx = np.diff(xs, axis=0)
        dy = np.diff(ys, axis=0)
        step = np.sqrt(dx ** 2 + dy ** 2)
        return float(np.nanmean(step) * 100.0)

    def _nanargmax_with_fill(self, arr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Per-column nanargmax with safe defaults."""
        arr = np.asarray(arr)
        idxs = np.full(arr.shape[1], -1, dtype=int)
        vals = np.full(arr.shape[1], np.nan, dtype=float)
        for u in range(arr.shape[1]):
            col = arr[:, u]
            if np.isfinite(col).any():
                i = int(np.nanargmax(col))
                idxs[u] = i
                vals[u] = col[i]
        return idxs, vals

    def _maybe_run_grid_eval(self, epoch_idx: int, mean_loss: float, mean_err: float) -> None:
        interval = getattr(self.options, "grid_eval_interval", 0)
        if interval is None or interval <= 0:
            return
        if (epoch_idx + 1) % interval != 0:
            return

        lags = list(getattr(self.options, "grid_eval_lags", [0, 1, 2, 3, 4, 5]))
        if 0 not in lags:
            lags = [0] + lags
        n_batches = int(getattr(self.options, "grid_eval_batches", 5))
        threshold = float(getattr(self.options, "grid_eval_threshold", 0.3))
        max_units = getattr(self.options, "grid_eval_max_units", None)
        res = int(getattr(self.options, "grid_eval_res", 20))

        Ng_eval = self.model.Ng if max_units is None else min(int(max_units), self.model.Ng)
        idxs = np.arange(Ng_eval)

        self.model.eval()
        with torch.no_grad():
            scores_60, scores_90, xs, ys, activations, scorer = predictive_gridness_analysis(
                self.model,
                self.trajectory_generator,
                self.options,
                lags=lags,
                res=res,
                n_batches=n_batches,
                Ng=Ng_eval,
                idxs=idxs,
            )
        self.model.train()

        lags_arr = np.array(lags)
        cm_step = self._cm_per_step(xs, ys)
        lag_cm = lags_arr * cm_step

        zero_mask = lags_arr == 0
        zero_scores = scores_60[zero_mask][0] if np.any(zero_mask) else np.full(Ng_eval, np.nan)
        positive_mask = lags_arr > 0
        if np.any(positive_mask):
            positive_scores = scores_60[positive_mask]
            best_idx, best_vals = self._nanargmax_with_fill(positive_scores)
            predictive_scores = np.full(Ng_eval, np.nan, dtype=float)
            predictive_scores[np.where(best_idx >= 0)] = best_vals[np.where(best_idx >= 0)]
            pos_lags_cm = lag_cm[positive_mask]
            best_shift_cm = np.full(Ng_eval, np.nan, dtype=float)
            valid = best_idx >= 0
            best_shift_cm[valid] = pos_lags_cm[best_idx[valid]]
        else:
            predictive_scores = np.full(Ng_eval, np.nan, dtype=float)
            best_shift_cm = np.full(Ng_eval, np.nan, dtype=float)

        valid_zero = np.isfinite(zero_scores)
        zero_fraction = float(np.mean(zero_scores[valid_zero] >= threshold)) if np.any(valid_zero) else 0.0
        valid_pred = np.isfinite(predictive_scores)
        predictive_fraction = float(np.mean(predictive_scores[valid_pred] >= threshold)) if np.any(valid_pred) else 0.0

        grid_entry: Dict[str, Any] = {
            "epoch": epoch_idx,
            "lags": lags,
            "lag_cm": lag_cm.tolist(),
            "cm_per_step": cm_step,
            "gridness_threshold": threshold,
            "zero_grid_fraction": zero_fraction,
            "predictive_grid_fraction": predictive_fraction,
            "mean_zero_gridness": float(np.nanmean(zero_scores)),
            "mean_best_positive_gridness": float(np.nanmean(predictive_scores)),
            "median_best_shift_cm": float(np.nanmedian(best_shift_cm)) if np.isfinite(best_shift_cm).any() else None,
            "mean_loss": mean_loss,
            "mean_err": mean_err,
        }
        self.grid_history.append(grid_entry)
        self._save_metrics_file()
        self._plot_training_curves()

    def _save_metrics_file(self) -> None:
        payload = {
            "epoch": self.history["epoch"],
            "loss": self.history["loss"],
            "err": self.history["err"],
            "grid_history": self.grid_history,
        }
        with open(self.metrics_path, "w") as f:
            json.dump(payload, f, indent=2)

    def _plot_training_curves(self) -> None:
        if not self.history["epoch"]:
            return
        fig, ax1 = plt.subplots(figsize=(8, 4.5))
        epochs = np.array(self.history["epoch"])
        ax1.plot(epochs, self.history["loss"], label="loss", color="#1f77b4")
        ax1.plot(epochs, self.history["err"], label="pos err (m)", color="#ff7f0e", linestyle="--")
        ax1.set_xlabel("epoch")
        ax1.set_ylabel("loss / position error")
        ax1.grid(alpha=0.25)

        ax2 = ax1.twinx()
        if self.grid_history:
            ge = np.array([g["epoch"] for g in self.grid_history])
            pred_frac = [g["predictive_grid_fraction"] for g in self.grid_history]
            zero_frac = [g["zero_grid_fraction"] for g in self.grid_history]
            ax2.plot(ge, pred_frac, label="predictive >= thr", color="#2ca02c")
            ax2.plot(ge, zero_frac, label="grid (0 shift)", color="#9467bd", linestyle=":")
            ax2.set_ylabel("fraction of units")
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")
        fig.tight_layout()
        fig.savefig(self.metrics_plot_path, dpi=200)
        plt.close(fig)
