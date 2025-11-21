# Copyright 2018 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Grid score calculations.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal
import scipy.ndimage as ndimage
import scipy.stats


def circle_mask(size, radius, in_val=1.0, out_val=0.0):
  """Calculating the grid scores with different radius."""
  sz = [math.floor(size[0] / 2), math.floor(size[1] / 2)]
  x = np.linspace(-sz[0], sz[1], size[1])
  x = np.expand_dims(x, 0)
  x = x.repeat(size[0], 0)
  y = np.linspace(-sz[0], sz[1], size[1])
  y = np.expand_dims(y, 1)
  y = y.repeat(size[1], 1)
  z = np.sqrt(x**2 + y**2)
  z = np.less_equal(z, radius)
  vfunc = np.vectorize(lambda b: b and in_val or out_val)
  return vfunc(z)


class GridScorer(object):
  """Class for scoring ratemaps given trajectories."""

  def __init__(self, nbins, coords_range, mask_parameters, min_max=False):
    """Scoring ratemaps given trajectories.
    Args:
      nbins: Number of bins per dimension in the ratemap.
      coords_range: Environment coordinates range.
      mask_parameters: parameters for the masks that analyze the angular
        autocorrelation of the 2D autocorrelation.
      min_max: Correction.
    """
    self._nbins = nbins
    self._min_max = min_max
    self._coords_range = coords_range
    self._corr_angles = [30, 45, 60, 90, 120, 135, 150]
    # Create all masks
    self._masks = [(self._get_ring_mask(mask_min, mask_max), (mask_min,
                                                              mask_max))
                   for mask_min, mask_max in mask_parameters]
    # Mask for hiding the parts of the SAC that are never used
    self._plotting_sac_mask = circle_mask(
        [self._nbins * 2 - 1, self._nbins * 2 - 1],
        self._nbins,
        in_val=1.0,
        out_val=np.nan)

  def calculate_ratemap(self, xs, ys, activations, statistic='mean'):
    return scipy.stats.binned_statistic_2d(
        xs,
        ys,
        activations,
        bins=self._nbins,
        statistic=statistic,
        range=self._coords_range)[0]

  def _get_ring_mask(self, mask_min, mask_max):
    n_points = [self._nbins * 2 - 1, self._nbins * 2 - 1]
    return (circle_mask(n_points, mask_max * self._nbins) *
            (1 - circle_mask(n_points, mask_min * self._nbins)))

  def grid_score_60(self, corr):
    if self._min_max:
      return np.minimum(corr[60], corr[120]) - np.maximum(
          corr[30], np.maximum(corr[90], corr[150]))
    else:
      return (corr[60] + corr[120]) / 2 - (corr[30] + corr[90] + corr[150]) / 3

  def grid_score_90(self, corr):
    return corr[90] - (corr[45] + corr[135]) / 2

  def calculate_sac(self, seq1):
    """Calculating spatial autocorrelogram."""
    seq2 = seq1

    def filter2(b, x):
      stencil = np.rot90(b, 2)
      return scipy.signal.convolve2d(x, stencil, mode='full')

    seq1 = np.nan_to_num(seq1)
    seq2 = np.nan_to_num(seq2)

    ones_seq1 = np.ones(seq1.shape)
    ones_seq1[np.isnan(seq1)] = 0
    ones_seq2 = np.ones(seq2.shape)
    ones_seq2[np.isnan(seq2)] = 0

    seq1[np.isnan(seq1)] = 0
    seq2[np.isnan(seq2)] = 0

    seq1_sq = np.square(seq1)
    seq2_sq = np.square(seq2)

    seq1_x_seq2 = filter2(seq1, seq2)
    sum_seq1 = filter2(seq1, ones_seq2)
    sum_seq2 = filter2(ones_seq1, seq2)
    sum_seq1_sq = filter2(seq1_sq, ones_seq2)
    sum_seq2_sq = filter2(ones_seq1, seq2_sq)
    n_bins = filter2(ones_seq1, ones_seq2)
    n_bins_sq = np.square(n_bins)

    std_seq1 = np.power(
        np.subtract(
            np.divide(sum_seq1_sq, n_bins),
            (np.divide(np.square(sum_seq1), n_bins_sq))), 0.5)
    std_seq2 = np.power(
        np.subtract(
            np.divide(sum_seq2_sq, n_bins),
            (np.divide(np.square(sum_seq2), n_bins_sq))), 0.5)
    covar = np.subtract(
        np.divide(seq1_x_seq2, n_bins),
        np.divide(np.multiply(sum_seq1, sum_seq2), n_bins_sq))
    x_coef = np.divide(covar, np.multiply(std_seq1, std_seq2))
    x_coef = np.real(x_coef)
    x_coef = np.nan_to_num(x_coef)
    return x_coef

  def rotated_sacs(self, sac, angles):
    return [
        scipy.ndimage.interpolation.rotate(sac, angle, reshape=False)
        for angle in angles
    ]

  def get_grid_scores_for_mask(self, sac, rotated_sacs, mask):
    """Calculate Pearson correlations of area inside mask at corr_angles."""
    masked_sac = sac * mask
    ring_area = np.sum(mask)
    # Calculate dc on the ring area
    masked_sac_mean = np.sum(masked_sac) / ring_area
    # Center the sac values inside the ring
    masked_sac_centered = (masked_sac - masked_sac_mean) * mask
    variance = np.sum(masked_sac_centered**2) / ring_area + 1e-5
    corrs = dict()
    for angle, rotated_sac in zip(self._corr_angles, rotated_sacs):
      masked_rotated_sac = (rotated_sac - masked_sac_mean) * mask
      cross_prod = np.sum(masked_sac_centered * masked_rotated_sac) / ring_area
      corrs[angle] = cross_prod / variance
    return self.grid_score_60(corrs), self.grid_score_90(corrs), variance

  def get_scores(self, rate_map):
    """Get summary of scrores for grid cells."""
    sac = self.calculate_sac(rate_map)
    rotated_sacs = self.rotated_sacs(sac, self._corr_angles)

    scores = [
        self.get_grid_scores_for_mask(sac, rotated_sacs, mask)
        for mask, mask_params in self._masks  # pylint: disable=unused-variable
    ]
    scores_60, scores_90, variances = map(np.asarray, zip(*scores))  # pylint: disable=unused-variable
    max_60_ind = np.argmax(scores_60)
    max_90_ind = np.argmax(scores_90)

    return (scores_60[max_60_ind], scores_90[max_90_ind],
            self._masks[max_60_ind][1], self._masks[max_90_ind][1], sac, max_60_ind)

  def plot_ratemap(self, ratemap, ax=None, title=None, *args, **kwargs):  # pylint: disable=keyword-arg-before-vararg
    """Plot ratemaps."""
    if ax is None:
      ax = plt.gca()
    # Plot the ratemap
    ax.imshow(ratemap, interpolation='none', *args, **kwargs)
    # ax.pcolormesh(ratemap, *args, **kwargs)
    ax.axis('off')
    if title is not None:
      ax.set_title(title)
  
    def plot_sac(self,
                 sac,
                 mask_params=None,
                 ax=None,
                 title=None,
                 *args,
                 **kwargs):  # pylint: disable=keyword-arg-before-vararg
      """Plot spatial autocorrelogram."""
      if ax is None:
        ax = plt.gca()
      # Plot the sac
      useful_sac = sac * self._plotting_sac_mask
      ax.imshow(useful_sac, interpolation='none', *args, **kwargs)
      # ax.pcolormesh(useful_sac, *args, **kwargs)
      # Plot a ring for the adequate mask
      if mask_params is not None:
        center = self._nbins - 1
        ax.add_artist(
            plt.Circle(
                (center, center),
                mask_params[0] * self._nbins,
                # lw=bump_size,
                fill=False,
                edgecolor='k'))
        ax.add_artist(
            plt.Circle(
                (center, center),
                mask_params[1] * self._nbins,
                # lw=bump_size,
                fill=False,
                edgecolor='k'))
      ax.axis('off')
      if title is not None:
        ax.set_title(title)

  def _predictive_scores_single(self, xs, ys, activations, lags, statistic='mean'):
    """Compute gridness scores as a function of time lag for a single unit.

    Args:
      xs, ys: Arrays shaped [T, B] with positions per time step and batch.
      activations: Array shaped [T, B] with unit activation over time/batch.
      lags: Iterable of integer lags (positive = predict future position).
      statistic: Aggregation for ratemap binning.

    Returns:
      (scores_60, scores_90): Arrays of shape [len(lags)].
    """
    xs = np.asarray(xs)
    ys = np.asarray(ys)
    activations = np.asarray(activations)
    assert xs.shape == ys.shape == activations.shape, 'xs, ys, activations must have same [T,B] shape'

    T = activations.shape[0]
    scores_60 = []
    scores_90 = []
    for lag in lags:
      if lag >= 0:
        # Align activations[t] with position[t+lag]
        if T - lag <= 1:
          scores_60.append(np.nan)
          scores_90.append(np.nan)
          continue
        xs_l = xs[lag:]
        ys_l = ys[lag:]
        a_l = activations[:T - lag]
      else:
        k = -lag
        if T - k <= 1:
          scores_60.append(np.nan)
          scores_90.append(np.nan)
          continue
        xs_l = xs[:T - k]
        ys_l = ys[:T - k]
        a_l = activations[k:]

      # Flatten and bin into ratemap with future (or past) positions
      rm = self.calculate_ratemap(xs_l.reshape(-1), ys_l.reshape(-1), a_l.reshape(-1), statistic=statistic)
      s60, s90, _, _, _, _ = self.get_scores(rm)
      scores_60.append(s60)
      scores_90.append(s90)

    return np.asarray(scores_60), np.asarray(scores_90)

  def predictive_grid_scores(self, xs, ys, activations, lags, unit_idx=None, statistic='mean'):
    """Predictive gridness across time lags.

    Computes gridness (60° and 90°) for activations aligned to future or past
    positions. Positive lag means "predictive" (activity at t vs position at t+lag).

    Args:
      xs, ys: Arrays of shape [T, B] (positions over time and batch).
      activations:
        - shape [T, B] for a single unit, or
        - shape [T, B, Ng] for multiple units.
      lags: Iterable of integer time lags to evaluate.
      unit_idx: If activations is 3D, optionally select a unit index. If None,
        returns scores for all units.
      statistic: Aggregation for ratemap binning.

    Returns:
      If activations is 2D or unit_idx is provided: (scores_60, scores_90)
        each of shape [len(lags)].
      If activations is 3D and unit_idx is None:
        (scores_60, scores_90) each of shape [len(lags), Ng].
    """
    xs = np.asarray(xs)
    ys = np.asarray(ys)
    activations = np.asarray(activations)

    if activations.ndim == 2:
      return self._predictive_scores_single(xs, ys, activations, lags, statistic)
    elif activations.ndim == 3:
      T, B, Ng = activations.shape
      if unit_idx is not None:
        return self._predictive_scores_single(xs, ys, activations[:, :, unit_idx], lags, statistic)
      # All units
      scores_60 = np.zeros((len(lags), Ng))
      scores_90 = np.zeros((len(lags), Ng))
      for u in range(Ng):
        s60, s90 = self._predictive_scores_single(xs, ys, activations[:, :, u], lags, statistic)
        scores_60[:, u] = s60
        scores_90[:, u] = s90
      return scores_60, scores_90
    else:
      raise ValueError('activations must have shape [T,B] or [T,B,Ng]')

  def best_predictive_lag(self, xs, ys, activations, lags, unit_idx=None, metric='60', statistic='mean'):
    """Return best lag(s) maximizing gridness.

    Args:
      xs, ys: [T,B]
      activations: [T,B] or [T,B,Ng]
      lags: iterable of ints
      unit_idx: optional unit index when activations is 3D
      metric: '60' or '90'
    Returns:
      If single unit: (best_lag, best_score)
      If Ng units: (best_lags, best_scores) with shape [Ng]
    """
    s60, s90 = self.predictive_grid_scores(xs, ys, activations, lags, unit_idx=unit_idx, statistic=statistic)
    if activations.ndim == 2 or unit_idx is not None:
      scores = s60 if metric == '60' else s90
      idx = int(np.nanargmax(scores))
      return lags[idx], scores[idx]
    else:
      scores = s60 if metric == '60' else s90  # [L, Ng]
      idxs = np.nanargmax(scores, axis=0)
      best_lags = np.array(lags)[idxs]
      best_scores = scores[idxs, np.arange(scores.shape[1])]
      return best_lags, best_scores

  def get_scores_with_shift(self, xs, ys, activations, shift, unit_idx=None, statistic='mean', return_maps=False):
    """Gridness at a specific time shift.

    Positive shift K aligns activity at t with position at t+K (predictive).
    Negative shift aligns activity at t with position at t-K (postdictive).

    Args:
      xs, ys: Arrays [T,B] with positions over time and batch.
      activations: [T,B] for one unit or [T,B,Ng] for many units.
      shift: int, time shift to evaluate.
      unit_idx: optional int, choose a single unit when activations is 3D.
      statistic: ratemap aggregation (default 'mean').
      return_maps: if True and single unit is selected, also return (ratemap, sac).

    Returns:
      If single unit: (score_60, score_90[, ratemap, sac]).
      If multiple units and unit_idx is None: (scores_60, scores_90), arrays [Ng].
    """
    xs = np.asarray(xs)
    ys = np.asarray(ys)
    activations = np.asarray(activations)

    T = xs.shape[0]
    if abs(shift) >= T:
      # Not enough overlap to compute anything meaningful
      if return_maps:
        return np.nan, np.nan, None, None
      return np.nan, np.nan

    if shift >= 0:
      xs_l = xs[shift:]
      ys_l = ys[shift:]
      if activations.ndim == 2:
        a_l = activations[:activations.shape[0] - shift]
      else:
        a_l = activations[:activations.shape[0] - shift]
    else:
      k = -shift
      xs_l = xs[:xs.shape[0] - k]
      ys_l = ys[:ys.shape[0] - k]
      if activations.ndim == 2:
        a_l = activations[k:]
      else:
        a_l = activations[k:]

    if xs_l.size == 0 or ys_l.size == 0:
      if return_maps:
        return np.nan, np.nan, None, None
      return np.nan, np.nan

    if activations.ndim == 2 or unit_idx is not None:
      if activations.ndim == 3:
        a_use = a_l[:, :, unit_idx]
      else:
        a_use = a_l
      rm = self.calculate_ratemap(xs_l.reshape(-1), ys_l.reshape(-1), a_use.reshape(-1), statistic=statistic)
      s60, s90, _, _, sac, _ = self.get_scores(rm)
      if return_maps:
        return s60, s90, rm, sac
      return s60, s90
    elif activations.ndim == 3:
      T, B, Ng = a_l.shape
      scores_60 = np.zeros((Ng,), dtype=np.float32)
      scores_90 = np.zeros((Ng,), dtype=np.float32)
      flat_x = xs_l.reshape(-1)
      flat_y = ys_l.reshape(-1)
      for u in range(Ng):
        rm = self.calculate_ratemap(flat_x, flat_y, a_l[:, :, u].reshape(-1), statistic=statistic)
        s60, s90, _, _, _, _ = self.get_scores(rm)
        scores_60[u] = s60
        scores_90[u] = s90
      return scores_60, scores_90
    else:
      raise ValueError('activations must have shape [T,B] or [T,B,Ng]')


def border_score(rm, res, box_width):
	# Find connected firing fields
    pix_area = 100**2*box_width**2/res**2
    rm_thresh =  rm>(rm.max()*0.3)
    rm_comps, ncomps = ndimage.measurements.label(rm_thresh)

    # Keep fields with area > 200cm^2
    masks = []
    nfields = 0
    for i in range(1,ncomps+1):
        mask = (rm_comps==i).reshape(res,res)
        if mask.sum()*pix_area > 200:
            masks.append(mask)
            nfields += 1
            
    # Max coverage of any one field over any one border
    cm_max = 0
    for mask in masks:
        mask = masks[0]
        n_cov = mask[0].mean()
        s_cov = mask[-1].mean()
        e_cov = mask[:,0].mean()
        w_cov = mask[:,-1].mean()
        cm = np.max([n_cov,s_cov,e_cov,w_cov])
        if cm>cm_max:
            cm_max = cm

    # Distance to nearest wall
    x,y = np.mgrid[:res,:res] + 1
    x = x.ravel()
    y = y.ravel()
    xmin = np.min(np.vstack([x,res+1-x]),0)
    ymin = np.min(np.vstack([y,res+1-y]),0)
    dweight = np.min(np.vstack([xmin,ymin]),0).reshape(res,res)
    dweight = dweight*box_width/res

    # Mean firing distance
    dms = []
    for mask in masks:
        field = rm[mask]
        field /= field.sum()   # normalize
        dm = (field*dweight[mask]).sum()
        dms.append(dm)
    dm = np.nanmean(dms) / (box_width/2)
    border_score = (cm_max-dm)/(cm_max+dm)
    return border_score, cm_max, dm
