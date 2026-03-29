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

  def _wrap_coords(self, coords, axis):
    lo, hi = self._coords_range[axis]
    period = hi - lo
    if not np.isfinite(period) or period <= 0:
      return np.asarray(coords, dtype=float)
    return np.mod(np.asarray(coords, dtype=float) - lo, period) + lo

  def _unwrap_coords(self, coords, axis):
    coords = np.asarray(coords, dtype=float)
    if coords.ndim != 2:
      raise ValueError('coords must have shape [T,B]')
    lo, hi = self._coords_range[axis]
    period = hi - lo
    if not np.isfinite(period) or period <= 0 or coords.shape[0] <= 1:
      return coords.copy()

    diffs = np.diff(coords, axis=0)
    half_period = period / 2.0
    diffs = diffs.copy()
    diffs[diffs > half_period] -= period
    diffs[diffs < -half_period] += period

    unwrapped = np.empty_like(coords, dtype=float)
    unwrapped[0] = coords[0]
    unwrapped[1:] = coords[0:1] + np.cumsum(diffs, axis=0)
    return unwrapped

  def _shift_positions_by_space(self, xs, ys, shift_cm, periodic=False):
    """Shift positions along each trajectory by signed arc length."""
    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)
    if xs.shape != ys.shape or xs.ndim != 2:
      raise ValueError('xs and ys must share shape [T,B]')

    shift_m = float(shift_cm) / 100.0
    if periodic:
      x_path = self._unwrap_coords(xs, axis=0)
      y_path = self._unwrap_coords(ys, axis=1)
    else:
      x_path = xs.copy()
      y_path = ys.copy()

    T, B = xs.shape
    shifted_x = np.full((T, B), np.nan, dtype=float)
    shifted_y = np.full((T, B), np.nan, dtype=float)
    valid_mask = np.zeros((T, B), dtype=bool)

    for b in range(B):
      x_b = x_path[:, b]
      y_b = y_path[:, b]
      finite = np.isfinite(x_b) & np.isfinite(y_b)
      if np.count_nonzero(finite) == 0:
        continue

      orig_idx = np.where(finite)[0]
      x_valid = x_b[finite]
      y_valid = y_b[finite]

      if x_valid.size == 1:
        if abs(shift_m) <= 1e-12:
          idx0 = orig_idx[0]
          shifted_x[idx0, b] = xs[idx0, b]
          shifted_y[idx0, b] = ys[idx0, b]
          valid_mask[idx0, b] = True
        continue

      seg_lengths = np.sqrt(np.diff(x_valid)**2 + np.diff(y_valid)**2)
      cumdist = np.concatenate(([0.0], np.cumsum(seg_lengths)))
      target = cumdist + shift_m
      in_bounds = (target >= cumdist[0]) & (target <= cumdist[-1])
      if not np.any(in_bounds):
        continue

      dist_unique, unique_idx = np.unique(cumdist, return_index=True)
      x_unique = x_valid[unique_idx]
      y_unique = y_valid[unique_idx]
      if dist_unique.size == 1:
        if abs(shift_m) <= 1e-12:
          shifted_x[orig_idx, b] = xs[orig_idx, b]
          shifted_y[orig_idx, b] = ys[orig_idx, b]
          valid_mask[orig_idx, b] = True
        continue

      interp_x = np.interp(target[in_bounds], dist_unique, x_unique)
      interp_y = np.interp(target[in_bounds], dist_unique, y_unique)
      if periodic:
        interp_x = self._wrap_coords(interp_x, axis=0)
        interp_y = self._wrap_coords(interp_y, axis=1)

      tgt_idx = orig_idx[in_bounds]
      shifted_x[tgt_idx, b] = interp_x
      shifted_y[tgt_idx, b] = interp_y
      valid_mask[tgt_idx, b] = True

    return shifted_x, shifted_y, valid_mask

  def _estimate_heading_unit_vectors(self, xs, ys, periodic=False):
    """Estimate per-sample heading unit vectors from trajectory tangents."""
    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)
    if xs.shape != ys.shape or xs.ndim != 2:
      raise ValueError('xs and ys must share shape [T,B]')

    if periodic:
      x_path = self._unwrap_coords(xs, axis=0)
      y_path = self._unwrap_coords(ys, axis=1)
    else:
      x_path = xs.copy()
      y_path = ys.copy()

    # how much the position is changing between neighboring timesteps
    dx = np.gradient(x_path, axis=0)
    dy = np.gradient(y_path, axis=0)

    # normalize dx and dy to the unit length
    norm = np.sqrt(dx**2 + dy**2) # find magnitude of vector

    #find which timesteps are safe to divide by 
    ux = np.full_like(dx, np.nan, dtype=float)
    uy = np.full_like(dy, np.nan, dtype=float)
    valid = np.isfinite(norm) & (norm > 1e-12)

    # finally do the normalization for valid timesteps
    ux[valid] = dx[valid] / norm[valid]
    uy[valid] = dy[valid] / norm[valid]

    T, B = xs.shape

    # handle missing data
    for b in range(B):
      # ensure every timestep will have a heading
      valid_b = np.isfinite(ux[:, b]) & np.isfinite(uy[:, b])
      if not np.any(valid_b):
        continue
      valid_idx = np.where(valid_b)[0]
      first = valid_idx[0]
      last = valid_idx[-1]
      for t in range(first - 1, -1, -1):
        ux[t, b] = ux[first, b]
        uy[t, b] = uy[first, b]
      prev = first
      for t in valid_idx[1:]:
        for fill_t in range(prev + 1, t):
          ux[fill_t, b] = ux[prev, b]
          uy[fill_t, b] = uy[prev, b]
        prev = t
      for t in range(last + 1, T):
        ux[t, b] = ux[last, b]
        uy[t, b] = uy[last, b]

      norm_b = np.sqrt(ux[:, b]**2 + uy[:, b]**2)
      good = np.isfinite(norm_b) & (norm_b > 1e-12)
      ux[good, b] /= norm_b[good]
      uy[good, b] /= norm_b[good]

    return ux, uy

  def _shift_positions_by_heading(self, xs, ys, shift_cm, periodic=False):
    """Project positions by signed distance along the local heading direction."""
    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)
    if xs.shape != ys.shape or xs.ndim != 2:
      raise ValueError('xs and ys must share shape [T,B]')

    # convert inputs to float np arrays
    shift_m = float(shift_cm) / 100.0

    # unwrap coordinates for accurate heading estimation, but keep original wrapped
    if periodic:
      x_base = self._unwrap_coords(xs, axis=0)
      y_base = self._unwrap_coords(ys, axis=1)
    else:
      x_base = xs.copy()
      y_base = ys.copy()

  
    # get the normalized heading unit vectors at each time step
    ux, uy = self._estimate_heading_unit_vectors(xs, ys, periodic=periodic)

    # only include samples where we have valid position AND heading information (help skil over bad data w/o dealing w a loop)
    valid_mask = (np.isfinite(x_base) & np.isfinite(y_base) &
                  np.isfinite(ux) & np.isfinite(uy))
    
    # allocate arrays for the shifted positions, initialized to nan for invalid samples
    shifted_x = np.full_like(x_base, np.nan, dtype=float)
    shifted_y = np.full_like(y_base, np.nan, dtype=float)

    # ux, uy is the direction the rat is facing. shift_m is the +d cm label (converted to meters). The addition moves the point forward along the heading by exactly that distance
    # actually apply the shift along the heading direction
    shifted_x[valid_mask] = x_base[valid_mask] + shift_m * ux[valid_mask]
    shifted_y[valid_mask] = y_base[valid_mask] + shift_m * uy[valid_mask]

    # deal with periodic wrapping if needed. only apply to valid samples since we want to keep invalid samples as nan
    if periodic:
      shifted_x[valid_mask] = self._wrap_coords(shifted_x[valid_mask], axis=0)
      shifted_y[valid_mask] = self._wrap_coords(shifted_y[valid_mask], axis=1)

    return shifted_x, shifted_y, valid_mask

  def aligned_samples(self, xs, ys, activations, shift, shift_mode='time', periodic=False, space_projection='path'):
    """Return flattened position/activity samples aligned by time or space."""
    xs = np.asarray(xs)
    ys = np.asarray(ys)
    activations = np.asarray(activations)

    if xs.shape != ys.shape or xs.ndim != 2:
      raise ValueError('xs and ys must share shape [T,B]')
    if activations.shape[:2] != xs.shape:
      raise ValueError('activations must have leading shape [T,B]')

    if shift_mode == 'time':
      shift_steps = int(round(float(shift)))
      if not np.isclose(shift_steps, float(shift)):
        raise ValueError('time shifts must be integer-valued')
      T = xs.shape[0]
      if abs(shift_steps) >= T:
        return np.array([]), np.array([]), np.array([])

      if shift_steps >= 0:
        xs_l = xs[shift_steps:]
        ys_l = ys[shift_steps:]
        a_l = activations[:T - shift_steps]
      else:
        k = -shift_steps
        xs_l = xs[:T - k]
        ys_l = ys[:T - k]
        a_l = activations[k:]

      flat_x = xs_l.reshape(-1)
      flat_y = ys_l.reshape(-1)
      if activations.ndim == 2:
        flat_a = a_l.reshape(-1)
      else:
        flat_a = a_l.reshape(-1, activations.shape[-1])
      return flat_x, flat_y, flat_a

    if shift_mode == 'space':
      if space_projection == 'path':
        shifted_x, shifted_y, valid_mask = self._shift_positions_by_space(xs, ys, shift, periodic=periodic)
      elif space_projection == 'heading':
        shifted_x, shifted_y, valid_mask = self._shift_positions_by_heading(xs, ys, shift, periodic=periodic)
      else:
        raise ValueError("space_projection must be 'path' or 'heading'")
      if not np.any(valid_mask):
        return np.array([]), np.array([]), np.array([])
      flat_x = shifted_x[valid_mask]
      flat_y = shifted_y[valid_mask]
      if activations.ndim == 2:
        flat_a = activations[valid_mask]
      else:
        flat_a = activations[valid_mask, :]
      return flat_x, flat_y, flat_a

    raise ValueError("shift_mode must be 'time' or 'space'")

  def _predictive_scores_single(self, xs, ys, activations, lags, statistic='mean', shift_mode='time', periodic=False, space_projection='path'):
    """Compute gridness scores as a function of time or spatial shift for one unit.

    Args:
      xs, ys: Arrays shaped [T, B] with positions per time step and batch.
      activations: Array shaped [T, B] with unit activation over time/batch.
      lags: Iterable of shift values. In time mode these are integer steps; in
        space mode they are signed centimeters along the trajectory.
      statistic: Aggregation for ratemap binning.
      shift_mode: 'time' or 'space'.
      periodic: Whether to wrap interpolated spatial shifts to the box.
      space_projection: In spatial mode, 'path' uses arc-length interpolation
        along the realized trajectory and 'heading' projects along the local
        movement direction.

    Returns:
      (scores_60, scores_90): Arrays of shape [len(lags)].
    """
    xs = np.asarray(xs)
    ys = np.asarray(ys)
    activations = np.asarray(activations)
    assert xs.shape == ys.shape == activations.shape, 'xs, ys, activations must have same [T,B] shape'

    scores_60 = []
    scores_90 = []
    for lag in lags:
      flat_x, flat_y, flat_a = self.aligned_samples(
          xs,
          ys,
          activations,
          lag,
          shift_mode=shift_mode,
          periodic=periodic,
          space_projection=space_projection)
      if flat_x.size <= 1 or flat_y.size <= 1 or flat_a.size <= 1:
        scores_60.append(np.nan)
        scores_90.append(np.nan)
        continue

      rm = self.calculate_ratemap(flat_x, flat_y, flat_a, statistic=statistic)
      s60, s90, _, _, _, _ = self.get_scores(rm)
      scores_60.append(s60)
      scores_90.append(s90)

    return np.asarray(scores_60), np.asarray(scores_90)

  def predictive_grid_scores(self, xs, ys, activations, lags, unit_idx=None, statistic='mean', shift_mode='time', periodic=False, space_projection='path'):
    """Predictive gridness across temporal or spatial shifts.

    Computes gridness (60° and 90°) for activations aligned to future/past
    positions in time mode or to forward/backward trajectory positions in
    space mode. Positive shifts are predictive in either case.

    Args:
      xs, ys: Arrays of shape [T, B] (positions over time and batch).
      activations:
        - shape [T, B] for a single unit, or
        - shape [T, B, Ng] for multiple units.
      lags: Iterable of shift values to evaluate.
      unit_idx: If activations is 3D, optionally select a unit index. If None,
        returns scores for all units.
      statistic: Aggregation for ratemap binning.
      shift_mode: 'time' or 'space'.
      periodic: Whether to wrap interpolated spatial shifts to the box.
      space_projection: In spatial mode, 'path' or 'heading'.

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
      return self._predictive_scores_single(
          xs,
          ys,
          activations,
          lags,
          statistic=statistic,
          shift_mode=shift_mode,
          periodic=periodic,
          space_projection=space_projection)
    elif activations.ndim == 3:
      T, B, Ng = activations.shape
      if unit_idx is not None:
        return self._predictive_scores_single(
            xs,
            ys,
            activations[:, :, unit_idx],
            lags,
            statistic=statistic,
            shift_mode=shift_mode,
            periodic=periodic,
            space_projection=space_projection)
      # All units
      scores_60 = np.zeros((len(lags), Ng))
      scores_90 = np.zeros((len(lags), Ng))
      for u in range(Ng):
        s60, s90 = self._predictive_scores_single(
            xs,
            ys,
            activations[:, :, u],
            lags,
            statistic=statistic,
            shift_mode=shift_mode,
            periodic=periodic,
            space_projection=space_projection)
        scores_60[:, u] = s60
        scores_90[:, u] = s90
      return scores_60, scores_90
    else:
      raise ValueError('activations must have shape [T,B] or [T,B,Ng]')

  def best_predictive_lag(self, xs, ys, activations, lags, unit_idx=None, metric='60', statistic='mean', shift_mode='time', periodic=False, space_projection='path'):
    """Return best shift(s) maximizing gridness.

    Args:
      xs, ys: [T,B]
      activations: [T,B] or [T,B,Ng]
      lags: iterable of shift values
      unit_idx: optional unit index when activations is 3D
      metric: '60' or '90'
    Returns:
      If single unit: (best_lag, best_score)
      If Ng units: (best_lags, best_scores) with shape [Ng]
    """
    s60, s90 = self.predictive_grid_scores(
        xs,
        ys,
        activations,
        lags,
        unit_idx=unit_idx,
        statistic=statistic,
        shift_mode=shift_mode,
        periodic=periodic,
        space_projection=space_projection)
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

  def get_scores_with_shift(self, xs, ys, activations, shift, unit_idx=None, statistic='mean', return_maps=False, shift_mode='time', periodic=False, space_projection='path'):
    """Gridness at a specific time-step or spatial shift.

    Args:
      xs, ys: Arrays [T,B] with positions over time and batch.
      activations: [T,B] for one unit or [T,B,Ng] for many units.
      shift: shift value to evaluate. In time mode this is integer-valued
        steps; in space mode it is centimeters along the trajectory.
      unit_idx: optional int, choose a single unit when activations is 3D.
      statistic: ratemap aggregation (default 'mean').
      return_maps: if True and single unit is selected, also return (ratemap, sac).
      shift_mode: 'time' or 'space'.
      periodic: Whether to wrap interpolated spatial shifts to the box.
      space_projection: In spatial mode, 'path' or 'heading'.

    Returns:
      If single unit: (score_60, score_90[, ratemap, sac]).
      If multiple units and unit_idx is None: (scores_60, scores_90), arrays [Ng].
    """
    xs = np.asarray(xs)
    ys = np.asarray(ys)
    activations = np.asarray(activations)

    flat_x, flat_y, flat_a = self.aligned_samples(
        xs,
        ys,
        activations,
        shift,
        shift_mode=shift_mode,
        periodic=periodic,
        space_projection=space_projection)
    if flat_x.size == 0 or flat_y.size == 0 or flat_a.size == 0:
      if return_maps:
        return np.nan, np.nan, None, None
      return np.nan, np.nan

    if activations.ndim == 2 or unit_idx is not None:
      if activations.ndim == 3:
        if flat_a.ndim != 2:
          raise ValueError('Expected flattened multi-unit activations with shape [N,Ng]')
        a_use = flat_a[:, unit_idx]
      else:
        a_use = flat_a
      rm = self.calculate_ratemap(flat_x, flat_y, a_use.reshape(-1), statistic=statistic)
      s60, s90, _, _, sac, _ = self.get_scores(rm)
      if return_maps:
        return s60, s90, rm, sac
      return s60, s90
    elif activations.ndim == 3:
      if flat_a.ndim != 2:
        raise ValueError('Expected flattened multi-unit activations with shape [N,Ng]')
      Ng = flat_a.shape[1]
      scores_60 = np.zeros((Ng,), dtype=np.float32)
      scores_90 = np.zeros((Ng,), dtype=np.float32)
      for u in range(Ng):
        rm = self.calculate_ratemap(flat_x, flat_y, flat_a[:, u], statistic=statistic)
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


_BAND_CACHE = {}


def _band_library(res, box_width, k_values):
  """Precompute normalized sinusoids for band scoring."""
  k_values = np.asarray(k_values, dtype=float)
  key = (int(res), float(box_width), tuple(np.round(k_values, 6)))
  if key in _BAND_CACHE:
    return _BAND_CACHE[key]

  xs = np.linspace(-box_width / 2, box_width / 2, res)
  ys = np.linspace(-box_width / 2, box_width / 2, res)
  X, Y = np.meshgrid(xs, ys, indexing='ij')

  combos = []
  sinusoids = []
  for kx in k_values:
    for ky in k_values:
      if kx == 0 and ky == 0:
        continue
      combos.append((float(kx), float(ky)))
      sinusoids.append(np.cos(2 * np.pi * (kx * X + ky * Y)).reshape(-1))

  if not sinusoids:
    sinusoids = [np.zeros((res * res,), dtype=float)]
    combos = [(0.0, 0.0)]

  S = np.stack(sinusoids, axis=0).astype(np.float32)
  S = S - np.mean(S, axis=1, keepdims=True)
  S_std = np.std(S, axis=1, ddof=1, keepdims=True)
  S = S / (S_std + 1e-8)

  kxky = np.asarray(combos, dtype=np.float32)
  _BAND_CACHE[key] = (S, kxky)
  return S, kxky


def band_scores(rate_maps, res, box_width, k_values=None):
  """Compute band scores for many rate maps using the NeurIPS 2024 definition.

  For each rate map Xi, the band score is:
    max_{kx,ky in K} corrcoef[Xi, X(kx,ky)]
  where X(kx,ky) is a 2D sinusoid with frequencies kx, ky.
  """
  if k_values is None:
    k_values = np.arange(0.0, 2.0 + 1e-6, 0.1)
  maps = np.asarray(rate_maps)
  if maps.ndim != 3:
    raise ValueError('rate_maps must have shape [N, res, res]')
  n_units = maps.shape[0]
  flat = maps.reshape(n_units, -1).astype(np.float32)
  flat = np.nan_to_num(flat)
  flat = flat - np.mean(flat, axis=1, keepdims=True)
  flat_std = np.std(flat, axis=1, ddof=1, keepdims=True)
  valid = flat_std.squeeze() > 1e-8
  flat = flat / (flat_std + 1e-8)

  S, kxky = _band_library(res, box_width, k_values)
  n = flat.shape[1]
  corr = (flat @ S.T) / max(n - 1, 1)

  best_idx = np.full((n_units,), -1, dtype=int)
  best_scores = np.full((n_units,), np.nan, dtype=np.float32)
  if n_units > 0 and corr.size > 0:
    best_idx = np.nanargmax(corr, axis=1)
    best_scores = corr[np.arange(n_units), best_idx]

  # Mark invalid (flat variance ~ 0) as NaN
  if np.any(~valid):
    best_scores = best_scores.astype(np.float32)
    best_scores[~valid] = np.nan

  best_kx = np.full((n_units,), np.nan, dtype=np.float32)
  best_ky = np.full((n_units,), np.nan, dtype=np.float32)
  if kxky.size > 0 and best_idx.size > 0:
    best_kx[best_idx >= 0] = kxky[best_idx[best_idx >= 0], 0]
    best_ky[best_idx >= 0] = kxky[best_idx[best_idx >= 0], 1]
  return best_scores, best_kx, best_ky


def band_score(rate_map, res, box_width, k_values=None):
  """Band score for a single rate map."""
  scores, best_kx, best_ky = band_scores(rate_map[np.newaxis, :, :], res, box_width, k_values=k_values)
  return float(scores[0]), float(best_kx[0]), float(best_ky[0])
