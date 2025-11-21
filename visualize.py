# -*- coding: utf-8 -*-
import numpy as np
from matplotlib import pyplot as plt

import scipy
import scipy.stats
from imageio import imsave
import cv2
from typing import List, Optional, Tuple

from scores import GridScorer


def concat_images(images, image_width, spacer_size):
    """ Concat image horizontally with spacer """
    spacer = np.ones([image_width, spacer_size, 4], dtype=np.uint8) * 255
    images_with_spacers = []

    image_size = len(images)

    for i in range(image_size):
        images_with_spacers.append(images[i])
        if i != image_size - 1:
            # Add spacer
            images_with_spacers.append(spacer)
    ret = np.hstack(images_with_spacers)
    return ret


def concat_images_in_rows(images, row_size, image_width, spacer_size=4):
    """ Concat images in rows """
    column_size = len(images) // row_size
    spacer_h = np.ones([spacer_size, image_width*column_size + (column_size-1)*spacer_size, 4],
                       dtype=np.uint8) * 255

    row_images_with_spacers = []

    for row in range(row_size):
        row_images = images[column_size*row:column_size*row+column_size]
        row_concated_images = concat_images(row_images, image_width, spacer_size)
        row_images_with_spacers.append(row_concated_images)

        if row != row_size-1:
            row_images_with_spacers.append(spacer_h)

    ret = np.vstack(row_images_with_spacers)
    return ret


def convert_to_colormap(im, cmap):
    im = cmap(im)
    im = np.uint8(im * 255)
    return im


def rgb(im, cmap='jet', smooth=True):
    cmap = plt.cm.get_cmap(cmap)
    np.seterr(invalid='ignore')  # ignore divide by zero err
    im = (im - np.min(im)) / (np.max(im) - np.min(im))
    if smooth:
        im = cv2.GaussianBlur(im, (3,3), sigmaX=1, sigmaY=0)
    im = cmap(im)
    im = np.uint8(im * 255)
    return im


def plot_ratemaps(activations, n_plots, cmap='jet', smooth=True, width=16):
    images = [rgb(im, cmap, smooth) for im in activations[:n_plots]]
    rm_fig = concat_images_in_rows(images, n_plots//width, activations.shape[-1])
    return rm_fig


def compute_ratemaps(model, trajectory_generator, options, res=20, n_avg=None, Ng=512, idxs=None):
    '''Compute spatial firing fields'''

    if not n_avg:
        n_avg = 1000 // options.sequence_length

    if not np.any(idxs):
        idxs = np.arange(Ng)
    idxs = idxs[:Ng]

    g = np.zeros([n_avg, options.batch_size * options.sequence_length, Ng])
    pos = np.zeros([n_avg, options.batch_size * options.sequence_length, 2])

    activations = np.zeros([Ng, res, res]) 
    counts  = np.zeros([res, res])

    for index in range(n_avg):
        inputs, pos_batch, _ = trajectory_generator.get_test_batch()
        g_batch = model.g(inputs).detach().cpu().numpy()
        
        pos_batch = np.reshape(pos_batch.cpu(), [-1, 2])
        g_batch = g_batch[:,:,idxs].reshape(-1, Ng)
        
        g[index] = g_batch
        pos[index] = pos_batch

        x_batch = (pos_batch[:,0] + options.box_width/2) / (options.box_width) * res
        y_batch = (pos_batch[:,1] + options.box_height/2) / (options.box_height) * res

        for i in range(options.batch_size*options.sequence_length):
            x = x_batch[i]
            y = y_batch[i]
            if x >=0 and x < res and y >=0 and y < res:
                counts[int(x), int(y)] += 1
                activations[:, int(x), int(y)] += g_batch[i, :]

    for x in range(res):
        for y in range(res):
            if counts[x, y] > 0:
                activations[:, x, y] /= counts[x, y]
                
    g = g.reshape([-1, Ng])
    pos = pos.reshape([-1, 2])

    # # scipy binned_statistic_2d is slightly slower
    # activations = scipy.stats.binned_statistic_2d(pos[:,0], pos[:,1], g.T, bins=res)[0]
    rate_map = activations.reshape(Ng, -1)

    return activations, rate_map, g, pos


def save_ratemaps(model, trajectory_generator, options, step, res=20, n_avg=None):
    if not n_avg:
        n_avg = 1000 // options.sequence_length
    activations, rate_map, g, pos = compute_ratemaps(model, trajectory_generator,
                                                     options, res=res, n_avg=n_avg)
    rm_fig = plot_ratemaps(activations, n_plots=len(activations))
    imdir = options.save_dir + "/" + options.run_ID
    imsave(imdir + "/" + str(step) + ".png", rm_fig)


def save_autocorr(sess, model, save_name, trajectory_generator, step, flags):
    starts = [0.2] * 10
    ends = np.linspace(0.4, 1.0, num=10)
    coord_range=((-1.1, 1.1), (-1.1, 1.1))
    masks_parameters = zip(starts, ends.tolist())
    latest_epoch_scorer = scores.GridScorer(20, coord_range, masks_parameters)
    
    res = dict()
    index_size = 100
    for _ in range(index_size):
      feed_dict = trajectory_generator.feed_dict(flags.box_width, flags.box_height)
      mb_res = sess.run({
          'pos_xy': model.target_pos,
          'bottleneck': model.g,
      }, feed_dict=feed_dict)
      res = utils.concat_dict(res, mb_res)
        
    filename = save_name + '/autocorrs_' + str(step) + '.pdf'
    imdir = flags.save_dir + '/'
    out = utils.get_scores_and_plot(
                latest_epoch_scorer, res['pos_xy'], res['bottleneck'],
                imdir, filename)


def collect_sequences(model, trajectory_generator, options, n_batches: int = 10, Ng: Optional[int] = None, idxs: Optional[np.ndarray] = None):
    """Collects multiple batches of sequences for predictive analysis.

    Returns:
      xs, ys: arrays of shape [T, B_total]
      activations: array of shape [T, B_total, Ng]
    """
    model.eval()
    if Ng is None:
        Ng = model.Ng
    if idxs is None:
        idxs = np.arange(Ng)
    idxs = idxs[:Ng]

    xs_list, ys_list, g_list = [], [], []
    T_ref = None
    for _ in range(n_batches):
        inputs, pos_batch, _ = trajectory_generator.get_test_batch()
        with np.errstate(all='ignore'):
            g_batch = model.g(inputs).detach().cpu().numpy()  # [T,B,Ng]
        pos_np = pos_batch.detach().cpu().numpy()  # [T,B,2]
        if T_ref is None:
            T_ref = g_batch.shape[0]
        # Sanity: enforce consistent T
        if g_batch.shape[0] != T_ref:
            T = min(T_ref, g_batch.shape[0])
            g_batch = g_batch[:T]
            pos_np = pos_np[:T]
            T_ref = T
        xs_list.append(pos_np[:, :, 0])
        ys_list.append(pos_np[:, :, 1])
        g_list.append(g_batch[:, :, idxs])

    xs = np.concatenate(xs_list, axis=1)
    ys = np.concatenate(ys_list, axis=1)
    activations = np.concatenate(g_list, axis=1)
    return xs, ys, activations


def predictive_gridness_analysis(model, trajectory_generator, options,
                                 lags: List[int],
                                 res: int = 20,
                                 n_batches: int = 20,
                                 Ng: int = 512,
                                 idxs: Optional[np.ndarray] = None):
    """Compute predictive gridness (60° and 90°) across lags for many units.

    Returns:
      scores_60: [L, Ng]
      scores_90: [L, Ng]
      xs, ys, activations: sequences used [T,B], [T,B], [T,B,Ng]
      scorer: GridScorer instance configured for these params
    """
    xs, ys, activations = collect_sequences(model, trajectory_generator, options, n_batches=n_batches, Ng=Ng, idxs=idxs)

    starts = [0.2] * 10
    ends = np.linspace(0.4, 1.0, num=10)
    coord_range = ((-options.box_width / 2, options.box_width / 2),
                   (-options.box_height / 2, options.box_height / 2))
    masks_parameters = zip(starts, ends.tolist())
    scorer = GridScorer(res, coord_range, masks_parameters)

    s60, s90 = scorer.predictive_grid_scores(xs, ys, activations, lags)
    return s60, s90, xs, ys, activations, scorer


def plot_predictive_gridness_per_cell(scores_60: np.ndarray,
                                      lags: List[int],
                                      scores_90: Optional[np.ndarray] = None,
                                      cell_indices: Optional[List[int]] = None,
                                      cols: int = 8,
                                      figsize: Tuple[int, int] = (16, 10),
                                      suptitle: Optional[str] = None):
    """Plot each selected cell's gridness across all shifts.

    Args:
      scores_60: [L, Ng]
      lags: list of ints
      scores_90: optional [L, Ng]
      cell_indices: list of unit ids to plot; if None, plot all
    Returns: matplotlib figure
    """
    L, Ng = scores_60.shape
    if cell_indices is None:
        cell_indices = list(range(Ng))
    n = len(cell_indices)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=figsize, squeeze=False)
    x = np.array(lags)
    for i, u in enumerate(cell_indices):
        r = i // cols
        c = i % cols
        ax = axes[r, c]
        ax.plot(x, scores_60[:, u], label='Grid-60', color='C0')
        if scores_90 is not None:
            ax.plot(x, scores_90[:, u], label='Grid-90', color='C1', alpha=0.7)
        ax.axhline(0, color='k', lw=0.5)
        ax.set_title(f'Cell {u}', fontsize=9)
        ax.set_xlabel('Shift (time)')
        ax.set_ylabel('Score')
        ax.grid(True, alpha=0.2)
        if i == 0:
            ax.legend(fontsize=7)
    # Hide unused axes
    for j in range(n, rows * cols):
        r = j // cols
        c = j % cols
        axes[r, c].axis('off')
    if suptitle:
        fig.suptitle(suptitle)
    fig.tight_layout()
    return fig


def plot_predictive_heatmap(scores_60: np.ndarray, lags: List[int], figsize: Tuple[int, int] = (10, 6), title: Optional[str] = None):
    """Plot a heatmap of predictive gridness for all units across shifts."""
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    im = ax.imshow(scores_60.T, aspect='auto', origin='lower',
                   extent=[min(lags), max(lags), -0.5, scores_60.shape[1]-0.5],
                   cmap='viridis')
    ax.set_xlabel('Shift (time)')
    ax.set_ylabel('Unit index')
    if title:
        ax.set_title(title)
    fig.colorbar(im, ax=ax, label='Grid-60 score')
    fig.tight_layout()
    return fig
