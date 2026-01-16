import os
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib import gridspec
import scipy.ndimage as ndi

from place_cells import PlaceCells
from model import RNN
from trajectory_generator import TrajectoryGenerator
from visualize import collect_sequences
from scores import GridScorer


def cm_per_step_from_positions(xs, ys):
    """Estimate average centimeters traveled per time step from positions.

    xs, ys: arrays [T, B]
    Returns: float, cm per step
    """
    dx = np.diff(xs, axis=0)
    dy = np.diff(ys, axis=0)
    step = np.sqrt(dx**2 + dy**2)  # meters
    m_per_step = np.nanmean(step)
    return float(m_per_step * 100.0)


def head_direction_tuning(xs, ys, activations_u, n_bins=36):
    """Compute head-direction tuning using movement direction as proxy.

    xs, ys: [T, B]
    activations_u: [T, B]
    Returns:
      bin_centers: [n_bins]
      tuning: [n_bins] mean activation per angle
      r: Rayleigh vector length (0..1)
    """
    dx = np.diff(xs, axis=0)
    dy = np.diff(ys, axis=0)
    thetas = np.arctan2(dy, dx)  # [T-1, B]
    a = activations_u[:-1]  # align with thetas

    # Flatten and bin
    th = thetas.reshape(-1)
    aa = a.reshape(-1)
    # Remove NaNs
    valid = np.isfinite(th) & np.isfinite(aa)
    th = th[valid]
    aa = aa[valid]
    bins = np.linspace(-np.pi, np.pi, n_bins + 1)
    inds = np.digitize(th, bins) - 1
    inds = np.clip(inds, 0, n_bins - 1)
    tuning = np.zeros(n_bins, dtype=np.float64)
    counts = np.zeros(n_bins, dtype=np.float64)
    for i, v in zip(inds, aa):
        tuning[i] += v
        counts[i] += 1
    counts[counts == 0] = 1
    tuning /= counts
    bin_centers = (bins[:-1] + bins[1:]) / 2

    # Rayleigh vector length r
    vec = np.exp(1j * th) * aa
    r = np.abs(np.sum(vec)) / np.sum(np.abs(aa) + 1e-8)
    return bin_centers, tuning, float(r)


def ratemap_and_sac_for_shift(scorer: GridScorer, xs, ys, activations_u, shift: int):
    s60, s90, rm, sac = scorer.get_scores_with_shift(xs, ys, activations_u, shift, statistic='mean', return_maps=True)
    return rm, sac, s60, s90


def build_scorer(res, options):
    starts = [0.2] * 10
    ends = np.linspace(0.4, 1.0, num=10)
    coord_range = ((-options.box_width / 2, options.box_width / 2),
                   (-options.box_height / 2, options.box_height / 2))
    masks_parameters = zip(starts, ends.tolist())
    return GridScorer(res, coord_range, masks_parameters)


def plot_predictive_row(fig, gs_row, scorer, xs, ys, activations_u, pos_shifts_cm, cm_per_step, res=20, title_prefix='Cell'):
    """Plot a single cell row replicating predictive grid cell panel.

    gs_row: GridSpec row with subgridspec layout.
    pos_shifts_cm: list of positive shifts in cm to show (future).
    """
    # Sub-layout: [image grid (2x(N+1))] [curve] [polar]
    n_cols_img = 1 + len(pos_shifts_cm)
    gs_imgs = gridspec.GridSpecFromSubplotSpec(2, n_cols_img, subplot_spec=gs_row[0], wspace=0.15, hspace=0.15)
    ax_curve = fig.add_subplot(gs_row[1])
    ax_polar = fig.add_subplot(gs_row[2], projection='polar')

    # Original (shift=0)
    rm0, sac0, _, _ = ratemap_and_sac_for_shift(scorer, xs, ys, activations_u, 0)
    # Pre-compute vmin/vmax for rate maps across all shown shifts for consistent color scaling
    rm_list = [rm0]
    sac_max = np.nanmax(np.abs(sac0)) if sac0 is not None else None
    for cm in pos_shifts_cm:
        steps = int(np.round(cm / cm_per_step)) if cm_per_step > 0 else 0
        rm_tmp, sac_tmp, _, _ = ratemap_and_sac_for_shift(scorer, xs, ys, activations_u, steps)
        if rm_tmp is not None:
            rm_list.append(rm_tmp)
        if sac_tmp is not None:
            sac_max = max(sac_max, np.nanmax(np.abs(sac_tmp))) if sac_max is not None else np.nanmax(np.abs(sac_tmp))
    rm_vmin = np.nanmin([np.nanmin(r) for r in rm_list]) if rm_list else None
    rm_vmax = np.nanmax([np.nanmax(r) for r in rm_list]) if rm_list else None

    ax_rm0 = fig.add_subplot(gs_imgs[0, 0])
    rm0_vis = ndi.gaussian_filter(rm0, sigma=0.6) if rm0 is not None else rm0
    ax_rm0.imshow(rm0_vis, cmap='jet', interpolation='nearest', vmin=rm_vmin, vmax=rm_vmax)
    ax_rm0.set_title('Original', fontsize=7)
    ax_rm0.axis('off')
    # Row labels on the left
    ax_rm0.text(-0.18, 0.5, 'Rate map', rotation=90, transform=ax_rm0.transAxes,
                ha='center', va='center', fontsize=8)
    ax_sac0 = fig.add_subplot(gs_imgs[1, 0])
    if sac0 is not None:
        sac0_vis = ndi.gaussian_filter(sac0, sigma=0.6)
        ax_sac0.imshow(sac0_vis, cmap='jet', interpolation='nearest', vmin=-sac_max, vmax=sac_max)
        # Crosshairs at center
        c = sac0.shape[0] // 2
        ax_sac0.axhline(c, color='k', lw=1.0, alpha=0.85)
        ax_sac0.axvline(c, color='k', lw=1.0, alpha=0.85)
    ax_sac0.axis('off')
    ax_sac0.text(-0.18, 0.5, 'Autocorr', rotation=90, transform=ax_sac0.transAxes,
                 ha='center', va='center', fontsize=8)

    # Shifted (future) panels
    for j, cm in enumerate(pos_shifts_cm, start=1):
        steps = int(np.round(cm / cm_per_step)) if cm_per_step > 0 else 0
        rm, sac, _, _ = ratemap_and_sac_for_shift(scorer, xs, ys, activations_u, steps)
        ax_rm = fig.add_subplot(gs_imgs[0, j])
        rm_vis = ndi.gaussian_filter(rm, sigma=0.6) if rm is not None else rm
        ax_rm.imshow(rm_vis, cmap='jet', interpolation='nearest', vmin=rm_vmin, vmax=rm_vmax)
        ax_rm.set_title(f'+{int(cm)} cm', fontsize=7)
        ax_rm.axis('off')
        ax_sac = fig.add_subplot(gs_imgs[1, j])
        if sac is not None:
            sac_vis = ndi.gaussian_filter(sac, sigma=0.6)
            ax_sac.imshow(sac_vis, cmap='jet', interpolation='nearest', vmin=-sac_max, vmax=sac_max)
            c = sac.shape[0] // 2
            ax_sac.axhline(c, color='k', lw=1.0, alpha=0.85)
            ax_sac.axvline(c, color='k', lw=1.0, alpha=0.85)
        ax_sac.axis('off')

    # Gridness vs projected position (sweep lags)
    max_cm = max(pos_shifts_cm) if pos_shifts_cm else 30
    # Symmetric range around zero (in steps)
    max_steps = int(np.ceil(max_cm / cm_per_step)) if cm_per_step > 0 else 10
    # Respect sequence length to avoid empty overlap
    T = xs.shape[0]
    max_allowed = max(1, T - 2)
    span = min(max_steps * 5, max_allowed)
    lags = list(range(-span, span + 1))
    s60 = []
    for k in lags:
        s, _ = scorer.get_scores_with_shift(xs, ys, activations_u, k, statistic='mean')
        s60.append(s)
    s60 = np.array(s60)
    x_cm = np.array(lags) * cm_per_step
    ax_curve.plot(x_cm, s60, color='#2C6BB0', lw=2.5)
    ax_curve.axhline(0, color='k', lw=0.8)
    ax_curve.axvline(0, color='k', lw=0.8, ls='--')
    ax_curve.set_xlabel('Position shift (cm)', fontsize=10)
    ax_curve.set_ylabel('Gridness (60°)', fontsize=10)
    ax_curve.set_title('Gridness vs. projected position', fontsize=11)
    # Clean styling and stronger grid
    ax_curve.minorticks_on()
    ax_curve.grid(True, which='major', alpha=0.35, lw=0.8)
    ax_curve.grid(True, which='minor', alpha=0.20, lw=0.5)
    for spine in ['top','right']:
        ax_curve.spines[spine].set_visible(False)
    # Reasonable limits and ticks
    if np.isfinite(s60).any():
        ymin = min(0.0, float(np.nanmin(s60)) - 0.02)
        ymax = float(np.nanmax(s60)) + 0.02
        ax_curve.set_ylim(ymin, ymax)
    ax_curve.tick_params(labelsize=9)
    # Past/Future labels above axis
    ax_curve.text(0.05, 1.02, 'Past', transform=ax_curve.transAxes, color='crimson', fontsize=9, ha='left', va='bottom')
    ax_curve.text(0.95, 1.02, 'Future', transform=ax_curve.transAxes, color='crimson', fontsize=9, ha='right', va='bottom')

    # Head direction tuning (movement-based proxy)
    th, tuning, r = head_direction_tuning(xs, ys, activations_u)
    ax_polar.plot(th, tuning, color='#2AA654', lw=2.5)
    # Tidy polar look
    ax_polar.set_xticklabels([])
    ax_polar.set_yticklabels([])
    ax_polar.grid(alpha=0.35, lw=0.8)
    # Small label near the top-left of the polar subplot
    ax_polar.text(0.05, 1.05, f'Head direction\nr = {r:.3f}', transform=ax_polar.transAxes,
                  ha='left', va='bottom', fontsize=10)
    return fig


def predictive_figure(options,
                      checkpoint_path: str,
                      res: int = 20,
                      n_batches: int = 50,
                      Ng: int = 512,
                      cells: list = None,
                      pos_shifts_cm=(5, 10, 15, 20),
                      save_path: str = None):
    """Replicate predictive grid cell panels for selected cells.

    - Aggregates many trajectories to build xs, ys, and activations.
    - Estimates cm-per-step to label the x-axis and select shift steps.
    - For each cell: shows original and several future-shift panels, gridness-vs-shift curve, and HD tuning.
    """
    # Init model and data pipeline
    place_cells = PlaceCells(options)
    model = RNN(options, place_cells).to(options.device)
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f'Checkpoint not found: {checkpoint_path}')
    model.load_state_dict(torch.load(checkpoint_path, map_location=options.device))
    model.eval()
    traj_gen = TrajectoryGenerator(options, place_cells)

    # Collect sequences to increase trajectories
    xs, ys, activations = collect_sequences(model, traj_gen, options, n_batches=n_batches, Ng=Ng)
    cm_step = cm_per_step_from_positions(xs, ys)

    scorer = build_scorer(res, options)

    # Choose cells
    Ng_avail = activations.shape[2]
    if cells is None or len(cells) == 0:
        # Rank by max predictive gridness over a coarse lag sweep
        coarse_lags = list(range(-10, 11))
        s60 = np.zeros((len(coarse_lags), Ng_avail))
        for u in range(Ng_avail):
            a_u = activations[:, :, u]
            vals = []
            for k in coarse_lags:
                s, _ = scorer.get_scores_with_shift(xs, ys, a_u, k)
                vals.append(s)
            s60[:, u] = vals
        order = np.argsort(np.nanmax(s60, axis=0))[::-1]
        cells = order[:3].tolist()

    # Layout: 3 rows (cells) x 3 blocks (images, curve, polar)
    n_rows = len(cells)
    fig = plt.figure(figsize=(15, 4.4 * n_rows))
    # Each row: left block width ratio larger for images, spacious center plot
    gs = gridspec.GridSpec(n_rows, 3, width_ratios=[4.0, 2.3, 1.0], wspace=0.5, hspace=0.6, figure=fig)

    for i, u in enumerate(cells):
        a_u = activations[:, :, u]
        row_spec = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=gs[i], width_ratios=[4.0, 2.3, 1.0])
        plot_predictive_row(fig, row_spec, scorer, xs, ys, a_u, pos_shifts_cm, cm_step, res=res, title_prefix=f'Cell #{u}')

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=200, bbox_inches='tight')
        print(f'Saved predictive figure to {save_path}')
    return fig, cells


def _parse_cells(s: str):
    if s is None or s.strip() == '':
        return None
    return [int(x) for x in s.split(',') if x]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint_path', required=True)
    parser.add_argument('--save_path', default=None)
    parser.add_argument('--batch_size', default=100, type=int)
    parser.add_argument('--sequence_length', default=20, type=int)
    parser.add_argument('--Np', default=256, type=int)
    parser.add_argument('--Ng', default=2048, type=int)
    parser.add_argument('--place_cell_rf', default=0.12, type=float)
    parser.add_argument('--surround_scale', default=2, type=float)
    parser.add_argument('--RNN_type', default='RNN')
    parser.add_argument('--activation', default='relu')
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--DoG', default=True, type=bool)
    parser.add_argument('--periodic', default=False, type=bool)
    parser.add_argument('--box_width', default=2.2, type=float)
    parser.add_argument('--box_height', default=2.2, type=float)
    parser.add_argument('--learning_rate', default=1e-4, type=float)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')

    parser.add_argument('--res', default=20, type=int)
    parser.add_argument('--n_batches', default=50, type=int, help='How many trajectory batches to aggregate (increase trajectories).')
    parser.add_argument('--Ng_use', default=512, type=int, help='How many units to analyze.')
    parser.add_argument('--cells', default=None, help='Comma separated indices of cells to plot; if omitted, pick top 3.')
    parser.add_argument('--pos_shifts_cm', default='5,10,15,20', help='Positive position shifts (cm) to display next to original.')

    args = parser.parse_args()
    cells = _parse_cells(args.cells)
    pos_shifts_cm = [int(x) for x in args.pos_shifts_cm.split(',') if x]

    class Opt:  # simple namespace for model/options compatibility
        pass
    options = Opt()
    for k, v in vars(args).items():
        if hasattr(options, k):
            continue
        setattr(options, k, v)
    # Ensure fields used by other modules
    options.run_ID = 'predictive_figure'
    options.save_dir = os.path.dirname(args.checkpoint_path) or '.'

    fig, cells_used = predictive_figure(options,
                                        args.checkpoint_path,
                                        res=args.res,
                                        n_batches=args.n_batches,
                                        Ng=args.Ng_use,
                                        cells=cells,
                                        pos_shifts_cm=pos_shifts_cm,
                                        save_path=args.save_path)
    if args.save_path is None:
        plt.show()


if __name__ == '__main__':
    main()
