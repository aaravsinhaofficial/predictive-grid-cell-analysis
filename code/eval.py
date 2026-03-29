import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib import gridspec
import os

from utils import generate_run_ID
from place_cells import PlaceCells
from trajectory_generator import TrajectoryGenerator
from model import RNN
from trainer import Trainer
from visualize import compute_ratemaps, plot_ratemaps, save_ratemaps
from visualize import predictive_gridness_analysis, plot_predictive_gridness_per_cell, plot_predictive_heatmap
from scores import GridScorer, border_score
from shift_utils import parse_shift_values, shift_axis_label
import argparse


def evaluate_model(options):
    """
    Full evaluation of a trained grid cell model.
    """
    # Initialize place cells and model
    place_cells = PlaceCells(options)
    model = RNN(options, place_cells)
    model = model.to(options.device)
    
    # Load trained model
    ckpt_dir = os.path.join(options.save_dir, options.run_ID)
    ckpt_path = os.path.join(ckpt_dir, 'most_recent_model.pth')
    
    # Override with specific path if needed
    if options.checkpoint_path:
        ckpt_path = options.checkpoint_path
    
    if os.path.exists(ckpt_path):
        model.load_state_dict(torch.load(ckpt_path, map_location=options.device))
        print(f"Loaded model from {ckpt_path}")
    else:
        print(f"No model found at {ckpt_path}")
        return
    
    model.eval()
    
    # Create trajectory generator
    trajectory_generator = TrajectoryGenerator(options, place_cells)
    
    # 1. Compute rate maps
    print("Computing rate maps...")
    res = 20  # Resolution of rate maps
    n_avg = 1000 // options.sequence_length  # Number of batches to average
    Ng = min(512, options.Ng)  # Number of grid cells to analyze
    
    activations, rate_map, g, pos = compute_ratemaps(
        model, trajectory_generator, options, 
        res=res, n_avg=n_avg, Ng=Ng
    )
    
    # 2. Calculate grid scores
    print("Calculating grid scores...")
    starts = [0.2] * 10
    ends = np.linspace(0.4, 1.0, num=10)
    coord_range = ((-options.box_width/2, options.box_width/2), 
                   (-options.box_height/2, options.box_height/2))
    masks_parameters = zip(starts, ends.tolist())
    
    scorer = GridScorer(res, coord_range, masks_parameters)
    
    # Calculate scores for each grid cell
    scores_60 = []
    scores_90 = []
    sacs = []
    border_scores = []
    
    n_cells_to_score = min(100, Ng)  # Score first 100 cells
    
    for i in range(n_cells_to_score):
        rm = activations[i]
        score_60, score_90, _, _, sac, _ = scorer.get_scores(rm)
        scores_60.append(score_60)
        scores_90.append(score_90)
        sacs.append(sac)
        
        # Calculate border score
        bs, _, _ = border_score(rm, res, options.box_width)
        border_scores.append(bs)
    
    scores_60 = np.array(scores_60)
    scores_90 = np.array(scores_90)
    border_scores = np.array(border_scores)
    
    # 3. Print summary statistics
    print("\n=== Evaluation Results ===")
    print(f"Grid scores (60°):")
    print(f"  Mean: {np.nanmean(scores_60):.3f}")
    print(f"  Std:  {np.nanstd(scores_60):.3f}")
    print(f"  Max:  {np.nanmax(scores_60):.3f}")
    print(f"  % > 0.3: {(scores_60 > 0.3).sum() / len(scores_60) * 100:.1f}%")
    
    print(f"\nGrid scores (90°):")
    print(f"  Mean: {np.nanmean(scores_90):.3f}")
    print(f"  Std:  {np.nanstd(scores_90):.3f}")
    
    print(f"\nBorder scores:")
    print(f"  Mean: {np.nanmean(border_scores):.3f}")
    print(f"  Std:  {np.nanstd(border_scores):.3f}")
    
    # 4. Predictive gridness (optional)
    pred_lags = parse_shift_values(options.predictive_lags, shift_mode=options.shift_mode)
    if pred_lags is not None and len(pred_lags) > 0:
        print(f"\nComputing predictive gridness across {options.shift_mode} shifts...")
        s60_lag, s90_lag, xs_seq, ys_seq, g_seq, scorer = predictive_gridness_analysis(
            model, trajectory_generator, options,
            lags=pred_lags,
            res=res,
            n_batches=options.n_pred_batches,
            Ng=Ng,
            shift_mode=options.shift_mode,
            space_projection=options.space_projection)

        # Plot per-cell shift responses for top cells by max score
        top_by_max = np.argsort(np.nanmax(s60_lag, axis=0))[::-1]
        n_plot = min(options.predictive_plot_cells, len(top_by_max))
        sel = top_by_max[:n_plot].tolist()
        fig = plot_predictive_gridness_per_cell(s60_lag, pred_lags, scores_90=s90_lag, cell_indices=sel,
                                                suptitle='Predictive gridness per cell',
                                                shift_mode=options.shift_mode)
        if options.checkpoint_path:
            save_dir = os.path.dirname(options.checkpoint_path)
        else:
            save_dir = ckpt_dir
        out_path = os.path.join(save_dir, 'predictive_gridness_per_cell.png')
        fig.savefig(out_path, dpi=150)
        print(f"Saved predictive per-cell plot to {out_path}")

        # Heatmap across all units
        fig_hm = plot_predictive_heatmap(
            s60_lag,
            pred_lags,
            title=f'Predictive gridness (60°) heatmap [{shift_axis_label(options.shift_mode)}]',
            shift_mode=options.shift_mode,
        )
        out_hm = os.path.join(save_dir, 'predictive_gridness_heatmap.png')
        fig_hm.savefig(out_hm, dpi=150)
        print(f"Saved predictive heatmap to {out_hm}")

    # 5. Visualize top grid cells
    print("\nGenerating visualizations...")
    
    # Sort by grid score
    grid_idx = np.argsort(scores_60)[::-1]
    
    # Plot top grid cells
    n_plots = 16
    fig = plt.figure(figsize=(16, 8))
    gs = gridspec.GridSpec(4, 8, figure=fig)
    
    for i in range(n_plots):
        idx = grid_idx[i]
        
        # Plot rate map
        ax1 = fig.add_subplot(gs[i // 4, (i % 4) * 2])
        ax1.imshow(activations[idx], cmap='jet', interpolation='nearest')
        ax1.set_title(f'Cell {idx}\nScore: {scores_60[idx]:.2f}', fontsize=8)
        ax1.axis('off')
        
        # Plot autocorrelogram
        ax2 = fig.add_subplot(gs[i // 4, (i % 4) * 2 + 1])
        ax2.imshow(sacs[idx], cmap='jet', interpolation='nearest')
        ax2.set_title('SAC', fontsize=8)
        ax2.axis('off')
    
    plt.suptitle('Top Grid Cells by Grid Score', fontsize=14)
    plt.tight_layout()
    
    # Save figure
    if options.checkpoint_path:
        save_dir = os.path.dirname(options.checkpoint_path)
    else:
        save_dir = ckpt_dir
    save_path = os.path.join(save_dir, 'evaluation_results.png')
    plt.savefig(save_path, dpi=150)
    print(f"Saved visualization to {save_path}")
    plt.show()
    
    # 6. Save detailed results
    results_path = os.path.join(save_dir, 'evaluation_results.npz')
    np.savez(results_path,
             activations=activations,
             scores_60=scores_60,
             scores_90=scores_90,
             border_scores=border_scores,
             positions=pos,
             grid_activations=g)
    print(f"Saved detailed results to {results_path}")
    
    return {
        'scores_60': scores_60,
        'scores_90': scores_90,
        'border_scores': border_scores,
        'activations': activations,
        'sacs': sacs
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_dir', default='models/', 
                        help='directory where models are saved')
    parser.add_argument('--run_ID', default=None,
                        help='specific run ID to evaluate')
    parser.add_argument('--checkpoint_path', default=None,
                        help='direct path to checkpoint file')
    parser.add_argument('--batch_size', default=100, type=int)  # Changed to match saved model
    parser.add_argument('--sequence_length', default=20, type=int)
    parser.add_argument('--Np', default=256, type=int)  # Changed to match saved model
    parser.add_argument('--Ng', default=2048, type=int)  # Changed to match saved model
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
    parser.add_argument('--device', 
                        default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--shift_mode', default='time', choices=['time', 'space'],
                        help='Interpret predictive shifts as time steps or direct spatial displacements.')
    parser.add_argument('--space_projection', default='path', choices=['path', 'heading'],
                        help='When --shift_mode space, use arc-length along the trajectory or heading-based projection.')
    parser.add_argument('--predictive_lags', default=None,
                        help="Comma-separated list or range 'start:end[:step]'. In time mode these are steps; in space mode they are centimeters.")
    parser.add_argument('--n_pred_batches', default=20, type=int,
                        help='How many trajectory batches to aggregate for predictive analysis (increases trajectories).')
    parser.add_argument('--predictive_plot_cells', default=16, type=int,
                        help='How many top cells (by max predictive gridness) to plot per-cell responses for.')
    
    options = parser.parse_args()
    
    # Set the checkpoint path to your trained model
    options.checkpoint_path = "/Users/aaravsinha/grid-pattern-formation/models_trained_aarav/steps_20_batch_100_RNN_2048_relu_rf_012_DoG_True_periodic_False_lr_00001_weight_decay_00001/most_recent_model.pth"
    
    # Generate run_ID if not provided
    if options.run_ID is None:
        options.run_ID = generate_run_ID(options)
    
    print(f"Evaluating model: {options.run_ID}")
    print(f"Using device: {options.device}")
    
    results = evaluate_model(options)
