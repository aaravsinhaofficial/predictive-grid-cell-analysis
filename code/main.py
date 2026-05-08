import numpy as np
#import tensorflow as tf
import torch.cuda
import os
import sys

#tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

from utils import generate_run_ID
from utils import load_example_npy_weights_into_model
from place_cells import PlaceCells
from trajectory_generator import TrajectoryGenerator
from model import RNN as FullRankRNN

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from model_low_rank import RNN as LowRankRNN
from trainer import Trainer
from visualize import save_ratemaps

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--save_dir',
                    # default='/mnt/fs2/bsorsch/grid_cells/models/',
                    default='models_trained_aarav/',
                    help='directory to save trained models')
parser.add_argument('--n_epochs',
                    default=5,
                    type=int,
                    help='number of training epochs')
parser.add_argument('--n_steps',
                    default=250,
                    type=int,
                    help='batches per epoch')
parser.add_argument('--batch_size',
                    default=100,
                    type=int,
                    help='number of trajectories per batch')
parser.add_argument('--sequence_length',
                    default=20,
                    type=int,
                    help='number of steps in trajectory')
parser.add_argument('--learning_rate',
                    default=1e-4,
                    type=float,
                    help='gradient descent learning rate')
parser.add_argument('--Np',
                    default=256,
                    type=int,
                    help='number of place cells')
parser.add_argument('--Ng',
                    default=2048,
                    type=int,
                    help='number of grid cells')
parser.add_argument('--place_cell_rf',
                    default=0.12,
                    type=float,
                    help='width of place cell center tuning curve (m)')
parser.add_argument('--surround_scale',
                    default=2,
                    type=float,
                    help='if DoG, ratio of sigma2^2 to sigma1^2')
parser.add_argument('--RNN_type',
                    default='RNN',
                    choices=['RNN', 'low_rank', 'LSTM'],
                    help='RNN, low_rank, or LSTM')
parser.add_argument('--rank',
                    default=8,
                    type=int,
                    help='rank K for --RNN_type low_rank')
parser.add_argument('--low_rank_factor_init',
                    default='balanced',
                    choices=['balanced', 'legacy'],
                    help='low-rank factor initialization for --RNN_type low_rank')
parser.add_argument('--low_rank_recurrent_gain',
                    default=1.0,
                    type=float,
                    help='gain applied to the realized low-rank recurrent matrix')
parser.add_argument('--low_rank_input_init_scale',
                    default=1.0,
                    type=float,
                    help='stddev for low-rank input weights')
parser.add_argument('--activation',
                    default='relu',
                    help='recurrent nonlinearity')
parser.add_argument('--weight_decay',
                    default=1e-4,
                    type=float,
                    help='strength of weight decay on recurrent weights')
parser.add_argument('--DoG',
                    default=True,
                    help='use difference of gaussians tuning curves')
parser.add_argument('--periodic',
                    default=False,
                    help='trajectories with periodic boundary conditions')
parser.add_argument('--box_width',
                    default=2.2,
                    type=float,
                    help='width of training environment')
parser.add_argument('--box_height',
                    default=2.2,
                    type=float,
                    help='height of training environment')
parser.add_argument('--trajectory_style',
                    default='random_walk',
                    choices=['random_walk', 'straight', 'per_step_random'],
                    help='motion regime: smooth random walk (default), straight with fixed speed, or new random heading/speed each step')
parser.add_argument('--trajectory_fixed_speed',
                    default=None,
                    type=float,
                    help='fixed forward speed in m/s when using --trajectory_style straight')
parser.add_argument('--trajectory_dt',
                    default=0.02,
                    type=float,
                    help='trajectory timestep (seconds)')
parser.add_argument('--trajectory_turn_sigma_scale',
                    default=1.0,
                    type=float,
                    help='scale on rotational noise; reduce for more predictable heading')
parser.add_argument('--trajectory_speed_scale',
                    default=1.0,
                    type=float,
                    help='scale on forward speed; increase for faster motion')
parser.add_argument('--trajectory_speed_max',
                    default=None,
                    type=float,
                    help='optional cap on forward speed (m/s)')
parser.add_argument('--trajectory_velocity_smoothing',
                    default=0.0,
                    type=float,
                    help='EMA factor in [0,1); >0 smooths speed changes')
parser.add_argument('--trajectory_border_region',
                    default=0.03,
                    type=float,
                    help='distance from wall (m) that triggers avoidance turn/slowdown')
parser.add_argument('--trajectory_wall_slowdown',
                    default=0.25,
                    type=float,
                    help='speed multiplier near walls (non-periodic envs)')
parser.add_argument('--trajectory_wall_turn_scale',
                    default=1.0,
                    type=float,
                    help='strength of wall-induced turn when non-periodic')
parser.add_argument('--device',
                    default='cuda' if torch.cuda.is_available() else 'cpu',
                    type=str,
                    help='device to use for training')

# Optional: path to a specific checkpoint to load (.pth)
parser.add_argument('--resume_from',
                    default=None,
                    help='path to a .pth checkpoint to load')

# Optional: evaluation only (no training). Saves ratemaps and exits.
parser.add_argument('--eval_only',
                    action='store_true',
                    help='only compute and save ratemaps, no training')
parser.add_argument('--eval_n_avg',
                    default=10,
                    type=int,
                    help='number of batches to average for eval-only ratemaps')

# Optional: load TF-era .npy weights directly into the PyTorch model
parser.add_argument('--weights_npy',
                    default=None,
                    help='path to example_trained_weights.npy (TF-era) to load')
parser.add_argument('--grid_eval_interval',
                    default=0,
                    type=int,
                    help='run predictive grid-cell diagnostics every N epochs (0 disables)')
parser.add_argument('--grid_eval_lags',
                    nargs='+',
                    default=[0, 1, 2, 3, 4, 5],
                    type=float,
                    help='shift values for predictive gridness scoring; steps in time mode, centimeters in space mode')
parser.add_argument('--grid_eval_shift_mode',
                    default='time',
                    choices=['time', 'space'],
                    help='align training diagnostics by time steps or direct spatial displacement')
parser.add_argument('--grid_eval_space_projection',
                    default='path',
                    choices=['path', 'heading'],
                    help='when grid-eval is in space mode, use arc-length path shifting or heading-based projection')
parser.add_argument('--grid_eval_batches',
                    default=5,
                    type=int,
                    help='batches to average for gridness diagnostics')
parser.add_argument('--grid_eval_threshold',
                    default=0.3,
                    type=float,
                    help='gridness cutoff used to count predictive/zero-lag units')
parser.add_argument('--grid_eval_strong_threshold',
                    default=0.5,
                    type=float,
                    help='higher gridness cutoff used for strong-grid emergence curves')
parser.add_argument('--grid_eval_min_shift_cm',
                    default=5.0,
                    type=float,
                    help='minimum spatial displacement used to classify predictive/retrospective units')
parser.add_argument('--grid_eval_max_units',
                    default=256,
                    type=int,
                    help='max grid units to score during training diagnostics; <=0 uses all units')
parser.add_argument('--grid_eval_res',
                    default=20,
                    type=int,
                    help='ratemap resolution for predictive gridness diagnostics')
parser.add_argument('--save_ratemaps_interval',
                    default=1,
                    type=int,
                    help='save ratemap mosaics every N epochs; <=0 disables per-epoch mosaics')

options = parser.parse_args()
options.run_ID = generate_run_ID(options)

print(f'Using device: {options.device}')

place_cells = PlaceCells(options)
if options.RNN_type == 'RNN':
    model = FullRankRNN(options, place_cells)
elif options.RNN_type == 'low_rank':
    model = LowRankRNN(options, place_cells)
elif options.RNN_type == 'LSTM':
    # model = LSTM(options, place_cells)
    raise NotImplementedError

# Put model on GPU if using GPU
model = model.to(options.device)

trajectory_generator = TrajectoryGenerator(options, place_cells)

trainer = Trainer(options, model, trajectory_generator)

# Optionally load an explicit checkpoint
if options.resume_from:
    import torch
    state = torch.load(options.resume_from, map_location=options.device)
    model.load_state_dict(state)
    print(f"Loaded checkpoint from {options.resume_from}")

# Optionally load TF-era weights (.npy) into the current model
if options.weights_npy:
    load_example_npy_weights_into_model(model, options.weights_npy)

# Eval-only path
if getattr(options, 'eval_only', False):
    save_ratemaps(model, trajectory_generator, options, step='eval', n_avg=options.eval_n_avg)
    raise SystemExit(0)

# Train (or continue training if a checkpoint was restored)
trainer.train(n_epochs=options.n_epochs, n_steps=options.n_steps)
