import numpy as np
#import tensorflow as tf
import torch.cuda

#tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

from utils import generate_run_ID
from utils import load_example_npy_weights_into_model
from place_cells import PlaceCells
from trajectory_generator import TrajectoryGenerator
from model import RNN
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
                    help='RNN or LSTM')
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

options = parser.parse_args()
options.run_ID = generate_run_ID(options)

print(f'Using device: {options.device}')

place_cells = PlaceCells(options)
if options.RNN_type == 'RNN':
    model = RNN(options, place_cells)
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
