#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Low-rank RNN model for the place-cell path-integration task."""

import torch

from LowRankRNN import LowRankRNN


class RNN(torch.nn.Module):
    """Drop-in low-rank recurrent model with the same API as code/model.py."""

    def __init__(self, options, place_cells=None):
        super(RNN, self).__init__()
        self.Ng = options.Ng
        self.Np = options.Np
        self.velocity_dim = getattr(options, "velocity_dim", 2)
        self.rank = int(getattr(options, "rank", getattr(options, "low_rank_rank", 8)))
        self.sequence_length = options.sequence_length
        self.weight_decay = options.weight_decay
        self.place_cells = place_cells

        self.encoder = torch.nn.Linear(self.Np, self.Ng, bias=False)
        self.RNN = LowRankRNN(
            input_size=self.velocity_dim,
            hidden_size=self.Ng,
            k=self.rank,
            nonlinearity=options.activation,
            factor_init=getattr(options, "low_rank_factor_init", "balanced"),
            recurrent_gain=getattr(options, "low_rank_recurrent_gain", 1.0),
            input_init_scale=getattr(options, "low_rank_input_init_scale", 1.0),
        )
        self.decoder = torch.nn.Linear(self.Ng, self.Np, bias=False)
        self.softmax = torch.nn.Softmax(dim=-1)

    def g(self, inputs):
        """
        Compute grid-cell activations.

        Args:
            inputs: Tuple (v, p0), with v shaped [T, B, velocity_dim] and
                p0 shaped [B, Np].

        Returns:
            Activations shaped [T, B, Ng].
        """
        v, p0 = inputs
        init_state = self.encoder(p0)[None]
        g, _ = self.RNN(v, init_state)
        return g

    def predict(self, inputs):
        """Predict place-cell logits for each trajectory step."""
        return self.decoder(self.g(inputs))

    def compute_loss(self, inputs, pc_outputs, pos):
        """
        Compute place-cell cross-entropy loss and decoded position error.

        This mirrors the full-rank model API so the existing Trainer can use
        either model class without a special low-rank training loop.
        """
        if self.place_cells is None:
            raise ValueError("Low-rank RNN needs place_cells to compute decoding error.")

        y = pc_outputs
        preds = self.predict(inputs)
        yhat = self.softmax(preds)
        ce_loss = -(y * torch.log(yhat.clamp_min(1e-12))).sum(-1).mean()

        # Regularize the realized recurrent matrix, not the raw factors. With
        # the low-rank initialization, factor L2 is enormous and can dominate
        # the task loss before the network learns path integration.
        W_rec = self.RNN.recurrent_gain * (self.RNN.M @ self.RNN.N / self.Ng)
        rec_reg_loss = self.weight_decay * (W_rec ** 2).sum()
        loss = ce_loss + rec_reg_loss
        self.last_loss_terms = {
            "ce_loss": float(ce_loss.detach().cpu()),
            "rec_reg_loss": float(rec_reg_loss.detach().cpu()),
        }

        pred_pos = self.place_cells.get_nearest_cell_pos(preds)
        err = torch.sqrt(((pos - pred_pos) ** 2).sum(-1)).mean()

        return loss, err
