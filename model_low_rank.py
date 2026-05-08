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
        loss = -(y * torch.log(yhat)).sum(-1).mean()

        loss += self.weight_decay * ((self.RNN.M ** 2).sum() + (self.RNN.N ** 2).sum())

        pred_pos = self.place_cells.get_nearest_cell_pos(preds)
        err = torch.sqrt(((pos - pred_pos) ** 2).sum(-1)).mean()

        return loss, err
