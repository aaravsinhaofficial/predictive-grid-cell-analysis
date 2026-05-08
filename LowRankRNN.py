#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  2 10:28:47 2025

@author: redmawt1
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

class LowRankRNN(nn.Module):
    def __init__(self, input_size, hidden_size, k, nonlinearity='tanh'):
        super(LowRankRNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.k = k

        # Input weight and bias
        self.weight_ih = nn.Parameter(torch.empty(hidden_size, input_size))
        self.bias_ih = nn.Parameter(torch.zeros(hidden_size))

        # Low-rank recurrent weight components: M (N x K), N (K x N)
        self.M = nn.Parameter(torch.empty(hidden_size, k))
        self.N = nn.Parameter(torch.empty(k, hidden_size))

        # Nonlinearity
        if nonlinearity == 'tanh':
            self.activation = torch.tanh
        elif nonlinearity == 'relu':
            self.activation = F.relu
        else:
            raise ValueError("Only 'tanh' and 'relu' are supported")

        self.reset_parameters()

    #def reset_parameters(self):
    #    std = 1.0 / math.sqrt(self.hidden_size)
    #   nn.init.normal_(self.weight_ih, mean=0.0, std=std)
    #    nn.init.normal_(self.M, mean=0.0, std=std)
    #    nn.init.normal_(self.N, mean=0.0, std=std)
    #    nn.init.zeros_(self.bias_ih)
        
    def reset_parameters(self):
        std = 1.0 /np.sqrt(self.k)*np.sqrt(self.hidden_size)
        nn.init.normal_(self.weight_ih, mean=0.0, std=1)
        nn.init.normal_(self.M, mean=0.0, std=std)
        nn.init.normal_(self.N, mean=0.0, std=1)
        nn.init.zeros_(self.bias_ih)

    def forward(self, input, h_0=None):
        """
        input: (seq_len, batch_size, input_size)
        h_0: (batch_size, hidden_size), (1, batch_size, hidden_size), or None
        Returns:
            output: (seq_len, batch_size, hidden_size)
            h_n: (1, batch_size, hidden_size)
        """
        seq_len, batch_size, _ = input.size()
        if h_0 is None:
            h_t = torch.zeros(batch_size, self.hidden_size, device=input.device)
        else:
            if h_0.dim() == 3:
                h_0 = h_0[0]
            h_t = h_0

        outputs = []
        # W_rec = self.M @ self.N  # Low-rank recurrent weight
        W_rec = self.M @ self.N / self.hidden_size 

        for t in range(seq_len):
            x_t = input[t]
            h_t = self.activation(
                F.linear(x_t, self.weight_ih, self.bias_ih) +
                F.linear(h_t, W_rec)
            )
            outputs.append(h_t.unsqueeze(0))

        output = torch.cat(outputs, dim=0)
        return output, h_t.unsqueeze(0)
