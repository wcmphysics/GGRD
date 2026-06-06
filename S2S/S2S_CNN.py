import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn

class MultiFoldResidualBlock(nn.Module):
    """
    A multiple-folded residual block. 
    The output of a fold becomes the input to the CNN of the next fold,
    while the shortcut connection always uses the original data input.
    """
    def __init__(self, channels, num_folds=3):
        super().__init__()
        self.folds = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(channels, channels, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv1d(channels, channels, kernel_size=3, padding=1)
            ) for _ in range(num_folds)
        ])
        self.relu = nn.ReLU()

    def forward(self, x):
        shortcut = x
        out = x
        for conv in self.folds:
            out = self.relu(shortcut + conv(out))
        return out

class S2S_CNN(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_dim=64, num_folds=3):
        super().__init__()
        # Initial projection to hidden dimension
        self.input_proj = nn.Conv1d(in_channels, hidden_dim, kernel_size=3, padding=1)
        
        # The multiple-folded residual network core
        self.multi_fold_core = MultiFoldResidualBlock(hidden_dim, num_folds=num_folds)
        
        # Final projection to output channels
        self.output_proj = nn.Conv1d(hidden_dim, out_channels, kernel_size=1)

    def forward(self, x):
        # x shape: (batch, in_channels, seq_len)
        x = self.input_proj(x)
        x = self.multi_fold_core(x)
        return self.output_proj(x)
