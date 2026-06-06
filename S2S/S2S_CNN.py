import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import torch

import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(x + self.conv(x))

class S2S_CNN(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_dim=64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(in_channels, hidden_dim, kernel_size=3, padding=1),
            ResidualBlock(hidden_dim),
            ResidualBlock(hidden_dim)
        )
        self.decoder = nn.Sequential(
            ResidualBlock(hidden_dim),
            nn.Conv1d(hidden_dim, out_channels, kernel_size=1)
        )

    def forward(self, x):
        # x shape: (batch, in_channels, seq_len)
        latent = self.encoder(x)
        return self.decoder(latent)










