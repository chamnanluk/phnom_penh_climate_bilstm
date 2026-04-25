"""
BiLSTM model for learning SARIMA residuals.

Hybrid idea:
    final_temperature_prediction = SARIMA_prediction + BiLSTM_residual_prediction
"""

from __future__ import annotations

import torch
import torch.nn as nn


class ResidualBiLSTM(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.25,
    ):
        super().__init__()

        lstm_dropout = dropout if num_layers > 1 else 0.0

        self.bilstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=lstm_dropout,
        )

        self.head = nn.Sequential(
            nn.Linear(hidden_size * 2, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output, _ = self.bilstm(x)
        last_state = output[:, -1, :]
        residual_pred = self.head(last_state)
        return residual_pred
