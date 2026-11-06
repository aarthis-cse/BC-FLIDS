"""
BC-FLIDS: Blockchain-Federated Learning Intrusion Detection System
model.py — BC_FLIDS model definition (matches original training code exactly)

Architecture (from original bc_flids_model.py):
  Input  : (batch, timesteps=1, features)
  BiLSTM : hidden=128, num_layers=2, bidirectional → output 256
  Attn   : custom AttentionLayer (Linear(256→1) → softmax → weighted sum)
  LayerNorm(256)
  FC     : 256 → 128 → ReLU → Dropout → 64 → ReLU → Dropout → num_classes
"""

import torch
import torch.nn as nn


class AttentionLayer(nn.Module):
    """
    Additive (Bahdanau-style) attention over LSTM outputs.
    Produces a single context vector via softmax-weighted sum.
    """
    def __init__(self, hidden_size: int):
        super().__init__()
        self.attention = nn.Linear(hidden_size * 2, 1)

    def forward(self, lstm_out: torch.Tensor):
        """
        lstm_out : (batch, seq_len, hidden*2)
        returns  : context (batch, hidden*2), attn_weights (batch, seq_len, 1)
        """
        attn_weights = torch.softmax(self.attention(lstm_out), dim=1)
        context = torch.sum(attn_weights * lstm_out, dim=1)
        return context, attn_weights


class BC_FLIDS(nn.Module):
    """
    BiLSTM + Attention intrusion detection model for IoV.
    Matches the architecture used to produce the paper's reported results.
    """
    def __init__(self,
                 input_size: int,
                 hidden_size: int = 128,
                 num_layers: int = 2,
                 num_classes: int = 13,
                 dropout: float = 0.3):
        super().__init__()

        self.bilstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.attention = AttentionLayer(hidden_size)

        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes),
        )

        self.layer_norm = nn.LayerNorm(hidden_size * 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x      : (batch, seq_len=1, features)
        returns: logits (batch, num_classes)
        """
        lstm_out, _ = self.bilstm(x)                       # (batch, 1, 256)
        context, _  = self.attention(lstm_out)              # (batch, 256)
        context     = self.layer_norm(context)
        return self.classifier(context)                     # (batch, num_classes)
