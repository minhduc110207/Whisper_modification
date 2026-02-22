"""
Positional encoding modules.
- ConvSPE: Convolutional Spatial-Positional Encoding (used in Frontend)
- RelativePositionalEncoding: For Temporal MHSA in Encoder
"""
import math
import torch
import torch.nn as nn


class RelativePositionalEncoding(nn.Module):
    """
    Relative Positional Encoding (RPE) for Temporal Multi-Head Self-Attention.

    Instead of absolute position, encodes relative distance between
    time steps, which is more natural for motion trajectory analysis.
    """

    def __init__(self, d_model: int, max_len: int = 2000):
        super().__init__()
        self.d_model = d_model

        # Learnable relative position bias
        self.rel_pos_bias = nn.Embedding(2 * max_len - 1, d_model)
        self.max_len = max_len

    def forward(self, seq_len: int) -> torch.Tensor:
        """
        Compute relative position encoding matrix.

        Args:
            seq_len: Length of the sequence

        Returns:
            (seq_len, seq_len, d_model) relative position encodings
        """
        positions = torch.arange(seq_len, device=self.rel_pos_bias.weight.device)
        # Relative distance matrix
        rel_dist = positions.unsqueeze(0) - positions.unsqueeze(1)  # (L, L)
        # Shift to non-negative indices
        rel_dist = rel_dist + self.max_len - 1
        rel_dist = rel_dist.clamp(0, 2 * self.max_len - 2)

        return self.rel_pos_bias(rel_dist)  # (L, L, d_model)


class SinusoidalPositionalEncoding(nn.Module):
    """
    Standard sinusoidal positional encoding as a fallback/baseline.
    """

    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, d_model)

        Returns:
            (B, T, d_model) with positional encoding added
        """
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)
