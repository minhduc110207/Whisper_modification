"""
Modified Encoder with Spatio-temporal Blocks.

Each Transformer Encoder block is restructured to process:
  1. Spatial Multi-Head Self-Attention (S-MHSA) - joint relationships within each frame
  2. Temporal Multi-Head Self-Attention (T-MHSA) - trajectory across time with RPE
  3. Feed-Forward Network (FFN)

All using Pre-Norm architecture (LayerNorm before attention/FFN).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from .positional import RelativePositionalEncoding


class SpatialAttention(nn.Module):
    """
    Spatial Multi-Head Self-Attention (S-MHSA).
    Computes attention between all joints within the SAME time frame.
    Learns handshape configurations.
    """

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.num_heads = num_heads
        self.d_head = d_model // num_heads
        self.scale = self.d_head ** -0.5

        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T', d_model) - each position encodes joint info from a patch

        Note: In the current patch-based design, spatial attention operates
        across the temporal sequence to capture inter-patch spatial relationships.
        """
        B, T, D = x.shape

        qkv = self.qkv(x).reshape(B, T, 3, self.num_heads, self.d_head)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, H, T, d_head)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        out = (attn @ v).transpose(1, 2).reshape(B, T, D)
        return self.proj(out)


class TemporalAttention(nn.Module):
    """
    Temporal Multi-Head Self-Attention (T-MHSA) with Relative Positional Encoding.
    Computes attention across time steps to learn motion trajectories.
    """

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1, max_len: int = 2000):
        super().__init__()
        self.num_heads = num_heads
        self.d_head = d_model // num_heads
        self.scale = self.d_head ** -0.5

        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

        # Relative positional encoding
        self.rpe = RelativePositionalEncoding(self.d_head, max_len=max_len)
        self.rpe_proj = nn.Linear(self.d_head, 1, bias=False)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: (B, T', d_model)
            mask: Optional attention mask

        Returns:
            (B, T', d_model)
        """
        B, T, D = x.shape

        qkv = self.qkv(x).reshape(B, T, 3, self.num_heads, self.d_head)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Standard attention
        attn = (q @ k.transpose(-2, -1)) * self.scale

        # Add relative positional bias
        rel_pos = self.rpe(T)  # (T, T, d_head)
        rel_bias = self.rpe_proj(rel_pos).squeeze(-1)  # (T, T)
        attn = attn + rel_bias.unsqueeze(0).unsqueeze(0)

        if mask is not None:
            attn = attn.masked_fill(~mask.unsqueeze(1).unsqueeze(2), float('-inf'))

        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        out = (attn @ v).transpose(1, 2).reshape(B, T, D)
        return self.proj(out)


class FeedForward(nn.Module):
    """Standard FFN with GELU activation."""

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SpatioTemporalBlock(nn.Module):
    """
    Single Spatio-temporal Transformer block.

    Pre-Norm architecture:
        x -> LN -> S-MHSA -> + -> LN -> T-MHSA -> + -> LN -> FFN -> +
         |_____________________| |____________________| |______________|
    """

    def __init__(
        self,
        d_model: int = 512,
        num_heads: int = 8,
        d_ff: int = 2048,
        dropout: float = 0.1,
    ):
        super().__init__()

        # Spatial attention
        self.norm_s = nn.LayerNorm(d_model)
        self.spatial_attn = SpatialAttention(d_model, num_heads, dropout)

        # Temporal attention
        self.norm_t = nn.LayerNorm(d_model)
        self.temporal_attn = TemporalAttention(d_model, num_heads, dropout)

        # Feed-forward
        self.norm_ff = nn.LayerNorm(d_model)
        self.ffn = FeedForward(d_model, d_ff, dropout)

    def forward(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x: (B, T', d_model)
            mask: Optional attention mask

        Returns:
            (B, T', d_model)
        """
        # Pre-Norm + S-MHSA
        x = x + self.spatial_attn(self.norm_s(x))

        # Pre-Norm + T-MHSA with RPE
        x = x + self.temporal_attn(self.norm_t(x), mask)

        # Pre-Norm + FFN
        x = x + self.ffn(self.norm_ff(x))

        return x


class SpatioTemporalEncoder(nn.Module):
    """
    Full encoder stack with multiple Spatio-temporal blocks.
    Replaces the original Whisper encoder.
    """

    def __init__(
        self,
        num_layers: int = 6,
        d_model: int = 512,
        num_heads: int = 8,
        d_ff: int = 2048,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.layers = nn.ModuleList([
            SpatioTemporalBlock(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])

        self.final_norm = nn.LayerNorm(d_model)

    def forward(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x: (B, T', d_model) from Frontend
            mask: Optional attention mask

        Returns:
            (B, T', d_model) encoder hidden states
        """
        for layer in self.layers:
            x = layer(x, mask)

        return self.final_norm(x)
