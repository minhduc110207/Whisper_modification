"""
New Frontend module replacing Whisper's Mel Spectrogram pipeline.
Converts skeletal 3D coordinate data into Transformer-compatible embeddings.

Architecture:
  1. Temporal Grouping (Patch Embedding) - groups P consecutive frames
  2. Linear Projection - maps to d_model dimension
  3. ConvSPE - Convolutional Spatial-Positional Encoding
  4. Batch Normalization + Spatial Dropout
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class TemporalPatchEmbedding(nn.Module):
    """
    Groups P consecutive frames into patches and projects
    to d_model dimension. Reduces sequence length from T to T/P.

    Input:  (B, T, 42, F)
    Output: (B, T/P, d_model)
    """

    def __init__(
        self,
        num_joints: int = 42,
        num_features: int = 7,
        patch_size: int = 4,
        d_model: int = 512,
    ):
        super().__init__()
        self.num_joints = num_joints
        self.num_features = num_features
        self.patch_size = patch_size
        self.d_model = d_model

        # Input dimension per patch = P * 42 * F
        input_dim = patch_size * num_joints * num_features

        self.projection = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, J, F) skeletal data

        Returns:
            (B, T', d_model) where T' = T // patch_size
        """
        B, T, J, C = x.shape  # C = num_features (avoid shadowing F = torch.nn.functional)

        # Pad T to be divisible by patch_size
        remainder = T % self.patch_size
        if remainder != 0:
            pad_len = self.patch_size - remainder
            x = F.pad(x, (0, 0, 0, 0, 0, pad_len))
            T = T + pad_len

        # Reshape: (B, T, J, C) -> (B, T/P, P*J*C)
        T_prime = T // self.patch_size
        x = x.reshape(B, T_prime, self.patch_size * J * C)

        # Project to d_model
        x = self.projection(x)  # (B, T', d_model)

        return x


class ConvSPE(nn.Module):
    """
    Convolutional Spatial-Positional Encoding.
    Uses depthwise separable convolutions to learn local spatial
    relationships between neighboring joints.

    Instead of fixed positional encoding, this learns relative
    spatial structure of the hand skeleton dynamically.
    """

    def __init__(self, d_model: int = 512, kernel_size: int = 31):
        super().__init__()

        # Depthwise convolution
        self.depthwise = nn.Conv1d(
            d_model, d_model,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=d_model,
            bias=False,
        )
        # Pointwise convolution
        self.pointwise = nn.Conv1d(d_model, d_model, kernel_size=1)
        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T', d_model)

        Returns:
            (B, T', d_model) with positional information encoded
        """
        # Conv1d expects (B, C, L)
        residual = x
        x = x.transpose(1, 2)  # (B, d_model, T')
        x = self.depthwise(x)
        x = self.activation(x)
        x = self.pointwise(x)
        x = x.transpose(1, 2)  # (B, T', d_model)

        return x + residual


class SpatialDropout1D(nn.Module):
    """
    Drops entire feature channels instead of individual elements.
    More effective for sequential data as it maintains temporal coherence.
    """

    def __init__(self, p: float = 0.15):
        super().__init__()
        self.p = p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, d_model)
        """
        if not self.training or self.p == 0:
            return x

        # Create channel-wise mask: (B, 1, d_model)
        mask = torch.ones(x.shape[0], 1, x.shape[2], device=x.device)
        mask = F.dropout(mask, p=self.p, training=True)
        return x * mask


class SignLanguageFrontend(nn.Module):
    """
    Complete frontend replacing Whisper's audio processing.

    Pipeline:
        Input (B, T, 42, F)
          -> Temporal Patch Embedding (B, T/P, d_model)
          -> ConvSPE positional encoding
          -> Batch Normalization
          -> Spatial Dropout
          -> Output (B, T/P, d_model)
    """

    def __init__(
        self,
        num_joints: int = 42,
        num_features: int = 7,
        patch_size: int = 4,
        d_model: int = 512,
        dropout: float = 0.1,
        spatial_dropout: float = 0.15,
    ):
        super().__init__()

        self.patch_embedding = TemporalPatchEmbedding(
            num_joints=num_joints,
            num_features=num_features,
            patch_size=patch_size,
            d_model=d_model,
        )

        self.conv_spe = ConvSPE(d_model=d_model)

        self.batch_norm = nn.BatchNorm1d(d_model)
        self.spatial_dropout = SpatialDropout1D(p=spatial_dropout)
        self.dropout = nn.Dropout(p=dropout)

        self.patch_size = patch_size

    def forward(
        self, x: torch.Tensor, input_lengths: Optional[torch.Tensor] = None
    ) -> tuple:
        """
        Args:
            x: (B, T, 42, F) raw skeletal features
            input_lengths: (B,) actual lengths before padding

        Returns:
            embeddings: (B, T', d_model)
            output_lengths: (B,) lengths after patch grouping
        """
        # Compute output lengths
        if input_lengths is not None:
            output_lengths = (input_lengths + self.patch_size - 1) // self.patch_size
        else:
            output_lengths = None

        # Patch embedding
        x = self.patch_embedding(x)  # (B, T', d_model)

        # Positional encoding via ConvSPE
        x = self.conv_spe(x)

        # Batch normalization (expects B, C, L)
        x = x.transpose(1, 2)  # (B, d_model, T')
        x = self.batch_norm(x)
        x = x.transpose(1, 2)  # (B, T', d_model)

        # Dropout
        x = self.spatial_dropout(x)
        x = self.dropout(x)

        return x, output_lengths
