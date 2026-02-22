"""Training pipeline: trainer, losses, and scheduling."""

from .trainer import WhisperSignTrainer
from .losses import HybridCTCAttentionLoss
from .scheduler import CosineWarmupScheduler

__all__ = [
    "WhisperSignTrainer",
    "HybridCTCAttentionLoss",
    "CosineWarmupScheduler",
]
