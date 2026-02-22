"""WhisperSign - Modified Whisper for Sign Language Recognition."""

__version__ = "0.1.0"

from .model import WhisperSignModel
from .training import WhisperSignTrainer, HybridCTCAttentionLoss
from .data import SignLanguageDataset, create_dataloaders

__all__ = [
    "WhisperSignModel",
    "WhisperSignTrainer",
    "HybridCTCAttentionLoss",
    "SignLanguageDataset",
    "create_dataloaders",
]
