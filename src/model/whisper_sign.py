"""
WhisperSign - Main model integrating all components.

Loads pretrained Whisper weights selectively, replaces the frontend,
restructures the encoder, and adds CTC + Attention two-pass decoder.
"""
import torch
import torch.nn as nn
from typing import Optional, Tuple, Dict, List

from .frontend import SignLanguageFrontend
from .encoder import SpatioTemporalEncoder
from .decoder import TwoPassDecoder


class WhisperSignModel(nn.Module):
    """
    Modified Whisper model for sign language recognition.

    Architecture:
        Skeletal Data (B, T, 42, F)
          -> SignLanguageFrontend -> (B, T', d_model)
          -> SpatioTemporalEncoder -> (B, T', d_model)
          -> TwoPassDecoder -> CTC log_probs + Attention logits
    """

    def __init__(self, config: dict):
        super().__init__()

        self.config = config
        frontend_cfg = config.get("frontend", {})
        encoder_cfg = config.get("encoder", {})
        decoder_cfg = config.get("decoder", {})

        d_model = frontend_cfg.get("d_model", 512)

        # Frontend: replaces Whisper's Mel Spectrogram + CNN
        self.frontend = SignLanguageFrontend(
            num_joints=frontend_cfg.get("num_joints", 42),
            num_features=frontend_cfg.get("num_features", 7),
            patch_size=frontend_cfg.get("patch_size", 4),
            d_model=d_model,
            dropout=frontend_cfg.get("dropout", 0.1),
            spatial_dropout=frontend_cfg.get("spatial_dropout", 0.15),
        )

        # Encoder: Spatio-temporal Transformer blocks
        self.encoder = SpatioTemporalEncoder(
            num_layers=encoder_cfg.get("num_layers", 6),
            d_model=d_model,
            num_heads=encoder_cfg.get("num_heads", 8),
            d_ff=encoder_cfg.get("d_ff", 2048),
            dropout=encoder_cfg.get("dropout", 0.1),
        )

        # Decoder: CTC + Attention two-pass
        self.decoder = TwoPassDecoder(
            d_model=d_model,
            vocab_size=decoder_cfg.get("vocab_size", 1296),
            num_heads=encoder_cfg.get("num_heads", 8),
            num_decoder_layers=encoder_cfg.get("num_layers", 6),
            d_ff=encoder_cfg.get("d_ff", 2048),
            dropout=encoder_cfg.get("dropout", 0.1),
        )

    def forward(
        self,
        features: torch.Tensor,
        feature_lengths: Optional[torch.Tensor] = None,
        target_tokens: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Full forward pass.

        Args:
            features: (B, T, 42, F) skeletal input
            feature_lengths: (B,) actual lengths before padding
            target_tokens: (B, T_dec) target tokens for attention decoder

        Returns:
            Dictionary with:
                - 'ctc_log_probs': (B, T', vocab_size)
                - 'att_logits': (B, T_dec, vocab_size) or None
                - 'encoder_output': (B, T', d_model)
                - 'output_lengths': (B,)
        """
        # Frontend
        x, output_lengths = self.frontend(features, feature_lengths)

        # Create encoder mask from output lengths
        encoder_mask = None
        if output_lengths is not None:
            B, T_prime = x.shape[0], x.shape[1]
            encoder_mask = torch.arange(T_prime, device=x.device).unsqueeze(0) < output_lengths.unsqueeze(1)

        # Encoder
        encoder_output = self.encoder(x, encoder_mask)

        # Decoder
        ctc_log_probs, att_logits = self.decoder(
            encoder_output, target_tokens, encoder_mask
        )

        return {
            "ctc_log_probs": ctc_log_probs,
            "att_logits": att_logits,
            "encoder_output": encoder_output,
            "output_lengths": output_lengths,
        }

    def decode(self, features: torch.Tensor, feature_lengths: Optional[torch.Tensor] = None) -> List[List[int]]:
        """
        Inference: decode skeletal input to sign glosses.

        Args:
            features: (B, T, 42, F)
            feature_lengths: (B,) optional

        Returns:
            List of decoded token sequences
        """
        self.eval()
        with torch.no_grad():
            x, output_lengths = self.frontend(features, feature_lengths)
            encoder_mask = None
            if output_lengths is not None:
                B, T_prime = x.shape[0], x.shape[1]
                encoder_mask = torch.arange(T_prime, device=x.device).unsqueeze(0) < output_lengths.unsqueeze(1)
            encoder_output = self.encoder(x, encoder_mask)
            return self.decoder.decode(encoder_output)

    def freeze_encoder(self):
        """Freeze encoder parameters (for Stage 1 training)."""
        for param in self.encoder.parameters():
            param.requires_grad = False

    def unfreeze_encoder(self):
        """Unfreeze encoder parameters (for Stage 2 training)."""
        for param in self.encoder.parameters():
            param.requires_grad = True

    def freeze_decoder(self):
        """Freeze attention decoder parameters."""
        for param in self.decoder.attention_decoder.parameters():
            param.requires_grad = False

    def unfreeze_decoder(self):
        """Unfreeze attention decoder parameters."""
        for param in self.decoder.attention_decoder.parameters():
            param.requires_grad = True

    def get_num_params(self, trainable_only: bool = True) -> int:
        """Count model parameters."""
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())

    @classmethod
    def from_config(cls, config_path: str) -> 'WhisperSignModel':
        """Create model from YAML config file."""
        import yaml
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return cls(config.get("model", config))

    def save_checkpoint(self, path: str, optimizer=None, epoch: int = 0, loss: float = 0.0):
        """Save model checkpoint."""
        checkpoint = {
            "model_state_dict": self.state_dict(),
            "config": self.config,
            "epoch": epoch,
            "loss": loss,
        }
        if optimizer is not None:
            checkpoint["optimizer_state_dict"] = optimizer.state_dict()
        torch.save(checkpoint, path)

    @classmethod
    def load_checkpoint(cls, path: str, device: str = "cpu") -> Tuple['WhisperSignModel', dict]:
        """Load model from checkpoint."""
        checkpoint = torch.load(path, map_location=device)
        model = cls(checkpoint["config"])
        model.load_state_dict(checkpoint["model_state_dict"])
        return model, checkpoint
