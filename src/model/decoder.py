"""
Decoder modules for sign language recognition.

Implements:
- CTCDecoder: Linear + Softmax head on top of encoder for CTC decoding (Pass 1)
- AttentionRescorer: Uses Whisper's original decoder for rescoring (Pass 2)
- TwoPassDecoder: Combines both for final predictions
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List


class CTCDecoder(nn.Module):
    """
    CTC (Connectionist Temporal Classification) decoder head.

    Placed on top of the encoder to produce frame-level predictions
    of sign glosses. Uses CTC's conditional independence assumption
    for fast, input-synchronous emission.

    Architecture: Linear projection from d_model -> vocab_size + softmax
    """

    def __init__(self, d_model: int = 512, vocab_size: int = 1296):
        super().__init__()

        self.projection = nn.Linear(d_model, vocab_size)
        self.vocab_size = vocab_size

    def forward(self, encoder_output: torch.Tensor) -> torch.Tensor:
        """
        Args:
            encoder_output: (B, T', d_model) from encoder

        Returns:
            log_probs: (B, T', vocab_size) log probabilities
        """
        logits = self.projection(encoder_output)
        return F.log_softmax(logits, dim=-1)

    def greedy_decode(self, log_probs: torch.Tensor) -> List[List[int]]:
        """
        Greedy CTC decoding: take argmax, remove blanks and repeated tokens.

        Args:
            log_probs: (B, T', vocab_size)

        Returns:
            List of decoded token sequences (one per batch)
        """
        predictions = log_probs.argmax(dim=-1)  # (B, T')
        results = []

        for b in range(predictions.shape[0]):
            tokens = predictions[b].tolist()
            # Remove blanks (index 0) and consecutive duplicates
            decoded = []
            prev_token = -1
            for token in tokens:
                if token != 0 and token != prev_token:  # 0 = blank
                    decoded.append(token)
                prev_token = token
            results.append(decoded)

        return results


class AttentionDecoder(nn.Module):
    """
    Attention-based decoder for rescoring CTC hypotheses.

    Uses a Transformer decoder with causal (diagonal) attention mask
    to rescore the CTC output in a single batched forward pass,
    much faster than full autoregressive decoding.
    """

    def __init__(
        self,
        vocab_size: int = 1296,
        d_model: int = 512,
        num_heads: int = 8,
        num_layers: int = 6,
        d_ff: int = 2048,
        dropout: float = 0.1,
        max_target_len: int = 200,
    ):
        super().__init__()

        self.d_model = d_model
        self.vocab_size = vocab_size

        # Token embedding
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Embedding(max_target_len, d_model)

        # Transformer decoder layers
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,  # Pre-Norm
        )
        self.decoder = nn.TransformerDecoder(
            decoder_layer, num_layers=num_layers
        )

        self.final_norm = nn.LayerNorm(d_model)
        self.output_proj = nn.Linear(d_model, vocab_size)

    def _generate_causal_mask(self, sz: int, device: torch.device) -> torch.Tensor:
        """Generate causal (upper triangular) attention mask."""
        mask = torch.triu(torch.ones(sz, sz, device=device), diagonal=1).bool()
        return mask

    def forward(
        self,
        encoder_output: torch.Tensor,
        target_tokens: torch.Tensor,
        encoder_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass for training / rescoring.

        Args:
            encoder_output: (B, T_enc, d_model) encoder hidden states
            target_tokens: (B, T_dec) target token indices
            encoder_mask: Optional mask for encoder output

        Returns:
            logits: (B, T_dec, vocab_size)
        """
        B, T_dec = target_tokens.shape

        # Token + positional embeddings
        positions = torch.arange(T_dec, device=target_tokens.device).unsqueeze(0)
        x = self.token_embedding(target_tokens) + self.pos_embedding(positions)

        # Causal mask
        causal_mask = self._generate_causal_mask(T_dec, target_tokens.device)

        # Transformer decoder
        x = self.decoder(
            tgt=x,
            memory=encoder_output,
            tgt_mask=causal_mask,
            memory_key_padding_mask=~encoder_mask if encoder_mask is not None else None,
        )

        x = self.final_norm(x)
        logits = self.output_proj(x)

        return logits


class TwoPassDecoder(nn.Module):
    """
    Two-Pass decoding combining CTC and Attention.

    Pass 1 (CTC): Fast, monotonic predictions from encoder
    Pass 2 (Attention): Rescore CTC hypotheses using decoder context

    Final output combines both passes for high accuracy + low latency.
    """

    def __init__(
        self,
        d_model: int = 512,
        vocab_size: int = 1296,
        num_heads: int = 8,
        num_decoder_layers: int = 6,
        d_ff: int = 2048,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.ctc_decoder = CTCDecoder(d_model, vocab_size)
        self.attention_decoder = AttentionDecoder(
            vocab_size=vocab_size,
            d_model=d_model,
            num_heads=num_heads,
            num_layers=num_decoder_layers,
            d_ff=d_ff,
            dropout=dropout,
        )

    def forward(
        self,
        encoder_output: torch.Tensor,
        target_tokens: Optional[torch.Tensor] = None,
        encoder_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            encoder_output: (B, T', d_model)
            target_tokens: (B, T_dec) for training, None for inference
            encoder_mask: Optional mask

        Returns:
            ctc_log_probs: (B, T', vocab_size)
            att_logits: (B, T_dec, vocab_size) or None
        """
        # Pass 1: CTC
        ctc_log_probs = self.ctc_decoder(encoder_output)

        # Pass 2: Attention (only if target tokens provided)
        att_logits = None
        if target_tokens is not None:
            att_logits = self.attention_decoder(
                encoder_output, target_tokens, encoder_mask
            )

        return ctc_log_probs, att_logits

    def decode(self, encoder_output: torch.Tensor) -> List[List[int]]:
        """
        Inference decoding using CTC greedy search.

        Args:
            encoder_output: (B, T', d_model)

        Returns:
            List of decoded sign gloss sequences
        """
        ctc_log_probs = self.ctc_decoder(encoder_output)
        return self.ctc_decoder.greedy_decode(ctc_log_probs)
