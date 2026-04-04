"""
Decoder modules for sign language recognition.

Implements:
- CTCDecoder: Linear + Softmax head on top of encoder for CTC decoding (Pass 1)
- AttentionDecoder: Transformer decoder for rescoring (Pass 2)
- TwoPassDecoder: Combines both for hybrid CTC-Attention decoding

Inference uses hybrid decode:
  1. CTC greedy decode -> initial hypothesis (monotonic alignment)
  2. Attention rescore -> context-aware refinement
  3. Combined scoring prevents hallucination loops
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

    Uses a Transformer decoder with causal (diagonal) attention mask.
    Supports both teacher-forced (training) and autoregressive (inference)
    forward passes.
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

    def score_hypothesis(
        self,
        encoder_output: torch.Tensor,
        hypothesis: torch.Tensor,
        encoder_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Score a CTC hypothesis using the attention decoder.

        Runs a single forward pass (non-autoregressive) to get log-probs
        for each token in the hypothesis given the encoder output.

        Args:
            encoder_output: (B, T_enc, d_model)
            hypothesis: (B, T_hyp) token indices from CTC
            encoder_mask: Optional (B, T_enc) mask

        Returns:
            log_probs: (B, T_hyp, vocab_size)
        """
        logits = self.forward(encoder_output, hypothesis, encoder_mask)
        return F.log_softmax(logits, dim=-1)


class TwoPassDecoder(nn.Module):
    """
    Two-Pass decoding combining CTC and Attention.

    Pass 1 (CTC): Fast, monotonic predictions from encoder
    Pass 2 (Attention): Rescore CTC hypotheses using decoder context

    Inference uses hybrid scoring:
        final_score = ctc_weight * ctc_score + (1 - ctc_weight) * att_score

    This prevents:
    - Hallucination loops (CTC enforces monotonic alignment)
    - Context loop / repetitive output (max length + repetition guard)
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
        self.vocab_size = vocab_size

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

    def decode(
        self,
        encoder_output: torch.Tensor,
        encoder_mask: Optional[torch.Tensor] = None,
        ctc_weight: float = 0.5,
        max_decode_length: int = 100,
    ) -> List[List[int]]:
        """
        Hybrid CTC-Attention inference decoding.

        Strategy:
          1. CTC greedy decode -> initial hypothesis (monotonic, no hallucination)
          2. Attention rescore -> refine with context
          3. Combined scoring selects best tokens

        The CTC branch enforces monotonic alignment, preventing the
        Attention decoder from generating infinite loops.

        Args:
            encoder_output: (B, T', d_model)
            encoder_mask: Optional (B, T') mask
            ctc_weight: Weight for CTC in combined scoring (0-1)
                        Higher = more monotonic/safe, Lower = more contextual
            max_decode_length: Maximum tokens to generate (prevents infinite loops)

        Returns:
            List of decoded sign gloss sequences
        """
        B = encoder_output.shape[0]

        # Pass 1: CTC greedy decode (guaranteed monotonic, no loops)
        ctc_log_probs = self.ctc_decoder(encoder_output)
        ctc_hypotheses = self.ctc_decoder.greedy_decode(ctc_log_probs)

        # If ctc_weight is 1.0, skip attention rescoring entirely
        if ctc_weight >= 1.0:
            return ctc_hypotheses

        # Pass 2: Attention rescoring of CTC hypotheses
        results = []
        for b in range(B):
            hyp = ctc_hypotheses[b]

            # If CTC produced empty hypothesis, keep it
            if len(hyp) == 0:
                results.append([])
                continue

            # Cap hypothesis length
            hyp = hyp[:max_decode_length]

            # Remove repetitive sequences (Fix 4: context loop guard)
            hyp = self._remove_repetitions(hyp)

            if ctc_weight <= 0.0:
                # Pure attention: just use CTC hypothesis directly
                # (attention rescoring still needs CTC to provide the hypothesis)
                results.append(hyp)
                continue

            # Score hypothesis with Attention decoder
            # We must shift the input right (prepend SOS) to match training
            # We use 0 (CTC blank) as the SOS token
            hyp_input = [0] + hyp[:-1] if len(hyp) > 0 else [0]
            hyp_tensor = torch.tensor([hyp_input], device=encoder_output.device)
            enc_out_b = encoder_output[b:b+1]
            enc_mask_b = encoder_mask[b:b+1] if encoder_mask is not None else None

            att_log_probs = self.attention_decoder.score_hypothesis(
                enc_out_b, hyp_tensor, enc_mask_b
            )  # (1, T_hyp, vocab_size)

            # Get CTC scores for each hypothesis token at each position
            # Average CTC log-prob across all time frames for each token
            ctc_lp = ctc_log_probs[b]  # (T', vocab_size)

            # Combined rescoring: check if attention agrees with CTC
            refined_hyp = []
            for t, token_id in enumerate(hyp):
                # CTC score: average log-prob of this token across encoder frames
                ctc_score = ctc_lp[:, token_id].max().item()

                # Attention score: log-prob of this token at this decode position
                att_score = att_log_probs[0, t, token_id].item()

                # Combined score
                combined = ctc_weight * ctc_score + (1 - ctc_weight) * att_score

                # Only keep token if combined score is reasonable
                # (negative log-prob, so higher = better, threshold at very poor)
                if combined > -20.0:
                    refined_hyp.append(token_id)

            results.append(refined_hyp)

        return results

    @staticmethod
    def _remove_repetitions(tokens: List[int], max_repeat: int = 3) -> List[int]:
        """
        Remove repetitive subsequences from decoded tokens.
        Prevents context loop hallucinations (Fix 4).

        If a token appears more than max_repeat times consecutively,
        keep only max_repeat occurrences. Also detect repeating n-grams.

        Args:
            tokens: List of token IDs
            max_repeat: Maximum allowed consecutive repetitions

        Returns:
            Cleaned token list
        """
        if len(tokens) <= max_repeat:
            return tokens

        # Remove excessive consecutive repeats
        cleaned = []
        repeat_count = 1
        for i, token in enumerate(tokens):
            if i > 0 and token == tokens[i - 1]:
                repeat_count += 1
            else:
                repeat_count = 1

            if repeat_count <= max_repeat:
                cleaned.append(token)

        # Detect repeating bigrams (e.g., [A, B, A, B, A, B])
        if len(cleaned) >= 6:
            for n in range(2, 4):  # Check n-grams of size 2-3
                if len(cleaned) >= n * 3:
                    last_ngrams = [
                        tuple(cleaned[-(n*k + n):-(n*k) or None])
                        for k in range(3)
                    ]
                    if len(set(last_ngrams)) == 1 and all(last_ngrams):
                        # All last 3 n-grams are identical -> truncate
                        cleaned = cleaned[:-(n * 2)]
                        break

        return cleaned

