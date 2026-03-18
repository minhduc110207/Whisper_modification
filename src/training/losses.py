"""
Hybrid CTC + Attention Loss function.

L = alpha * L_CTC + (1 - alpha) * L_Attention

- L_CTC: CTC loss for monotonic temporal alignment
- L_Attention: Cross-entropy loss for context-aware rescoring

Supports dynamic alpha scheduling: start with high CTC weight (0.8)
to enforce monotonic alignment, then decay to lower alpha (0.3) for
context-aware refinement.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict


class DynamicAlphaScheduler:
    """
    Linearly decays alpha from alpha_start to alpha_end over total_steps.

    Strategy: Start Stage 2 with alpha=0.8 (force CTC alignment learning)
    and decay to alpha=0.3 by end of training (allow Attention refinement).

    This prevents the Attention decoder from learning lazy language priors
    before the latent space has stabilized.

    Args:
        alpha_start: Initial CTC weight (default 0.8)
        alpha_end: Final CTC weight (default 0.3)
        total_steps: Number of steps for full decay
    """

    def __init__(
        self,
        alpha_start: float = 0.8,
        alpha_end: float = 0.3,
        total_steps: int = 1000,
    ):
        self.alpha_start = alpha_start
        self.alpha_end = alpha_end
        self.total_steps = max(total_steps, 1)
        self.current_step = 0

    def step(self) -> float:
        """Return current alpha and advance one step."""
        progress = min(self.current_step / self.total_steps, 1.0)
        alpha = self.alpha_start + (self.alpha_end - self.alpha_start) * progress
        self.current_step += 1
        return alpha

    def get_alpha(self) -> float:
        """Return current alpha without advancing."""
        progress = min(self.current_step / self.total_steps, 1.0)
        return self.alpha_start + (self.alpha_end - self.alpha_start) * progress


class HybridCTCAttentionLoss(nn.Module):
    """
    Combined loss function for two-pass decoding.

    Args:
        alpha: Default weight for CTC loss (default 0.3)
            - Higher alpha -> prioritize temporal alignment
            - Lower alpha -> prioritize contextual accuracy
        blank_id: CTC blank token id
    """

    def __init__(self, alpha: float = 0.3, blank_id: int = 0):
        super().__init__()
        self.alpha = alpha
        self.ctc_loss = nn.CTCLoss(blank=blank_id, zero_infinity=True)
        self.ce_loss = nn.CrossEntropyLoss(ignore_index=-1)

    def forward(
        self,
        ctc_log_probs: torch.Tensor,
        att_logits: Optional[torch.Tensor],
        labels: torch.Tensor,
        output_lengths: torch.Tensor,
        label_lengths: torch.Tensor,
        att_targets: Optional[torch.Tensor] = None,
        alpha_override: Optional[float] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute hybrid loss.

        Args:
            ctc_log_probs: (B, T', vocab_size) from CTC decoder
            att_logits: (B, T_dec, vocab_size) from attention decoder, or None
            labels: (sum(label_lengths),) concatenated labels for CTC
            output_lengths: (B,) encoder output lengths
            label_lengths: (B,) label lengths
            att_targets: (B, T_dec) attention decoder targets, or None
            alpha_override: If provided, overrides self.alpha for this call
                           (used by DynamicAlphaScheduler)

        Returns:
            Dictionary with 'total', 'ctc', 'attention' losses, and 'alpha'
        """
        alpha = alpha_override if alpha_override is not None else self.alpha

        # CTC Loss: expects (T', B, vocab_size)
        ctc_input = ctc_log_probs.transpose(0, 1)  # (T', B, V)
        loss_ctc = self.ctc_loss(
            ctc_input,
            labels,
            output_lengths,
            label_lengths,
        )

        result = {"ctc": loss_ctc, "alpha": alpha}

        # Attention Loss (if available)
        if att_logits is not None and att_targets is not None:
            # Reshape for cross-entropy: (B*T_dec, V) vs (B*T_dec,)
            B, T_dec, V = att_logits.shape
            loss_att = self.ce_loss(
                att_logits.reshape(-1, V),
                att_targets.reshape(-1),
            )
            result["attention"] = loss_att
            result["total"] = alpha * loss_ctc + (1 - alpha) * loss_att
        else:
            result["attention"] = torch.tensor(0.0, device=ctc_log_probs.device)
            # Use alpha even if attention is missing for consistency in logging/scaling
            result["total"] = alpha * loss_ctc

        return result
