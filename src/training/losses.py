"""
Hybrid CTC + Attention Loss function.

L = alpha * L_CTC + (1 - alpha) * L_Attention

- L_CTC: CTC loss for monotonic temporal alignment
- L_Attention: Cross-entropy loss for context-aware rescoring
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict


class HybridCTCAttentionLoss(nn.Module):
    """
    Combined loss function for two-pass decoding.

    Args:
        alpha: Weight for CTC loss (default 0.3)
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

        Returns:
            Dictionary with 'total', 'ctc', and 'attention' losses
        """
        # CTC Loss: expects (T', B, vocab_size)
        ctc_input = ctc_log_probs.transpose(0, 1)  # (T', B, V)
        loss_ctc = self.ctc_loss(
            ctc_input,
            labels,
            output_lengths,
            label_lengths,
        )

        result = {"ctc": loss_ctc}

        # Attention Loss (if available)
        if att_logits is not None and att_targets is not None:
            # Reshape for cross-entropy: (B*T_dec, V) vs (B*T_dec,)
            B, T_dec, V = att_logits.shape
            loss_att = self.ce_loss(
                att_logits.reshape(-1, V),
                att_targets.reshape(-1),
            )
            result["attention"] = loss_att
            result["total"] = self.alpha * loss_ctc + (1 - self.alpha) * loss_att
        else:
            result["attention"] = torch.tensor(0.0, device=ctc_log_probs.device)
            result["total"] = loss_ctc

        return result
