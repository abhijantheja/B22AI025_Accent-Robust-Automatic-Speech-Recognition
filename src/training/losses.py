"""
Loss functions for adversarial ASR training.

Total loss = L_CTC + lambda_sup * L_accent_sup + lambda_adv * L_accent_adv
Note: The GRL inside AccentClassifier negates gradients for the adversarial head,
so we ADD lambda_adv * L_accent_adv (not subtract) in the total loss.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class CTCAccentLoss(nn.Module):
    """
    Combined CTC + Accent classification loss.

    Args:
        blank_id: CTC blank token index
        lambda_sup: Weight for supervised accent loss
        lambda_adv: Weight for adversarial accent loss
        label_smoothing: Smoothing for accent CE loss
    """

    def __init__(
        self,
        blank_id: int = 0,
        lambda_sup: float = 0.1,
        lambda_adv: float = 1.0,
        label_smoothing: float = 0.0,
    ):
        super().__init__()
        self.ctc_loss = nn.CTCLoss(blank=blank_id, reduction="mean", zero_infinity=True)
        self.ce_loss = nn.CrossEntropyLoss(
            label_smoothing=label_smoothing,
            ignore_index=-1,
        )
        self.lambda_sup = lambda_sup
        self.lambda_adv = lambda_adv

    def forward(
        self,
        ctc_logits: torch.Tensor,          # (B, T', vocab)
        labels: torch.Tensor,              # (B, L)  padded with -100
        input_lengths: torch.Tensor,       # (B,)
        label_lengths: torch.Tensor,       # (B,)
        supervised_accent_logits: torch.Tensor,    # (B, num_accents)
        adversarial_accent_logits: Optional[torch.Tensor],  # (B, num_accents) or None
        accent_ids: torch.Tensor,          # (B,)  -1 for unlabeled
        use_adversarial: bool = True,
    ) -> Tuple[torch.Tensor, dict]:

        # ---- CTC Loss ----
        log_probs = F.log_softmax(ctc_logits, dim=-1).transpose(0, 1)  # (T', B, V)
        # CTC requires input_lengths <= T' (logit time steps)
        input_lengths = torch.clamp(input_lengths, max=log_probs.shape[0])
        # Replace -100 padding with 0 for CTC
        labels_ctc = labels.clone()
        labels_ctc[labels_ctc == -100] = 0
        # Clamp label lengths to actual non-padding length
        label_lengths = torch.clamp(label_lengths, max=labels_ctc.shape[1])

        loss_ctc = self.ctc_loss(log_probs, labels_ctc, input_lengths, label_lengths)

        # ---- Supervised Accent Loss ----
        # Only on samples with known accent (accent_id >= 0)
        labeled_mask = accent_ids >= 0
        loss_sup = torch.tensor(0.0, device=ctc_logits.device)
        if labeled_mask.any():
            loss_sup = self.ce_loss(
                supervised_accent_logits[labeled_mask],
                accent_ids[labeled_mask],
            )

        # ---- Adversarial Accent Loss ----
        # Gradients are reversed by GRL, so minimizing this pushes encoder to be accent-invariant
        loss_adv = torch.tensor(0.0, device=ctc_logits.device)
        if use_adversarial and adversarial_accent_logits is not None and labeled_mask.any():
            loss_adv = self.ce_loss(
                adversarial_accent_logits[labeled_mask],
                accent_ids[labeled_mask],
            )

        # ---- Total Loss ----
        total_loss = loss_ctc + self.lambda_sup * loss_sup + self.lambda_adv * loss_adv

        loss_dict = {
            "total": total_loss.item(),
            "ctc": loss_ctc.item(),
            "accent_sup": loss_sup.item(),
            "accent_adv": loss_adv.item(),
        }

        return total_loss, loss_dict
