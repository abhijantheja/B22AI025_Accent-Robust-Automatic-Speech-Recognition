"""
Accent Classifier for Domain-Adversarial Training.
Takes encoder representations and predicts accent class.
"""

import torch
import torch.nn as nn
from .gradient_reversal import GradientReversalLayer


class AccentClassifier(nn.Module):
    """
    Two-headed accent classifier:
    1. Supervised head: trained on labeled accent data
    2. Adversarial head: trained with GRL to make encoder accent-invariant
    """

    def __init__(
        self,
        input_dim: int,
        num_accents: int,
        hidden_dim: int = 256,
        dropout: float = 0.1,
        alpha: float = 1.0,
    ):
        super().__init__()
        self.num_accents = num_accents
        self.grl = GradientReversalLayer(alpha=alpha)

        self.adversarial_head = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_accents),
        )

        self.supervised_head = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_accents),
        )

    def forward(self, hidden_states: torch.Tensor, use_adversarial: bool = True):
        # Pool over time dimension
        pooled = hidden_states.mean(dim=1)  # (B, D)

        supervised_logits = self.supervised_head(pooled)

        if use_adversarial:
            reversed_features = self.grl(pooled)
            adversarial_logits = self.adversarial_head(reversed_features)
        else:
            adversarial_logits = None

        return supervised_logits, adversarial_logits

    def set_alpha(self, alpha: float):
        self.grl.set_alpha(alpha)
