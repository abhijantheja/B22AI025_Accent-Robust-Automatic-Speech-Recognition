"""
Main Adversarial ASR Model.
Combines Wav2Vec 2.0 encoder, CTC decoder, and accent classifier with GRL.

Architecture:
    Raw Audio -> Wav2Vec 2.0 Encoder -> Hidden States
                                          |
                          +---------------+---------------+
                          |                               |
                    CTC Decoder                   Accent Classifier
                    (ASR output)             (GRL -> Adversarial head)
                                             (Direct -> Supervised head)
"""

import torch
import torch.nn as nn
from transformers import Wav2Vec2Model, Wav2Vec2ForCTC, Wav2Vec2Config
from .accent_classifier import AccentClassifier
from .gradient_reversal import GradientReversalLayer


class AdversarialASRModel(nn.Module):
    """
    Wav2Vec 2.0 + Domain-Adversarial Training for Accent-Robust ASR.

    Training loss = L_ctc + lambda_sup * L_accent_sup - lambda_adv * L_accent_adv
    The GRL inside AccentClassifier handles the sign reversal automatically.
    """

    def __init__(
        self,
        model_name: str = "facebook/wav2vec2-base",
        vocab_size: int = 32,
        num_accents: int = 8,
        hidden_dim: int = 256,
        dropout: float = 0.1,
        alpha: float = 1.0,
        freeze_feature_extractor: bool = True,
    ):
        super().__init__()
        self.model_name = model_name
        self.num_accents = num_accents

        # Load full Wav2Vec2ForCTC so we get the pretrained lm_head weights
        _full = Wav2Vec2ForCTC.from_pretrained(model_name, ignore_mismatched_sizes=True)
        self.wav2vec2 = _full.wav2vec2

        if freeze_feature_extractor:
            self.wav2vec2.feature_extractor._freeze_parameters()

        encoder_dim = self.wav2vec2.config.hidden_size  # 768 base / 1024 large

        # CTC head — copy pretrained weights if vocab size matches, else random-init
        self.dropout = nn.Dropout(dropout)
        pretrained_vocab = _full.lm_head.out_features
        self.lm_head = nn.Linear(encoder_dim, vocab_size)
        if pretrained_vocab == vocab_size:
            self.lm_head.weight.data.copy_(_full.lm_head.weight.data)
            self.lm_head.bias.data.copy_(_full.lm_head.bias.data)
        del _full  # free memory

        # Accent classifier with GRL
        self.accent_classifier = AccentClassifier(
            input_dim=encoder_dim,
            num_accents=num_accents,
            hidden_dim=hidden_dim,
            dropout=dropout,
            alpha=alpha,
        )

    def forward(
        self,
        input_values: torch.Tensor,
        attention_mask: torch.Tensor = None,
        use_adversarial: bool = True,
    ):
        """
        Args:
            input_values: Raw waveform (B, T)
            attention_mask: Padding mask (B, T)
            use_adversarial: Whether to apply adversarial training

        Returns:
            logits: CTC logits (B, T', vocab_size)
            supervised_accent_logits: Accent logits from supervised head (B, num_accents)
            adversarial_accent_logits: Accent logits from GRL head (B, num_accents) or None
        """
        outputs = self.wav2vec2(
            input_values=input_values,
            attention_mask=attention_mask,
        )
        hidden_states = outputs.last_hidden_state  # (B, T', D)

        # CTC logits
        logits = self.lm_head(self.dropout(hidden_states))  # (B, T', vocab_size)

        # Accent classification (with optional GRL)
        supervised_logits, adversarial_logits = self.accent_classifier(
            hidden_states, use_adversarial=use_adversarial
        )

        return logits, supervised_logits, adversarial_logits

    def get_encoder_output(self, input_values: torch.Tensor, attention_mask=None):
        """Get encoder hidden states (for Grad-CAM and analysis)."""
        outputs = self.wav2vec2(input_values=input_values, attention_mask=attention_mask)
        return outputs.last_hidden_state

    def set_alpha(self, alpha: float):
        """Update GRL alpha during training."""
        self.accent_classifier.set_alpha(alpha)

    @classmethod
    def from_pretrained_asr(cls, checkpoint_path: str, **kwargs):
        """Load model from saved checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        model = cls(**checkpoint["model_config"])
        model.load_state_dict(checkpoint["model_state_dict"])
        return model
