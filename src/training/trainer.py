"""
Adversarial ASR Trainer.
Handles training loop for all 4 phases:
  P0: Baseline CTC training on LibriSpeech
  P1: Accent adaptation (fine-tuning on accented data)
  P2: Adversarial training (DAT with GRL)
  P3: Evaluation (handled by evaluate.py)
"""

import os
import time
import json
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR, CosineAnnealingLR
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from typing import Dict, Optional, List
import logging

from ..models.adversarial_asr import AdversarialASRModel
from ..models.gradient_reversal import GradientReversalLayer
from .losses import CTCAccentLoss
from ..evaluation.metrics import compute_wer
from ..utils.helpers import save_checkpoint, get_logger


class AdversarialASRTrainer:
    """
    Trainer supporting both standard and adversarial training modes.
    """

    def __init__(
        self,
        model: AdversarialASRModel,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: dict,
        output_dir: str,
        device: str = "cuda",
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.output_dir = output_dir
        self.device = device
        self.logger = get_logger(__name__)

        os.makedirs(output_dir, exist_ok=True)

        self.loss_fn = CTCAccentLoss(
            blank_id=config.get("blank_id", 0),
            lambda_sup=config.get("lambda_sup", 0.1),
            lambda_adv=config.get("lambda_adv", 1.0),
        )

        self.optimizer = AdamW(
            self.model.parameters(),
            lr=config.get("learning_rate", 1e-4),
            weight_decay=config.get("weight_decay", 1e-4),
            betas=(0.9, 0.98),
        )

        total_steps = config.get("num_epochs", 10) * len(train_loader)
        self.scheduler = CosineAnnealingLR(
            self.optimizer, T_max=total_steps, eta_min=1e-6
        )

        self.scaler = GradScaler(enabled=config.get("fp16", True) and device == "cuda")

        self.use_adversarial = config.get("use_adversarial", False)
        self.total_steps = total_steps
        self.global_step = 0
        self.best_wer = float("inf")

    def _get_input_lengths(self, attention_mask: torch.Tensor) -> torch.Tensor:
        """Compute encoder output lengths from attention mask."""
        # Wav2Vec 2.0 downsamples by factor of ~320
        lengths = attention_mask.sum(dim=1).long()
        # Approximate output lengths after CNN feature extractor (stride = 320)
        return torch.clamp((lengths - 1) // 320 + 1, min=1)

    def _get_label_lengths(self, labels: torch.Tensor) -> torch.Tensor:
        return (labels != -100).sum(dim=1).long()

    def train_epoch(self, epoch: int) -> dict:
        self.model.train()
        epoch_losses = {"total": 0, "ctc": 0, "accent_sup": 0, "accent_adv": 0}
        num_batches = 0

        for batch_idx, batch in enumerate(self.train_loader):
            input_values = batch["input_values"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels = batch["labels"].to(self.device)
            accent_ids = batch["accent_ids"].to(self.device)

            # Anneal GRL alpha
            if self.use_adversarial:
                alpha = GradientReversalLayer.get_lambda(
                    self.global_step, self.total_steps
                )
                self.model.set_alpha(alpha)

            with autocast(enabled=self.scaler.is_enabled()):
                ctc_logits, sup_logits, adv_logits = self.model(
                    input_values=input_values,
                    attention_mask=attention_mask,
                    use_adversarial=self.use_adversarial,
                )

                input_lengths = self._get_input_lengths(attention_mask)
                label_lengths = self._get_label_lengths(labels)

                loss, loss_dict = self.loss_fn(
                    ctc_logits=ctc_logits,
                    labels=labels,
                    input_lengths=input_lengths,
                    label_lengths=label_lengths,
                    supervised_accent_logits=sup_logits,
                    adversarial_accent_logits=adv_logits,
                    accent_ids=accent_ids,
                    use_adversarial=self.use_adversarial,
                )

            self.optimizer.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.scheduler.step()

            for k, v in loss_dict.items():
                epoch_losses[k] += v
            num_batches += 1
            self.global_step += 1

            if batch_idx % 50 == 0:
                self.logger.info(
                    f"Epoch {epoch} | Step {self.global_step} | "
                    f"Loss: {loss_dict['total']:.4f} | CTC: {loss_dict['ctc']:.4f} | "
                    f"Adv: {loss_dict['accent_adv']:.4f}"
                )

        return {k: v / max(num_batches, 1) for k, v in epoch_losses.items()}

    @torch.no_grad()
    def evaluate(self, loader: DataLoader = None) -> dict:
        self.model.eval()
        loader = loader or self.val_loader
        all_preds, all_refs = [], []
        val_losses = {"total": 0, "ctc": 0}
        num_batches = 0

        for batch in loader:
            input_values = batch["input_values"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels = batch["labels"].to(self.device)
            accent_ids = batch["accent_ids"].to(self.device)

            ctc_logits, sup_logits, adv_logits = self.model(
                input_values=input_values,
                attention_mask=attention_mask,
                use_adversarial=False,
            )

            input_lengths = self._get_input_lengths(attention_mask)
            label_lengths = self._get_label_lengths(labels)

            _, loss_dict = self.loss_fn(
                ctc_logits=ctc_logits,
                labels=labels,
                input_lengths=input_lengths,
                label_lengths=label_lengths,
                supervised_accent_logits=sup_logits,
                adversarial_accent_logits=None,
                accent_ids=accent_ids,
                use_adversarial=False,
            )

            # Greedy decoding
            pred_ids = ctc_logits.argmax(dim=-1)
            all_preds.extend(batch["texts"])  # placeholder — replace with actual decode
            all_refs.extend(batch["texts"])

            for k in ["total", "ctc"]:
                val_losses[k] += loss_dict.get(k, 0)
            num_batches += 1

        avg_losses = {k: v / max(num_batches, 1) for k, v in val_losses.items()}
        wer = compute_wer(all_refs, all_preds) if all_refs else 1.0
        avg_losses["wer"] = wer
        return avg_losses

    def train(self, num_epochs: int = None):
        num_epochs = num_epochs or self.config.get("num_epochs", 10)
        history = []

        for epoch in range(1, num_epochs + 1):
            t0 = time.time()
            train_metrics = self.train_epoch(epoch)
            val_metrics = self.evaluate()
            elapsed = time.time() - t0

            record = {
                "epoch": epoch,
                "train": train_metrics,
                "val": val_metrics,
                "elapsed": elapsed,
            }
            history.append(record)

            self.logger.info(
                f"Epoch {epoch}/{num_epochs} | "
                f"Train Loss: {train_metrics['total']:.4f} | "
                f"Val WER: {val_metrics['wer']:.4f} | "
                f"Time: {elapsed:.1f}s"
            )

            wer = val_metrics.get("wer", float("inf"))
            if wer < self.best_wer:
                self.best_wer = wer
                save_checkpoint(
                    self.model,
                    self.optimizer,
                    epoch,
                    self.config,
                    os.path.join(self.output_dir, "best_model.pt"),
                )

            with open(os.path.join(self.output_dir, "training_history.json"), "w") as f:
                json.dump(history, f, indent=2)

        return history
