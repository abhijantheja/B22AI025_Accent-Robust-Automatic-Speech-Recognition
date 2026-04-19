"""
Phase 2: Domain-Adversarial Training (DAT) for accent-invariant representations.
Uses Gradient Reversal Layer to prevent encoder from learning accent-specific features.

Loss = L_CTC + lambda_sup * L_accent_sup + lambda_adv * L_accent_adv
      (GRL in AccentClassifier handles the sign reversal internally)

Usage:
    python scripts/train_adversarial.py \
        --pretrained results/phase1_adaptation/best_model.pt \
        --config configs/adversarial_config.yaml
"""

import os
import sys
import argparse
import yaml
import json
import torch
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.adversarial_asr import AdversarialASRModel
from src.data.dataset import AccentedSpeechDataset, collate_fn, create_dataloader
from src.data.preprocessing import AudioPreprocessor
from src.data.sampling import AccentStratifiedSampler
from src.training.trainer import AdversarialASRTrainer
from src.utils.helpers import set_seed, get_logger, load_checkpoint, format_number, count_parameters
from torch.utils.data import DataLoader


def parse_args():
    parser = argparse.ArgumentParser(description="Phase 2: Domain-Adversarial Training")
    parser.add_argument("--config", type=str, default="configs/adversarial_config.yaml")
    parser.add_argument("--pretrained", type=str, default=None,
                        help="Path to Phase 1 checkpoint")
    parser.add_argument("--train_manifest", type=str, default=None)
    parser.add_argument("--val_manifest", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="results/phase2_adversarial")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lambda_adv", type=float, default=None)
    parser.add_argument("--lambda_sup", type=float, default=None)
    return parser.parse_args()


def plot_training_curves(history: list, output_dir: str):
    """Plot and save training loss curves."""
    epochs = [h["epoch"] for h in history]
    train_total = [h["train"]["total"] for h in history]
    train_ctc = [h["train"]["ctc"] for h in history]
    train_adv = [h["train"]["accent_adv"] for h in history]
    val_wer = [h["val"].get("wer", 0) * 100 for h in history]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    axes[0, 0].plot(epochs, train_total, "b-o", label="Total Loss")
    axes[0, 0].set_title("Total Training Loss")
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("Loss")
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(epochs, train_ctc, "g-o", label="CTC Loss")
    axes[0, 1].set_title("CTC Loss")
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].set_ylabel("Loss")
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].plot(epochs, train_adv, "r-o", label="Adversarial Loss")
    axes[1, 0].set_title("Adversarial (Accent) Loss")
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].set_ylabel("Loss")
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].plot(epochs, val_wer, "m-o", label="Val WER (%)")
    axes[1, 1].set_title("Validation WER (%)")
    axes[1, 1].set_xlabel("Epoch")
    axes[1, 1].set_ylabel("WER (%)")
    axes[1, 1].grid(True, alpha=0.3)

    plt.suptitle("Phase 2: Adversarial Training Curves", fontsize=14)
    plt.tight_layout()
    path = os.path.join(output_dir, "training_curves.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Training curves saved: {path}")


def main():
    args = parse_args()
    logger = get_logger("train_adversarial")

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    if args.train_manifest:
        config["train_manifest"] = args.train_manifest
    if args.val_manifest:
        config["val_manifest"] = args.val_manifest
    if args.output_dir:
        config["output_dir"] = args.output_dir
    if args.lambda_adv is not None:
        config["lambda_adv"] = args.lambda_adv
    if args.lambda_sup is not None:
        config["lambda_sup"] = args.lambda_sup

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")
    logger.info(f"Adversarial lambda: adv={config.get('lambda_adv', 1.0)}, sup={config.get('lambda_sup', 0.1)}")

    preprocessor = AudioPreprocessor(processor_name=config.get("model_name", "facebook/wav2vec2-base"))

    # Stratified dataloader for accent balance
    train_dataset = AccentedSpeechDataset(
        manifest_path=config["train_manifest"],
        preprocessor=preprocessor,
        split="train",
    )
    sampler = AccentStratifiedSampler(train_dataset, replacement=True)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.get("batch_size", 8),
        sampler=sampler,
        num_workers=config.get("num_workers", 4),
        collate_fn=collate_fn,
        pin_memory=True,
    )

    val_loader = create_dataloader(
        manifest_path=config["val_manifest"],
        preprocessor=preprocessor,
        split="val",
        batch_size=config.get("batch_size", 8),
        shuffle=False,
    )

    # Model
    model = AdversarialASRModel(
        model_name=config.get("model_name", "facebook/wav2vec2-base"),
        vocab_size=config.get("vocab_size", 32),
        num_accents=config.get("num_accents", 8),
        alpha=0.0,  # Start with alpha=0, annealed during training
    )

    if args.pretrained and os.path.exists(args.pretrained):
        model, _, start_epoch = load_checkpoint(args.pretrained, model, device=device)
        logger.info(f"Loaded Phase 1 checkpoint: {args.pretrained}")
    else:
        logger.warning("No Phase 1 checkpoint; training from scratch with adversarial.")

    logger.info(f"Trainable parameters: {format_number(count_parameters(model))}")

    # Enable adversarial training
    config["use_adversarial"] = True
    config["learning_rate"] = config.get("adversarial_lr", 1e-4)

    trainer = AdversarialASRTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        output_dir=config["output_dir"],
        device=device,
    )

    logger.info("Starting Phase 2: Domain-Adversarial Training (GRL + CTC)")
    history = trainer.train(num_epochs=config.get("num_epochs", 20))

    logger.info(f"Phase 2 complete. Best WER: {trainer.best_wer:.4f}")
    plot_training_curves(history, config["output_dir"])


if __name__ == "__main__":
    main()
