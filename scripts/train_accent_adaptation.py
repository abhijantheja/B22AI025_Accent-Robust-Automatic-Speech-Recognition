"""
Phase 1: Fine-tune baseline model on accented speech datasets.
Adapts the model to non-native accents WITHOUT adversarial training.
Datasets: CommonVoice, L2-ARCTIC, AESRC2020.

Usage:
    python scripts/train_accent_adaptation.py \
        --pretrained results/phase0_baseline/best_model.pt \
        --config configs/adversarial_config.yaml
"""

import os
import sys
import argparse
import yaml
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.adversarial_asr import AdversarialASRModel
from src.data.dataset import create_dataloader
from src.data.preprocessing import AudioPreprocessor
from src.data.sampling import AccentStratifiedSampler
from src.training.trainer import AdversarialASRTrainer
from src.utils.helpers import set_seed, get_logger, load_checkpoint


def parse_args():
    parser = argparse.ArgumentParser(description="Phase 1: Accent Adaptation")
    parser.add_argument("--config", type=str, default="configs/adversarial_config.yaml")
    parser.add_argument("--pretrained", type=str, default=None,
                        help="Path to Phase 0 checkpoint")
    parser.add_argument("--train_manifest", type=str, default=None)
    parser.add_argument("--val_manifest", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="results/phase1_adaptation")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    logger = get_logger("train_accent_adaptation")

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    if args.train_manifest:
        config["train_manifest"] = args.train_manifest
    if args.val_manifest:
        config["val_manifest"] = args.val_manifest
    if args.output_dir:
        config["output_dir"] = args.output_dir

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")

    preprocessor = AudioPreprocessor(processor_name=config.get("model_name", "facebook/wav2vec2-base"))

    # Build dataset first (needed for stratified sampler)
    from src.data.dataset import AccentedSpeechDataset, collate_fn
    from torch.utils.data import DataLoader

    train_dataset = AccentedSpeechDataset(
        manifest_path=config["train_manifest"],
        preprocessor=preprocessor,
        split="train",
    )

    # Accent-stratified sampling to balance accent distribution
    sampler = AccentStratifiedSampler(train_dataset, replacement=True)
    dist = sampler.get_accent_distribution()
    logger.info(f"Accent sampling distribution: {dist}")

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

    # Load pretrained baseline
    model = AdversarialASRModel(
        model_name=config.get("model_name", "facebook/wav2vec2-base"),
        vocab_size=config.get("vocab_size", 32),
        num_accents=config.get("num_accents", 8),
    )

    if args.pretrained and os.path.exists(args.pretrained):
        model, _, start_epoch = load_checkpoint(args.pretrained, model, device=device)
        logger.info(f"Loaded pretrained model from {args.pretrained} (epoch {start_epoch})")
    else:
        logger.warning("No pretrained checkpoint found; training from scratch.")

    # Phase 1: no adversarial yet
    config["use_adversarial"] = False
    config["lambda_adv"] = 0.0
    config["learning_rate"] = config.get("finetune_lr", 3e-5)  # Lower LR for fine-tuning

    trainer = AdversarialASRTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        output_dir=config["output_dir"],
        device=device,
    )

    logger.info("Starting Phase 1: Accent adaptation with stratified sampling")
    history = trainer.train(num_epochs=config.get("num_epochs_phase1", 5))
    logger.info(f"Phase 1 complete. Best WER: {trainer.best_wer:.4f}")
