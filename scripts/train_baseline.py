"""
Phase 0: Train baseline Wav2Vec 2.0 ASR model on LibriSpeech.
No adversarial training — standard CTC fine-tuning.

Usage:
    python scripts/train_baseline.py --config configs/baseline_config.yaml
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
from src.training.trainer import AdversarialASRTrainer
from src.utils.helpers import set_seed, get_logger, count_parameters, format_number


def parse_args():
    parser = argparse.ArgumentParser(description="Phase 0: Baseline ASR Training")
    parser.add_argument("--config", type=str, default="configs/baseline_config.yaml")
    parser.add_argument("--train_manifest", type=str, default=None)
    parser.add_argument("--val_manifest", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="results/phase0_baseline")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    logger = get_logger("train_baseline")

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    # Override with CLI args
    if args.train_manifest:
        config["train_manifest"] = args.train_manifest
    if args.val_manifest:
        config["val_manifest"] = args.val_manifest
    if args.output_dir:
        config["output_dir"] = args.output_dir

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")

    # Data
    preprocessor = AudioPreprocessor(processor_name=config.get("model_name", "facebook/wav2vec2-base"))

    train_loader = create_dataloader(
        manifest_path=config["train_manifest"],
        preprocessor=preprocessor,
        split="train",
        batch_size=config.get("batch_size", 8),
        num_workers=config.get("num_workers", 4),
        shuffle=True,
    )

    val_loader = create_dataloader(
        manifest_path=config["val_manifest"],
        preprocessor=preprocessor,
        split="val",
        batch_size=config.get("batch_size", 8),
        num_workers=config.get("num_workers", 4),
        shuffle=False,
    )

    # Model
    model = AdversarialASRModel(
        model_name=config.get("model_name", "facebook/wav2vec2-base"),
        vocab_size=config.get("vocab_size", 32),
        num_accents=config.get("num_accents", 8),
        freeze_feature_extractor=config.get("freeze_feature_extractor", True),
    )

    logger.info(f"Model parameters: {format_number(count_parameters(model))}")

    # Disable adversarial for baseline
    config["use_adversarial"] = False
    config["lambda_adv"] = 0.0

    trainer = AdversarialASRTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        output_dir=config["output_dir"],
        device=device,
    )

    logger.info("Starting Phase 0: Baseline training on LibriSpeech")
    history = trainer.train(num_epochs=config.get("num_epochs", 10))

    logger.info(f"Training complete. Best WER: {trainer.best_wer:.4f}")
    logger.info(f"Model saved to: {config['output_dir']}/best_model.pt")


if __name__ == "__main__":
    main()
