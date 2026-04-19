"""
Utility functions: seeding, checkpointing, logging.
"""

import os
import random
import logging
import numpy as np
import torch
from typing import Optional


def set_seed(seed: int = 42):
    """Set random seed for full reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Create a logger with console handler."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(level)
        handler = logging.StreamHandler()
        handler.setLevel(level)
        formatter = logging.Formatter(
            "[%(asctime)s] %(levelname)s %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


def save_checkpoint(
    model,
    optimizer,
    epoch: int,
    config: dict,
    path: str,
    extra: Optional[dict] = None,
):
    """Save model checkpoint."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "model_config": config,
    }
    if extra:
        checkpoint.update(extra)
    torch.save(checkpoint, path)
    logging.getLogger(__name__).info(f"Checkpoint saved: {path}")


def load_checkpoint(path: str, model, optimizer=None, device: str = "cpu"):
    """Load model from checkpoint."""
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    epoch = checkpoint.get("epoch", 0)
    return model, optimizer, epoch


def count_parameters(model) -> int:
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def format_number(n: int) -> str:
    """Format large numbers with M/K suffix."""
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"
    elif n >= 1_000:
        return f"{n / 1_000:.1f}K"
    return str(n)
