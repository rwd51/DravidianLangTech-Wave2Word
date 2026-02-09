# Training utilities for fine-tuning

import os
import logging
import math
from pathlib import Path
from typing import Dict, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

logger = logging.getLogger(__name__)


def get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps):
    """
    Create a schedule with a learning rate that linearly increases during a warmup period
    and then linearly decreases.
    """

    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(
            0.0,
            float(num_training_steps - current_step)
            / float(max(1, num_training_steps - num_warmup_steps)),
        )

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def setup_training_config(
    model, train_dataloader, num_epochs, warmup_steps, learning_rate_range
):
    """
    Setup optimizer, scheduler, and training configuration.

    Returns:
        optimizer, scheduler, total_training_steps, training_config
    """
    # Unpack learning rate range
    lr_min, lr_max = learning_rate_range
    lr = (lr_min + lr_max) / 2  # Use average learning rate

    # Get trainable parameters
    trainable_params = [p for p in model.parameters() if p.requires_grad]

    logger.info(
        f"Number of trainable parameters: {sum(p.numel() for p in trainable_params):,}"
    )

    # Setup optimizer
    optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=1e-5)

    # Calculate total training steps
    num_update_steps_per_epoch = len(train_dataloader)
    total_training_steps = int(num_update_steps_per_epoch * num_epochs)

    # Setup scheduler
    scheduler = get_linear_schedule_with_warmup(
        optimizer, warmup_steps, total_training_steps
    )

    training_config = {
        "lr": lr,
        "lr_min": lr_min,
        "lr_max": lr_max,
        "total_steps": total_training_steps,
        "warmup_steps": warmup_steps,
        "num_epochs": num_epochs,
    }

    return optimizer, scheduler, total_training_steps, training_config


def compute_wer(predicted: str, reference: str) -> float:
    """
    Compute Word Error Rate (WER).
    WER = (S + D + I) / N
    where S = substitutions, D = deletions, I = insertions, N = number of reference words
    """
    pred_words = predicted.split()
    ref_words = reference.split()

    # Dynamic programming for edit distance
    d = {}

    for i in range(len(pred_words) + 1):
        d[i, 0] = i
    for j in range(len(ref_words) + 1):
        d[0, j] = j

    for i in range(1, len(pred_words) + 1):
        for j in range(1, len(ref_words) + 1):
            if pred_words[i - 1] == ref_words[j - 1]:
                d[i, j] = d[i - 1, j - 1]
            else:
                d[i, j] = min(d[i - 1, j], d[i, j - 1], d[i - 1, j - 1]) + 1

    wer = (
        float(d[len(pred_words), len(ref_words)]) / len(ref_words)
        if len(ref_words) > 0
        else 0.0
    )
    return wer


class TrainingMetrics:
    """Track training metrics."""

    def __init__(self, log_interval: int = 100):
        self.log_interval = log_interval
        self.train_losses = []
        self.val_losses = []
        self.wers = []
        self.step = 0

    def update_train(self, loss: float):
        """Update training loss."""
        self.train_losses.append(loss)
        self.step += 1

    def get_train_loss(self) -> float:
        """Get average training loss over log interval."""
        if len(self.train_losses) == 0:
            return 0.0
        return np.mean(self.train_losses[-self.log_interval :])

    def add_val_metrics(self, val_loss: float, wer: float):
        """Add validation metrics."""
        self.val_losses.append(val_loss)
        self.wers.append(wer)

    def get_last_val_metrics(self) -> Tuple[float, float]:
        """Get last validation loss and WER."""
        if len(self.val_losses) == 0:
            return 0.0, 0.0
        return self.val_losses[-1], self.wers[-1]


def save_checkpoint(
    model, optimizer, scheduler, config, metrics, checkpoint_dir, step=None
):
    """Save checkpoint."""
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = checkpoint_dir / "checkpoint.pt"

    logger.info(f"Saving checkpoint to {checkpoint_path}")

    torch.save(
        {
            "step": step,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "config": config,
            "metrics": {
                "train_losses": metrics.train_losses,
                "val_losses": metrics.val_losses,
                "wers": metrics.wers,
            },
        },
        checkpoint_path,
    )


def load_checkpoint(model, optimizer, scheduler, checkpoint_dir):
    """Load checkpoint."""
    checkpoint_path = Path(checkpoint_dir) / "checkpoint.pt"

    if not checkpoint_path.exists():
        logger.warning(f"Checkpoint not found at {checkpoint_path}")
        return 0, None

    logger.info(f"Loading checkpoint from {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(checkpoint["model_state"], strict=False)

    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state"])
    if scheduler is not None:
        scheduler.load_state_dict(checkpoint["scheduler_state"])

    metrics = {
        "train_losses": checkpoint.get("metrics", {}).get("train_losses", []),
        "val_losses": checkpoint.get("metrics", {}).get("val_losses", []),
        "wers": checkpoint.get("metrics", {}).get("wers", []),
    }

    return checkpoint.get("step", 0), metrics


def log_metrics(
    step: int,
    train_loss: float,
    val_loss: Optional[float] = None,
    wer: Optional[float] = None,
    lr: Optional[float] = None,
):
    """Log training metrics."""
    msg = f"Step {step:6d} | Train Loss: {train_loss:.4f}"

    if val_loss is not None:
        msg += f" | Val Loss: {val_loss:.4f}"
    if wer is not None:
        msg += f" | WER: {wer:.4f}"
    if lr is not None:
        msg += f" | LR: {lr:.2e}"

    logger.info(msg)


def get_device(device_name: str = "cuda") -> torch.device:
    """Get device."""
    if device_name == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda:0")
        logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        logger.info("Using CPU")

    return device
