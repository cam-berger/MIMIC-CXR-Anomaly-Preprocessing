#!/usr/bin/env python3
"""
Train Masked Autoencoder (MAE) for chest X-ray anomaly detection.

This script implements the self-supervised pretraining pipeline described
in MODEL_TRAINING_RESEARCH.md:

Phase 1: MAE Pretraining on normal cohort
- Model learns to reconstruct masked patches
- Uses ViT encoder with asymmetric decoder
- 75% masking ratio (optimal for medical images)

Usage:
    # Quick test with small model
    python train_mae.py --config debug --train-dir output/preprocessed/normal_train

    # Full training with ViT-Base
    python train_mae.py --config base --epochs 800 --batch-size 64 \\
        --train-dir output/preprocessed/normal_train \\
        --val-dir output/preprocessed/normal_val

    # Resume from checkpoint
    python train_mae.py --resume checkpoints/mae_epoch_100.pt

Example:
    python train_mae.py \\
        --train-dir output/preprocessed/normal_train \\
        --val-dir output/preprocessed/normal_val \\
        --output-dir output/models \\
        --epochs 800 \\
        --batch-size 64

Data format (per PREPROCESSED_DATA_SCHEMA.md):
    output/preprocessed/{cohort_name}/
    ├── images.h5              # HDF5: /images/{idx}, /metadata/{idx}, /index
    ├── structured.parquet     # Clinical features
    ├── text.parquet           # Reports and summaries
    └── manifest.json          # Processing statistics
"""

import argparse
import json
import logging
import math
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.models.mae import MaskedAutoencoder, mae_vit_base_patch16, mae_vit_small_patch16
from src.models.dataset import MIMICCXRDataset, PreprocessedMAEDataset, get_mae_augmentations
from src.models.config import (
    TrainingConfig, MAEConfig,
    get_debug_config, get_fast_config, get_base_config
)
from src.models.anomaly import (
    ReconstructionAnomalyDetector,
    EmbeddingAnomalyDetector,
    EnsembleAnomalyDetector,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_lr_scheduler(
    optimizer: optim.Optimizer,
    config: MAEConfig,
    steps_per_epoch: int,
) -> optim.lr_scheduler._LRScheduler:
    """
    Create learning rate scheduler with warmup and cosine decay.

    Args:
        optimizer: Optimizer
        config: MAE configuration
        steps_per_epoch: Number of steps per epoch

    Returns:
        Learning rate scheduler
    """
    warmup_steps = config.warmup_epochs * steps_per_epoch
    total_steps = config.epochs * steps_per_epoch

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            # Linear warmup
            return step / warmup_steps
        else:
            # Cosine decay
            progress = (step - warmup_steps) / (total_steps - warmup_steps)
            return config.min_lr / config.base_lr + \
                   (1 - config.min_lr / config.base_lr) * \
                   0.5 * (1 + math.cos(math.pi * progress))

    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def create_model(config: MAEConfig, device: str) -> MaskedAutoencoder:
    """Create MAE model based on configuration."""
    logger.info(f"Creating {config.model_type} model...")

    model = MaskedAutoencoder(**config.get_model_kwargs())
    model = model.to(device)

    # Count parameters
    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model parameters: {n_params:,} total, {n_trainable:,} trainable")

    return model


def create_dataloaders(
    config: TrainingConfig,
) -> tuple[DataLoader, Optional[DataLoader]]:
    """Create training and validation dataloaders.

    Supports the preprocessed directory format:
        output/preprocessed/{cohort_name}/
        ├── images.h5
        ├── structured.parquet
        └── text.parquet
    """
    # Check for preprocessed directory (new format)
    if config.train_dir and config.train_dir.exists():
        logger.info(f"Loading training data from: {config.train_dir}")

        train_dataset = PreprocessedMAEDataset(
            config.train_dir,
            training=True,
            target_size=(config.mae.img_size, config.mae.img_size),
        )

        val_dataset = None
        if config.val_dir and config.val_dir.exists():
            logger.info(f"Loading validation data from: {config.val_dir}")
            val_dataset = PreprocessedMAEDataset(
                config.val_dir,
                training=False,
                target_size=(config.mae.img_size, config.mae.img_size),
            )

    # Legacy: Direct HDF5 path
    elif config.train_hdf5 and config.train_hdf5.exists():
        logger.info(f"Loading data from HDF5: {config.train_hdf5}")

        train_transform = get_mae_augmentations(
            target_size=(config.mae.img_size, config.mae.img_size),
            training=True,
        )
        val_transform = get_mae_augmentations(
            target_size=(config.mae.img_size, config.mae.img_size),
            training=False,
        )

        # For legacy HDF5, assume it's in a directory with images.h5
        train_dir = config.train_hdf5.parent
        train_dataset = MIMICCXRDataset(
            train_dir,
            transform=train_transform,
            target_size=(config.mae.img_size, config.mae.img_size),
        )

        val_dataset = None
        if config.val_hdf5 and config.val_hdf5.exists():
            val_dir = config.val_hdf5.parent
            val_dataset = MIMICCXRDataset(
                val_dir,
                transform=val_transform,
                target_size=(config.mae.img_size, config.mae.img_size),
            )
    else:
        raise ValueError(
            "Must provide --train-dir (preprocessed directory) or --train-hdf5 for data loading.\n"
            "Expected format: output/preprocessed/{cohort_name}/ with images.h5, structured.parquet, text.parquet"
        )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.mae.batch_size,
        shuffle=True,
        num_workers=config.mae.num_workers,
        pin_memory=config.mae.pin_memory,
        drop_last=True,
    )

    val_loader = None
    if val_dataset:
        val_loader = DataLoader(
            val_dataset,
            batch_size=config.mae.batch_size,
            shuffle=False,
            num_workers=config.mae.num_workers,
            pin_memory=config.mae.pin_memory,
        )

    logger.info(f"Training samples: {len(train_dataset):,}")
    if val_dataset:
        logger.info(f"Validation samples: {len(val_dataset):,}")

    return train_loader, val_loader


def train_epoch(
    model: MaskedAutoencoder,
    dataloader: DataLoader,
    optimizer: optim.Optimizer,
    scheduler: optim.lr_scheduler._LRScheduler,
    scaler: Optional[GradScaler],
    config: MAEConfig,
    device: str,
    epoch: int,
) -> dict:
    """Train for one epoch."""
    model.train()

    total_loss = 0.0
    num_batches = 0
    start_time = time.time()

    progress = tqdm(dataloader, desc=f"Epoch {epoch}")
    for batch_idx, batch in enumerate(progress):
        # Handle different batch formats
        if isinstance(batch, dict):
            images = batch["image"].to(device)
        elif isinstance(batch, (list, tuple)):
            images = batch[0].to(device)
        else:
            images = batch.to(device)

        optimizer.zero_grad()

        # Forward pass with mixed precision
        if scaler is not None:
            with autocast():
                loss, pred, mask = model(images)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss, pred, mask = model(images)
            loss.backward()
            optimizer.step()

        scheduler.step()

        # Track metrics
        total_loss += loss.item()
        num_batches += 1

        # Update progress bar
        if batch_idx % config.log_interval == 0:
            current_lr = scheduler.get_last_lr()[0]
            progress.set_postfix({
                "loss": f"{loss.item():.4f}",
                "lr": f"{current_lr:.2e}",
            })

    epoch_time = time.time() - start_time
    avg_loss = total_loss / num_batches

    return {
        "loss": avg_loss,
        "time": epoch_time,
        "lr": scheduler.get_last_lr()[0],
    }


@torch.no_grad()
def validate(
    model: MaskedAutoencoder,
    dataloader: DataLoader,
    config: MAEConfig,
    device: str,
) -> dict:
    """Validate model."""
    model.eval()

    total_loss = 0.0
    num_batches = 0

    for batch in tqdm(dataloader, desc="Validating"):
        if isinstance(batch, dict):
            images = batch["image"].to(device)
        elif isinstance(batch, (list, tuple)):
            images = batch[0].to(device)
        else:
            images = batch.to(device)

        loss, pred, mask = model(images)

        total_loss += loss.item()
        num_batches += 1

    avg_loss = total_loss / num_batches
    return {"loss": avg_loss}


def save_checkpoint(
    model: MaskedAutoencoder,
    optimizer: optim.Optimizer,
    scheduler: optim.lr_scheduler._LRScheduler,
    scaler: Optional[GradScaler],
    epoch: int,
    config: TrainingConfig,
    metrics: dict,
    is_best: bool = False,
) -> Path:
    """Save training checkpoint."""
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "config": {
            "mae": config.mae.__dict__,
        },
        "metrics": metrics,
    }

    if scaler is not None:
        checkpoint["scaler_state_dict"] = scaler.state_dict()

    # Save regular checkpoint
    checkpoint_path = config.checkpoint_dir / f"mae_epoch_{epoch:04d}.pt"
    torch.save(checkpoint, checkpoint_path)

    # Save best model
    if is_best:
        best_path = config.output_dir / "mae_best.pt"
        torch.save(checkpoint, best_path)
        logger.info(f"Saved best model to {best_path}")

    # Save latest
    latest_path = config.checkpoint_dir / "mae_latest.pt"
    torch.save(checkpoint, latest_path)

    return checkpoint_path


def load_checkpoint(
    checkpoint_path: Path,
    model: MaskedAutoencoder,
    optimizer: Optional[optim.Optimizer] = None,
    scheduler: Optional[optim.lr_scheduler._LRScheduler] = None,
    scaler: Optional[GradScaler] = None,
) -> int:
    """Load checkpoint and return starting epoch."""
    logger.info(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    if scheduler is not None and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    if scaler is not None and "scaler_state_dict" in checkpoint:
        scaler.load_state_dict(checkpoint["scaler_state_dict"])

    return checkpoint["epoch"]


def train(config: TrainingConfig) -> MaskedAutoencoder:
    """
    Main training loop.

    Args:
        config: Training configuration

    Returns:
        Trained model
    """
    set_seed(config.seed)
    device = config.device

    # Create model
    model = create_model(config.mae, device)

    # Create dataloaders
    train_loader, val_loader = create_dataloaders(config)

    # Create optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.mae.base_lr,
        betas=config.mae.betas,
        weight_decay=config.mae.weight_decay,
    )

    # Create scheduler
    steps_per_epoch = len(train_loader)
    scheduler = get_lr_scheduler(optimizer, config.mae, steps_per_epoch)

    # Mixed precision scaler
    scaler = GradScaler() if config.mae.mixed_precision and device == "cuda" else None

    # Training history
    history = {"train_loss": [], "val_loss": [], "lr": []}
    best_val_loss = float("inf")
    start_epoch = 0

    # Log training info
    logger.info("=" * 60)
    logger.info("MAE Training Configuration")
    logger.info("=" * 60)
    logger.info(f"Model: {config.mae.model_type}")
    logger.info(f"Mask ratio: {config.mae.mask_ratio}")
    logger.info(f"Epochs: {config.mae.epochs}")
    logger.info(f"Batch size: {config.mae.batch_size}")
    logger.info(f"Learning rate: {config.mae.base_lr}")
    logger.info(f"Device: {device}")
    logger.info(f"Mixed precision: {config.mae.mixed_precision}")
    logger.info("=" * 60)

    # Training loop
    for epoch in range(start_epoch, config.mae.epochs):
        # Train
        train_metrics = train_epoch(
            model, train_loader, optimizer, scheduler, scaler,
            config.mae, device, epoch
        )
        history["train_loss"].append(train_metrics["loss"])
        history["lr"].append(train_metrics["lr"])

        # Validate
        val_metrics = {"loss": None}
        if val_loader is not None and (epoch + 1) % config.mae.eval_interval == 0:
            val_metrics = validate(model, val_loader, config.mae, device)
            history["val_loss"].append(val_metrics["loss"])

            is_best = val_metrics["loss"] < best_val_loss
            if is_best:
                best_val_loss = val_metrics["loss"]
        else:
            is_best = False

        # Log
        log_msg = (
            f"Epoch {epoch:4d}/{config.mae.epochs} | "
            f"Train Loss: {train_metrics['loss']:.4f} | "
            f"LR: {train_metrics['lr']:.2e} | "
            f"Time: {train_metrics['time']:.1f}s"
        )
        if val_metrics["loss"] is not None:
            log_msg += f" | Val Loss: {val_metrics['loss']:.4f}"
        logger.info(log_msg)

        # Save checkpoint
        if (epoch + 1) % config.mae.save_interval == 0 or is_best:
            save_checkpoint(
                model, optimizer, scheduler, scaler,
                epoch, config,
                {"train_loss": train_metrics["loss"], "val_loss": val_metrics["loss"]},
                is_best=is_best,
            )

    # Save final model
    final_path = config.output_dir / "mae_final.pt"
    torch.save({
        "model_state_dict": model.state_dict(),
        "config": config.mae.__dict__,
        "history": history,
    }, final_path)
    logger.info(f"Saved final model to {final_path}")

    # Save training history
    history_path = config.output_dir / "training_history.json"
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    logger.info(f"Saved training history to {history_path}")

    return model


def fit_anomaly_detector(
    model: MaskedAutoencoder,
    train_loader: DataLoader,
    config: TrainingConfig,
) -> EnsembleAnomalyDetector:
    """
    Fit anomaly detector on training data.

    Args:
        model: Trained MAE model
        train_loader: Training data loader
        config: Training configuration

    Returns:
        Fitted ensemble detector
    """
    logger.info("Fitting anomaly detector on training data...")

    detector = EnsembleAnomalyDetector(
        model,
        device=config.device,
        weights=config.anomaly.ensemble_weights,
    )

    detector.fit(train_loader, percentile=config.anomaly.threshold_percentile)

    # Save detector parameters
    detector_path = config.output_dir / "anomaly_detector.pt"
    torch.save({
        "threshold": detector.threshold,
        "score_means": detector.score_means,
        "score_stds": detector.score_stds,
        "weights": detector.weights,
        "recon_threshold": detector.recon_detector.threshold,
        "recon_train_errors": detector.recon_detector.train_errors,
        "knn_threshold": detector.knn_detector.threshold,
        "gmm_threshold": detector.gmm_detector.threshold,
    }, detector_path)
    logger.info(f"Saved anomaly detector to {detector_path}")

    return detector


def main():
    parser = argparse.ArgumentParser(
        description="Train MAE for chest X-ray anomaly detection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Configuration preset
    parser.add_argument(
        "--config", type=str, default="base",
        choices=["debug", "fast", "base"],
        help="Configuration preset (default: base)"
    )

    # Data paths (new preprocessed format)
    parser.add_argument(
        "--train-dir", type=Path, default=None,
        help="Path to training preprocessed directory (e.g., output/preprocessed/normal_train)"
    )
    parser.add_argument(
        "--val-dir", type=Path, default=None,
        help="Path to validation preprocessed directory (e.g., output/preprocessed/normal_val)"
    )

    # Legacy data paths (backwards compatibility)
    parser.add_argument(
        "--train-hdf5", type=Path, default=None,
        help="[Legacy] Path to training HDF5 file"
    )
    parser.add_argument(
        "--val-hdf5", type=Path, default=None,
        help="[Legacy] Path to validation HDF5 file"
    )
    parser.add_argument(
        "--data-dir", type=Path, default=Path("output/preprocessed"),
        help="[Legacy] Base directory for preprocessed data"
    )

    # Output paths
    parser.add_argument(
        "--output-dir", type=Path, default=Path("output/models"),
        help="Directory for output models"
    )
    parser.add_argument(
        "--checkpoint-dir", type=Path, default=Path("output/checkpoints"),
        help="Directory for checkpoints"
    )

    # Training overrides
    parser.add_argument("--epochs", type=int, default=None, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=None, help="Batch size")
    parser.add_argument("--lr", type=float, default=None, help="Learning rate")
    parser.add_argument("--mask-ratio", type=float, default=None, help="Mask ratio")
    parser.add_argument("--img-size", type=int, default=None, help="Input image size (default: 224)")
    parser.add_argument("--patch-size", type=int, default=None, help="Patch size (default: 16)")

    # Resumption
    parser.add_argument(
        "--resume", type=Path, default=None,
        help="Path to checkpoint to resume from"
    )

    # Hardware
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use"
    )
    parser.add_argument(
        "--num-workers", type=int, default=None,
        help="Number of data loading workers"
    )

    # Phases
    parser.add_argument(
        "--skip-pretrain", action="store_true",
        help="Skip pretraining (load from checkpoint)"
    )
    parser.add_argument(
        "--skip-anomaly", action="store_true",
        help="Skip anomaly detector fitting"
    )

    args = parser.parse_args()

    # Load configuration preset
    if args.config == "debug":
        config = get_debug_config()
    elif args.config == "fast":
        config = get_fast_config()
    else:
        config = get_base_config()

    # Apply overrides
    # New preprocessed directory format (preferred)
    if args.train_dir:
        config.train_dir = args.train_dir
    if args.val_dir:
        config.val_dir = args.val_dir

    # Legacy paths
    if args.train_hdf5:
        config.train_hdf5 = args.train_hdf5
    if args.val_hdf5:
        config.val_hdf5 = args.val_hdf5
    if args.data_dir:
        config.data_dir = args.data_dir
    if args.output_dir:
        config.output_dir = args.output_dir
    if args.checkpoint_dir:
        config.checkpoint_dir = args.checkpoint_dir
    if args.epochs:
        config.mae.epochs = args.epochs
    if args.batch_size:
        config.mae.batch_size = args.batch_size
    if args.lr:
        config.mae.base_lr = args.lr
    if args.mask_ratio:
        config.mae.mask_ratio = args.mask_ratio
    if args.img_size:
        config.mae.img_size = args.img_size
    if args.patch_size:
        config.mae.patch_size = args.patch_size
    if args.num_workers:
        config.mae.num_workers = args.num_workers
    config.device = args.device

    # Create directories
    config.output_dir.mkdir(parents=True, exist_ok=True)
    config.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Save configuration
    config_path = config.output_dir / "config.json"
    with open(config_path, "w") as f:
        json.dump({
            "mae": config.mae.__dict__,
            "anomaly": config.anomaly.__dict__,
            "device": config.device,
            "seed": config.seed,
        }, f, indent=2, default=str)
    logger.info(f"Saved configuration to {config_path}")

    # Train
    if not args.skip_pretrain:
        if args.resume:
            # Resume training
            model = create_model(config.mae, config.device)
            train_loader, val_loader = create_dataloaders(config)

            optimizer = optim.AdamW(
                model.parameters(),
                lr=config.mae.base_lr,
                betas=config.mae.betas,
                weight_decay=config.mae.weight_decay,
            )
            steps_per_epoch = len(train_loader)
            scheduler = get_lr_scheduler(optimizer, config.mae, steps_per_epoch)
            scaler = GradScaler() if config.mae.mixed_precision else None

            start_epoch = load_checkpoint(
                args.resume, model, optimizer, scheduler, scaler
            )
            logger.info(f"Resuming from epoch {start_epoch}")

            # Continue training
            model = train(config)
        else:
            model = train(config)
    else:
        # Load from checkpoint
        if args.resume:
            model = create_model(config.mae, config.device)
            load_checkpoint(args.resume, model)
        else:
            # Load best model
            best_path = config.output_dir / "mae_best.pt"
            if best_path.exists():
                model = create_model(config.mae, config.device)
                load_checkpoint(best_path, model)
            else:
                raise ValueError("No model found. Run training first or provide --resume")

    # Fit anomaly detector
    if not args.skip_anomaly:
        train_loader, _ = create_dataloaders(config)
        detector = fit_anomaly_detector(model, train_loader, config)

    logger.info("Training complete!")


if __name__ == "__main__":
    main()
