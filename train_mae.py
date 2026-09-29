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

    # Resume from checkpoint (configuration and data paths come from the checkpoint)
    python train_mae.py --resume output/checkpoints/mae_latest.pt

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
import sys
import time
from dataclasses import fields
from pathlib import Path
from typing import Optional

import torch
import torch.optim as optim
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.models.mae import MaskedAutoencoder
from src.models.dataset import MIMICCXRDataset, PreprocessedMAEDataset, get_mae_augmentations
from src.models.config import (
    IMAGE_MODES,
    LEGACY_IMAGE_MODE,
    TrainingConfig, MAEConfig,
    get_debug_config, get_fast_config, get_base_config
)
from src.models.anomaly import EnsembleAnomalyDetector
from src.training import (
    ConsecutiveSkipGuard,
    assert_params_finite,
    backward_and_step,
    build_warmup_cosine_scheduler,
    is_cuda_device,
    load_checkpoint_states,
    save_checkpoint_files,
    set_seed,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# CLI flag -> MAEConfig field
CLI_OVERRIDES = {
    "epochs": "epochs",
    "batch_size": "batch_size",
    "lr": "base_lr",
    "mask_ratio": "mask_ratio",
    "img_size": "img_size",
    "patch_size": "patch_size",
    "image_mode": "image_mode",
    "num_workers": "num_workers",
}

# Resuming continues the saved schedule/model, so these cannot change
RESUME_LOCKED_FIELDS = {
    "epochs", "batch_size", "base_lr", "mask_ratio", "img_size", "patch_size", "image_mode",
}

PATH_FIELDS = ("train_dir", "val_dir", "output_dir", "checkpoint_dir")


def mae_augmentation_kwargs(config: MAEConfig) -> dict:
    """Augmentation settings from MAEConfig for get_mae_augmentations."""
    return {
        "crop_scale": config.crop_scale,
        "horizontal_flip": config.horizontal_flip,
        "rotation_degrees": config.rotation_degrees,
        "gaussian_blur": config.gaussian_blur,
    }


def apply_overrides(config: TrainingConfig, args: argparse.Namespace, resuming: bool) -> None:
    """Apply CLI overrides. Explicit zeros count (e.g. --num-workers 0)."""
    for arg, field_name in CLI_OVERRIDES.items():
        value = getattr(args, arg)
        if value is None:
            continue
        current = getattr(config.mae, field_name)
        if resuming and field_name in RESUME_LOCKED_FIELDS and value != current:
            raise ValueError(
                f"--{arg.replace('_', '-')} {value} differs from the checkpoint's value "
                f"({current}). Resuming continues the saved run; start a new run to change it."
            )
        setattr(config.mae, field_name, value)


def config_from_checkpoint(checkpoint: dict) -> TrainingConfig:
    """Rebuild the training configuration stored in a checkpoint."""
    saved = checkpoint["config"]
    mae_saved = saved.get("mae", saved)
    known = {f.name for f in fields(MAEConfig)}
    config = TrainingConfig()
    config.mae = MAEConfig(**{k: v for k, v in mae_saved.items() if k in known})
    if "image_mode" not in mae_saved:
        # Saved before image_mode existed: that model was trained on center crops
        config.mae.image_mode = LEGACY_IMAGE_MODE
    for key in PATH_FIELDS:
        if saved.get(key) is not None:
            setattr(config, key, Path(saved[key]))
    return config


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
            image_mode=config.mae.image_mode,
            augmentation=mae_augmentation_kwargs(config.mae),
        )

        val_dataset = None
        if config.val_dir and config.val_dir.exists():
            logger.info(f"Loading validation data from: {config.val_dir}")
            val_dataset = PreprocessedMAEDataset(
                config.val_dir,
                training=False,
                target_size=(config.mae.img_size, config.mae.img_size),
                image_mode=config.mae.image_mode,
            )

    # Legacy: Direct HDF5 path
    elif config.train_hdf5 and config.train_hdf5.exists():
        logger.info(f"Loading data from HDF5: {config.train_hdf5}")

        train_transform = get_mae_augmentations(
            target_size=(config.mae.img_size, config.mae.img_size),
            training=True,
            mode=config.mae.image_mode,
            **mae_augmentation_kwargs(config.mae),
        )
        val_transform = get_mae_augmentations(
            target_size=(config.mae.img_size, config.mae.img_size),
            training=False,
            mode=config.mae.image_mode,
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
    scheduler: optim.lr_scheduler.LRScheduler,
    scaler: GradScaler,
    config: MAEConfig,
    device: str,
    epoch: int,
) -> dict:
    """
    Train for one epoch.

    Batches with a non-finite loss or gradient are skipped, never stepped
    (backward_and_step). The scheduler advances every batch.
    """
    model.train()

    device_type = torch.device(device).type
    total_loss = 0.0
    num_steps = 0
    skipped = 0
    guard = ConsecutiveSkipGuard(config.max_consecutive_skips)
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

        # Forward pass with mixed precision
        with torch.autocast(device_type=device_type, enabled=scaler.is_enabled()):
            loss, pred, mask = model(images)

        stepped = False
        if torch.isfinite(loss):
            stepped, _ = backward_and_step(loss, optimizer, scaler, config.grad_clip)

        if stepped:
            total_loss += loss.item()
            num_steps += 1
        else:
            skipped += 1

        scheduler.step()
        guard.update(stepped, context=f"epoch {epoch}, batch {batch_idx}")

        # Update progress bar
        if batch_idx % config.log_interval == 0:
            current_lr = scheduler.get_last_lr()[0]
            progress.set_postfix({
                "loss": f"{loss.item():.4f}",
                "lr": f"{current_lr:.2e}",
                "skipped": skipped,
            })

    assert_params_finite([model])

    epoch_time = time.time() - start_time
    avg_loss = total_loss / max(num_steps, 1)
    if skipped:
        logger.warning(f"Epoch {epoch}: {skipped} batches skipped (non-finite loss or gradients)")

    return {
        "loss": avg_loss,
        "time": epoch_time,
        "lr": scheduler.get_last_lr()[0],
        "skipped": skipped,
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

    avg_loss = total_loss / max(num_batches, 1)
    return {"loss": avg_loss}


def config_to_dict(config: TrainingConfig) -> dict:
    """Configuration stored in checkpoints (MAE settings + data/output paths)."""
    return {
        "mae": dict(config.mae.__dict__),
        **{key: str(getattr(config, key)) if getattr(config, key) is not None else None
           for key in PATH_FIELDS},
    }


def save_checkpoint(
    model: MaskedAutoencoder,
    optimizer: optim.Optimizer,
    scheduler: optim.lr_scheduler.LRScheduler,
    scaler: GradScaler,
    epoch: int,
    config: TrainingConfig,
    metrics: dict,
    best_val_loss: float,
    history: dict,
    is_best: bool = False,
) -> Path:
    """Save training checkpoint (``epoch`` is the epoch just completed)."""
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "config": config_to_dict(config),
        "metrics": metrics,
        "best_val_loss": best_val_loss,
        "history": history,
    }
    if scaler.is_enabled():
        checkpoint["scaler_state_dict"] = scaler.state_dict()

    return save_checkpoint_files(
        checkpoint,
        config.checkpoint_dir,
        prefix="mae",
        epoch=epoch,
        best_path=config.output_dir / "mae_best.pt" if is_best else None,
    )


def train(config: TrainingConfig, resume_checkpoint: Optional[dict] = None) -> MaskedAutoencoder:
    """
    Main training loop.

    Args:
        config: Training configuration
        resume_checkpoint: Loaded checkpoint to resume from (continues after
            its last completed epoch, with its best validation loss and history)

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
    scheduler = build_warmup_cosine_scheduler(
        optimizer,
        warmup_steps=config.mae.warmup_epochs * steps_per_epoch,
        total_steps=config.mae.epochs * steps_per_epoch,
        min_lr_ratio=config.mae.min_lr / config.mae.base_lr,
    )

    # Mixed precision scaler (disabled: plain full-precision steps)
    scaler = GradScaler(enabled=config.mae.mixed_precision and is_cuda_device(device))

    # Training history
    history = {"epoch": [], "train_loss": [], "val_epoch": [], "val_loss": [], "lr": []}
    best_val_loss = float("inf")
    start_epoch = 0

    # Resume from checkpoint if provided
    if resume_checkpoint is not None:
        load_checkpoint_states(resume_checkpoint, model, optimizer, scheduler, scaler)
        start_epoch = resume_checkpoint["epoch"] + 1
        best_val_loss = resume_checkpoint.get("best_val_loss", best_val_loss)
        history = resume_checkpoint.get("history", history)
        logger.info(f"Resuming training at epoch {start_epoch}")

    # Log training info
    logger.info("=" * 60)
    logger.info("MAE Training Configuration")
    logger.info("=" * 60)
    logger.info(f"Model: {config.mae.model_type}")
    logger.info(f"Mask ratio: {config.mae.mask_ratio}")
    logger.info(f"Epochs: {config.mae.epochs}")
    logger.info(f"Batch size: {config.mae.batch_size}")
    logger.info(f"Learning rate: {config.mae.base_lr}")
    logger.info(f"Image: {config.mae.img_size}px, mode={config.mae.image_mode}")
    logger.info(f"Device: {device}")
    logger.info(f"Mixed precision: {scaler.is_enabled()}")
    logger.info("=" * 60)

    # Training loop
    for epoch in range(start_epoch, config.mae.epochs):
        # Train
        train_metrics = train_epoch(
            model, train_loader, optimizer, scheduler, scaler,
            config.mae, device, epoch
        )
        history["epoch"].append(epoch)
        history["train_loss"].append(train_metrics["loss"])
        history["lr"].append(train_metrics["lr"])

        # Validate
        val_metrics = {"loss": None}
        if val_loader is not None and (epoch + 1) % config.mae.eval_interval == 0:
            val_metrics = validate(model, val_loader, config.mae, device)
            history["val_epoch"].append(epoch)
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
                best_val_loss, history,
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
        "--config", type=str, default=None,
        choices=["debug", "fast", "base"],
        help="Configuration preset (default: base; ignored with --resume)"
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
        "--data-dir", type=Path, default=None,
        help="[Legacy] Base directory for preprocessed data"
    )

    # Output paths
    parser.add_argument(
        "--output-dir", type=Path, default=None,
        help="Directory for output models (default: output/models)"
    )
    parser.add_argument(
        "--checkpoint-dir", type=Path, default=None,
        help="Directory for checkpoints (default: output/checkpoints)"
    )

    # Training overrides
    parser.add_argument("--epochs", type=int, default=None, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=None, help="Batch size")
    parser.add_argument("--lr", type=float, default=None, help="Learning rate")
    parser.add_argument("--mask-ratio", type=float, default=None, help="Mask ratio")
    parser.add_argument("--img-size", type=int, default=None, help="Input image size (default: 224)")
    parser.add_argument("--patch-size", type=int, default=None, help="Patch size (default: 16)")
    parser.add_argument(
        "--image-mode", type=str, default=None, choices=IMAGE_MODES,
        help="resize: full radiograph (default); center_crop: legacy crop from native resolution"
    )

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

    # Load configuration: from the checkpoint when resuming, else from a preset
    resume_checkpoint = None
    if args.resume is not None:
        resume_checkpoint = torch.load(args.resume, map_location="cpu")
        config = config_from_checkpoint(resume_checkpoint)
        if args.config is not None:
            logger.info("--config is ignored when resuming; using the checkpoint's configuration")
    elif args.config == "debug":
        config = get_debug_config()
    elif args.config == "fast":
        config = get_fast_config()
    else:
        config = get_base_config()

    # Apply paths
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
    config.device = args.device

    # Apply overrides
    try:
        apply_overrides(config, args, resuming=resume_checkpoint is not None and not args.skip_pretrain)
    except ValueError as e:
        parser.error(str(e))

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
        model = train(config, resume_checkpoint=resume_checkpoint)
    else:
        # Load from checkpoint
        if resume_checkpoint is not None:
            model = create_model(config.mae, config.device)
            load_checkpoint_states(resume_checkpoint, model)
        else:
            # Load best model
            best_path = config.output_dir / "mae_best.pt"
            if best_path.exists():
                best_checkpoint = torch.load(best_path, map_location="cpu")
                model = create_model(config_from_checkpoint(best_checkpoint).mae, config.device)
                load_checkpoint_states(best_checkpoint, model)
            else:
                raise ValueError("No model found. Run training first or provide --resume")

    # Fit anomaly detector
    if not args.skip_anomaly:
        train_loader, _ = create_dataloaders(config)
        detector = fit_anomaly_detector(model, train_loader, config)

    logger.info("Training complete!")


if __name__ == "__main__":
    main()
