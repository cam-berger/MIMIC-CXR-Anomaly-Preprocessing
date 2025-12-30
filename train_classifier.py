#!/usr/bin/env python3
"""
Train multimodal classifier for CheXpert pathology classification.

This script implements the classification pipeline:
- Multimodal fusion of images, text, and structured clinical data
- CLIP-style contrastive learning for image-text alignment
- Supervised contrastive learning for label-based clustering
- Asymmetric focal loss for imbalanced multi-label classification

Usage:
    # Quick test
    python train_classifier.py --config debug \
        --train-dir output/preprocessed/anomalous_train \
        --val-dir output/preprocessed/anomalous_val \
        --chexpert-csv /path/to/mimic-cxr-2.0.0-chexpert.csv.gz

    # Full training with pretrained MAE
    python train_classifier.py --config base \
        --train-dir output/preprocessed/anomalous_train \
        --val-dir output/preprocessed/anomalous_val \
        --chexpert-csv /path/to/mimic-cxr-2.0.0-chexpert.csv.gz \
        --mae-checkpoint output/models/mae_final.pt

    # Resume from checkpoint
    python train_classifier.py --resume output/checkpoints/classifier_latest.pt
"""

import argparse
import json
import logging
import math
import os
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, List

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score
from tqdm import tqdm

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.models.multimodal import MultimodalClassifier
from src.models.classification_dataset import (
    MultimodalClassificationDataset,
    collate_multimodal,
)
from src.models.losses import MultiTaskLoss, AsymmetricFocalLoss, CLIPLoss, SupConLoss

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


@dataclass
class ClassifierConfig:
    """Configuration for multimodal classifier training."""

    # Model
    embed_dim: int = 768
    struct_input_dim: int = 44
    struct_hidden_dim: int = 256
    contrastive_dim: int = 128
    num_labels: int = 12
    img_size: int = 224

    # Training
    epochs: int = 50
    batch_size: int = 16  # Reduced from 32 for stability
    base_lr: float = 5e-5  # Reduced from 1e-4 to prevent gradient explosion
    min_lr: float = 1e-6
    weight_decay: float = 0.05
    warmup_epochs: int = 5
    freeze_mae_epochs: int = 5
    lr_decay: float = 0.9  # Layer-wise LR decay factor

    # Loss weights
    cls_weight: float = 1.0
    clip_weight: float = 0.3  # Re-enabled with FP32 numerical stability fixes
    supcon_weight: float = 0.3  # Re-enabled with FP32 numerical stability fixes

    # Data
    num_workers: int = 4
    pin_memory: bool = True

    # Logging/saving
    log_interval: int = 50
    eval_interval: int = 1
    save_interval: int = 5

    # Mixed precision
    mixed_precision: bool = True


@dataclass
class TrainingConfig:
    """Full training configuration."""

    classifier: ClassifierConfig = field(default_factory=ClassifierConfig)

    # Paths
    train_dir: Optional[Path] = None
    val_dir: Optional[Path] = None
    chexpert_csv: Optional[Path] = None
    mae_checkpoint: Optional[Path] = None
    output_dir: Path = Path("output/models")
    checkpoint_dir: Path = Path("output/checkpoints")

    # Hardware
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 42


def get_debug_config() -> TrainingConfig:
    """Debug configuration for quick testing."""
    config = TrainingConfig()
    config.classifier.epochs = 2
    config.classifier.batch_size = 4
    config.classifier.num_workers = 0
    config.classifier.eval_interval = 1
    config.classifier.save_interval = 1
    config.classifier.warmup_epochs = 1
    config.classifier.freeze_mae_epochs = 100  # Keep MAE frozen to avoid OOM
    return config


def get_fast_config() -> TrainingConfig:
    """Fast configuration for development."""
    config = TrainingConfig()
    config.classifier.epochs = 10
    config.classifier.batch_size = 16
    config.classifier.warmup_epochs = 2
    config.classifier.freeze_mae_epochs = 2
    return config


def get_base_config() -> TrainingConfig:
    """Base configuration for full training."""
    return TrainingConfig()


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def create_model(config: TrainingConfig) -> MultimodalClassifier:
    """Create multimodal classifier."""
    logger.info("Creating MultimodalClassifier...")

    model = MultimodalClassifier(
        mae_checkpoint=config.mae_checkpoint,
        num_labels=config.classifier.num_labels,
        embed_dim=config.classifier.embed_dim,
        struct_input_dim=config.classifier.struct_input_dim,
        struct_hidden_dim=config.classifier.struct_hidden_dim,
        contrastive_dim=config.classifier.contrastive_dim,
        freeze_mae_epochs=config.classifier.freeze_mae_epochs,
        img_size=config.classifier.img_size,
    )

    model = model.to(config.device)

    # Count parameters
    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model parameters: {n_params:,} total, {n_trainable:,} trainable")

    return model


def create_dataloaders(
    config: TrainingConfig,
) -> tuple[DataLoader, Optional[DataLoader]]:
    """Create training and validation dataloaders."""
    if config.train_dir is None or not config.train_dir.exists():
        raise ValueError(f"Training directory not found: {config.train_dir}")
    if config.chexpert_csv is None or not config.chexpert_csv.exists():
        raise ValueError(f"CheXpert CSV not found: {config.chexpert_csv}")

    logger.info(f"Loading training data from: {config.train_dir}")

    train_dataset = MultimodalClassificationDataset(
        preprocessed_dir=config.train_dir,
        chexpert_csv=config.chexpert_csv,
        target_size=(config.classifier.img_size, config.classifier.img_size),
        augment=True,
    )

    val_dataset = None
    if config.val_dir and config.val_dir.exists():
        logger.info(f"Loading validation data from: {config.val_dir}")
        val_dataset = MultimodalClassificationDataset(
            preprocessed_dir=config.val_dir,
            chexpert_csv=config.chexpert_csv,
            target_size=(config.classifier.img_size, config.classifier.img_size),
            augment=False,
        )

    # Use 'spawn' to avoid HDF5 fork issues when num_workers > 0
    mp_context = 'spawn' if config.classifier.num_workers > 0 else None

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.classifier.batch_size,
        shuffle=True,
        num_workers=config.classifier.num_workers,
        pin_memory=config.classifier.pin_memory,
        drop_last=True,
        collate_fn=collate_multimodal,
        multiprocessing_context=mp_context,
    )

    val_loader = None
    if val_dataset:
        val_loader = DataLoader(
            val_dataset,
            batch_size=config.classifier.batch_size,
            shuffle=False,
            num_workers=config.classifier.num_workers,
            pin_memory=config.classifier.pin_memory,
            collate_fn=collate_multimodal,
            multiprocessing_context=mp_context,
        )

    logger.info(f"Training samples: {len(train_dataset):,}")
    if val_dataset:
        logger.info(f"Validation samples: {len(val_dataset):,}")

    # Log label frequencies
    frequencies = train_dataset.get_label_frequencies()
    logger.info("Label frequencies:")
    for label, freq in frequencies.items():
        logger.info(f"  {label}: {freq:.3f}")

    return train_loader, val_loader


def create_optimizer_with_llrd(
    model: MultimodalClassifier,
    config: ClassifierConfig,
) -> optim.Optimizer:
    """
    Create optimizer with layer-wise learning rate decay.

    Deeper layers get lower learning rates to preserve pretrained features.
    """
    # Get parameter groups
    groups = model.get_layer_groups()

    # Apply layer-wise LR decay
    param_groups = []
    for i, group_params in enumerate(groups):
        # Earlier groups (deeper) get lower LR
        lr_scale = config.lr_decay ** (len(groups) - 1 - i)
        param_groups.append({
            "params": group_params,
            "lr": config.base_lr * lr_scale,
        })

    logger.info("Layer-wise learning rates:")
    for i, pg in enumerate(param_groups):
        logger.info(f"  Group {i}: lr={pg['lr']:.2e}")

    return optim.AdamW(
        param_groups,
        betas=(0.9, 0.999),
        weight_decay=config.weight_decay,
    )


def get_lr_scheduler(
    optimizer: optim.Optimizer,
    config: ClassifierConfig,
    steps_per_epoch: int,
) -> optim.lr_scheduler._LRScheduler:
    """Create learning rate scheduler with warmup and cosine decay."""
    warmup_steps = config.warmup_epochs * steps_per_epoch
    total_steps = config.epochs * steps_per_epoch

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / warmup_steps
        else:
            progress = (step - warmup_steps) / (total_steps - warmup_steps)
            return config.min_lr / config.base_lr + \
                   (1 - config.min_lr / config.base_lr) * \
                   0.5 * (1 + math.cos(math.pi * progress))

    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def _params_finite(model: nn.Module) -> bool:
    """Check if all model parameters are finite (no NaN/Inf)."""
    for p in model.parameters():
        if p.requires_grad and not torch.isfinite(p).all():
            return False
    return True


def _copy_params(model: nn.Module) -> Dict[str, torch.Tensor]:
    """Create a CPU copy of model parameters for checkpoint."""
    return {n: p.data.clone().cpu() for n, p in model.named_parameters() if p.requires_grad}


def _restore_params(model: nn.Module, saved: Dict[str, torch.Tensor]) -> None:
    """Restore model parameters from checkpoint."""
    for n, p in model.named_parameters():
        if n in saved and p.requires_grad:
            p.data.copy_(saved[n].to(p.device))


def train_epoch(
    model: MultimodalClassifier,
    dataloader: DataLoader,
    optimizer: optim.Optimizer,
    scheduler: optim.lr_scheduler._LRScheduler,
    scaler: Optional[GradScaler],
    loss_fn: MultiTaskLoss,
    config: ClassifierConfig,
    device: str,
    epoch: int,
) -> Dict[str, float]:
    """Train for one epoch with bulletproof NaN handling."""
    model.train()
    model.set_epoch(epoch)

    total_losses = {"total": 0.0, "cls": 0.0, "clip": 0.0, "supcon": 0.0}
    num_batches = 0
    nan_batches = 0
    nan_study_ids = []  # Track which studies cause NaN for debugging
    weight_reverts = 0
    consecutive_corruptions = 0  # Track consecutive weight corruptions for circuit breaker
    start_time = time.time()

    # Weight integrity fuse: save last-good params checkpoint
    last_good_params = _copy_params(model)

    progress = tqdm(dataloader, desc=f"Epoch {epoch}")
    for batch_idx, batch in enumerate(progress):
        # Move to device
        images = batch["image"].to(device)
        text_tokens = batch["text_tokens"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        structured = batch["structured"].to(device)
        labels = batch["labels"].to(device)
        label_mask = batch["label_mask"].to(device)
        study_ids = batch["study_id"].tolist()  # For NaN debugging

        # Replace NaN/Inf in structured features with 0
        structured = torch.nan_to_num(structured, nan=0.0, posinf=0.0, neginf=0.0)

        # Clear gradients (set_to_none=True is faster and cleaner)
        optimizer.zero_grad(set_to_none=True)

        # Forward pass
        if scaler is not None:
            with autocast():
                logits, clip_emb, supcon_emb, _, _, text_emb = model(
                    images, text_tokens, structured, attention_mask,
                    return_embeddings=True,
                )
                losses = loss_fn(
                    logits, clip_emb, supcon_emb,
                    labels, label_mask, text_emb,
                )

            # ===== HARD GATE: Never backward/step on invalid batch =====
            total_loss = losses["total"]
            is_finite = torch.isfinite(total_loss)
            is_valid = losses.get("valid", True)

            if (not is_finite) or (not is_valid):
                nan_batches += 1
                nan_study_ids.extend(study_ids)
                logger.warning(
                    f"NaN/Inf loss at batch {batch_idx}, skipping | "
                    f"study_ids: {study_ids[:4]}... | finite={is_finite}, valid={is_valid}"
                )
                optimizer.zero_grad(set_to_none=True)
                continue

            # Backward pass (only for valid batches)
            scaler.scale(total_loss).backward()

            # Gradient clipping before optimizer step
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            # Optimizer step
            scaler.step(optimizer)
            scaler.update()

            # CRITICAL: Clamp CLIP logit_scale to prevent parameter overflow
            # The raw logit_scale parameter can grow unboundedly during AdamW updates,
            # causing exp(logit_scale) to overflow to Inf and corrupt model weights
            loss_fn.clamp_logit_scale()

            # ===== WEIGHT INTEGRITY FUSE: Check params after step =====
            if not _params_finite(model):
                weight_reverts += 1
                consecutive_corruptions += 1
                logger.error(
                    f"[WEIGHT CORRUPTION] Non-finite params after batch {batch_idx}! "
                    f"Reverting to last-good checkpoint. study_ids: {study_ids[:4]}..."
                )

                # Restore model weights
                _restore_params(model, last_good_params)

                # FIX #1: Reset GradScaler state to break cascade failures
                # CRITICAL: Do NOT create new GradScaler() - that loses state synchronization!
                # Instead, reset using load_state_dict() to restore initial state
                # See: tests/baseline_metrics.md for cascade failure analysis
                initial_state = {
                    'scale': 65536.0,  # Default initial scale (2^16)
                    'growth_factor': 2.0,
                    'backoff_factor': 0.5,
                    'growth_interval': 2000,
                    '_growth_tracker': 0,
                }
                scaler.load_state_dict(initial_state)

                # Reset optimizer state completely - Adam will reinitialize on next step
                optimizer.zero_grad(set_to_none=True)
                for group in optimizer.param_groups:
                    for p in group['params']:
                        if p in optimizer.state:
                            del optimizer.state[p]  # Clear entire state, Adam will reinit

                # Circuit breaker: stop epoch if too many consecutive corruptions
                if consecutive_corruptions >= 10:
                    logger.error(f"[CIRCUIT BREAKER] {consecutive_corruptions} consecutive corruptions! Breaking epoch.")
                    break

                continue
            else:
                consecutive_corruptions = 0  # Reset counter on successful batch

            # Update last-good checkpoint
            last_good_params = _copy_params(model)

        else:
            # Non-mixed-precision path
            logits, clip_emb, supcon_emb, _, _, text_emb = model(
                images, text_tokens, structured, attention_mask,
                return_embeddings=True,
            )
            losses = loss_fn(
                logits, clip_emb, supcon_emb,
                labels, label_mask, text_emb,
            )

            # ===== HARD GATE: Never backward/step on invalid batch =====
            total_loss = losses["total"]
            is_finite = torch.isfinite(total_loss)
            is_valid = losses.get("valid", True)

            if (not is_finite) or (not is_valid):
                nan_batches += 1
                nan_study_ids.extend(study_ids)
                logger.warning(
                    f"NaN/Inf loss at batch {batch_idx}, skipping | "
                    f"study_ids: {study_ids[:4]}... | finite={is_finite}, valid={is_valid}"
                )
                optimizer.zero_grad(set_to_none=True)
                continue

            # Backward pass (only for valid batches)
            total_loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            # Optimizer step
            optimizer.step()

            # ===== WEIGHT INTEGRITY FUSE: Check params after step =====
            if not _params_finite(model):
                weight_reverts += 1
                consecutive_corruptions += 1
                logger.error(
                    f"[WEIGHT CORRUPTION] Non-finite params after batch {batch_idx}! "
                    f"Reverting to last-good checkpoint. study_ids: {study_ids[:4]}..."
                )

                # Restore model weights
                _restore_params(model, last_good_params)

                # Reset optimizer state completely - Adam will reinitialize on next step
                optimizer.zero_grad(set_to_none=True)
                for group in optimizer.param_groups:
                    for p in group['params']:
                        if p in optimizer.state:
                            del optimizer.state[p]  # Clear entire state, Adam will reinit

                # Circuit breaker: stop epoch if too many consecutive corruptions
                if consecutive_corruptions >= 10:
                    logger.error(f"[CIRCUIT BREAKER] {consecutive_corruptions} consecutive corruptions! Breaking epoch.")
                    break

                continue
            else:
                consecutive_corruptions = 0  # Reset counter on successful batch

            # Update last-good checkpoint
            last_good_params = _copy_params(model)

        scheduler.step()

        # Track metrics (skip 'valid' flag which is not a tensor)
        for k, v in losses.items():
            if k != "valid":
                total_losses[k] += v.item()
        num_batches += 1

        # Update progress bar
        if batch_idx % config.log_interval == 0:
            current_lr = scheduler.get_last_lr()[0]
            progress.set_postfix({
                "loss": f"{total_loss.item():.4f}",
                "cls": f"{losses['cls'].item():.4f}",
                "lr": f"{current_lr:.2e}",
                "nan": nan_batches,
            })

    epoch_time = time.time() - start_time
    avg_losses = {k: v / max(num_batches, 1) for k, v in total_losses.items()}
    avg_losses["time"] = epoch_time
    avg_losses["lr"] = scheduler.get_last_lr()[0]
    avg_losses["nan_batches"] = nan_batches
    avg_losses["weight_reverts"] = weight_reverts

    if nan_batches > 0:
        logger.warning(f"Epoch {epoch}: {nan_batches} batches skipped due to NaN/Inf loss")
        # Log unique study_ids that caused NaN (for debugging which samples are bad)
        unique_nan_studies = list(set(nan_study_ids))[:20]  # First 20 unique
        logger.warning(f"Epoch {epoch}: NaN study_ids sample: {unique_nan_studies}")

    if weight_reverts > 0:
        logger.error(f"Epoch {epoch}: {weight_reverts} weight corruptions reverted!")

    return avg_losses


@torch.no_grad()
def validate(
    model: MultimodalClassifier,
    dataloader: DataLoader,
    loss_fn: MultiTaskLoss,
    device: str,
) -> Dict[str, float]:
    """Validate model and compute metrics."""
    model.eval()

    # Clear GPU cache before validation to prevent OOM
    if device == "cuda":
        torch.cuda.empty_cache()

    total_losses = {"total": 0.0, "cls": 0.0, "clip": 0.0, "supcon": 0.0}
    all_logits = []
    all_labels = []
    all_masks = []
    num_batches = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validating"):
            images = batch["image"].to(device)
            text_tokens = batch["text_tokens"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            structured = batch["structured"].to(device)
            labels = batch["labels"].to(device)
            label_mask = batch["label_mask"].to(device)

            # Forward pass
            logits, clip_emb, supcon_emb, _, _, text_emb = model(
                images, text_tokens, structured, attention_mask,
                return_embeddings=True,
            )
            losses = loss_fn(
                logits, clip_emb, supcon_emb,
                labels, label_mask, text_emb,
            )

            # Track losses (skip 'valid' boolean flag)
            for k, v in losses.items():
                if k != 'valid':
                    total_losses[k] += v.item()
            num_batches += 1

            # Collect predictions (move to CPU immediately)
            all_logits.append(logits.cpu())
            all_labels.append(labels.cpu())
            all_masks.append(label_mask.cpu())

            # Explicitly delete GPU tensors to free memory
            del images, text_tokens, attention_mask, structured, labels, label_mask
            del logits, clip_emb, supcon_emb, text_emb, losses

            # Clear cache periodically (every 50 batches)
            if device == "cuda" and num_batches % 50 == 0:
                torch.cuda.empty_cache()

    # Average losses
    avg_losses = {k: v / num_batches for k, v in total_losses.items()}

    # Compute AUROC/AUPRC per label
    all_logits = torch.cat(all_logits, dim=0).numpy()
    all_labels = torch.cat(all_labels, dim=0).numpy()
    all_masks = torch.cat(all_masks, dim=0).numpy()

    # Handle NaN/Inf in logits
    if np.any(~np.isfinite(all_logits)):
        logger.warning("NaN/Inf detected in validation logits - model may be corrupted")
        all_logits = np.nan_to_num(all_logits, nan=0.0, posinf=10.0, neginf=-10.0)

    probs = 1 / (1 + np.exp(-np.clip(all_logits, -20, 20)))  # Clip to prevent overflow

    label_names = MultimodalClassificationDataset.PATHOLOGY_LABELS
    aurocs = {}
    auprcs = {}

    for i, label_name in enumerate(label_names):
        # Only compute for samples with valid labels
        valid_mask = all_masks[:, i] > 0
        if valid_mask.sum() > 0:
            y_true = all_labels[valid_mask, i]
            y_pred = probs[valid_mask, i]

            # Skip if predictions contain NaN
            if np.any(~np.isfinite(y_pred)):
                logger.warning(f"Skipping {label_name} due to NaN predictions")
                continue

            # Need both classes present
            if len(np.unique(y_true)) > 1:
                aurocs[label_name] = roc_auc_score(y_true, y_pred)
                auprcs[label_name] = average_precision_score(y_true, y_pred)

    # Mean metrics
    if aurocs:
        avg_losses["auroc_mean"] = np.mean(list(aurocs.values()))
        avg_losses["auprc_mean"] = np.mean(list(auprcs.values()))

    avg_losses["aurocs"] = aurocs
    avg_losses["auprcs"] = auprcs

    return avg_losses


def save_checkpoint(
    model: MultimodalClassifier,
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
            "classifier": config.classifier.__dict__,
        },
        "metrics": {k: v for k, v in metrics.items() if not isinstance(v, dict)},
    }

    if scaler is not None:
        checkpoint["scaler_state_dict"] = scaler.state_dict()

    # Save regular checkpoint
    checkpoint_path = config.checkpoint_dir / f"classifier_epoch_{epoch:04d}.pt"
    torch.save(checkpoint, checkpoint_path)

    # Save best model
    if is_best:
        best_path = config.output_dir / "classifier_best.pt"
        torch.save(checkpoint, best_path)
        logger.info(f"Saved best model to {best_path}")

    # Save latest
    latest_path = config.checkpoint_dir / "classifier_latest.pt"
    torch.save(checkpoint, latest_path)

    return checkpoint_path


def load_checkpoint(
    checkpoint_path: Path,
    model: MultimodalClassifier,
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


def train(config: TrainingConfig) -> MultimodalClassifier:
    """Main training loop."""
    set_seed(config.seed)
    device = config.device

    # Create model
    model = create_model(config)

    # Create dataloaders
    train_loader, val_loader = create_dataloaders(config)

    # Create optimizer with layer-wise LR decay
    optimizer = create_optimizer_with_llrd(model, config.classifier)

    # Create scheduler
    steps_per_epoch = len(train_loader)
    scheduler = get_lr_scheduler(optimizer, config.classifier, steps_per_epoch)

    # Create loss function
    loss_fn = MultiTaskLoss(
        cls_weight=config.classifier.cls_weight,
        clip_weight=config.classifier.clip_weight,
        supcon_weight=config.classifier.supcon_weight,
    )

    # Mixed precision scaler
    scaler = GradScaler() if config.classifier.mixed_precision and device == "cuda" else None

    # Training history
    history = {
        "train_loss": [], "val_loss": [],
        "auroc_mean": [], "auprc_mean": [], "lr": []
    }
    best_auroc = 0.0
    start_epoch = 0

    # Log training info
    logger.info("=" * 60)
    logger.info("Multimodal Classifier Training")
    logger.info("=" * 60)
    logger.info(f"Epochs: {config.classifier.epochs}")
    logger.info(f"Batch size: {config.classifier.batch_size}")
    logger.info(f"Learning rate: {config.classifier.base_lr}")
    logger.info(f"Loss weights: cls={config.classifier.cls_weight}, "
                f"clip={config.classifier.clip_weight}, supcon={config.classifier.supcon_weight}")
    logger.info(f"Device: {device}")
    logger.info("=" * 60)

    # Training loop
    for epoch in range(start_epoch, config.classifier.epochs):
        # Train
        train_metrics = train_epoch(
            model, train_loader, optimizer, scheduler, scaler,
            loss_fn, config.classifier, device, epoch
        )
        history["train_loss"].append(train_metrics["total"])
        history["lr"].append(train_metrics["lr"])

        # Log training metrics
        logger.info(
            f"Epoch {epoch:4d}/{config.classifier.epochs} | "
            f"Train Loss: {train_metrics['total']:.4f} "
            f"(cls={train_metrics['cls']:.4f}, clip={train_metrics['clip']:.4f}, "
            f"supcon={train_metrics['supcon']:.4f}) | "
            f"LR: {train_metrics['lr']:.2e} | "
            f"Time: {train_metrics['time']:.1f}s"
        )

        # Validate
        is_best = False
        if val_loader is not None and (epoch + 1) % config.classifier.eval_interval == 0:
            val_metrics = validate(model, val_loader, loss_fn, device)
            history["val_loss"].append(val_metrics["total"])

            if "auroc_mean" in val_metrics:
                history["auroc_mean"].append(val_metrics["auroc_mean"])
                history["auprc_mean"].append(val_metrics["auprc_mean"])

                is_best = val_metrics["auroc_mean"] > best_auroc
                if is_best:
                    best_auroc = val_metrics["auroc_mean"]

                logger.info(
                    f"  Val Loss: {val_metrics['total']:.4f} | "
                    f"AUROC: {val_metrics['auroc_mean']:.4f} | "
                    f"AUPRC: {val_metrics['auprc_mean']:.4f}"
                    f"{' (best)' if is_best else ''}"
                )

                # Log per-label metrics
                if val_metrics.get("aurocs"):
                    for label, auroc in val_metrics["aurocs"].items():
                        logger.info(f"    {label}: AUROC={auroc:.4f}")

        # Save checkpoint
        if (epoch + 1) % config.classifier.save_interval == 0 or is_best:
            save_checkpoint(
                model, optimizer, scheduler, scaler,
                epoch, config, train_metrics,
                is_best=is_best,
            )

    # Save final model
    final_path = config.output_dir / "classifier_final.pt"
    torch.save({
        "model_state_dict": model.state_dict(),
        "config": config.classifier.__dict__,
        "history": history,
    }, final_path)
    logger.info(f"Saved final model to {final_path}")

    # Save training history
    history_path = config.output_dir / "classifier_history.json"
    with open(history_path, "w") as f:
        json.dump({
            k: v for k, v in history.items()
            if not isinstance(v, dict) or all(not isinstance(vv, dict) for vv in v)
        }, f, indent=2)
    logger.info(f"Saved training history to {history_path}")

    return model


def main():
    parser = argparse.ArgumentParser(
        description="Train multimodal classifier for CheXpert pathology classification",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Configuration preset
    parser.add_argument(
        "--config", type=str, default="base",
        choices=["debug", "fast", "base"],
        help="Configuration preset (default: base)"
    )

    # Data paths
    parser.add_argument(
        "--train-dir", type=Path, required=True,
        help="Path to training preprocessed directory"
    )
    parser.add_argument(
        "--val-dir", type=Path, default=None,
        help="Path to validation preprocessed directory"
    )
    parser.add_argument(
        "--chexpert-csv", type=Path, required=True,
        help="Path to mimic-cxr-2.0.0-chexpert.csv.gz"
    )

    # Model paths
    parser.add_argument(
        "--mae-checkpoint", type=Path, default=None,
        help="Path to pretrained MAE checkpoint"
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
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--img-size", type=int, default=None)
    parser.add_argument("--cls-weight", type=float, default=None)
    parser.add_argument("--clip-weight", type=float, default=None)
    parser.add_argument("--supcon-weight", type=float, default=None)
    parser.add_argument("--freeze-mae-epochs", type=int, default=None,
                        help="Number of epochs to keep MAE encoder frozen (default: 5, use 100 to keep frozen)")

    # Resumption
    parser.add_argument(
        "--resume", type=Path, default=None,
        help="Path to checkpoint to resume from"
    )

    # Hardware
    parser.add_argument(
        "--device", type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument("--num-workers", type=int, default=None)

    args = parser.parse_args()

    # Load configuration preset
    if args.config == "debug":
        config = get_debug_config()
    elif args.config == "fast":
        config = get_fast_config()
    else:
        config = get_base_config()

    # Apply paths
    config.train_dir = args.train_dir
    config.val_dir = args.val_dir
    config.chexpert_csv = args.chexpert_csv
    config.mae_checkpoint = args.mae_checkpoint
    config.output_dir = args.output_dir
    config.checkpoint_dir = args.checkpoint_dir
    config.device = args.device

    # Apply overrides
    if args.epochs:
        config.classifier.epochs = args.epochs
    if args.batch_size:
        config.classifier.batch_size = args.batch_size
    if args.lr:
        config.classifier.base_lr = args.lr
    if args.img_size:
        config.classifier.img_size = args.img_size
    if args.cls_weight:
        config.classifier.cls_weight = args.cls_weight
    if args.clip_weight:
        config.classifier.clip_weight = args.clip_weight
    if args.supcon_weight:
        config.classifier.supcon_weight = args.supcon_weight
    if args.freeze_mae_epochs is not None:
        config.classifier.freeze_mae_epochs = args.freeze_mae_epochs
    if args.num_workers is not None:
        config.classifier.num_workers = args.num_workers

    # Create directories
    config.output_dir.mkdir(parents=True, exist_ok=True)
    config.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Save configuration
    config_path = config.output_dir / "classifier_config.json"
    with open(config_path, "w") as f:
        json.dump({
            "classifier": config.classifier.__dict__,
            "device": config.device,
            "seed": config.seed,
            "train_dir": str(config.train_dir),
            "val_dir": str(config.val_dir) if config.val_dir else None,
            "chexpert_csv": str(config.chexpert_csv),
            "mae_checkpoint": str(config.mae_checkpoint) if config.mae_checkpoint else None,
        }, f, indent=2)
    logger.info(f"Saved configuration to {config_path}")

    # Train
    model = train(config)
    logger.info("Training complete!")


if __name__ == "__main__":
    main()
