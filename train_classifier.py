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

    # Resume from checkpoint (configuration and data paths come from the checkpoint)
    python train_classifier.py --resume output/checkpoints/classifier_latest.pt
"""

import argparse
import json
import logging
import sys
import time
from dataclasses import asdict, fields
from pathlib import Path
from typing import Optional, Dict

import numpy as np
import torch
import torch.optim as optim
from torch.cuda.amp import GradScaler
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
from src.models.losses import MultiTaskLoss
from src.models.config import (
    IMAGE_MODES,
    ClassifierConfig,
    ClassifierTrainingConfig,
    get_classifier_config,
)
from src.training import (
    ConsecutiveSkipGuard,
    assert_params_finite,
    backward_and_step,
    build_warmup_cosine_scheduler,
    check_trainable_params_in_optimizer,
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


# CLI flag -> ClassifierConfig field
CLI_OVERRIDES = {
    "epochs": "epochs",
    "batch_size": "batch_size",
    "lr": "base_lr",
    "img_size": "img_size",
    "image_mode": "image_mode",
    "text_model": "text_model_name",
    "cls_weight": "cls_weight",
    "clip_weight": "clip_weight",
    "supcon_weight": "supcon_weight",
    "freeze_mae_epochs": "freeze_mae_epochs",
    "num_workers": "num_workers",
}

# Fields that define the run. Resuming continues the saved LR schedule, so a
# different value here would silently train a different run.
RESUME_LOCKED_FIELDS = {
    "epochs", "batch_size", "base_lr", "img_size", "image_mode", "text_model_name",
    "cls_weight", "clip_weight", "supcon_weight", "freeze_mae_epochs",
}

PATH_FIELDS = ("train_dir", "val_dir", "chexpert_csv", "mae_checkpoint", "output_dir", "checkpoint_dir")


def config_to_dict(config: ClassifierTrainingConfig) -> dict:
    """JSON-serializable configuration (stored in checkpoints and classifier_config.json)."""
    return {
        "classifier": asdict(config.classifier),
        **{key: str(getattr(config, key)) if getattr(config, key) is not None else None
           for key in PATH_FIELDS},
        "device": config.device,
        "seed": config.seed,
    }


def config_from_dict(saved: dict) -> ClassifierTrainingConfig:
    """Rebuild the training configuration saved by config_to_dict."""
    known = {f.name for f in fields(ClassifierConfig)}
    classifier = ClassifierConfig(**{k: v for k, v in saved["classifier"].items() if k in known})
    config = ClassifierTrainingConfig(classifier=classifier)
    for key in PATH_FIELDS:
        if saved.get(key) is not None:
            setattr(config, key, Path(saved[key]))
    config.seed = saved.get("seed", config.seed)
    return config


def apply_overrides(config: ClassifierTrainingConfig, args: argparse.Namespace, resuming: bool) -> None:
    """Apply CLI overrides. Explicit zeros count (e.g. --clip-weight 0)."""
    for arg, field_name in CLI_OVERRIDES.items():
        value = getattr(args, arg)
        if value is None:
            continue
        current = getattr(config.classifier, field_name)
        if resuming and field_name in RESUME_LOCKED_FIELDS and value != current:
            raise ValueError(
                f"--{arg.replace('_', '-')} {value} differs from the checkpoint's value "
                f"({current}). Resuming continues the saved run; start a new run to change it."
            )
        setattr(config.classifier, field_name, value)


def create_model(config: ClassifierTrainingConfig, load_mae_weights: bool = True) -> MultimodalClassifier:
    """Create multimodal classifier."""
    logger.info("Creating MultimodalClassifier...")
    c = config.classifier

    model = MultimodalClassifier(
        mae_checkpoint=config.mae_checkpoint if load_mae_weights else None,
        num_labels=c.num_labels,
        embed_dim=c.embed_dim,
        struct_input_dim=c.struct_input_dim,
        struct_hidden_dim=c.struct_hidden_dim,
        contrastive_dim=c.contrastive_dim,
        freeze_mae_epochs=c.freeze_mae_epochs,
        text_model_name=c.text_model_name,
        img_size=c.img_size,
        struct_log_transform=MultimodalClassificationDataset.structured_log_transform_flags(),
    )

    if model.mae_image_mode is not None and model.mae_image_mode != c.image_mode:
        logger.warning(
            f"The MAE checkpoint was pretrained with image_mode={model.mae_image_mode!r} "
            f"but this run uses {c.image_mode!r}, so the encoder sees anatomy at a "
            f"different scale than in pretraining. Pass --image-mode "
            f"{model.mae_image_mode} to match it, or re-pretrain the MAE."
        )

    model = model.to(config.device)

    # Count parameters
    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model parameters: {n_params:,} total, {n_trainable:,} trainable at epoch 0")

    return model


def create_dataloaders(
    config: ClassifierTrainingConfig,
) -> tuple[DataLoader, Optional[DataLoader]]:
    """Create training and validation dataloaders."""
    if config.train_dir is None or not config.train_dir.exists():
        raise ValueError(f"Training directory not found: {config.train_dir}")
    if config.chexpert_csv is None or not config.chexpert_csv.exists():
        raise ValueError(f"CheXpert CSV not found: {config.chexpert_csv}")

    c = config.classifier
    target_size = (c.img_size, c.img_size)
    logger.info(f"Loading training data from: {config.train_dir}")

    train_dataset = MultimodalClassificationDataset(
        preprocessed_dir=config.train_dir,
        chexpert_csv=config.chexpert_csv,
        target_size=target_size,
        augment=True,
        image_mode=c.image_mode,
    )

    val_dataset = None
    if config.val_dir and config.val_dir.exists():
        logger.info(f"Loading validation data from: {config.val_dir}")
        val_dataset = MultimodalClassificationDataset(
            preprocessed_dir=config.val_dir,
            chexpert_csv=config.chexpert_csv,
            target_size=target_size,
            augment=False,
            image_mode=c.image_mode,
        )

    # Use 'spawn' to avoid HDF5 fork issues when num_workers > 0
    mp_context = 'spawn' if c.num_workers > 0 else None

    train_loader = DataLoader(
        train_dataset,
        batch_size=c.batch_size,
        shuffle=True,
        num_workers=c.num_workers,
        pin_memory=c.pin_memory,
        drop_last=True,
        collate_fn=collate_multimodal,
        multiprocessing_context=mp_context,
    )

    val_loader = None
    if val_dataset:
        val_loader = DataLoader(
            val_dataset,
            batch_size=c.batch_size,
            shuffle=False,
            num_workers=c.num_workers,
            pin_memory=c.pin_memory,
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


def check_text_tokens(
    datasets: list[MultimodalClassificationDataset],
    model: MultimodalClassifier,
) -> None:
    """Fail fast if the preprocessed token ids come from a different tokenizer."""
    from transformers import AutoTokenizer

    model_name = model.text_encoder.model_name
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    for dataset in datasets:
        dataset.validate_text_tokens(
            tokenizer.cls_token_id, model.text_encoder.vocab_size, model_name
        )


def create_optimizer_with_llrd(
    model: MultimodalClassifier,
    config: ClassifierConfig,
    loss_fn: MultiTaskLoss,
    steps_per_epoch: int,
) -> optim.Optimizer:
    """
    Create AdamW with per-block layer-wise learning rate decay.

    Deeper encoder blocks get lower learning rates to preserve pretrained
    features (see MultimodalClassifier.get_layer_groups). MAE encoder groups
    hold LR 0 while frozen and warm up over ``unfreeze_warmup_epochs`` after
    unfreezing. Learnable loss parameters (the CLIP logit scale) get their own
    group without weight decay.
    """
    unfreeze_step = config.freeze_mae_epochs * steps_per_epoch
    ramp_steps = config.unfreeze_warmup_epochs * steps_per_epoch

    param_groups = []
    for group in model.get_layer_groups():
        param_group = {
            "name": group["name"],
            "params": group["params"],
            "lr": config.base_lr * config.lr_decay ** group["decay_exponent"],
        }
        if group["image_encoder"]:
            param_group["start_step"] = unfreeze_step
            param_group["ramp_steps"] = ramp_steps
        param_groups.append(param_group)

    loss_params = [p for p in loss_fn.parameters() if p.requires_grad]
    if loss_params:
        # Weight decay would pull the CLIP logit scale toward 0 (temperature 1)
        param_groups.append({
            "name": "loss", "params": loss_params, "lr": config.base_lr, "weight_decay": 0.0,
        })

    logger.info("Layer-wise learning rates:")
    for pg in param_groups:
        delay = f" (after step {pg['start_step']})" if pg.get("start_step") else ""
        logger.info(f"  {pg['name']}: lr={pg['lr']:.2e}{delay}")

    return optim.AdamW(
        param_groups,
        betas=(0.9, 0.999),
        weight_decay=config.weight_decay,
    )


def _group_lr(optimizer: optim.Optimizer, name: str) -> float:
    return next(g["lr"] for g in optimizer.param_groups if g.get("name") == name)


def train_epoch(
    model: MultimodalClassifier,
    dataloader: DataLoader,
    optimizer: optim.Optimizer,
    scheduler: optim.lr_scheduler.LRScheduler,
    scaler: GradScaler,
    loss_fn: MultiTaskLoss,
    config: ClassifierConfig,
    device: str,
    epoch: int,
) -> Dict[str, float]:
    """
    Train for one epoch.

    A batch with a non-finite loss is skipped before backward; a batch with a
    non-finite gradient is skipped at the optimizer step (backward_and_step).
    The scheduler advances every batch so the schedule tracks data seen.
    """
    model.train()
    loss_fn.train()
    model.set_epoch(epoch)
    check_trainable_params_in_optimizer([model, loss_fn], optimizer)

    device_type = torch.device(device).type
    total_losses = {"total": 0.0, "cls": 0.0, "clip": 0.0, "supcon": 0.0}
    num_steps = 0
    skipped_loss = 0
    skipped_grad = 0
    skipped_study_ids = []  # Track which studies produce non-finite losses
    guard = ConsecutiveSkipGuard(config.max_consecutive_skips)
    start_time = time.time()

    progress = tqdm(dataloader, desc=f"Epoch {epoch}")
    for batch_idx, batch in enumerate(progress):
        # Move to device
        images = batch["image"].to(device)
        text_tokens = batch["text_tokens"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        structured = batch["structured"].to(device)
        labels = batch["labels"].to(device)
        label_mask = batch["label_mask"].to(device)
        study_ids = batch["study_id"].tolist()

        with torch.autocast(device_type=device_type, enabled=scaler.is_enabled()):
            logits, clip_emb, supcon_emb, _, _, text_emb = model(
                images, text_tokens, structured, attention_mask,
                return_embeddings=True,
            )
            losses = loss_fn(logits, clip_emb, supcon_emb, labels, label_mask, text_emb)

        total_loss = losses["total"]
        if not losses["valid"] or not torch.isfinite(total_loss):
            stepped = False
            skipped_loss += 1
            skipped_study_ids.extend(study_ids)
            logger.warning(
                f"Non-finite loss at batch {batch_idx}, skipping | study_ids: {study_ids[:4]}..."
            )
        else:
            stepped, _ = backward_and_step(total_loss, optimizer, scaler, config.grad_clip)
            if stepped:
                loss_fn.clamp_logit_scale()
                for k in total_losses:
                    total_losses[k] += losses[k].item()
                num_steps += 1
            else:
                skipped_grad += 1

        scheduler.step()
        guard.update(stepped, context=f"epoch {epoch}, batch {batch_idx}")

        if batch_idx % config.log_interval == 0:
            progress.set_postfix({
                "loss": f"{total_loss.item():.4f}",
                "cls": f"{losses['cls'].item():.4f}",
                "lr": f"{_group_lr(optimizer, 'head'):.2e}",
                "skipped": skipped_loss + skipped_grad,
            })

    # Skipped steps never apply an update, so this should be unreachable;
    # fail loudly rather than train on (or save) corrupted weights.
    assert_params_finite([model, loss_fn])

    epoch_time = time.time() - start_time
    avg_losses = {k: v / max(num_steps, 1) for k, v in total_losses.items()}
    avg_losses["time"] = epoch_time
    avg_losses["lr"] = _group_lr(optimizer, "head")
    avg_losses["steps"] = num_steps
    avg_losses["skipped_loss"] = skipped_loss
    avg_losses["skipped_grad"] = skipped_grad

    if skipped_loss > 0:
        logger.warning(f"Epoch {epoch}: {skipped_loss} batches skipped due to non-finite loss")
        unique_studies = list(set(skipped_study_ids))[:20]
        logger.warning(f"Epoch {epoch}: non-finite-loss study_ids sample: {unique_studies}")
    if skipped_grad > 0:
        logger.warning(
            f"Epoch {epoch}: {skipped_grad} steps skipped due to non-finite gradients"
            f"{' (includes GradScaler loss-scale calibration)' if scaler.is_enabled() else ''}"
        )

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
    loss_fn.eval()

    # Release cached training allocations before switching to validation batches
    if is_cuda_device(device):
        torch.cuda.empty_cache()

    total_losses = {"total": 0.0, "cls": 0.0, "clip": 0.0, "supcon": 0.0}
    all_logits = []
    all_labels = []
    all_masks = []
    num_batches = 0

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

        # Explicitly delete GPU tensors to free memory. empty_cache() is
        # intentionally NOT called in-loop: it only releases the allocator
        # cache back to the OS (not live tensors) and adds overhead.
        del images, text_tokens, attention_mask, structured, labels, label_mask
        del logits, clip_emb, supcon_emb, text_emb, losses

    # Average losses
    avg_losses = {k: v / max(num_batches, 1) for k, v in total_losses.items()}

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
    scheduler: optim.lr_scheduler.LRScheduler,
    scaler: GradScaler,
    loss_fn: MultiTaskLoss,
    epoch: int,
    config: ClassifierTrainingConfig,
    metrics: dict,
    best_auroc: float,
    history: dict,
    is_best: bool = False,
) -> Path:
    """Save training checkpoint (``epoch`` is the epoch just completed)."""
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "loss_fn_state_dict": loss_fn.state_dict(),
        "config": config_to_dict(config),
        "metrics": {k: v for k, v in metrics.items() if not isinstance(v, dict)},
        "best_auroc": best_auroc,
        "history": history,
    }
    if scaler.is_enabled():
        checkpoint["scaler_state_dict"] = scaler.state_dict()

    return save_checkpoint_files(
        checkpoint,
        config.checkpoint_dir,
        prefix="classifier",
        epoch=epoch,
        best_path=config.output_dir / "classifier_best.pt" if is_best else None,
    )


def save_config_json(config: ClassifierTrainingConfig) -> None:
    config_path = config.output_dir / "classifier_config.json"
    with open(config_path, "w") as f:
        json.dump(config_to_dict(config), f, indent=2)
    logger.info(f"Saved configuration to {config_path}")


def train(
    config: ClassifierTrainingConfig,
    resume_checkpoint: Optional[dict] = None,
) -> MultimodalClassifier:
    """Main training loop.

    Args:
        config: Training configuration.
        resume_checkpoint: Loaded checkpoint to resume from (continues after
            its last completed epoch, with its best AUROC and history).
    """
    set_seed(config.seed)
    device = config.device
    c = config.classifier

    train_loader, val_loader = create_dataloaders(config)
    c.struct_input_dim = len(MultimodalClassificationDataset.STRUCTURED_FEATURES)

    # Create model (on resume, all weights come from the checkpoint)
    model = create_model(config, load_mae_weights=resume_checkpoint is None)
    check_text_tokens(
        [train_loader.dataset] + ([val_loader.dataset] if val_loader is not None else []),
        model,
    )

    # Create loss function (it has a learnable CLIP logit scale)
    loss_fn = MultiTaskLoss(
        cls_weight=c.cls_weight,
        clip_weight=c.clip_weight,
        supcon_weight=c.supcon_weight,
    ).to(device)

    # Create optimizer with layer-wise LR decay, and the LR schedule
    steps_per_epoch = len(train_loader)
    optimizer = create_optimizer_with_llrd(model, c, loss_fn, steps_per_epoch)
    scheduler = build_warmup_cosine_scheduler(
        optimizer,
        warmup_steps=c.warmup_epochs * steps_per_epoch,
        total_steps=c.epochs * steps_per_epoch,
        min_lr_ratio=c.min_lr / c.base_lr,
    )

    # Mixed precision scaler (disabled: plain full-precision steps)
    scaler = GradScaler(enabled=c.mixed_precision and is_cuda_device(device))

    # Training history
    history = {
        "epoch": [], "train_loss": [], "lr": [],
        "val_epoch": [], "val_loss": [], "auroc_mean": [], "auprc_mean": [],
    }
    best_auroc = 0.0
    start_epoch = 0

    if resume_checkpoint is not None:
        load_checkpoint_states(
            resume_checkpoint, model, optimizer, scheduler, scaler,
            extra_modules={"loss_fn": loss_fn},
        )
        start_epoch = resume_checkpoint["epoch"] + 1
        best_auroc = resume_checkpoint.get("best_auroc", best_auroc)
        history = resume_checkpoint.get("history", history)
        logger.info(
            f"Resuming classifier training at epoch {start_epoch} "
            f"(best AUROC so far: {best_auroc:.4f})"
        )
    else:
        model.struct_encoder.fit_normalization(train_loader.dataset.structured_matrix())

    # Saved after every config change above (struct_input_dim, resume)
    save_config_json(config)

    # Log training info
    logger.info("=" * 60)
    logger.info("Multimodal Classifier Training")
    logger.info("=" * 60)
    logger.info(f"Epochs: {c.epochs}")
    logger.info(f"Batch size: {c.batch_size}")
    logger.info(f"Learning rate: {c.base_lr}")
    logger.info(f"Image: {c.img_size}px, mode={c.image_mode}")
    logger.info(f"Loss weights: cls={c.cls_weight}, "
                f"clip={c.clip_weight}, supcon={c.supcon_weight}")
    logger.info(f"Device: {device} (mixed precision: {scaler.is_enabled()})")
    logger.info("=" * 60)

    # Training loop
    for epoch in range(start_epoch, c.epochs):
        # Train
        train_metrics = train_epoch(
            model, train_loader, optimizer, scheduler, scaler,
            loss_fn, c, device, epoch
        )
        history["epoch"].append(epoch)
        history["train_loss"].append(train_metrics["total"])
        history["lr"].append(train_metrics["lr"])

        # Log training metrics
        logger.info(
            f"Epoch {epoch:4d}/{c.epochs} | "
            f"Train Loss: {train_metrics['total']:.4f} "
            f"(cls={train_metrics['cls']:.4f}, clip={train_metrics['clip']:.4f}, "
            f"supcon={train_metrics['supcon']:.4f}) | "
            f"LR: {train_metrics['lr']:.2e} | "
            f"Time: {train_metrics['time']:.1f}s"
        )

        # Validate
        is_best = False
        if val_loader is not None and (epoch + 1) % c.eval_interval == 0:
            val_metrics = validate(model, val_loader, loss_fn, device)
            history["val_epoch"].append(epoch)
            history["val_loss"].append(val_metrics["total"])

            if "auroc_mean" in val_metrics:
                history["auroc_mean"].append(float(val_metrics["auroc_mean"]))
                history["auprc_mean"].append(float(val_metrics["auprc_mean"]))

                is_best = val_metrics["auroc_mean"] > best_auroc
                if is_best:
                    best_auroc = float(val_metrics["auroc_mean"])

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
        if (epoch + 1) % c.save_interval == 0 or is_best:
            save_checkpoint(
                model, optimizer, scheduler, scaler, loss_fn,
                epoch, config, train_metrics, best_auroc, history,
                is_best=is_best,
            )

    # Save final model
    final_path = config.output_dir / "classifier_final.pt"
    torch.save({
        "model_state_dict": model.state_dict(),
        "loss_fn_state_dict": loss_fn.state_dict(),
        "config": config_to_dict(config),
        "history": history,
    }, final_path)
    logger.info(f"Saved final model to {final_path}")

    # Save training history
    history_path = config.output_dir / "classifier_history.json"
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    logger.info(f"Saved training history to {history_path}")

    return model


def main():
    parser = argparse.ArgumentParser(
        description="Train multimodal classifier for CheXpert pathology classification",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Configuration preset
    parser.add_argument(
        "--config", type=str, default=None,
        choices=["debug", "fast", "base"],
        help="Configuration preset (default: base; ignored with --resume)"
    )

    # Data paths
    parser.add_argument(
        "--train-dir", type=Path, default=None,
        help="Path to training preprocessed directory (required unless resuming)"
    )
    parser.add_argument(
        "--val-dir", type=Path, default=None,
        help="Path to validation preprocessed directory"
    )
    parser.add_argument(
        "--chexpert-csv", type=Path, default=None,
        help="Path to mimic-cxr-2.0.0-chexpert.csv.gz (required unless resuming)"
    )

    # Model paths
    parser.add_argument(
        "--mae-checkpoint", type=Path, default=None,
        help="Path to pretrained MAE checkpoint"
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
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--img-size", type=int, default=None)
    parser.add_argument("--image-mode", type=str, default=None, choices=IMAGE_MODES,
                        help="resize: full radiograph (default); center_crop: legacy crop "
                             "from native resolution")
    parser.add_argument("--text-model", type=str, default=None,
                        help="Text encoder (must match the preprocessing tokenizer; "
                             "default: emilyalsentzer/Bio_ClinicalBERT)")
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

    # Load configuration: from the checkpoint when resuming, else from a preset
    resume_checkpoint = None
    if args.resume is not None:
        resume_checkpoint = torch.load(args.resume, map_location="cpu")
        config = config_from_dict(resume_checkpoint["config"])
        if args.config is not None:
            logger.info("--config is ignored when resuming; using the checkpoint's configuration")
    else:
        config = get_classifier_config(args.config or "base")

    # Apply paths
    for key in PATH_FIELDS:
        value = getattr(args, key)
        if value is not None:
            setattr(config, key, value)
    if config.train_dir is None or config.chexpert_csv is None:
        parser.error("--train-dir and --chexpert-csv are required")
    config.device = args.device

    # Apply overrides
    try:
        apply_overrides(config, args, resuming=resume_checkpoint is not None)
    except ValueError as e:
        parser.error(str(e))

    # Create directories
    config.output_dir.mkdir(parents=True, exist_ok=True)
    config.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Train
    model = train(config, resume_checkpoint=resume_checkpoint)
    logger.info("Training complete!")


if __name__ == "__main__":
    main()
