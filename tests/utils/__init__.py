"""
Shared helpers for tests that run the real classifier and training loop.

The classifier is built at img_size=32 (4 patches) so the real ViT-B + BERT
stack runs in seconds on CPU.
"""

import torch

import train_classifier as tc
from src.models.config import ClassifierConfig
from src.models.losses import MultiTaskLoss
from src.models.multimodal import MultimodalClassifier
from src.training import build_warmup_cosine_scheduler

IMG = 32
N_STRUCT = 5
N_LABELS = 12
SEQ = 16


def make_classifier(freeze_mae_epochs: int = 1) -> MultimodalClassifier:
    torch.manual_seed(0)
    return MultimodalClassifier(
        num_labels=N_LABELS,
        struct_input_dim=N_STRUCT,
        freeze_mae_epochs=freeze_mae_epochs,
        img_size=IMG,
    )


def make_batch(batch_size: int = 4) -> dict:
    ids = torch.randint(1000, 5000, (batch_size, SEQ))
    ids[:, 0] = 101  # [CLS]
    return {
        "image": torch.randn(batch_size, 3, IMG, IMG),
        "text_tokens": ids,
        "attention_mask": torch.ones(batch_size, SEQ),
        "structured": torch.randn(batch_size, N_STRUCT),
        "labels": torch.randint(0, 2, (batch_size, N_LABELS)).float(),
        "label_mask": torch.ones(batch_size, N_LABELS),
        "study_id": torch.arange(batch_size),
    }


def make_training(model: MultimodalClassifier, steps_per_epoch: int, **config_overrides):
    config = ClassifierConfig(
        epochs=3, warmup_epochs=1, log_interval=10_000, **config_overrides
    )
    loss_fn = MultiTaskLoss()
    optimizer = tc.create_optimizer_with_llrd(model, config, loss_fn, steps_per_epoch)
    scheduler = build_warmup_cosine_scheduler(
        optimizer,
        warmup_steps=config.warmup_epochs * steps_per_epoch,
        total_steps=config.epochs * steps_per_epoch,
        min_lr_ratio=config.min_lr / config.base_lr,
    )
    return config, loss_fn, optimizer, scheduler
