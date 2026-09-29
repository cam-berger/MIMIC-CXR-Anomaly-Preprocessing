"""
Training-loop helpers shared by the MAE and classifier training scripts.

The optimizer step here replaces the earlier per-step weight-integrity fuse,
GradScaler reset and Adam-state wipe. Those recovered from a symptom: a
parameter outside the optimizer (e.g. text_clip_proj) kept its gradient
forever, and clip_grad_norm_(model.parameters()) spread a single non-finite
value from it into every later update. Clipping over exactly the optimizer's
parameters and skipping non-finite steps removes that failure mode, and
check_trainable_params_in_optimizer makes it impossible to reintroduce
silently.
"""

import logging
import math
import shutil
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler

logger = logging.getLogger(__name__)


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def is_cuda_device(device) -> bool:
    """True for any CUDA device spec: "cuda", "cuda:1", torch.device("cuda", 0)."""
    return torch.device(device).type == "cuda"


# =============================================================================
# Learning-rate schedule
# =============================================================================

def warmup_cosine_factor(
    step: int,
    warmup_steps: int,
    total_steps: int,
    min_ratio: float,
) -> float:
    """
    LR multiplier: linear warmup from 0, then cosine decay to ``min_ratio``.

    Safe when ``total_steps == warmup_steps`` (no division by zero) and for
    steps past ``total_steps`` (holds at ``min_ratio``).
    """
    if step < warmup_steps:
        return step / warmup_steps
    decay_steps = max(1, total_steps - warmup_steps)
    progress = min(1.0, (step - warmup_steps) / decay_steps)
    return min_ratio + (1.0 - min_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))


def build_warmup_cosine_scheduler(
    optimizer: optim.Optimizer,
    warmup_steps: int,
    total_steps: int,
    min_lr_ratio: float,
) -> optim.lr_scheduler.LambdaLR:
    """
    Warmup + cosine schedule, stepped once per batch.

    A param group may carry ``start_step`` and ``ramp_steps``: its LR is held
    at 0 until ``start_step`` and then ramped linearly over ``ramp_steps``.
    This is for parameters that start training mid-run (an encoder unfrozen
    after N epochs): they start with empty Adam state, and Adam's first
    updates move every weight by roughly the full LR regardless of gradient
    size, so they need their own warmup.
    """
    def make_lambda(start_step: int, ramp_steps: int):
        def lr_lambda(step: int) -> float:
            if step < start_step:
                return 0.0
            factor = warmup_cosine_factor(step, warmup_steps, total_steps, min_lr_ratio)
            if start_step > 0 and ramp_steps > 0:
                factor *= min(1.0, (step - start_step + 1) / ramp_steps)
            return factor
        return lr_lambda

    lambdas = [
        make_lambda(group.get("start_step", 0), group.get("ramp_steps", 0))
        for group in optimizer.param_groups
    ]
    return optim.lr_scheduler.LambdaLR(optimizer, lambdas)


# =============================================================================
# Optimizer step
# =============================================================================

def optimizer_params(optimizer: optim.Optimizer) -> list[nn.Parameter]:
    """All parameters the optimizer updates."""
    return [p for group in optimizer.param_groups for p in group["params"]]


def backward_and_step(
    loss: torch.Tensor,
    optimizer: optim.Optimizer,
    scaler: GradScaler,
    max_grad_norm: float,
) -> tuple[bool, float]:
    """
    Backward pass and optimizer step that never applies a non-finite update.

    Gradients are unscaled and clipped over exactly the parameters the
    optimizer updates, so a parameter outside the optimizer can neither
    dominate the clip norm nor poison it. A non-finite gradient norm skips
    the update: with AMP, ``GradScaler.step`` skips it and backs off the loss
    scale; without AMP it is skipped here. Gradients are always cleared.

    Args:
        loss: Scalar loss (unscaled).
        optimizer: Optimizer to step.
        scaler: GradScaler; pass ``GradScaler(enabled=False)`` for full precision.
        max_grad_norm: Gradient clipping threshold.

    Returns:
        (stepped, grad_norm): whether the update was applied, and the
        pre-clipping gradient norm (inf/nan when skipped).
    """
    scaler.scale(loss).backward()

    params = [p for p in optimizer_params(optimizer) if p.grad is not None]
    if not params:
        raise RuntimeError(
            "The loss has no gradient path to any parameter in the optimizer."
        )

    scaler.unscale_(optimizer)
    grad_norm = torch.nn.utils.clip_grad_norm_(params, max_grad_norm)
    stepped = bool(torch.isfinite(grad_norm))

    if scaler.is_enabled():
        scaler.step(optimizer)
        scaler.update()
    elif stepped:
        optimizer.step()

    optimizer.zero_grad(set_to_none=True)
    return stepped, float(grad_norm)


class ConsecutiveSkipGuard:
    """
    Fail loudly when training stops making progress.

    Occasional skipped batches (a bad sample, GradScaler calibrating its loss
    scale) are expected. A long unbroken run of them means something
    systematic is wrong, and silently continuing hides it.
    """

    def __init__(self, limit: int):
        self.limit = limit
        self.count = 0

    def update(self, stepped: bool, context: str = "") -> None:
        self.count = 0 if stepped else self.count + 1
        if self.limit and self.count >= self.limit:
            raise RuntimeError(
                f"{self.count} consecutive batches were skipped (non-finite loss or "
                f"gradients){': ' + context if context else ''}. Training is not "
                f"making progress; check the data and learning rate."
            )


def check_trainable_params_in_optimizer(
    modules: Sequence[nn.Module],
    optimizer: optim.Optimizer,
) -> None:
    """
    Raise if a parameter with ``requires_grad=True`` is missing from the optimizer.

    Such a parameter still receives gradients, but the optimizer neither
    updates it nor clears its ``.grad``, so gradients accumulate on it
    indefinitely. Call after anything that changes ``requires_grad`` (e.g.
    unfreezing an encoder).
    """
    in_optimizer = {id(p) for p in optimizer_params(optimizer)}
    missing = [
        f"{type(module).__name__}.{name}"
        for module in modules
        for name, p in module.named_parameters()
        if p.requires_grad and id(p) not in in_optimizer
    ]
    if missing:
        preview = ", ".join(missing[:10]) + (" ..." if len(missing) > 10 else "")
        raise RuntimeError(
            f"{len(missing)} trainable parameter(s) are not in the optimizer: {preview}. "
            f"Add them to a parameter group or set requires_grad=False."
        )


def assert_params_finite(modules: Sequence[nn.Module]) -> None:
    """Raise if any parameter is NaN/Inf (cheap enough to call once per epoch)."""
    for module in modules:
        for name, p in module.named_parameters():
            if not torch.isfinite(p).all():
                raise RuntimeError(f"Non-finite values in parameter {type(module).__name__}.{name}")


# =============================================================================
# Checkpoints
# =============================================================================

def _atomic_save(obj: dict, path: Path) -> None:
    tmp = path.with_name(path.name + ".tmp")
    torch.save(obj, tmp)
    tmp.replace(path)


def _atomic_copy(src: Path, dst: Path) -> None:
    tmp = dst.with_name(dst.name + ".tmp")
    shutil.copyfile(src, tmp)
    tmp.replace(dst)


def save_checkpoint_files(
    checkpoint: dict,
    checkpoint_dir: Path,
    prefix: str,
    epoch: int,
    best_path: Optional[Path] = None,
) -> Path:
    """
    Write ``{prefix}_epoch_{epoch:04d}.pt`` and ``{prefix}_latest.pt``, plus
    ``best_path`` if given. Writes go through a temp file, so an interrupted
    save never leaves a truncated ``latest`` checkpoint behind.
    """
    checkpoint_path = checkpoint_dir / f"{prefix}_epoch_{epoch:04d}.pt"
    _atomic_save(checkpoint, checkpoint_path)
    _atomic_copy(checkpoint_path, checkpoint_dir / f"{prefix}_latest.pt")
    if best_path is not None:
        _atomic_copy(checkpoint_path, best_path)
        logger.info(f"Saved best model to {best_path}")
    return checkpoint_path


def load_checkpoint_states(
    checkpoint: dict,
    model: nn.Module,
    optimizer: Optional[optim.Optimizer] = None,
    scheduler: Optional[optim.lr_scheduler.LRScheduler] = None,
    scaler: Optional[GradScaler] = None,
    extra_modules: Optional[dict[str, nn.Module]] = None,
) -> None:
    """
    Restore model/optimizer/scheduler/scaler (and extra modules such as a loss
    with learnable parameters, stored under ``"{name}_state_dict"``).

    A disabled GradScaler saves an empty state that an enabled one refuses to
    load, so scaler state is only restored when both sides use AMP.
    """
    model.load_state_dict(checkpoint["model_state_dict"])

    for name, module in (extra_modules or {}).items():
        key = f"{name}_state_dict"
        if key in checkpoint:
            module.load_state_dict(checkpoint[key])

    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    if scheduler is not None and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    if scaler is not None and scaler.is_enabled() and checkpoint.get("scaler_state_dict"):
        scaler.load_state_dict(checkpoint["scaler_state_dict"])
