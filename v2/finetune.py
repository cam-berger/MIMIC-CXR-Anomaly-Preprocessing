"""
M3: fine-tune an encoder end to end (docs/V2_PLAN.md §6).

    python -m v2.finetune --model rad-dino --size 518 --seed 0        # rented GPU
    python -m v2.finetune --model rad-dino --size 224 --seed 0        # local GPU / Apple MPS
    python -m v2.finetune --model rad-dino --size 224 --smoke         # 1-3 min check on real data

Fixed recipe: masked BCE (uncertain labels ignored); AdamW (weight decay 0.05);
head lr 1e-3, encoder lr 2e-5 with layer-wise decay 0.75 (DenseNet: 1e-4, no
decay); 1 warmup epoch then cosine; 10 epochs; effective batch 32; bf16 on
CUDA; gradient clipping 1.0. The epoch with the best val macro AUROC is kept,
and val/test predictions are written once, at the end, from that checkpoint.

The head is one linear layer on the pooled features. With --head attention
(§6 M5b amendment) it is M4b's h0 head instead, on the CLS token and the patch
tokens averaged to a 12 x 12 grid: it starts at random, with its token
normalization fixed from the frozen encoder on 512 training studies.

Outputs: runs/{run}/{last.pt,best.pt,history.json} and predictions/{run}.parquet
(run {model}_{size}_ft_s{seed}, or _ft_attn_s{seed} with --head attention).
Interrupted? Re-run the same command with --resume.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Sampler

from .common import (
    LABELS,
    MIN_COUNT_VAL,
    TEST,
    TRAIN,
    VAL,
    amp_dtype,
    default_out_dir,
    git_commit,
    label_matrix,
    load_manifest,
    measurable,
    pick_device,
    seed_everything,
    setup_logging,
    targets_and_mask,
    write_predictions,
)
from .data import ImageDataset, image_paths, loader_kwargs
from .embed import save_npz
from .encoders import REGISTRY, DinoEncoder, Encoder, SiglipEncoder, load_encoder
from .evaluate import auroc

logger = logging.getLogger("v2.finetune")

NO_DECAY_NAMES = ("position_embeddings", "cls_token", "mask_token", "register_tokens", "position_embedding")
TOKEN_STATS_STUDIES = 512  # --head attention: training studies whose frozen tokens fix the head's normalization


def _start_at_priors(bias: torch.Tensor, prevalence: Optional[np.ndarray]) -> None:
    """Start each logit at its label's prior: smaller, saner first gradients."""
    if prevalence is not None:
        p = np.clip(prevalence, 0.01, 0.99)
        with torch.no_grad():
            bias.copy_(torch.as_tensor(np.log(p / (1 - p)), dtype=torch.float32))


class Classifier(nn.Module):
    """Encoder features -> one linear layer -> 12 logits."""

    def __init__(self, encoder: Encoder, prevalence: Optional[np.ndarray] = None):
        super().__init__()
        self.encoder = encoder
        self.head = nn.Linear(encoder.dim, len(LABELS))
        _start_at_priors(self.head.bias, prevalence)

    def features(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(x))


class AttentionClassifier(nn.Module):
    """Encoder tokens (CLS + patch tokens averaged to grid x grid cells) -> M4b's h0 head -> 12 logits."""

    def __init__(self, encoder: Encoder, grid: int, token_mean: np.ndarray, token_std: np.ndarray,
                 prevalence: Optional[np.ndarray] = None):
        super().__init__()
        from .attend import TokenAttentionHead  # attend imports this module

        self.encoder, self.grid = encoder, grid
        g = encoder.token_grid(grid)
        self.head = TokenAttentionHead(1 + g * g, encoder.token_dim, len(LABELS), token_mean=token_mean,
                                       token_std=token_std)
        _start_at_priors(self.head.bias, prevalence)

    def features(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder.tokens(x, self.grid)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(x))


@torch.inference_mode()
def token_stats(encoder: Encoder, loader: DataLoader, device, dtype, grid: int) -> tuple[np.ndarray, np.ndarray]:
    """Per-dimension mean and SD of the frozen encoder's tokens over `loader`'s images (every position)."""
    encoder.eval()
    autocast = torch.autocast(device.type, dtype=dtype) if dtype is not None else nullcontext()
    total, sq, n = 0.0, 0.0, 0
    for batch in loader:
        with autocast:
            toks = encoder.tokens(batch[0].to(device, non_blocking=True), grid)
        x = toks.float().reshape(-1, toks.shape[-1]).double().cpu()
        total, sq, n = total + x.sum(0), sq + (x * x).sum(0), n + len(x)
    mean = total / n
    std = torch.sqrt(torch.clamp(sq / n - mean ** 2, min=0.0))
    return mean.float().numpy(), np.maximum(std.float().numpy(), 1e-6)


def masked_bce(logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    loss = F.binary_cross_entropy_with_logits(logits.float(), targets, reduction="none")
    m = mask.float()
    return (loss * m).sum() / m.sum().clamp(min=1.0)


def _no_decay(name: str, p: torch.Tensor) -> bool:
    return p.ndim <= 1 or any(key in name for key in NO_DECAY_NAMES)


def _dino_layer(name: str, n_layers: int) -> int:
    """0 for embeddings, i + 1 for transformer block i, n_layers + 1 for the final norm."""
    if name.startswith("embeddings."):
        return 0
    if name.startswith("encoder.layer."):
        return int(name.split(".")[2]) + 1
    return n_layers + 1


def param_groups(model: nn.Module, encoder_lr: float, head_lr: float, weight_decay: float, llrd: float) -> list:
    """AdamW groups: layer-wise lr decay for ViTs (BEiT/DINOv2 recipe), no decay on norms/biases/tokens."""
    groups: dict[tuple, dict] = {}

    def add(name: str, p: torch.Tensor, lr: float) -> None:
        if not p.requires_grad:
            return
        wd = 0.0 if _no_decay(name, p) else weight_decay
        key = (lr, wd)
        if key not in groups:
            groups[key] = {"params": [], "names": [], "lr": lr, "weight_decay": wd}
        groups[key]["params"].append(p)
        groups[key]["names"].append(name)

    for name, p in model.head.named_parameters():
        add(f"head.{name}", p, head_lr)
    enc = model.encoder
    if isinstance(enc, DinoEncoder):
        n_layers = enc.backbone.config.num_hidden_layers
        for name, p in enc.backbone.named_parameters():
            add(name, p, encoder_lr * llrd ** (n_layers + 1 - _dino_layer(name, n_layers)))
    else:
        for name, p in enc.named_parameters():
            add(name, p, encoder_lr)
    return list(groups.values())


def warmup_cosine(warmup_steps: int, total_steps: int):
    def fn(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / max(1, warmup_steps)
        progress = min(1.0, (step - warmup_steps) / max(1, total_steps - warmup_steps))
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    return fn


class EpochShuffle(Sampler):
    """Shuffles with seed + epoch, so a resumed run sees the same order as an uninterrupted one."""

    def __init__(self, n: int, seed: int):
        self.n, self.seed, self.epoch = n, seed, 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __iter__(self):
        g = torch.Generator().manual_seed(self.seed * 10_000 + self.epoch)
        return iter(torch.randperm(self.n, generator=g).tolist())

    def __len__(self) -> int:
        return self.n


@torch.inference_mode()
def predict(model: nn.Module, loader: DataLoader, device, dtype, with_features: bool = False):
    """Logits (and the head's input features) in dataset order."""
    model.eval()
    autocast = torch.autocast(device.type, dtype=dtype) if dtype is not None else nullcontext()
    logits, feats = [], []
    for batch in loader:
        x = batch[0].to(device, non_blocking=True)
        with autocast:
            f = model.features(x)
            out = model.head(f)
        logits.append(out.float().cpu().numpy())
        if with_features:
            feats.append(f.float().cpu().numpy())
    logits = np.concatenate(logits) if logits else np.zeros((0, len(LABELS)), np.float32)
    feats = np.concatenate(feats) if feats else None
    return logits, feats


def val_macro_auroc(logits: np.ndarray, raw: np.ndarray) -> tuple[float, dict]:
    Y, M = targets_and_mask(raw, "v2")
    classes = measurable(raw, "v2", MIN_COUNT_VAL)
    per = {c: auroc(Y[M[:, k], k], logits[M[:, k], k]) for c, k in ((c, LABELS.index(c)) for c in classes)}
    finite = [v for v in per.values() if np.isfinite(v)]
    return (float(np.mean(finite)) if finite else float("nan")), per


def make_grad_scaler():
    try:
        return torch.amp.GradScaler("cuda")
    except (AttributeError, TypeError):  # torch < 2.3
        return torch.cuda.amp.GradScaler()


def _rng_state() -> dict:
    return {"torch": torch.get_rng_state(), "numpy": np.random.get_state()}


def _set_rng_state(state: dict) -> None:
    torch.set_rng_state(state["torch"])
    np.random.set_state(state["numpy"])


def build_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="rad-dino", choices=sorted(REGISTRY))
    parser.add_argument("--size", type=int, help="Input size in px (default: the model's native size)")
    parser.add_argument("--out", default=None, help="v2 output dir (default: $OUTPUT_PATH/v2 or output/v2)")
    parser.add_argument("--run", help="Run name (default: {model}_{size}_ft_s{seed}, _ft_attn_s{seed} with "
                                      "--head attention)")
    parser.add_argument("--head", default="mean", choices=["mean", "attention"],
                        help="mean: a linear layer on the pooled features; attention: M4b's h0 head on the tokens "
                             "(§6 M5b amendment)")
    parser.add_argument("--grid", type=int, default=12, help="--head attention: patch tokens averaged to G x G")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--warmup-epochs", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=16, help="Per step; x --accum = effective batch")
    parser.add_argument("--accum", type=int, default=2)
    parser.add_argument("--encoder-lr", type=float, help="Default 2e-5 (ViT) or 1e-4 (DenseNet)")
    parser.add_argument("--head-lr", type=float, default=1e-3)
    parser.add_argument("--llrd", type=float, default=0.75, help="Layer-wise lr decay (ViTs only)")
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--clip", type=float, default=1.0)
    parser.add_argument("--precision", default="auto", choices=["auto", "bf16", "fp16", "fp32"])
    parser.add_argument("--device", default="auto", help="auto | cuda | mps | cpu")
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument("--grad-checkpointing", action="store_true", help="Less memory, ~30%% slower")
    parser.add_argument("--max-hours", type=float, default=0, help="Stop training after this long (0 = no limit)")
    parser.add_argument("--steps-per-epoch", type=int, default=0, help="Cap optimizer steps per epoch (tests)")
    parser.add_argument("--max-bad-steps", type=float, default=0.01,
                        help="Abort if more than this share of steps have non-finite loss or gradients")
    parser.add_argument("--save-embeddings", action="store_true",
                        help="Also save the fine-tuned encoder's features for every study (for v2.fuse)")
    parser.add_argument("--resume", action="store_true", help="Continue from runs/{run}/last.pt")
    parser.add_argument("--smoke", action="store_true", help="64 train / 32 val / 32 test studies, 2 epochs")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)
    if args.smoke:
        args.epochs = min(args.epochs, 2)
        args.batch_size = min(args.batch_size, 8)
        args.accum = 1
    if args.head == "attention" and args.save_embeddings:
        parser.error("--save-embeddings needs --head mean: the attention head has no single feature vector")
    return args


def main(argv: Optional[list[str]] = None) -> str:
    args = build_args(argv)
    setup_logging(args.verbose)
    seed_everything(args.seed)
    out_dir = Path(args.out) if args.out else default_out_dir()
    device = pick_device(args.device)
    dtype = amp_dtype(device, args.precision)

    # --- data -------------------------------------------------------------
    manifest = load_manifest(out_dir)
    splits = {s: manifest[manifest["split"] == s].sort_values("study_id").reset_index(drop=True)
              for s in (TRAIN, VAL, TEST)}
    if args.smoke:
        for s, n in ((TRAIN, 64), (VAL, 32), (TEST, 32)):
            splits[s] = splits[s].head(n)
    if len(splits[TRAIN]) == 0 or len(splits[VAL]) == 0:
        raise SystemExit("Need train and val studies in the manifest")

    enc = load_encoder(args.model, args.size, device)
    if isinstance(enc, SiglipEncoder):  # fine-tune the image tower only
        for p in enc.model.text_model.parameters():
            p.requires_grad_(False)
    if args.grad_checkpointing:
        backbone = getattr(enc, "backbone", None) or getattr(enc, "model", None)
        if backbone is None or not hasattr(backbone, "gradient_checkpointing_enable"):
            raise SystemExit(f"--grad-checkpointing is not supported for {enc.name}")
        backbone.gradient_checkpointing_enable()
    encoder_lr = args.encoder_lr or (1e-4 if REGISTRY[args.model].kind == "densenet" else 2e-5)
    run = args.run or f"{enc.name}_{enc.size}_ft{'_attn' if args.head == 'attention' else ''}_s{args.seed}" + \
        ("_smoke" if args.smoke else "")
    run_dir = out_dir / "runs" / run
    run_dir.mkdir(parents=True, exist_ok=True)

    raw_tr = label_matrix(splits[TRAIN])
    y_tr, m_tr = targets_and_mask(raw_tr, "v2")
    prevalence = (y_tr * m_tr).sum(0) / np.maximum(m_tr.sum(0), 1)
    eval_tf = enc.eval_transform()
    eval_kw = loader_kwargs(args.workers, device, persistent=False)
    if args.head == "attention":
        if enc.token_dim is None:
            raise SystemExit(f"--head attention needs patch tokens, which {enc.name} doesn't give")
        rows = splits[TRAIN].head(TOKEN_STATS_STUDIES)
        stats_dl = DataLoader(ImageDataset(image_paths(out_dir, rows), eval_tf), batch_size=args.batch_size * 2,
                              shuffle=False, **eval_kw)
        token_mean, token_std = token_stats(enc, stats_dl, device, dtype, args.grid)
        model = AttentionClassifier(enc, args.grid, token_mean, token_std, prevalence).to(device)
        logger.info("Attention head on %d tokens (grid %d), normalized from %d training studies",
                    1 + enc.token_grid(args.grid) ** 2, enc.token_grid(args.grid), len(rows))
    else:
        model = Classifier(enc, prevalence).to(device)

    train_ds = ImageDataset(image_paths(out_dir, splits[TRAIN]), enc.train_transform(), y_tr, m_tr)
    sampler = EpochShuffle(len(train_ds), args.seed)
    kw = loader_kwargs(args.workers, device)
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler,
                          drop_last=len(train_ds) > args.batch_size, **kw)
    val_dl = DataLoader(ImageDataset(image_paths(out_dir, splits[VAL]), eval_tf), batch_size=args.batch_size * 2,
                        shuffle=False, **eval_kw)
    raw_val = label_matrix(splits[VAL])

    # --- optimization -----------------------------------------------------
    groups = param_groups(model, encoder_lr, args.head_lr, args.weight_decay, args.llrd)
    trainable = [p for g in groups for p in g["params"]]
    optimizer = torch.optim.AdamW([{k: v for k, v in g.items() if k != "names"} for g in groups],
                                  betas=(0.9, 0.999))
    steps_per_epoch = math.ceil(len(train_dl) / args.accum)
    if args.steps_per_epoch:
        steps_per_epoch = min(steps_per_epoch, args.steps_per_epoch)
    total_steps = steps_per_epoch * args.epochs
    warmup_steps = int(round(args.warmup_epochs * steps_per_epoch))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, warmup_cosine(warmup_steps, total_steps))
    scaler = make_grad_scaler() if dtype == torch.float16 else None
    autocast = (lambda: torch.autocast(device.type, dtype=dtype)) if dtype is not None else nullcontext

    config = {k: v for k, v in vars(args).items()}
    config.update({"run": run, "encoder_lr": encoder_lr, "device": str(device),
                   "precision": str(dtype).replace("torch.", "") if dtype else "fp32",
                   "steps_per_epoch": steps_per_epoch, "total_steps": total_steps, "warmup_steps": warmup_steps,
                   "n_train": len(train_ds), "n_val": len(splits[VAL]), "commit": git_commit(), **enc.describe()})
    logger.info("Run %s: %s at %d px on %s (%s), %d train / %d val studies, %d steps (%d per epoch)",
                run, enc.name, enc.size, device, config["precision"], len(train_ds), len(splits[VAL]),
                total_steps, steps_per_epoch)
    logger.info("Param groups: %s", [(round(g["lr"], 8), g["weight_decay"], len(g["params"])) for g in groups])

    state = {"epoch": -1, "step": 0, "best": -math.inf, "best_epoch": -1, "history": [], "bad_steps": 0}
    last_path, best_path = run_dir / "last.pt", run_dir / "best.pt"
    if args.resume and last_path.exists():
        ckpt = torch.load(last_path, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        if scaler is not None and ckpt.get("scaler"):
            scaler.load_state_dict(ckpt["scaler"])
        state = ckpt["state"]
        _set_rng_state(ckpt["rng"])
        logger.info("Resumed from %s after epoch %d (best val %.4f)", last_path, state["epoch"], state["best"])
    (run_dir / "config.json").write_text(json.dumps(config, indent=2, default=str))

    # --- train ------------------------------------------------------------
    start_time = time.time()
    stopped_early = False
    for epoch in range(state["epoch"] + 1, args.epochs):
        model.train()
        sampler.set_epoch(epoch)
        optimizer.zero_grad(set_to_none=True)
        t0, seen, loss_sum, loss_n, epoch_steps = time.time(), 0, 0.0, 0, 0
        for i, (x, y, m, _) in enumerate(train_dl):
            x, y, m = x.to(device, non_blocking=True), y.to(device), m.to(device)
            with autocast():
                logits = model(x)
            loss = masked_bce(logits, y, m)
            if torch.isfinite(loss):
                (scaler.scale(loss / args.accum) if scaler else loss / args.accum).backward()
                loss_sum += loss.item()
                loss_n += 1
            else:
                state["bad_steps"] += 1
                logger.warning("Non-finite loss at epoch %d batch %d: skipped", epoch, i)
            seen += len(x)
            if (i + 1) % args.accum and i + 1 != len(train_dl):
                continue
            if scaler:
                scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(trainable, args.clip)
            if scaler:
                scaler.step(optimizer)  # skips the step itself if gradients are inf/nan
                scaler.update()
            elif torch.isfinite(grad_norm):
                optimizer.step()
            else:
                state["bad_steps"] += 1
                logger.warning("Non-finite gradient norm at epoch %d step %d: skipped", epoch, state["step"])
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()
            state["step"] += 1
            epoch_steps += 1
            if state["step"] > 20 and state["bad_steps"] > args.max_bad_steps * state["step"]:
                raise SystemExit(f"{state['bad_steps']} bad steps out of {state['step']}: aborting. "
                                 "Check the data and learning rate before spending more GPU time.")
            if epoch_steps % 50 == 0:
                rate = seen / (time.time() - t0)
                logger.info("  epoch %d step %d/%d  loss %.4f  lr %.2e  %.1f img/s", epoch, epoch_steps,
                            steps_per_epoch, loss_sum / max(loss_n, 1), scheduler.get_last_lr()[0], rate)
            if epoch_steps >= steps_per_epoch:
                break
            if args.max_hours and time.time() - start_time > args.max_hours * 3600:
                stopped_early = True
                break

        val_logits, _ = predict(model, val_dl, device, dtype)
        val_auc, per = val_macro_auroc(val_logits, raw_val)
        val_loss = float(masked_bce(torch.as_tensor(val_logits), *map(torch.as_tensor, targets_and_mask(raw_val))))
        record = {"epoch": epoch, "train_loss": loss_sum / max(loss_n, 1), "val_loss": val_loss,
                  "val_macro_auroc": val_auc, "val_classes": len(per), "seconds": round(time.time() - t0, 1),
                  "steps": state["step"]}
        state["history"].append(record)
        improved = np.isfinite(val_auc) and val_auc > state["best"]
        if improved or state["best_epoch"] < 0:
            state["best"], state["best_epoch"] = (val_auc if np.isfinite(val_auc) else state["best"]), epoch
            torch.save({"model": model.state_dict(), "config": config, "epoch": epoch, "val_macro_auroc": val_auc},
                       best_path)
        state["epoch"] = epoch
        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict() if scaler else None,
                    "state": state, "rng": _rng_state(), "config": config}, last_path)
        (run_dir / "history.json").write_text(json.dumps(state["history"], indent=2))
        logger.info("Epoch %d: train loss %.4f, val loss %.4f, val macro AUROC %.4f (%d classes)%s, %.0fs",
                    epoch, record["train_loss"], val_loss, val_auc, len(per), "  *best*" if improved else "",
                    record["seconds"])
        if stopped_early:
            logger.warning("--max-hours reached: stopping after epoch %d", epoch)
            break

    # --- predict once, from the best epoch --------------------------------
    best = torch.load(best_path, map_location="cpu", weights_only=False)
    model.load_state_dict(best["model"])
    logger.info("Best epoch %d (val macro AUROC %.4f): predicting val and test", best["epoch"],
                best["val_macro_auroc"])
    rows = pd.concat([splits[VAL], splits[TEST]], ignore_index=True)
    dl = DataLoader(ImageDataset(image_paths(out_dir, rows), eval_tf), batch_size=args.batch_size * 2,
                    shuffle=False, **eval_kw)
    logits, _ = predict(model, dl, device, dtype)
    write_predictions(out_dir, run, rows["study_id"], rows["split"], logits, {
        "kind": "finetune", "head": args.head, **enc.describe(), "best_epoch": best["epoch"],
        "best_val_macro_auroc": best["val_macro_auroc"], "stopped_early": stopped_early,
        "history": state["history"], "config": config,
    })

    if args.save_embeddings:
        every = manifest if not args.smoke else pd.concat(list(splits.values()), ignore_index=True)
        every = every.sort_values("study_id").reset_index(drop=True)
        dl = DataLoader(ImageDataset(image_paths(out_dir, every), eval_tf), batch_size=args.batch_size * 2,
                        shuffle=False, **eval_kw)
        _, feats = predict(model, dl, device, dtype, with_features=True)
        path = out_dir / "embeddings" / f"{run}.npz"
        save_npz(path, every["study_id"].to_numpy(), feats,
                 {**enc.describe(), "encoder": enc.name, "finetune_run": run, "commit": git_commit()})
        logger.info("Wrote fine-tuned embeddings %s", path)
    logger.info("Next: python -m v2.evaluate --run %s", run)
    return run


if __name__ == "__main__":
    main()
