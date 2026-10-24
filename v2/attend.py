"""
M4b part 1: can text steer where the model looks? (docs/ANALYSIS.md, M4b)

    python -m v2.embed --model rad-dino --size 518 --tokens 12     # first: cached patch tokens
    python -m v2.attend --tokens output/v2/embeddings/rad-dino_518_tokens12.npy

Three small attention heads on the frozen tokens, with M4's text features
(TF-IDF + SVD per source, fit on train), 5 seeds each:

    h0   12 class queries attend over the image tokens (image only)
    h1   h0 + a linear text term in each class's logit: late fusion on the same head
    h2   h1 + text-conditioned queries, so the text moves where each class looks. The
         steering layer starts at zero, so with the same seed h2 starts as exactly h1

Writes predictions/{encoder}_{size}_attn_{arm}.parquet (the mean of the seeds'
logits) and reports/m4b_attention_{encoder}_{size}.md: the head validity check,
gate G4b (h2 minus h1 on test_clean), the reported contrasts and how far the text
moves h2's attention. Only aggregates are logged.

Each finished arm and seed is cached in runs/attend/{encoder}_{size}/, keyed on the
inputs and the recipe, so an interrupted run resumes where it stopped. Delete that
folder to retrain from scratch.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import logging
import math
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler

from .common import LABELS, MIN_COUNT_VAL, TEST, TRAIN, VAL, default_out_dir, label_matrix, load_manifest, measurable
from .common import pick_device, seed_everything, setup_logging, targets_and_mask, write_predictions
from .context import TEXT_SOURCES
from .embed import embeddings_path, load_npz, load_tokens
from .evaluate import _get, compare_runs, evaluate_run, fmt, fmt_ci, g4b, split_scheme
from .finetune import masked_bce, val_macro_auroc, warmup_cosine
from .fuse import text_block

logger = logging.getLogger("v2.attend")

CACHE_VERSION = 1  # bump when the head or its training changes, so cached seeds are not reused
ARMS = ("h0", "h1", "h2")
TEXT_ARMS = ("h1", "h2")
STEER_ARMS = ("h2",)
SEEDS = (0, 1, 2, 3, 4)


@dataclass(frozen=True)
class Recipe:
    """The pre-registered head and training recipe (M4b)."""

    width: int = 256
    heads: int = 4
    steer_hidden: int = 64
    dropout: float = 0.1
    lr: float = 1e-3
    weight_decay: float = 0.05
    batch_size: int = 256
    max_epochs: int = 30
    patience: int = 5
    warmup_epochs: int = 1
    clip: float = 1.0


class TokenAttentionHead(nn.Module):
    """
    Per-class queries attend over one study's tokens (multi-head); each class's pooled vector has its own
    linear readout. With `text_dim`, a linear text term joins each logit (h1); with `steer`, an MLP of the
    text also shifts every class query (h2).
    """

    def __init__(self, n_tokens: int, token_dim: int, n_classes: int, text_dim: int = 0, steer: bool = False,
                 recipe: Recipe = Recipe(), token_mean: Optional[np.ndarray] = None,
                 token_std: Optional[np.ndarray] = None):
        super().__init__()
        d = recipe.width
        if d % recipe.heads:
            raise ValueError(f"width {d} is not divisible by {recipe.heads} heads")
        if steer and not text_dim:
            raise ValueError("steering needs text")
        self.n_classes, self.width, self.heads = n_classes, d, recipe.heads
        self.register_buffer("token_mean", torch.as_tensor(
            np.zeros(token_dim) if token_mean is None else token_mean, dtype=torch.float32))
        self.register_buffer("token_std", torch.as_tensor(
            np.ones(token_dim) if token_std is None else token_std, dtype=torch.float32))
        self.key = nn.Linear(token_dim, d)
        self.value = nn.Linear(token_dim, d)
        self.position = nn.Parameter(torch.randn(n_tokens, d) * 0.02)  # where each token sits: lets a query pick places
        self.queries = nn.Parameter(torch.randn(n_classes, d) * 0.02)
        self.readout = nn.Parameter(torch.randn(n_classes, d) * 0.02)
        self.bias = nn.Parameter(torch.zeros(n_classes))
        self.dropout = nn.Dropout(recipe.dropout)
        self.text = nn.Linear(text_dim, n_classes) if text_dim else None
        self.steer = None
        if steer:  # built last, so with the same seed every other parameter starts as in h1
            self.steer = nn.Sequential(nn.Linear(text_dim, recipe.steer_hidden), nn.GELU(),
                                       nn.Linear(recipe.steer_hidden, n_classes * d))
            nn.init.zeros_(self.steer[2].weight)
            nn.init.zeros_(self.steer[2].bias)

    def forward(self, tokens: torch.Tensor, text: Optional[torch.Tensor] = None, return_attention: bool = False):
        x = (tokens.float() - self.token_mean) / self.token_std
        B, T, _ = x.shape
        C, d, h = self.n_classes, self.width, self.heads
        dh = d // h
        k = (self.key(x) + self.position).view(B, T, h, dh).transpose(1, 2)        # [B, h, T, dh]
        v = self.value(x).view(B, T, h, dh).transpose(1, 2)
        q = self.queries.expand(B, C, d)
        if self.steer is not None:
            q = q + self.steer(text).view(B, C, d)
        q = q.reshape(B, C, h, dh).transpose(1, 2)                                 # [B, h, C, dh]
        att = torch.softmax(q @ k.transpose(-1, -2) / math.sqrt(dh), dim=-1)       # [B, h, C, T]
        z = (att @ v).transpose(1, 2).reshape(B, C, d)
        logits = (self.dropout(z) * self.readout).sum(-1) + self.bias
        if self.text is not None:
            logits = logits + self.text(text)
        return (logits, att.mean(dim=1)) if return_attention else logits


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------

@dataclass
class Inputs:
    ids: np.ndarray            # study ids, in token-file order
    tokens: np.ndarray         # float16 memmap [n, 1 + g*g, width]
    split: np.ndarray          # manifest split per row
    raw: np.ndarray            # raw CheXpert values [n, 12]
    text: Optional[np.ndarray]  # standardized M4 text features [n, k] (float32), or None without context
    text_stats: dict
    token_mean: np.ndarray
    token_std: np.ndarray
    meta: dict

    def rows(self, split: str) -> np.ndarray:
        return np.flatnonzero(self.split == split)


def token_stats(tokens: np.ndarray, rows: np.ndarray, max_studies: int = 2000, seed: int = 0,
                chunk: int = 256) -> tuple[np.ndarray, np.ndarray]:
    """Per-dimension mean and std of the tokens of up to `max_studies` training studies (all positions)."""
    rng = np.random.default_rng(seed)
    pick = np.sort(rng.choice(rows, size=min(max_studies, len(rows)), replace=False))
    total = np.zeros(tokens.shape[-1])
    sq = np.zeros(tokens.shape[-1])
    n = 0
    for start in range(0, len(pick), chunk):
        x = np.asarray(tokens[pick[start:start + chunk]], dtype=np.float64).reshape(-1, tokens.shape[-1])
        total += x.sum(0)
        sq += (x * x).sum(0)
        n += len(x)
    mean = total / n
    std = np.sqrt(np.maximum(sq / n - mean ** 2, 0.0))
    return mean.astype(np.float32), np.maximum(std, 1e-6).astype(np.float32)


def text_features(ctx_path: Optional[Path], ids: np.ndarray, train: np.ndarray) -> tuple[Optional[np.ndarray], dict]:
    """M4's text block for every source (fit on train), standardized on train; None without a context table."""
    if ctx_path is None or not ctx_path.exists():
        return None, {}
    ctx = pd.read_parquet(ctx_path).drop_duplicates("study_id").set_index("study_id").reindex(ids)
    ctx = ctx.rename_axis("study_id").reset_index()
    blocks, stats = [], {}
    for src in TEXT_SOURCES:
        if not ctx[src].fillna("").str.strip().ne("").any():
            logger.warning("No %s text in the context: left out of the text features", src)
            continue
        block, stats[src] = text_block(ctx[src], train)
        blocks.append(block)
    if not blocks:
        return None, stats
    X = np.concatenate(blocks, axis=1)
    return StandardScaler().fit(X[train]).transform(X).astype(np.float32), stats


def load_inputs(out_dir: Path, tokens_path: Path, ctx_path: Optional[Path]) -> Inputs:
    manifest = load_manifest(out_dir)
    ids, tokens, meta = load_tokens(tokens_path)
    man = manifest.drop_duplicates("study_id").set_index("study_id").reindex(ids)
    unknown = int(man["split"].isna().sum())
    if unknown:
        raise SystemExit(f"{unknown} studies in {tokens_path.name} are not in the manifest: tokens from other data?")
    split = man["split"].to_numpy()
    train = split == TRAIN
    if not train.any() or not (split == VAL).any():
        raise SystemExit("Need train and val tokens to fit the heads")
    raw = label_matrix(man.reset_index())
    text, stats = text_features(ctx_path, ids, train)
    mean, std = token_stats(tokens, np.flatnonzero(train))
    return Inputs(ids, tokens, split, raw, text, stats, mean, std, meta)


def check_alignment(inputs: Inputs, emb_path: Path) -> Optional[float]:
    """
    Token 0 must equal the matching slice of the pooled embeddings from v2.embed (same images, same order),
    to float16 precision. Returns the largest relative difference, or None if those embeddings don't exist.
    """
    sl = inputs.meta.get("pooled_slice")
    if sl is None or not emb_path.exists():
        return None
    ids, emb, _ = load_npz(emb_path)
    pos = pd.Series(np.arange(len(ids)), index=ids)
    common = np.isin(inputs.ids, ids)
    if not common.any():
        raise SystemExit(f"{emb_path.name} shares no studies with the tokens")
    want = emb[pos.loc[inputs.ids[common]].to_numpy(), sl[0]:sl[1]]
    got = np.asarray(inputs.tokens[np.flatnonzero(common), 0], dtype=np.float32)
    rel = float(np.max(np.abs(got - want)) / max(float(np.max(np.abs(want))), 1e-6))
    if rel > 1e-2:
        raise SystemExit(f"token 0 differs from {emb_path.name} by {rel:.2%} of its range: the tokens and the M1 "
                         "embeddings don't describe the same images (different data, transform or encoder version?)")
    return rel


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def _batch(inputs: Inputs, rows: np.ndarray, device) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    tokens = torch.from_numpy(np.ascontiguousarray(inputs.tokens[rows])).to(device).float()
    text = None if inputs.text is None else torch.from_numpy(inputs.text[rows]).to(device)
    return tokens, text


def build_head(arm: str, inputs: Inputs, recipe: Recipe, seed: int) -> TokenAttentionHead:
    seed_everything(seed)
    text_dim = inputs.text.shape[1] if arm in TEXT_ARMS else 0
    return TokenAttentionHead(inputs.tokens.shape[1], inputs.tokens.shape[2], len(LABELS), text_dim,
                              arm in STEER_ARMS, recipe, inputs.token_mean, inputs.token_std)


@torch.inference_mode()
def predict_head(model: TokenAttentionHead, inputs: Inputs, rows: np.ndarray, device, batch_size: int = 512,
                 text_off: bool = False, attention: bool = False):
    """Logits [len(rows), 12] in `rows` order (and head-averaged attention maps with `attention`)."""
    model.eval()
    logits, maps = [], []
    for start in range(0, len(rows), batch_size):
        tokens, text = _batch(inputs, rows[start:start + batch_size], device)
        if text is not None and text_off:
            text = torch.zeros_like(text)  # the train mean, after standardization
        out = model(tokens, text, return_attention=attention)
        if attention:
            out, att = out
            maps.append(att.float().cpu().numpy())
        logits.append(out.float().cpu().numpy())
    logits = np.concatenate(logits) if logits else np.zeros((0, len(LABELS)), np.float32)
    return (logits, np.concatenate(maps)) if attention else logits


def fit_head(arm: str, inputs: Inputs, seed: int, recipe: Recipe, device, train_rows: Optional[np.ndarray] = None,
             val_rows: Optional[np.ndarray] = None) -> tuple[TokenAttentionHead, dict]:
    """
    Train one arm with one seed; keeps the epoch with the best val macro AUROC (M4b). Only when val
    has no class with MIN_COUNT_VAL positives and negatives (tiny test data) does it select by val loss.
    """
    train_rows = inputs.rows(TRAIN) if train_rows is None else train_rows
    val_rows = inputs.rows(VAL) if val_rows is None else val_rows
    model = build_head(arm, inputs, recipe, seed).to(device)
    torch.manual_seed(seed + 1)  # the same dropout stream in every arm
    shuffle = np.random.default_rng(seed)
    Y, M = targets_and_mask(inputs.raw, "v2")
    decay = [p for n, p in model.named_parameters() if p.ndim > 1 and n not in ("position", "queries")]
    rest = [p for n, p in model.named_parameters() if not (p.ndim > 1 and n not in ("position", "queries"))]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": recipe.weight_decay},
                             {"params": rest, "weight_decay": 0.0}], lr=recipe.lr)
    steps = math.ceil(len(train_rows) / recipe.batch_size)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, warmup_cosine(recipe.warmup_epochs * steps, recipe.max_epochs * steps))
    raw_val = inputs.raw[val_rows]
    by_auroc = bool(measurable(raw_val, "v2", MIN_COUNT_VAL))
    Y_val, M_val = (torch.from_numpy(a) for a in targets_and_mask(raw_val, "v2"))
    best = {"epoch": -1, "score": -np.inf, "val_macro_auroc": float("nan"), "state": copy.deepcopy(model.state_dict())}
    history = []
    for epoch in range(recipe.max_epochs):
        model.train()
        order = shuffle.permutation(train_rows)
        total, seen = 0.0, 0
        for start in range(0, len(order), recipe.batch_size):
            rows = np.sort(order[start:start + recipe.batch_size])  # sorted reads from the memmap
            tokens, text = _batch(inputs, rows, device)
            loss = masked_bce(model(tokens, text), torch.from_numpy(Y[rows]).to(device),
                              torch.from_numpy(M[rows]).to(device))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), recipe.clip)
            opt.step()
            sched.step()
            total += loss.item() * len(rows)
            seen += len(rows)
        val_logits = predict_head(model, inputs, val_rows, device)
        auc, _ = val_macro_auroc(val_logits, raw_val)
        val_loss = float(masked_bce(torch.from_numpy(val_logits), Y_val, M_val))
        history.append({"epoch": epoch, "train_loss": round(total / max(seen, 1), 5), "val_loss": round(val_loss, 5),
                        "val_macro_auroc": None if not np.isfinite(auc) else round(float(auc), 5)})
        score = auc if by_auroc else -val_loss
        if np.isfinite(score) and score > best["score"]:
            best = {"epoch": epoch, "score": float(score), "val_macro_auroc": float(auc),
                    "state": copy.deepcopy(model.state_dict())}
        elif epoch - best["epoch"] >= recipe.patience:
            break
    model.load_state_dict(best["state"])
    info = {"seed": seed, "best_epoch": best["epoch"], "epochs_run": len(history),
            "best_val_macro_auroc": round(best["val_macro_auroc"], 5) if np.isfinite(best["val_macro_auroc"]) else None,
            "selected_by": "val macro AUROC" if by_auroc else "val loss (val has no class with enough labels)",
            "history": history}
    return model, info


def attention_shift(model: TokenAttentionHead, inputs: Inputs, rows: np.ndarray, device) -> np.ndarray:
    """Per class, the mean total variation distance between attention with each study's text and with none."""
    _, with_text = predict_head(model, inputs, rows, device, attention=True)
    _, without = predict_head(model, inputs, rows, device, text_off=True, attention=True)
    return 0.5 * np.abs(with_text - without).sum(axis=-1).mean(axis=0)


def _digest(*arrays) -> str:
    h = hashlib.sha1()
    for a in arrays:
        h.update(np.ascontiguousarray(a).tobytes())
    return h.hexdigest()[:16]


def seed_key(arm: str, seed: int, inputs: Inputs, eval_rows: np.ndarray, recipe: Recipe, tokens_path: Path) -> dict:
    """What a cached seed must match: the arm, seed, recipe, token file, the studies scored and the text features."""
    return {"version": CACHE_VERSION, "arm": arm, "seed": seed, "recipe": asdict(recipe),
            "tokens": Path(tokens_path).name, "tokens_shape": list(inputs.tokens.shape),
            "eval_ids": _digest(inputs.ids[eval_rows]),
            "text": _digest(inputs.text) if arm in TEXT_ARMS and inputs.text is not None else None}


def load_seed(path: Path, key: dict) -> Optional[tuple[np.ndarray, dict, Optional[np.ndarray]]]:
    if not path.exists():
        return None
    with np.load(path, allow_pickle=False) as f:
        if json.loads(str(f["key"])) != key:
            return None
        shift = f["shift"] if f["shift"].size else None
        return f["logits"], json.loads(str(f["info"])), shift


def save_seed(path: Path, key: dict, logits: np.ndarray, info: dict, shift: Optional[np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp.npz")
    np.savez(tmp, logits=logits, info=np.array(json.dumps(info)), key=np.array(json.dumps(key)),
             shift=np.zeros(0) if shift is None else shift)
    os.replace(tmp, path)


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def write_report(out_dir: Path, prefix: str, runs: dict, infos: dict, shift: Optional[np.ndarray], recipe: Recipe,
                 alignment: Optional[float], n_boot: int, seconds: float) -> tuple[Path, dict]:
    reps = {arm: evaluate_run(out_dir, run, n_boot) for arm, run in runs.items()}
    gate = g4b(out_dir, prefix, n_boot, {run: reps[arm] for arm, run in runs.items()})

    def cell(arm, test_set, metric):
        res = _get(reps[arm], test_set, metric) if arm in reps else None
        return fmt_ci(res["macro_auroc"], res["macro_auroc_ci"]) if res else "-"

    meta = json.loads((Path(out_dir) / "predictions" / f"{next(iter(runs.values()))}.json").read_text())
    bench = split_scheme(out_dir) == "official"
    title = f"# M5: attention heads on the benchmark ({prefix}; M4b's recipe)" if bench else \
        f"# M4b: text-steered attention ({prefix})"
    lines = [title, "",
             f"Frozen `{meta.get('encoder')}` tokens ({meta.get('token_grid')} x {meta.get('token_grid')} grid + "
             f"CLS) with M4's text features; seeds {meta.get('seeds')}, mean of their logits (docs/ANALYSIS.md, M4b).",
             ""]
    if recipe != Recipe():
        changed = {k: v for k, v in asdict(recipe).items() if asdict(Recipe())[k] != v}
        lines += [f"**Recipe differs from the pre-registered one:** {changed}. Not a G4b result.", ""]
    if alignment is not None:
        lines += [f"Token 0 vs the M1 embeddings: largest relative difference {alignment:.1e} (float16 rounding).", ""]
    lines += ["## 1. Head validity check (read first)", "", f"- {gate['check']}", "",
              "## 2. Arms", "", "| Arm | Run | val | test_clean | test_v1 (v1 metric) | Best epoch per seed |",
              "|---|---|---|---|---|---|"]
    for arm, run in runs.items():
        epochs = ", ".join(str(i["best_epoch"]) for i in infos.get(arm, []))
        lines.append(f"| {arm} | `{run}` | {cell(arm, 'val', 'v2')} | {cell(arm, 'test_clean', 'v2')} | "
                     f"{cell(arm, 'test_v1', 'v1')} | {epochs} |")
    still = [f"{arm} seed {i['seed']}" for arm, seeds in infos.items() for i in seeds
             if i["best_epoch"] == recipe.max_epochs - 1]
    if still:
        lines += ["", f"Best epoch is the last allowed one for {', '.join(still)}: those heads may still have been "
                  "improving (the recipe caps training at " f"{recipe.max_epochs} epochs)."]
    diff = fmt_ci(gate["cmp"]["diff"], gate["cmp"]["diff_ci"]) if gate["cmp"] else "-"
    if bench:
        lines += ["", "## 3. Gate G4b", "", "Not here: G4b is defined on the v2 cohort, and M5 drops `h2`. "
                  "The comparisons with published results are in reports/lineup.md.", ""]
    else:
        lines += ["", "## 3. Gate G4b", "", f"**G4b** (`h2` minus `h1` on test_clean): {diff} -> **{gate['verdict']}**",
                  ""]

    full = {arm: f"{prefix}_attn_{arm}" for arm in ARMS}
    contrasts = [("h0", gate["probe"], "test_clean", "v2", "head capacity"),
                 ("h1", "h0", "test_clean", "v2", "text on the attention head"),
                 ("h2", f"{prefix}_ctx_image_text", "test_clean", "v2", "against today's late fusion"),
                 ("h2", "h1", "test_v1", "v2", "G4b's contrast on the second test set"),
                 ("h2", "h1", "test_v1", "v1", "the same, mentioned findings only (biased toward 0 for text)")]
    if gate["check_ok"]:
        lines += ["## 4. Reported contrasts (not gates; uncorrected)", "", "| Contrast | Set | Difference [95% CI] | What it asks |",
                  "|---|---|---|---|"]
        for a, b, test_set, metric, what in contrasts:
            run_a, run_b = full.get(a, a), full.get(b, b)
            if not all((Path(out_dir) / "predictions" / f"{r}.parquet").exists() for r in (run_a, run_b)):
                continue
            try:
                c = compare_runs(out_dir, run_a, run_b, test_set, metric, n_boot)
            except SystemExit as exc:
                logger.warning("%s - %s on %s skipped: %s", run_a, run_b, test_set, exc)
                continue
            where = test_set + (", v1 metric" if metric == "v1" else "")
            lines.append(f"| `{run_a}` minus `{run_b}` | {where} | {fmt_ci(c['diff'], c['diff_ci'])} | {what} |")
        if "h1" in reps and "h2" in reps:
            p1 = {c["label"]: c for c in _get(reps["h1"], "test_clean", "v2")["per_class"]}
            p2 = {c["label"]: c for c in _get(reps["h2"], "test_clean", "v2")["per_class"]}
            lines += ["", "## 5. Per class on test_clean (AUROC)", "", "| Class | Measurable | h1 | h2 | h2 - h1 |",
                      "|---|---|---|---|---|"]
            for label in LABELS:
                a, b = p1[label]["auroc"], p2[label]["auroc"]
                d = f"{b - a:+.3f}" if np.isfinite(a) and np.isfinite(b) else "-"
                lines.append(f"| {label} | {'yes' if p1[label]['measurable'] else ''} | {fmt(a)} | {fmt(b)} | {d} |")
    else:
        lines += ["Contrasts and per-class results are held back with G4b until the head is inspected.", ""]
    if shift is not None:
        lines += ["", "## 6. How far the text moves h2's attention", "",
                  "Total variation distance between each class's attention map with the study's text and with the "
                  f"text at its train mean, on test_clean, averaged over studies and seeds: **{float(shift.mean()):.3f}** "
                  "overall (0: h2 does not steer; 1: it looks somewhere else entirely).", "",
                  "| " + " | ".join(LABELS) + " |", "|" + "---|" * len(LABELS),
                  "| " + " | ".join(f"{v:.3f}" for v in shift) + " |"]
    lines += ["", f"Wall time {seconds / 60:.1f} min."]
    path = Path(out_dir) / "reports" / f"m4b_attention_{prefix}.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")
    return path, gate


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(argv: Optional[list[str]] = None) -> dict:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tokens", required=True, help="embeddings/{model}_{size}_tokens{G}.npy from v2.embed --tokens")
    parser.add_argument("--context", help="context.parquet from v2.context (default: OUT/context.parquet)")
    parser.add_argument("--embeddings", help="Pooled M1 embeddings for the alignment check "
                                             "(default: embeddings/{model}_{size}.npz if it exists)")
    parser.add_argument("--out", default=None, help="v2 output dir (default: $OUTPUT_PATH/v2 or output/v2)")
    parser.add_argument("--arms", default=",".join(ARMS), help=f"Comma-separated subset of {', '.join(ARMS)}")
    parser.add_argument("--seeds", default=",".join(map(str, SEEDS)))
    parser.add_argument("--max-epochs", type=int, default=Recipe.max_epochs)
    parser.add_argument("--patience", type=int, default=Recipe.patience)
    parser.add_argument("--batch-size", type=int, default=Recipe.batch_size)
    parser.add_argument("--lr", type=float, default=Recipe.lr, help="Tests and smoke only: any change is flagged")
    parser.add_argument("--device", default="auto", help="auto | cuda | mps | cpu")
    parser.add_argument("--n-boot", type=int, default=1000)
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)
    setup_logging(args.verbose)
    t0 = time.time()
    out_dir = Path(args.out) if args.out else default_out_dir()
    arms = [a.strip() for a in args.arms.split(",")]
    if set(arms) - set(ARMS):
        raise SystemExit(f"Unknown arms {sorted(set(arms) - set(ARMS))}; choose from {ARMS}")
    seeds = [int(s) for s in args.seeds.split(",")]
    recipe = Recipe(max_epochs=args.max_epochs, patience=args.patience, batch_size=args.batch_size, lr=args.lr)
    ctx_path = Path(args.context) if args.context else out_dir / "context.parquet"
    if any(a in TEXT_ARMS for a in arms) and not ctx_path.exists():
        raise SystemExit(f"{ctx_path} not found: run `python -m v2.context` first (h1 and h2 need the text)")

    inputs = load_inputs(out_dir, Path(args.tokens), ctx_path)
    if any(a in TEXT_ARMS for a in arms) and inputs.text is None:
        raise SystemExit(f"{ctx_path} holds no text: h1 and h2 can't run")
    meta = inputs.meta
    prefix = f"{meta['encoder']}_{meta['size']}"
    emb_path = Path(args.embeddings) if args.embeddings else embeddings_path(out_dir, meta["encoder"], meta["size"])
    alignment = check_alignment(inputs, emb_path)
    device = pick_device(args.device)
    logger.info("Tokens %s on %s: %d studies x %d tokens x %d; text features %s; train %d, val %d",
                Path(args.tokens).name, device, *inputs.tokens.shape,
                None if inputs.text is None else inputs.text.shape[1], len(inputs.rows(TRAIN)), len(inputs.rows(VAL)))

    eval_rows = np.flatnonzero(np.isin(inputs.split, [VAL, TEST]))
    clean_ids = load_manifest(out_dir).query("split == @TEST and in_test_clean")["study_id"]
    clean_rows = np.flatnonzero(np.isin(inputs.ids, clean_ids))
    runs, infos, shifts = {}, {}, []
    for arm in arms:
        logits = np.zeros((len(eval_rows), len(LABELS)))
        infos[arm] = []
        for seed in seeds:
            key = seed_key(arm, seed, inputs, eval_rows, recipe, Path(args.tokens))
            cache = out_dir / "runs" / "attend" / prefix / f"{arm}_s{seed}.npz"
            cached = load_seed(cache, key)
            if cached is not None:
                seed_logits, info, shift = cached
                logger.info("  %s seed %d: finished earlier (%s)", arm, seed, cache)
            else:
                start = time.time()
                model, info = fit_head(arm, inputs, seed, recipe, device)
                seed_logits = predict_head(model, inputs, eval_rows, device)
                shift = attention_shift(model, inputs, clean_rows, device) if arm in STEER_ARMS and len(clean_rows) \
                    else None
                info["seconds"] = round(time.time() - start, 1)
                save_seed(cache, key, seed_logits, info, shift)
                logger.info("  %s seed %d: best epoch %d of %d (by %s), val macro AUROC %s, %.0fs", arm, seed,
                            info["best_epoch"], info["epochs_run"], info["selected_by"],
                            fmt(info["best_val_macro_auroc"] if info["best_val_macro_auroc"] is not None
                                else float("nan")), info["seconds"])
            logits += seed_logits / len(seeds)
            if shift is not None:
                shifts.append(shift)
            infos[arm].append(info)
        run = f"{prefix}_attn_{arm}"
        write_predictions(out_dir, run, inputs.ids[eval_rows], inputs.split[eval_rows], logits, {
            "kind": "attend", "arm": arm, "tokens": str(args.tokens), "context": str(ctx_path) if arm in TEXT_ARMS else None,
            **{k: meta.get(k) for k in ("encoder", "size", "revision", "torch", "transformers")},
            "token_grid": meta.get("grid"), "seeds": seeds, "recipe": asdict(recipe),
            "text_features": inputs.text_stats if arm in TEXT_ARMS else None, "per_seed": infos[arm],
            "image_features": "frozen",
        })
        runs[arm] = run
    shift = np.mean(shifts, axis=0) if shifts else None
    path, gate = write_report(out_dir, prefix, runs, infos, shift, recipe, alignment, args.n_boot, time.time() - t0)
    logger.info("Head validity check: %s", gate["check"])
    if split_scheme(out_dir) != "official":  # M5 drops h2: G4b is defined on the v2 cohort only
        logger.info("G4b (h2 - h1, test_clean): %s -> %s",
                    fmt_ci(gate["cmp"]["diff"], gate["cmp"]["diff_ci"]) if gate["cmp"] else "-", gate["verdict"])
    if shift is not None:
        logger.info("Mean attention shift from text (h2, test_clean): %.3f", float(shift.mean()))
    logger.info("Report: %s", path)
    return {"runs": runs, "gate": gate, "report": path, "shift": shift}


if __name__ == "__main__":
    main()
