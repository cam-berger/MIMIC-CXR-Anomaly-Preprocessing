"""
M6: fine-tune a language model on one text source (docs/ANALYSIS.md, M6).

    python -m v2.textft --encoder biomedbert --out output/v2_bench        # the indication, seed 0
    python -m v2.textft --report --out output/v2_bench                    # reports/m6_text.md and gate G6

The encoder and a linear head on its [CLS] token predict the 12 findings from that
text alone: masked BCE (uncertain labels ignored), AdamW (lr 2e-5, weight decay
0.01), batch 32, 64 tokens, 3 epochs (10% warm-up, then cosine), keeping the epoch
with the best val macro AUROC. Studies without text get an empty input, as in M4.
Writes predictions/{encoder}_ft_{source}_s{seed}.parquet (val and test). Only
aggregates are logged.

--report reads the M6 runs already written and writes reports/m6_text.md: red
flags, gate G6 (the medical encoder with the better val score minus bert, both
fine-tuned on the indication), the reported comparisons and the adoption rule.
"""

from __future__ import annotations

import argparse
import logging
import math
import time
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

from .common import LABELS, TEST, TRAIN, VAL, amp_dtype, default_out_dir, git_commit, label_matrix, load_manifest
from .common import pick_device, seed_everything, setup_logging, targets_and_mask, write_predictions
from .context import TEXT_SOURCES
from .evaluate import _get, compare_runs, evaluate_run, fmt, fmt_ci
from .finetune import _start_at_priors, masked_bce, val_macro_auroc, warmup_cosine
from .textenc import MEDICAL, TEXT_ENCODERS, TextEncoder, context_texts, load_text_encoder

logger = logging.getLogger("v2.textft")


@dataclass(frozen=True)
class TextRecipe:
    """The pre-registered fine-tuning recipe (M6)."""

    lr: float = 2e-5
    weight_decay: float = 0.01
    batch_size: int = 32
    max_tokens: int = 64
    epochs: int = 3
    warmup: float = 0.1
    dropout: float = 0.1
    clip: float = 1.0


class TextClassifier(nn.Module):
    """A text encoder, then dropout and one linear layer on the [CLS] token -> 12 logits."""

    def __init__(self, enc: TextEncoder, prevalence: Optional[np.ndarray] = None, dropout: float = 0.1):
        super().__init__()
        self.enc = enc
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(enc.width, len(LABELS))
        _start_at_priors(self.head.bias, prevalence)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        return self.head(self.dropout(self.enc(input_ids, attention_mask)[:, 0]))


@torch.inference_mode()
def predict_text(model: TextClassifier, texts: list[str], rows: np.ndarray, max_tokens: int, device, dtype=None,
                 batch_size: int = 128) -> np.ndarray:
    """Logits [len(rows), 12] in the order of `rows`."""
    model.eval()
    autocast = torch.autocast(device.type, dtype=dtype) if dtype is not None else nullcontext()
    out = []
    for start in range(0, len(rows), batch_size):
        batch = model.enc.tokenize([texts[i] for i in rows[start:start + batch_size]], max_tokens)
        with autocast:
            logits = model(batch["input_ids"].to(device), batch["attention_mask"].to(device))
        out.append(logits.float().cpu().numpy())
    return np.concatenate(out) if out else np.zeros((0, len(LABELS)), np.float32)


def fit_text(enc: TextEncoder, texts: list[str], raw: np.ndarray, split: np.ndarray, recipe: TextRecipe, seed: int,
             device, dtype=None, steps_per_epoch: int = 0, max_bad_steps: float = 0.01) -> tuple[TextClassifier, dict]:
    """Fine-tune on the train rows; returns the model at its best val epoch and the training record."""
    seed_everything(seed)
    train, val = np.flatnonzero(split == TRAIN), np.flatnonzero(split == VAL)
    Y, M = targets_and_mask(raw, "v2")
    prevalence = (Y[train] * M[train]).sum(0) / np.maximum(M[train].sum(0), 1)
    model = TextClassifier(enc, prevalence, recipe.dropout).to(device)
    # BERT's recipe: no weight decay on biases and LayerNorm weights (the 1-D parameters).
    params = list(model.parameters())
    optimizer = torch.optim.AdamW([{"params": [p for p in params if p.ndim > 1], "weight_decay": recipe.weight_decay},
                                   {"params": [p for p in params if p.ndim <= 1], "weight_decay": 0.0}], lr=recipe.lr)
    per_epoch = math.ceil(len(train) / recipe.batch_size)
    if steps_per_epoch:
        per_epoch = min(per_epoch, steps_per_epoch)
    total = per_epoch * recipe.epochs
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, warmup_cosine(int(recipe.warmup * total), total))
    autocast = (lambda: torch.autocast(device.type, dtype=dtype)) if dtype is not None else nullcontext
    rng = np.random.default_rng(seed)
    best, best_state, best_epoch, history, steps, bad = -math.inf, None, -1, [], 0, 0
    Yt, Mt = torch.as_tensor(Y, dtype=torch.float32), torch.as_tensor(M)
    for epoch in range(recipe.epochs):
        model.train()
        order, t0, loss_sum, n_loss = rng.permutation(train), time.time(), 0.0, 0
        for b in range(per_epoch):
            rows = order[b * recipe.batch_size:(b + 1) * recipe.batch_size]
            batch = enc.tokenize([texts[i] for i in rows], recipe.max_tokens)
            with autocast():
                logits = model(batch["input_ids"].to(device), batch["attention_mask"].to(device))
            loss = masked_bce(logits, Yt[rows].to(device), Mt[rows].to(device))
            steps += 1
            if not torch.isfinite(loss):
                bad += 1
                optimizer.zero_grad(set_to_none=True)
                if steps > 20 and bad > max_bad_steps * steps:
                    raise SystemExit(f"{bad} non-finite losses in {steps} steps: aborting")
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), recipe.clip)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            loss_sum, n_loss = loss_sum + loss.item(), n_loss + 1
            if (b + 1) % 500 == 0:
                logger.info("  epoch %d step %d/%d  loss %.4f  %.0f texts/s", epoch, b + 1, per_epoch,
                            loss_sum / max(n_loss, 1), (b + 1) * recipe.batch_size / (time.time() - t0))
        val_logits = predict_text(model, texts, val, recipe.max_tokens, device, dtype)
        val_auc, per = val_macro_auroc(val_logits, raw[val])
        record = {"epoch": epoch, "train_loss": loss_sum / max(n_loss, 1), "val_macro_auroc": val_auc,
                  "val_classes": len(per), "seconds": round(time.time() - t0, 1)}
        history.append(record)
        improved = np.isfinite(val_auc) and val_auc > best
        if improved or best_state is None:
            best, best_epoch = (val_auc if np.isfinite(val_auc) else best), epoch
            best_state = {k: v.detach().to("cpu", copy=True) for k, v in model.state_dict().items()}
        logger.info("Epoch %d: train loss %.4f, val macro AUROC %.4f (%d classes)%s, %.0fs", epoch,
                    record["train_loss"], val_auc, len(per), "  *best*" if improved else "", record["seconds"])
    model.load_state_dict(best_state)
    return model, {"best_epoch": best_epoch, "best_val_macro_auroc": best, "history": history,
                   "steps": steps, "bad_steps": bad, "per_epoch": per_epoch}


# ---------------------------------------------------------------------------
# Report: gate G6 and the comparisons (M6)
# ---------------------------------------------------------------------------

def ft_run(encoder: str, source: str = "indication", seed: int = 0) -> str:
    return f"{encoder}_ft_{source}_s{seed}"


def write_report(out_dir: Path, prefix: str = "rad-dino_518", image_run: str = "rad-dino_518_ft_attn_s0",
                 n_boot: int = 1000, general: str = "bert", medical: tuple = MEDICAL) -> dict:
    from .fuse import text_flags
    from .lineup import PUBLISHED

    pred = Path(out_dir) / "predictions"
    exists = lambda run: (pred / f"{run}.parquet").exists()  # noqa: E731
    reps: dict[str, dict] = {}

    def rep(run: str) -> Optional[dict]:
        if run not in reps:
            reps[run] = evaluate_run(out_dir, run, n_boot) if exists(run) else None
        return reps[run]

    def res(run: str, test_set: str = "test_clean", metric: str = "v2") -> Optional[dict]:
        r = rep(run)
        return _get(r, test_set, metric) if r else None

    def val_score(run: str) -> float:
        r = res(run, "val")
        return r["macro_auroc"] if r and np.isfinite(r["macro_auroc"]) else -math.inf

    def cell(r: Optional[dict]) -> str:
        return fmt_ci(r["macro_auroc"], r["macro_auroc_ci"]) if r else "-"

    def cmp_cells(a: str, b: str) -> list[str]:
        if not (exists(a) and exists(b)):
            return ["-", "-"]
        return [fmt_ci(c["diff"], c["diff_ci"]) for c in
                (compare_runs(out_dir, a, b, "test_clean", m, n_boot) for m in ("v2", "u0"))]

    lm = [e for e in TEXT_ENCODERS if exists(ft_run(e)) or exists(f"{prefix}_ctx_indication_{e}")]
    lines = [f"# M6: a language model for the pre-imaging text ({prefix})", "",
             "Macro AUROC [95% patient-bootstrap CI] on the official test split, v2 policy unless noted; "
             "differences are paired (docs/ANALYSIS.md, M6).", "", "## Red flags (read these first)", ""]
    image = res(f"{prefix}_ctx_image")
    flags = []
    for e in lm:
        for run in (f"{prefix}_ctx_indication_{e}", f"{prefix}_ctx_text_prior_labels_{e}", ft_run(e)):
            flags += text_flags(run, res(run), image)
    lines += [f"- **RED FLAG:** {f}. Don't read on until the extraction is inspected." for f in flags] or ["- none"]

    lines += ["", "## Text alone: the indication", "",
              "| Run | Encoder | How | val | Official test | Positive vs rest |", "|---|---|---|---|---|---|"]
    rows = [(f"{prefix}_ctx_indication", "TF-IDF", "probe")]
    rows += [(f"{prefix}_ctx_indication_{e}", e, "frozen + probe") for e in lm]
    rows += [(ft_run(e), e, "fine-tuned") for e in lm]
    for run, enc, how in rows:
        if exists(run):
            lines.append(f"| `{run}` | {enc} | {how} | {cell(res(run, 'val'))} | {cell(res(run))} | "
                         f"{cell(res(run, metric='u0'))} |")
    bert_pub = next(r for r in PUBLISHED if r.name == "BERT")
    lines += ["", f"MMBT's fine-tuned BERT, published, positive vs rest: {fmt(np.mean(list(bert_pub.per_class.values())))} "
              "over our 12 findings."]

    # Gate G6: the medical encoder with the better val score minus the general one, both fine-tuned on the indication.
    meds = [e for e in medical if exists(ft_run(e))]
    med = max(meds, key=lambda e: val_score(ft_run(e))) if meds else None
    gate = {"medical": med, "cmp": None, "verdict": f"not run: needs {general} and a medical encoder, fine-tuned"}
    if med and exists(ft_run(general)):
        c = compare_runs(out_dir, ft_run(med), ft_run(general), "test_clean", "v2", n_boot)
        gate["cmp"] = c
        gate["verdict"] = ("n/a: no measurable classes" if not np.isfinite(c["diff"]) else
                           "PASS: medical pretraining helps" if c["a_better"] else
                           "FAIL: no detectable gain from medical pretraining")
    lines += ["", "## Gate G6: medical context", ""]
    if gate["cmp"]:
        u0 = compare_runs(out_dir, ft_run(med), ft_run(general), "test_clean", "u0", n_boot)
        lines += [f"**G6** (`{ft_run(med)}`, the medical encoder with the better val score, minus `{ft_run(general)}`): "
                  f"{fmt_ci(gate['cmp']['diff'], gate['cmp']['diff_ci'])} -> **{gate['verdict']}**. "
                  f"Positive vs rest: {fmt_ci(u0['diff'], u0['diff_ci'])}."]
    else:
        lines.append(f"**G6**: {gate['verdict']}")

    lines += ["", "## Reported comparisons (not gates; uncorrected)", "",
              "| Comparison | What it asks | Difference | Positive vs rest |", "|---|---|---|---|"]
    lines.append(f"| `{ft_run(general)}` minus `{prefix}_ctx_indication` | 1. is our text gap the encoder? | "
                 + " | ".join(cmp_cells(ft_run(general), f"{prefix}_ctx_indication")) + " |")
    stacks = [(f"{image_run}_stack_indication", "TF-IDF")]
    stacks += [(f"{image_run}_stack_indication_{e}", f"{e}, frozen") for e in lm]
    stacks += [(f"{image_run}_stack_{e}_ft_indication", f"{e}, fine-tuned") for e in lm]
    stacks = [(run, how) for run, how in stacks if exists(run)]
    best = max(stacks, key=lambda s: val_score(s[0]))[0] if stacks else None
    base = f"{image_run}_stack_indication"
    if best and best != base:
        lines.append(f"| `{best}` minus `{base}` | 2. does a better text model move the like-for-like result? | "
                     + " | ".join(cmp_cells(best, base)) + " |")
    for e in lm:
        both = f"{prefix}_ctx_image_indication_tfidf-{e}"
        if exists(both):
            lines.append(f"| `{both}` minus `{prefix}_ctx_image_indication` | 3. does {e} add on top of TF-IDF? | "
                         + " | ".join(cmp_cells(both, f"{prefix}_ctx_image_indication")) + " |")

    lines += ["", f"## Image + indication: `{image_run}` with each text run, through the val combiner", "",
              "| Stacked run | Text | val | Official test | Positive vs rest |", "|---|---|---|---|---|"]
    for run, how in stacks:
        lines.append(f"| `{run}`{' **(best by val)**' if run == best else ''} | {how} | {cell(res(run, 'val'))} | "
                     f"{cell(res(run))} | {cell(res(run, metric='u0'))} |")
    lines += ["", f"Adoption rule (M6): the text representation for later work is the one behind the stack with "
              f"the best val score: **{'TF-IDF' if best == base else best or '-'}**. MMBT, published: 0.794."]

    lines += ["", "## All earlier text, frozen encoders", "", "| Run | val | Official test | Positive vs rest |",
              "|---|---|---|---|"]
    for run in [f"{prefix}_ctx_text_prior_labels", f"{image_run}_stack_text_prior_labels",
                *(r for e in lm for r in (f"{prefix}_ctx_text_prior_labels_{e}",
                                          f"{image_run}_stack_text_prior_labels_{e}"))]:
        if exists(run):
            lines.append(f"| `{run}` | {cell(res(run, 'val'))} | {cell(res(run))} | {cell(res(run, metric='u0'))} |")

    ours = res(ft_run(general), metric="u0")
    if ours:
        lines += ["", f"## 4. Per class: our fine-tuned {general} against MMBT's BERT (positive vs rest)", "",
                  "Cardiomegaly first: if MMBT's labels draw on the indication, its BERT should lead most on the "
                  "findings that indications name.", "", "| Class | Ours | MMBT's BERT | Difference |", "|---|---|---|---|"]
        per = {c["label"]: c["auroc"] for c in ours["per_class"] if c["measurable"]}
        for label in sorted(per, key=lambda lb: (lb != "Cardiomegaly", lb)):
            theirs = bert_pub.per_class.get(label)
            if theirs is not None:
                lines.append(f"| {label} | {per[label]:.3f} | {theirs:.3f} | {per[label] - theirs:+.3f} |")
    lines += ["", "Caveat (M6): `clinicalbert` was pretrained on MIMIC-III notes, which may include some of these "
              "patients' reports as unlabeled text; read a large lead for it alone with suspicion."]
    path = Path(out_dir) / "reports" / "m6_text.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")
    return {"path": path, "gate": gate, "flags": flags, "best": best}


def main(argv: Optional[list[str]] = None) -> Optional[str]:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--encoder", choices=sorted(TEXT_ENCODERS))
    parser.add_argument("--source", default="indication", choices=list(TEXT_SOURCES))
    parser.add_argument("--out", default=None, help="v2 output dir (default: $OUTPUT_PATH/v2 or output/v2)")
    parser.add_argument("--context", help="context.parquet from v2.context (default: OUT/context.parquet)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--run", help="Run name (default: {encoder}_ft_{source}_s{seed})")
    parser.add_argument("--epochs", type=int, default=TextRecipe.epochs)
    parser.add_argument("--lr", type=float, default=TextRecipe.lr)
    parser.add_argument("--batch-size", type=int, default=TextRecipe.batch_size)
    parser.add_argument("--steps-per-epoch", type=int, default=0, help="Cap optimizer steps per epoch (tests)")
    parser.add_argument("--device", default="auto", help="auto | cuda | mps | cpu")
    parser.add_argument("--precision", default="auto", choices=["auto", "bf16", "fp16", "fp32"])
    parser.add_argument("--report", action="store_true", help="Write reports/m6_text.md from the existing runs")
    parser.add_argument("--prefix", default="rad-dino_518", help="--report: the frozen image embeddings' run prefix")
    parser.add_argument("--image-run", default="rad-dino_518_ft_attn_s0", help="--report: the stacks' image run")
    parser.add_argument("--n-boot", type=int, default=1000)
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)
    setup_logging(args.verbose)
    out_dir = Path(args.out) if args.out else default_out_dir()
    if args.report:
        out = write_report(out_dir, args.prefix, args.image_run, args.n_boot)
        for flag in out["flags"]:
            logger.warning("RED FLAG: %s", flag)
        cmp = out["gate"]["cmp"]
        logger.info("G6 (%s - bert, fine-tuned on the indication): %s -> %s", out["gate"]["medical"] or "-",
                    fmt_ci(cmp["diff"], cmp["diff_ci"]) if cmp else "-", out["gate"]["verdict"])
        logger.info("Report: %s", out["path"])
        return None
    if not args.encoder:
        parser.error("--encoder is required unless --report")

    recipe = TextRecipe(epochs=args.epochs, lr=args.lr, batch_size=args.batch_size)
    ctx = context_texts(out_dir, args.context)
    manifest = load_manifest(out_dir).drop_duplicates("study_id").set_index("study_id").reindex(ctx["study_id"])
    split = manifest["split"].to_numpy()
    raw = label_matrix(manifest.reset_index())
    texts = ctx[args.source].tolist()
    device = pick_device(args.device)
    dtype = amp_dtype(device, args.precision)
    enc = load_text_encoder(args.encoder, device)
    run = args.run or ft_run(args.encoder, args.source, args.seed)
    logger.info("Run %s: %s on %s (%s), %d train / %d val studies, %d with %s text", run, enc.spec.repo, device,
                str(dtype).replace("torch.", "") if dtype else "fp32", int((split == TRAIN).sum()),
                int((split == VAL).sum()), sum(bool(t) for t in texts), args.source)
    model, info = fit_text(enc, texts, raw, split, recipe, args.seed, device, dtype, args.steps_per_epoch)
    rows = np.flatnonzero(np.isin(split, [VAL, TEST]))
    logits = predict_text(model, texts, rows, recipe.max_tokens, device, dtype)
    write_predictions(out_dir, run, ctx["study_id"].to_numpy()[rows], split[rows], logits, {
        "kind": "text_finetune", "milestone": "M6", "arm": args.source, "text_encoder": args.encoder,
        "stack_tag": f"{args.encoder}_ft_{args.source}", "seed": args.seed, "recipe": asdict(recipe),
        **enc.describe(), "commit": git_commit(), "n_train": int((split == TRAIN).sum()), **info,
    })
    logger.info("Best epoch %d (val macro AUROC %.4f); wrote predictions/%s.parquet", info["best_epoch"],
                info["best_val_macro_auroc"], run)
    return run


if __name__ == "__main__":
    main()
