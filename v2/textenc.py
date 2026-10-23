"""
M6: a language model for the text written before the X-ray (docs/V2_PLAN.md §6 M6).

    python -m v2.textenc --encoder biomedbert --out output/v2_bench

For each text source in context.parquet (indication, chief complaint, prior report)
it writes embeddings/text_{encoder}_{source}.npz: per study, the mean of the
encoder's last-layer token embeddings (zeros where the source is empty) and
whether the source had text. v2.fuse --text-encoder uses them in place of TF-IDF;
v2.textft fine-tunes the same encoders. A source whose file already covers the
studies is skipped, so an interrupted run resumes by source. Only counts and
timings are logged.

Encoders, with pinned revisions:

    bert          google-bert/bert-base-uncased: Wikipedia and books (MMBT's backbone)
    biomedbert    microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext: PubMed
    clinicalbert  emilyalsentzer/Bio_ClinicalBERT: MIMIC-III clinical notes (see the plan's caveat)
    tiny-bert     test only: a random two-layer BERT with a hashing tokenizer
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
import zlib
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from .common import amp_dtype, default_out_dir, git_commit, load_manifest, pick_device, seed_everything, setup_logging
from .context import TEXT_SOURCES

logger = logging.getLogger("v2.textenc")


@dataclass(frozen=True)
class TextSpec:
    name: str
    repo: str          # Hugging Face repo id, or "random" for the test encoder
    revision: str
    note: str


TEXT_ENCODERS: dict[str, TextSpec] = {
    spec.name: spec
    for spec in (
        TextSpec("bert", "google-bert/bert-base-uncased", "86b5e0934494bd15c9632b12f734a8a67f723594",
                 "general English: Wikipedia and books"),
        TextSpec("biomedbert", "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext",
                 "e1354b7a3a09615f6aba48dfad4b7a613eef7062", "biomedical literature: PubMed abstracts and full text"),
        TextSpec("clinicalbert", "emilyalsentzer/Bio_ClinicalBERT", "d5892b39a4adaed74b92212a44081509db72f87b",
                 "clinical notes: MIMIC-III (may include some of these patients' reports, unlabeled)"),
        TextSpec("tiny-bert", "random", "-", "test only: random tiny BERT"),
    )
}
MEDICAL = ("biomedbert", "clinicalbert")
# Tokens kept per source: indications and complaints are short, prior reports are not.
MAX_TOKENS = {"indication": 128, "chief_complaint": 128, "prior_report": 512}


class HashWordTokenizer:
    """Test tokenizer: [CLS] words [SEP], words hashed to stable ids, 0 for padding (BERT's pad id)."""

    def __init__(self, vocab_size: int):
        self.vocab_size = vocab_size

    def __call__(self, texts: list[str], max_length: int) -> dict:
        rows = [[1] + [3 + zlib.crc32(w.encode()) % (self.vocab_size - 3) for w in t.lower().split()][: max_length - 2]
                + [2] for t in texts]
        width = max(len(r) for r in rows)
        ids = torch.zeros(len(rows), width, dtype=torch.long)
        for i, r in enumerate(rows):
            ids[i, : len(r)] = torch.tensor(r, dtype=torch.long)
        return {"input_ids": ids, "attention_mask": (ids != 0).long()}

    def token_counts(self, texts: list[str]) -> np.ndarray:
        return np.array([len(t.split()) + 2 for t in texts])


class TextEncoder(nn.Module):
    """A BERT-style encoder and its tokenizer: tokenize() pads to the longest text in the batch."""

    def __init__(self, spec: TextSpec, model: nn.Module, tokenizer):
        super().__init__()
        self.spec, self.model, self.tokenizer = spec, model, tokenizer
        self.width = int(model.config.hidden_size)

    @property
    def name(self) -> str:
        return self.spec.name

    def tokenize(self, texts: list[str], max_length: int) -> dict:
        if isinstance(self.tokenizer, HashWordTokenizer):
            return self.tokenizer(texts, max_length)
        enc = self.tokenizer(texts, padding="longest", truncation=True, max_length=max_length, return_tensors="pt")
        return {"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"]}

    def token_counts(self, texts: list[str]) -> np.ndarray:
        """Tokens per text before truncation (to report how many were cut)."""
        if isinstance(self.tokenizer, HashWordTokenizer):
            return self.tokenizer.token_counts(texts)
        # Counting means encoding without truncation: verbose=False keeps the tokenizer from warning about
        # sequences longer than the model takes (tokenize() and embed_texts() always truncate).
        return np.array([len(ids) for ids in self.tokenizer(texts, add_special_tokens=True, verbose=False)["input_ids"]])

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Last-layer hidden states [B, T, width]."""
        return self.model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state

    def describe(self) -> dict:
        import transformers

        return {"text_encoder": self.spec.name, "repo": self.spec.repo, "revision": self.spec.revision,
                "width": self.width, "torch": torch.__version__, "transformers": transformers.__version__}


def load_text_encoder(name: str, device=None) -> TextEncoder:
    from transformers import AutoModel, AutoTokenizer, BertConfig, BertModel

    if name not in TEXT_ENCODERS:
        raise SystemExit(f"Unknown text encoder {name!r}; choose from {sorted(TEXT_ENCODERS)}")
    spec = TEXT_ENCODERS[name]
    if spec.repo == "random":
        torch.manual_seed(0)
        cfg = BertConfig(vocab_size=512, hidden_size=32, num_hidden_layers=2, num_attention_heads=2,
                         intermediate_size=64, max_position_embeddings=512)
        enc = TextEncoder(spec, BertModel(cfg, add_pooling_layer=False), HashWordTokenizer(cfg.vocab_size))
    else:
        tokenizer = AutoTokenizer.from_pretrained(spec.repo, revision=spec.revision)
        model = AutoModel.from_pretrained(spec.repo, revision=spec.revision)
        enc = TextEncoder(spec, model, tokenizer)
    return enc.to(device) if device is not None else enc


def mean_pool(hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Average of the token embeddings, padding left out: [B, T, D] -> [B, D]."""
    m = mask.unsqueeze(-1).to(hidden.dtype)
    return (hidden * m).sum(1) / m.sum(1).clamp(min=1.0)


@torch.inference_mode()
def embed_texts(enc: TextEncoder, texts: list[str], max_length: int, batch_size: int, device,
                dtype=None) -> np.ndarray:
    """Mean-pooled embeddings [n, width] (float32), in input order; texts run longest first in padded batches."""
    enc.eval()
    out = np.zeros((len(texts), enc.width), dtype=np.float32)
    order = np.argsort([-len(t) for t in texts], kind="stable")
    autocast = torch.autocast(device.type, dtype=dtype) if dtype is not None else nullcontext()
    for start in range(0, len(order), batch_size):
        rows = order[start:start + batch_size]
        batch = enc.tokenize([texts[i] for i in rows], max_length)
        ids, mask = batch["input_ids"].to(device), batch["attention_mask"].to(device)
        with autocast:
            pooled = mean_pool(enc(ids, mask).float(), mask)
        out[rows] = pooled.cpu().numpy()
    return out


# ---------------------------------------------------------------------------
# Files
# ---------------------------------------------------------------------------

def text_path(out_dir: Path, encoder: str, source: str) -> Path:
    return Path(out_dir) / "embeddings" / f"text_{encoder}_{source}.npz"


def save_text(path: Path, ids: np.ndarray, emb: np.ndarray, present: np.ndarray, meta: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp.npz")
    np.savez(tmp, study_id=np.asarray(ids, dtype=np.int64), emb=np.asarray(emb, dtype=np.float32),
             present=np.asarray(present, dtype=bool), meta=np.array(json.dumps(meta)))
    os.replace(tmp, path)


def load_text(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """(study_ids, embeddings [n, width], has-text mask, meta)."""
    if not Path(path).exists():
        raise SystemExit(f"{path} not found: run `python -m v2.textenc --encoder ...` first")
    with np.load(path, allow_pickle=False) as f:
        return (f["study_id"].astype(np.int64), f["emb"].astype(np.float32), f["present"].astype(bool),
                json.loads(str(f["meta"])))


def lm_block(out_dir: Path, encoder: str, source: str, ids: np.ndarray, train: np.ndarray) -> tuple[np.ndarray, dict]:
    """
    One source's frozen embeddings as probe features, in the row order of `ids`: standardized with the train
    rows that have text, zeros where there is none, plus a presence indicator (the TF-IDF block's layout).
    """
    t_ids, emb, present, meta = load_text(text_path(out_dir, encoder, source))
    pos = pd.Series(np.arange(len(t_ids)), index=t_ids).reindex(ids)
    missing = int(pos.isna().sum())
    if missing:
        raise SystemExit(f"{missing} studies have no {source} embedding from {encoder}: re-run v2.textenc")
    pos = pos.to_numpy(np.int64)
    X, has = emb[pos].astype(np.float64), present[pos]
    fit = np.asarray(train, dtype=bool) & has
    if fit.sum() < 3:
        return has[:, None].astype(np.float64), {"encoder": encoder, "train_docs": int(fit.sum()), "width": 0}
    mean, std = X[fit].mean(0), X[fit].std(0)
    X = (X - mean) / np.maximum(std, 1e-6)
    X[~has] = 0.0
    return np.concatenate([X, has[:, None]], axis=1), {"encoder": encoder, "train_docs": int(fit.sum()),
                                                        "width": int(emb.shape[1]), "revision": meta.get("revision")}


def context_texts(out_dir: Path, ctx_path: Optional[Path] = None) -> pd.DataFrame:
    """The manifest's studies (sorted by study_id) with their text sources; no text is the empty string."""
    manifest = load_manifest(out_dir)
    ctx_path = Path(ctx_path) if ctx_path else Path(out_dir) / "context.parquet"
    if not ctx_path.exists():
        raise SystemExit(f"{ctx_path} not found: run `python -m v2.context` first")
    ctx = pd.read_parquet(ctx_path, columns=["study_id", *TEXT_SOURCES]).drop_duplicates("study_id")
    ids = np.sort(manifest["study_id"].drop_duplicates().to_numpy(np.int64))
    ctx = ctx.set_index("study_id").reindex(ids).rename_axis("study_id").reset_index()
    for src in TEXT_SOURCES:
        ctx[src] = ctx[src].fillna("").astype(str).str.strip()
    return ctx


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--encoder", required=True, choices=sorted(TEXT_ENCODERS))
    parser.add_argument("--out", default=None, help="v2 output dir (default: $OUTPUT_PATH/v2 or output/v2)")
    parser.add_argument("--context", help="context.parquet from v2.context (default: OUT/context.parquet)")
    parser.add_argument("--sources", default=",".join(TEXT_SOURCES))
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="auto", help="auto | cuda | mps | cpu")
    parser.add_argument("--precision", default="auto", choices=["auto", "bf16", "fp16", "fp32"])
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)
    setup_logging(args.verbose)
    seed_everything(0)
    out_dir = Path(args.out) if args.out else default_out_dir()
    ctx = context_texts(out_dir, args.context)
    ids = ctx["study_id"].to_numpy(np.int64)
    sources = [s.strip() for s in args.sources.split(",")]
    unknown = [s for s in sources if s not in TEXT_SOURCES]
    if unknown:
        raise SystemExit(f"Unknown sources {unknown}; choose from {list(TEXT_SOURCES)}")

    device = pick_device(args.device)
    enc = None
    for src in sources:
        path = text_path(out_dir, args.encoder, src)
        if path.exists():
            done_ids, _, _, done_meta = load_text(path)
            if done_meta.get("revision") == TEXT_ENCODERS[args.encoder].revision and np.array_equal(done_ids, ids):
                logger.info("%s already covers these %d studies; delete it to recompute", path.name, len(ids))
                continue
        if enc is None:
            enc = load_text_encoder(args.encoder, device)
            logger.info("Text encoder %s (%s) on %s, width %d", enc.name, enc.spec.repo, device, enc.width)
        texts = ctx[src].tolist()
        present = np.array([bool(t) for t in texts])
        rows = np.flatnonzero(present)
        emb = np.zeros((len(texts), enc.width), dtype=np.float32)
        t0 = time.time()
        if len(rows):
            kept = [texts[i] for i in rows]
            cut = int((enc.token_counts(kept) > MAX_TOKENS[src]).sum())
            emb[rows] = embed_texts(enc, kept, MAX_TOKENS[src], args.batch_size, device,
                                    amp_dtype(device, args.precision))
            logger.info("%s: %d of %d studies have text; %d cut at %d tokens; %.1f texts/s", src, len(rows), len(texts),
                        cut, MAX_TOKENS[src], len(rows) / max(time.time() - t0, 1e-9))
        else:
            logger.warning("%s: no study has text", src)
        save_text(path, ids, emb, present, {**enc.describe(), "source": src, "max_tokens": MAX_TOKENS[src],
                                            "n": int(len(ids)), "with_text": int(len(rows)), "commit": git_commit()})
        logger.info("Wrote %s", path)


if __name__ == "__main__":
    main()
