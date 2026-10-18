"""
M2: zero-shot classification with a vision-language model (docs/V2_PLAN.md §6).

    python -m v2.embed --model medsiglip            # image embeddings first
    python -m v2.zero_shot --model medsiglip

For each class, score = cos(image, "<class>") - cos(image, "no <class>"). One
prompt template is chosen for all classes by val macro AUROC (never on a test
set). No labels are used for anything else.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch

from .common import (
    LABELS,
    MIN_COUNT_VAL,
    TEST,
    VAL,
    default_out_dir,
    label_matrix,
    load_manifest,
    measurable,
    pick_device,
    setup_logging,
    targets_and_mask,
    write_predictions,
)
from .embed import embeddings_path, load_npz
from .encoders import REGISTRY, SiglipEncoder, load_encoder
from .evaluate import auroc

logger = logging.getLogger("v2.zero_shot")

# (positive, negative) prompt pairs; {} is the finding.
TEMPLATES = {
    "plain": ("{}", "no {}"),
    "xray": ("a chest x-ray showing {}", "a chest x-ray showing no {}"),
    "evidence": ("{} is present", "no evidence of {}"),
}
# How each class is phrased in a prompt. "Pleural Other" is mostly pleural thickening.
PHRASES = {label: label.lower() for label in LABELS} | {"Pleural Other": "pleural thickening"}


def _normalize(x: np.ndarray) -> np.ndarray:
    return x / np.clip(np.linalg.norm(x, axis=1, keepdims=True), 1e-12, None)


def text_embeddings(enc: SiglipEncoder, template: tuple[str, str]) -> tuple[np.ndarray, np.ndarray]:
    pos = [template[0].format(PHRASES[label]) for label in LABELS]
    neg = [template[1].format(PHRASES[label]) for label in LABELS]
    t = enc.encode_text(pos + neg).float().cpu().numpy()
    return _normalize(t[: len(LABELS)]), _normalize(t[len(LABELS):])


def zero_shot_scores(img: np.ndarray, pos: np.ndarray, neg: np.ndarray) -> np.ndarray:
    """[n, 12] = cos(image, positive prompt) - cos(image, negative prompt)."""
    z = _normalize(img.astype(np.float64))
    return z @ pos.T.astype(np.float64) - z @ neg.T.astype(np.float64)


def macro_val_auroc(scores: np.ndarray, raw: np.ndarray) -> float:
    Y, M = targets_and_mask(raw, "v2")
    classes = measurable(raw, "v2", MIN_COUNT_VAL)
    vals = [auroc(Y[M[:, k], k], scores[M[:, k], k]) for k in (LABELS.index(c) for c in classes)]
    return float(np.nanmean(vals)) if vals else float("nan")


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="medsiglip", choices=[n for n, s in REGISTRY.items() if s.kind == "siglip"])
    parser.add_argument("--size", type=int, help="Image size used by v2.embed (default: native)")
    parser.add_argument("--embeddings", help="Image embeddings .npz (default: embeddings/{model}_{size}.npz)")
    parser.add_argument("--out", default=None, help="v2 output dir (default: $OUTPUT_PATH/v2 or output/v2)")
    parser.add_argument("--run", help="Run name (default: {model}_{size}_zeroshot)")
    parser.add_argument("--device", default="auto")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)
    setup_logging(args.verbose)

    out_dir = Path(args.out) if args.out else default_out_dir()
    size = args.size or REGISTRY[args.model].native_size
    emb_path = Path(args.embeddings) if args.embeddings else embeddings_path(out_dir, args.model, size)
    if not emb_path.exists():
        raise SystemExit(f"{emb_path} not found: run `python -m v2.embed --model {args.model}` first")
    ids, emb, meta = load_npz(emb_path)
    if meta.get("encoder") != args.model:
        raise SystemExit(f"{emb_path} was made with {meta.get('encoder')!r}, not {args.model!r}")

    manifest = load_manifest(out_dir)
    pos_index = pd.Series(np.arange(len(ids)), index=ids)
    rows = manifest[manifest["split"].isin([VAL, TEST]) & manifest["study_id"].isin(ids)]
    rows = rows.sort_values("study_id").reset_index(drop=True)
    img = emb[pos_index.loc[rows["study_id"]].to_numpy()]
    is_val = (rows["split"] == VAL).to_numpy()

    enc = load_encoder(args.model, size, pick_device(args.device))
    with torch.inference_mode():
        texts = {name: text_embeddings(enc, tpl) for name, tpl in TEMPLATES.items()}

    raw_val = label_matrix(rows[is_val])
    val_scores = {name: macro_val_auroc(zero_shot_scores(img[is_val], *t), raw_val) for name, t in texts.items()}
    for name, score in val_scores.items():
        logger.info("Template %-9s val macro AUROC %.4f", name, score)
    finite = {k: v for k, v in val_scores.items() if np.isfinite(v)}
    chosen = max(finite, key=finite.get) if finite else "xray"
    logger.info("Chosen template: %s", chosen)

    run = args.run or f"{args.model}_{size}_zeroshot"
    write_predictions(out_dir, run, rows["study_id"], rows["split"], zero_shot_scores(img, *texts[chosen]), {
        "kind": "zero_shot", "embeddings": str(emb_path), **{k: meta.get(k) for k in ("encoder", "size", "revision", "torch", "transformers")},
        "template": {"name": chosen, "positive": TEMPLATES[chosen][0], "negative": TEMPLATES[chosen][1]},
        "val_macro_auroc_by_template": val_scores, "phrases": PHRASES,
    })
    logger.info("Next: python -m v2.evaluate --run %s", run)


if __name__ == "__main__":
    main()
