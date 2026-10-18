"""
M1: frozen embeddings for every manifest study (docs/V2_PLAN.md §6).

    python -m v2.embed --model rad-dino --size 224        # CPU/MPS pilot
    python -m v2.embed --model rad-dino --size 518        # GPU

Writes embeddings/{model}_{size}.npz with `study_id`, `emb` (float32) and a
JSON `meta` string. Progress is saved to a .partial.npz every few minutes, so
an interrupted run resumes where it stopped.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from .common import (
    DEFAULT_SEED,
    TRAIN,
    amp_dtype,
    default_out_dir,
    git_commit,
    load_manifest,
    pick_device,
    setup_logging,
)
from .data import ImageDataset, image_paths, loader_kwargs
from .encoders import REGISTRY, Encoder, load_encoder

logger = logging.getLogger("v2.embed")


def embeddings_path(out_dir: Path, name: str, size: int) -> Path:
    return Path(out_dir) / "embeddings" / f"{name}_{size}.npz"


def save_npz(path: Path, study_ids: np.ndarray, emb: np.ndarray, meta: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp.npz")
    np.savez(tmp, study_id=np.asarray(study_ids, dtype=np.int64), emb=np.asarray(emb, dtype=np.float32),
             meta=np.array(json.dumps(meta)))
    os.replace(tmp, path)


def load_npz(path: Path) -> tuple[np.ndarray, np.ndarray, dict]:
    with np.load(path, allow_pickle=False) as f:
        return f["study_id"].astype(np.int64), f["emb"].astype(np.float32), json.loads(str(f["meta"]))


def select_rows(manifest: pd.DataFrame, splits: list[str], max_train: int, smoke: bool, seed: int) -> pd.DataFrame:
    rows = manifest[manifest["split"].isin(splits)]
    if max_train and (rows["split"] == TRAIN).sum() > max_train:
        train = rows[rows["split"] == TRAIN].sample(n=max_train, random_state=seed)
        rows = pd.concat([train, rows[rows["split"] != TRAIN]])
    if smoke:
        rows = rows.groupby("split", group_keys=False).head(16)
    return rows.sort_values("study_id").reset_index(drop=True)


@torch.inference_mode()
def run_embedding(
    enc: Encoder,
    rows: pd.DataFrame,
    out_dir: Path,
    batch_size: int,
    workers: int,
    device,
    precision: str = "auto",
    partial: Optional[Path] = None,
    meta: Optional[dict] = None,
    save_every_s: float = 300.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Embed `rows` (resuming from `partial` if it exists); returns (study_ids, emb)."""
    done_ids: list[np.ndarray] = []
    done_emb: list[np.ndarray] = []
    if partial is not None and partial.exists():
        ids, emb, old_meta = load_npz(partial)
        if old_meta.get("encoder") == enc.name and old_meta.get("size") == enc.size:
            done_ids.append(ids)
            done_emb.append(emb)
            logger.info("Resuming: %d studies already embedded", len(ids))
    done = set(np.concatenate(done_ids).tolist()) if done_ids else set()
    todo = rows[~rows["study_id"].isin(done)]

    dtype = amp_dtype(device, precision)
    autocast = torch.autocast(device.type, dtype=dtype) if dtype is not None else nullcontext()
    ds = ImageDataset(image_paths(out_dir, todo), enc.eval_transform())
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, **loader_kwargs(workers, device))
    todo_ids = todo["study_id"].to_numpy(np.int64)

    new_ids, new_emb = [], []
    start = last_save = time.time()
    n_done = 0
    for x, idx in dl:
        with autocast:
            feats = enc(x.to(device, non_blocking=True))
        feats = feats.float().cpu().numpy()
        if not np.isfinite(feats).all():
            raise RuntimeError(f"non-finite embeddings for studies {todo_ids[idx.numpy()][:5]}")
        new_emb.append(feats)
        new_ids.append(todo_ids[idx.numpy()])
        n_done += len(idx)
        now = time.time()
        if now - last_save > save_every_s or n_done == len(todo):
            rate = n_done / max(now - start, 1e-6)
            logger.info("  %d / %d  (%.1f img/s, ~%.0f min left)", n_done, len(todo), rate,
                        (len(todo) - n_done) / max(rate, 1e-6) / 60)
            if partial is not None and n_done < len(todo):
                save_npz(partial, np.concatenate(done_ids + new_ids), np.concatenate(done_emb + new_emb),
                         meta or {})
            last_save = now

    all_ids = np.concatenate(done_ids + new_ids) if (done_ids or new_ids) else np.zeros(0, np.int64)
    all_emb = np.concatenate(done_emb + new_emb) if (done_emb or new_emb) else np.zeros((0, enc.dim), np.float32)
    order = np.argsort(all_ids)
    return all_ids[order], all_emb[order]


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, choices=sorted(REGISTRY))
    parser.add_argument("--size", type=int, help="Input size in px (default: the model's native size)")
    parser.add_argument("--out", default=None, help="v2 output dir (default: $OUTPUT_PATH/v2 or output/v2)")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=min(4, os.cpu_count() or 1))
    parser.add_argument("--device", default="auto", help="auto | cuda | mps | cpu")
    parser.add_argument("--precision", default="auto", choices=["auto", "bf16", "fp16", "fp32"])
    parser.add_argument("--splits", default="train,val,test")
    parser.add_argument("--max-train", type=int, default=0, help="Embed only N training studies (CPU pilots)")
    parser.add_argument("--smoke", action="store_true", help="16 studies per split, to test the setup")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)
    setup_logging(args.verbose)

    out_dir = Path(args.out) if args.out else default_out_dir()
    manifest = load_manifest(out_dir)
    rows = select_rows(manifest, args.splits.split(","), args.max_train, args.smoke, args.seed)
    device = pick_device(args.device)
    enc = load_encoder(args.model, args.size, device)
    logger.info("Encoder %s at %d px on %s: %d studies -> %d-d embeddings", enc.name, enc.size, device,
                len(rows), enc.dim)

    suffix = "_smoke" if args.smoke else ""
    final = embeddings_path(out_dir, enc.name, enc.size).with_name(f"{enc.name}_{enc.size}{suffix}.npz")
    if final.exists():
        done_ids, _, done_meta = load_npz(final)
        same = (done_meta.get("encoder"), done_meta.get("size")) == (enc.name, enc.size)
        if same and rows["study_id"].isin(done_ids).all():
            logger.info("%s already covers these %d studies; delete it to recompute", final, len(rows))
            return
    partial = final.with_name(final.stem + ".partial.npz")
    meta = {**enc.describe(), "commit": git_commit(), "n": int(len(rows)), "max_train": args.max_train,
            "device": str(device), "precision": args.precision}
    ids, emb = run_embedding(enc, rows, out_dir, args.batch_size, args.workers, device, args.precision,
                             partial, meta)
    save_npz(final, ids, emb, meta)
    if partial.exists():
        partial.unlink()
    logger.info("Wrote %s (%d x %d)", final, *emb.shape)


if __name__ == "__main__":
    main()
