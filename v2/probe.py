"""
M1: linear probes on frozen embeddings (docs/ANALYSIS.md).

    python -m v2.probe --embeddings output/v2/embeddings/rad-dino_224.npz

Per class: standardize the features (fit on train), fit an L2 logistic
regression on train, choose C from --Cs by val AUROC. Writes predictions for
the val and test studies to predictions/{run}.parquet.
"""

from __future__ import annotations

import argparse
import logging
import os
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from .common import (
    LABELS,
    TEST,
    TRAIN,
    VAL,
    default_out_dir,
    label_matrix,
    load_manifest,
    setup_logging,
    targets_and_mask,
    write_predictions,
)
from .embed import load_npz
from .evaluate import auroc

logger = logging.getLogger("v2.probe")

# Widened on 2026-10-19: with {0.01, ..., 10} every class of the first real M1 probes chose 0.01.
DEFAULT_CS = (0.0001, 0.001, 0.01, 0.1, 1.0, 10.0)


@dataclass
class LabelFit:
    label: str
    C: Optional[float]
    val_auroc: dict
    coef: np.ndarray
    intercept: float
    note: str = ""


@dataclass
class Probe:
    scaler: StandardScaler
    fits: list = field(default_factory=list)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Logits, [n, 12]."""
        Z = self.scaler.transform(X)
        W = np.stack([f.coef for f in self.fits], axis=1)
        b = np.array([f.intercept for f in self.fits])
        return Z @ W + b

    def summary(self) -> dict:
        return {f.label: {"C": f.C, "val_auroc": f.val_auroc, **({"note": f.note} if f.note else {})}
                for f in self.fits}


def edge_note(Cs, raw: dict, best_C) -> str:
    """A note when the chosen C sits at an edge of the grid and beats its neighbour (ties prefer small C)."""
    Cs = sorted(Cs)
    if len(Cs) < 2:
        return ""
    if best_C == Cs[0] and raw.get(best_C, np.nan) > raw.get(Cs[1], -np.inf):
        return "best C is the smallest tried: the grid may be too narrow"
    if best_C == Cs[-1]:
        return "best C is the largest tried: the grid may be too narrow"
    return ""


def _fit_label(label, X, y, m, Xv, yv, mv, Cs, max_iter) -> LabelFit:
    y, Xl = y[m], X[m]
    dim = X.shape[1]
    if len(np.unique(y)) < 2:
        return LabelFit(label, None, {}, np.zeros(dim), 0.0, "single class in train: constant score")
    best, best_auc, scores, raw = None, -np.inf, {}, {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        for C in Cs:
            clf = LogisticRegression(C=C, max_iter=max_iter)
            clf.fit(Xl, y)
            a = auroc(yv[mv], clf.decision_function(Xv[mv])) if mv.any() else float("nan")
            raw[C] = a
            scores[str(C)] = None if np.isnan(a) else round(float(a), 4)
            if best is None or (np.isfinite(a) and a > best_auc):
                best, best_auc = (C, clf), (a if np.isfinite(a) else best_auc)
    note = edge_note(Cs, raw, best[0]) if np.isfinite(best_auc) else f"val AUROC undefined: kept C={best[0]}"
    C, clf = best
    return LabelFit(label, C, scores, clf.coef_[0].astype(np.float64), float(clf.intercept_[0]), note)


def jobs_for_memory(n_jobs: int, n_rows: int, n_cols: int, share: float = 0.6) -> int:
    """
    Parallel workers that fit in `share` of the RAM. Each copies its label's rows of the standardized training
    matrix (float64): 2.6 GB at M5's ~220k studies, which a dozen workers can't afford on a laptop.
    """
    jobs = (os.cpu_count() or 1) if n_jobs in (-1, None) else n_jobs
    try:
        ram = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    except (AttributeError, OSError, ValueError):
        return jobs
    cap = max(1, int(share * ram // max(1, int(n_rows * n_cols * 8 * 1.2))))
    if cap < jobs:
        logger.info("Probe: %d parallel fits instead of %d, to fit in memory (%.1f GB of training features each)",
                    cap, jobs, n_rows * n_cols * 8 / 1e9)
    return min(jobs, cap)


def fit_probe(
    X_tr: np.ndarray, raw_tr: np.ndarray, X_va: np.ndarray, raw_va: np.ndarray,
    Cs=DEFAULT_CS, n_jobs: int = -1, max_iter: int = 2000, chosen: Optional[dict] = None,
) -> Probe:
    """
    One logistic regression per label on standardized features; C chosen on val AUROC. `chosen` fixes a label's
    C instead (M7 refits a probe at the C each label picked, which gives back the same fit).
    """
    Y_tr, M_tr = targets_and_mask(raw_tr, "v2")
    Y_va, M_va = targets_and_mask(raw_va, "v2")
    scaler = StandardScaler().fit(X_tr)
    Z_tr, Z_va = scaler.transform(X_tr), scaler.transform(X_va)
    # Keep the Cs ordered small -> large so "first best" prefers stronger regularization on ties.
    Cs = sorted(Cs)
    chosen = chosen or {}
    fits = Parallel(n_jobs=jobs_for_memory(n_jobs, *Z_tr.shape))(
        delayed(_fit_label)(label, Z_tr, Y_tr[:, k], M_tr[:, k], Z_va, Y_va[:, k], M_va[:, k],
                            [chosen[label]] if chosen.get(label) is not None else Cs, max_iter)
        for k, label in enumerate(LABELS)
    )
    return Probe(scaler, list(fits))


def split_xy(manifest: pd.DataFrame, ids: np.ndarray, emb: np.ndarray, extra: Optional[np.ndarray] = None):
    """Align embeddings with manifest rows; returns per-split (rows, X)."""
    pos = pd.Series(np.arange(len(ids)), index=ids)
    rows = manifest[manifest["study_id"].isin(pos.index)].sort_values("study_id").reset_index(drop=True)
    X = emb[pos.loc[rows["study_id"]].to_numpy()]
    if extra is not None:
        X = np.concatenate([X, extra[pos.loc[rows["study_id"]].to_numpy()]], axis=1)
    out = {}
    for split in (TRAIN, VAL, TEST):
        sel = (rows["split"] == split).to_numpy()
        out[split] = (rows[sel].reset_index(drop=True), X[sel])
    return out


def run_probe(out_dir: Path, emb_path: Path, run: Optional[str], Cs, n_jobs: int, max_train: int, seed: int) -> str:
    manifest = load_manifest(out_dir)
    ids, emb, meta = load_npz(emb_path)
    parts = split_xy(manifest, ids, emb)
    del emb  # the splits hold their own copies; at M5's size this frees 1.4 GB before the fits
    missing = len(manifest) - sum(len(r) for r, _ in parts.values())
    if missing:
        logger.warning("%d manifest studies have no embedding (expected only for --max-train/--smoke runs)", missing)
    (tr_rows, X_tr), (va_rows, X_va), (te_rows, X_te) = parts[TRAIN], parts[VAL], parts[TEST]
    if max_train and len(tr_rows) > max_train:
        pick = np.sort(np.random.RandomState(seed).choice(len(tr_rows), max_train, replace=False))
        tr_rows, X_tr = tr_rows.iloc[pick].reset_index(drop=True), X_tr[pick]
    if len(tr_rows) == 0 or len(va_rows) == 0:
        raise SystemExit("Need train and val embeddings to fit a probe")
    logger.info("Probe on %s: train %d, val %d, test %d, dim %d", emb_path.name, len(tr_rows), len(va_rows),
                len(te_rows), X_tr.shape[1])

    probe = fit_probe(X_tr, label_matrix(tr_rows), X_va, label_matrix(va_rows), Cs, n_jobs)
    for f in probe.fits:
        logger.info("  %-27s C=%-5s val AUROC by C: %s %s", f.label, f.C, f.val_auroc, f.note)
    for side, edge, wider in (("smallest", min(Cs), (min(Cs) / 100, min(Cs) / 10)),
                              ("largest", max(Cs), (max(Cs) * 10, max(Cs) * 100))):
        n = sum("the grid may be too narrow" in f.note and f.C == edge for f in probe.fits)
        if n > len(probe.fits) // 2:
            logger.warning("%d of %d labels chose the %s C (%s). C is selected on val only, so it is safe "
                           "to widen: --Cs %s", n, len(probe.fits), side, edge,
                           ",".join(f"{c:g}" for c in sorted({*Cs, *wider})))

    run = run or f"{emb_path.stem}_probe"
    rows = pd.concat([va_rows, te_rows], ignore_index=True)
    scores = probe.predict(np.concatenate([X_va, X_te]))
    write_predictions(out_dir, run, rows["study_id"], rows["split"], scores, {
        "kind": "probe", "embeddings": str(emb_path), **{k: meta.get(k) for k in ("encoder", "size", "revision", "torch", "transformers")},
        "n_train": int(len(tr_rows)), "Cs": list(Cs), "per_label": probe.summary(),
    })
    return run


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--embeddings", required=True, help="embeddings/{model}_{size}.npz from v2.embed")
    parser.add_argument("--out", default=None, help="v2 output dir (default: $OUTPUT_PATH/v2 or output/v2)")
    parser.add_argument("--run", help="Run name (default: {embeddings stem}_probe)")
    parser.add_argument("--Cs", default=",".join(str(c) for c in DEFAULT_CS))
    parser.add_argument("--jobs", type=int, default=min(12, os.cpu_count() or 1))
    parser.add_argument("--max-train", type=int, default=0, help="Fit on a random subset of N training studies")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)
    setup_logging(args.verbose)
    out_dir = Path(args.out) if args.out else default_out_dir()
    run = run_probe(out_dir, Path(args.embeddings), args.run, [float(c) for c in args.Cs.split(",")],
                    args.jobs, args.max_train, args.seed)
    logger.info("Next: python -m v2.evaluate --run %s", run)


if __name__ == "__main__":
    main()
