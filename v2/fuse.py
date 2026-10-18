"""
M4: do pre-imaging triage vitals add anything to the image? (docs/V2_PLAN.md §6)

    python -m v2.fuse --embeddings output/v2/embeddings/rad-dino_518.npz --cohort-dir output/cohorts

Features: the image embedding, plus age, sex and ED triage temperature, heart
rate, respiratory rate, SpO2, blood pressure and acuity from v1's cohort files,
each with a missing-value indicator. Vitals count only when the ED arrival
(intime) is at or before the X-ray; otherwise they are treated as missing.

Fits the same probe twice on the same studies, image only and image + vitals,
writes both prediction files, and runs the paired comparison on test_clean.
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from .common import TEST, TRAIN, VAL, default_out_dir, label_matrix, load_manifest, setup_logging, write_predictions
from .embed import load_npz
from .evaluate import compare_runs, fmt_ci
from .probe import DEFAULT_CS, fit_probe, split_xy

logger = logging.getLogger("v2.fuse")

# Triage columns in v1's cohort files, with plausible ranges (outside -> missing).
VITALS = {
    "triage_temperature": (90.0, 110.0),   # Fahrenheit
    "triage_heartrate": (20.0, 250.0),
    "triage_resprate": (4.0, 60.0),
    "triage_o2sat": (50.0, 100.0),
    "triage_sbp": (50.0, 260.0),
    "triage_dbp": (20.0, 180.0),
    "triage_acuity": (1.0, 5.0),
}


def read_cohort_vitals(cohort_dir: Path) -> pd.DataFrame:
    paths = [cohort_dir / f"anomalous_{s}.parquet" for s in ("train", "val")]
    if not all(p.exists() for p in paths):
        raise SystemExit(f"Need {paths[0].name} and {paths[1].name} in {cohort_dir}")
    frames = []
    for p in paths:
        cols = pq.read_schema(p).names
        wanted = [c for c in ("study_id", "anchor_age", "gender", "intime", *VITALS) if c in cols]
        frames.append(pd.read_parquet(p, columns=wanted))
    df = pd.concat(frames, ignore_index=True).drop_duplicates("study_id")
    for col in ("anchor_age", "gender", "intime", *VITALS):
        if col not in df.columns:
            logger.warning("Cohort files have no %s column: treated as missing", col)
            df[col] = np.nan
    return df


def vitals_features(manifest: pd.DataFrame, cohort: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Per manifest study: raw features with NaN for missing, and counts for the log."""
    df = manifest[["study_id", "study_datetime"]].merge(cohort, on="study_id", how="left")
    intime = pd.to_datetime(df["intime"], errors="coerce")
    before = intime.notna() & df["study_datetime"].notna() & (intime <= df["study_datetime"])
    stats = {"studies": len(df), "with_ed_stay": int(intime.notna().sum()),
             "ed_arrival_before_xray": int(before.sum())}
    feats = pd.DataFrame({"study_id": df["study_id"]})
    feats["age"] = pd.to_numeric(df["anchor_age"], errors="coerce")
    feats["sex_male"] = df["gender"].map({"M": 1.0, "F": 0.0})
    for col, (lo, hi) in VITALS.items():
        v = pd.to_numeric(df[col], errors="coerce").where(before)
        stats[f"{col}_out_of_range"] = int(((v < lo) | (v > hi)).sum())
        feats[col] = v.where((v >= lo) & (v <= hi))
    return feats, stats


def impute_with_indicators(feats: pd.DataFrame, train_mask: np.ndarray) -> np.ndarray:
    """Median-impute each column (medians from train) and append a missing indicator per column."""
    values = feats.drop(columns="study_id").to_numpy(np.float64)
    missing = np.isnan(values)
    medians = np.nanmedian(np.where(train_mask[:, None], values, np.nan), axis=0)
    medians = np.where(np.isnan(medians), 0.0, medians)
    filled = np.where(missing, medians, values)
    return np.concatenate([filled, missing.astype(np.float64)], axis=1)


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--embeddings", required=True, help="Image embeddings (.npz from v2.embed or v2.finetune)")
    parser.add_argument("--cohort-dir", help="v1 cohort dir with anomalous_{train,val}.parquet (default: COHORT_DIR)")
    parser.add_argument("--out", default=None, help="v2 output dir (default: $OUTPUT_PATH/v2 or output/v2)")
    parser.add_argument("--run", help="Run-name prefix (default: {embeddings stem})")
    parser.add_argument("--jobs", type=int, default=min(12, os.cpu_count() or 1))
    parser.add_argument("--n-boot", type=int, default=1000)
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)
    setup_logging(args.verbose)
    out_dir = Path(args.out) if args.out else default_out_dir()  # also loads .env
    args.cohort_dir = args.cohort_dir or os.environ.get("COHORT_DIR")
    if not args.cohort_dir:
        raise SystemExit("M4 needs v1's cohort files: pass --cohort-dir or set COHORT_DIR")

    manifest = load_manifest(out_dir)
    ids, emb, meta = load_npz(Path(args.embeddings))
    feats, stats = vitals_features(manifest, read_cohort_vitals(Path(args.cohort_dir)))
    logger.info("Vitals: %s", stats)
    feats = feats.set_index("study_id").reindex(ids).rename_axis("study_id").reset_index()
    split_of = manifest.set_index("study_id")["split"].reindex(ids)
    extra = impute_with_indicators(feats, (split_of == TRAIN).to_numpy())

    prefix = args.run or Path(args.embeddings).stem
    base_meta = {"kind": "fuse", "embeddings": str(args.embeddings),
                 **{k: meta.get(k) for k in ("encoder", "size", "revision", "torch", "transformers")}, "vitals_stats": stats}
    runs = {}
    for name, x_extra in (("imageonly", None), ("vitals", extra)):
        parts = split_xy(manifest, ids, emb, x_extra)
        (tr_rows, X_tr), (va_rows, X_va), (te_rows, X_te) = parts[TRAIN], parts[VAL], parts[TEST]
        probe = fit_probe(X_tr, label_matrix(tr_rows), X_va, label_matrix(va_rows), DEFAULT_CS, args.jobs)
        rows = pd.concat([va_rows, te_rows], ignore_index=True)
        run = f"{prefix}_fuse_{name}"
        write_predictions(out_dir, run, rows["study_id"], rows["split"], probe.predict(np.concatenate([X_va, X_te])),
                          {**base_meta, "features": name, "per_label": probe.summary()})
        runs[name] = run

    res = compare_runs(out_dir, runs["vitals"], runs["imageonly"], "test_clean", "v2", args.n_boot)
    verdict = "adopt vitals" if res["a_better"] else "vitals not adopted (CI includes 0 or is negative)"
    logger.info("Image + vitals - image only, test_clean: %s -> %s", fmt_ci(res["diff"], res["diff_ci"]), verdict)


if __name__ == "__main__":
    main()
