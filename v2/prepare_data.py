"""
M0: build the v2 manifest, resized images and data audit (docs/V2_PLAN.md §6).

    python -m v2.prepare_data --mimic-cxr-jpg /Volumes/drive/mimic-cxr-jpg/2.1.0 \\
        --cohort-dir output/cohorts --out output/v2

Reads mimic-cxr-2.0.0-{metadata,split,chexpert}.csv.gz, the JPGs, and (if
given) v1's cohort files. Writes, under --out:

    manifest.parquet           one row per study: ids, view, split flags, 12 raw labels, image path
    images/{study_id}.png      8-bit grayscale, shorter side --image-size px
    measurable_classes.json    measurable classes per test set and metric (§3.2)
    audit.md, audit.json       counts, overlaps, drops, RAD-DINO check, gate G0

Re-running is safe: existing PNGs are kept, so an interrupted run resumes.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from PIL import Image

from .common import (
    LABELS,
    MIN_COUNT,
    TEST,
    TRAIN,
    VAL,
    DEFAULT_SEED,
    git_commit,
    jpg_path,
    label_counts,
    label_matrix,
    measurable,
    mimic_csv,
    resolve_mimic_root,
    default_out_dir,
    setup_logging,
)

logger = logging.getLogger("v2.prepare_data")

FRONTAL_RANK = {"PA": 0, "AP": 1}
OFFICIAL_TEST = ("validate", "test")
G0_MIN_CLASSES = 6


# ---------------------------------------------------------------------------
# Reading MIMIC-CXR-JPG
# ---------------------------------------------------------------------------

def read_mimic_tables(root: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """metadata, split and chexpert tables, with only the columns v2 needs."""
    meta = pd.read_csv(
        mimic_csv(root, "metadata"),
        usecols=["dicom_id", "subject_id", "study_id", "ViewPosition", "StudyDate", "StudyTime"],
        dtype={"dicom_id": str, "subject_id": np.int64, "study_id": np.int64, "ViewPosition": str},
    )
    split = pd.read_csv(
        mimic_csv(root, "split"),
        usecols=["dicom_id", "subject_id", "split"],
        dtype={"dicom_id": str, "subject_id": np.int64, "split": str},
    )
    chexpert = pd.read_csv(
        mimic_csv(root, "chexpert"),
        dtype={"subject_id": np.int64, "study_id": np.int64, **{label: np.float64 for label in LABELS}},
    )
    return meta, split, chexpert


def study_datetime(date: pd.Series, time: pd.Series) -> pd.Series:
    """StudyDate (YYYYMMDD) + StudyTime (HHMMSS.fff stored as a number, so leading zeros are lost)."""
    t = pd.to_numeric(time, errors="coerce")
    hhmmss = t.dropna().astype(np.int64).astype(str).str.zfill(6).reindex(t.index)  # missing time -> NaT
    day = pd.to_numeric(date, errors="coerce").astype("Int64").astype(str)
    return pd.to_datetime(day + hhmmss, format="%Y%m%d%H%M%S", errors="coerce")


def select_frontal(meta: pd.DataFrame) -> pd.DataFrame:
    """One frontal image per study: PA if present, else AP; ties broken by lowest dicom_id."""
    frontal = meta[meta["ViewPosition"].isin(FRONTAL_RANK.keys())].copy()
    frontal["_rank"] = frontal["ViewPosition"].map(FRONTAL_RANK)
    frontal = frontal.sort_values(["study_id", "_rank", "dicom_id"], kind="mergesort")
    chosen = frontal.drop_duplicates("study_id", keep="first").drop(columns="_rank")
    chosen = chosen.rename(columns={"ViewPosition": "view"})
    chosen["study_datetime"] = study_datetime(chosen.pop("StudyDate"), chosen.pop("StudyTime"))
    return chosen.reset_index(drop=True)


def official_split_by_patient(split: pd.DataFrame) -> pd.Series:
    """Official split per patient. MIMIC-CXR splits by patient; any non-train image counts."""
    order = {"train": 0, "validate": 1, "test": 2}
    ranked = split.assign(_r=split["split"].map(order).fillna(0))
    per_patient = ranked.groupby("subject_id")["_r"].max()
    names = {v: k for k, v in order.items()}
    return per_patient.map(names)


# ---------------------------------------------------------------------------
# Population and splits
# ---------------------------------------------------------------------------

def read_cohort(cohort_dir: Path) -> tuple[pd.DataFrame, set]:
    """v1's abnormal cohort (train + val) and the set of v1 validation study_ids."""
    files = {name: cohort_dir / f"anomalous_{name}.parquet" for name in ("train", "val")}
    missing = [str(p) for p in files.values() if not p.exists()]
    if missing:
        raise SystemExit(f"Missing v1 cohort files: {missing}")
    parts = []
    for name, path in files.items():
        df = pd.read_parquet(path, columns=["subject_id", "study_id"])
        df["_v1_split"] = name
        parts.append(df)
    cohort = pd.concat(parts, ignore_index=True)
    both = cohort.groupby("study_id")["_v1_split"].nunique()
    if (both > 1).any():
        logger.warning("%d studies are in both v1 train and val; treating them as val", int((both > 1).sum()))
    v1_val_studies = set(cohort.loc[cohort["_v1_split"] == "val", "study_id"].astype(np.int64))
    cohort = cohort.drop_duplicates("study_id")[["subject_id", "study_id"]]
    return cohort.astype(np.int64), v1_val_studies


def abnormal_studies(chexpert: pd.DataFrame) -> pd.DataFrame:
    """Studies with at least one positive pathology (the v1 cohort's first filter)."""
    positive = (chexpert[list(LABELS)] == 1.0).any(axis=1)
    return chexpert.loc[positive, ["subject_id", "study_id"]].drop_duplicates("study_id")


def assign_splits(
    df: pd.DataFrame,
    official: pd.Series,
    v1_val_studies: set,
    val_frac: float,
    seed: int,
) -> pd.DataFrame:
    """
    Add official_split, in_test_clean, in_test_v1 and split (§3.1).

    test_clean: studies of patients in the official validate or test split.
    test_v1:    studies in v1's validation file.
    split:      'test' for every study of a test_clean or test_v1 patient; the
                remaining patients go to val (val_frac, seeded) or train.
    """
    df = df.copy()
    df["official_split"] = df["subject_id"].map(official).fillna("unknown")
    df["in_test_clean"] = df["official_split"].isin(OFFICIAL_TEST)
    df["in_test_v1"] = df["study_id"].isin(v1_val_studies)

    test_patients = set(df.loc[df["in_test_clean"] | df["in_test_v1"], "subject_id"])
    rest = np.array(sorted(set(df["subject_id"]) - test_patients), dtype=np.int64)
    rng = np.random.RandomState(seed)
    n_val = int(round(val_frac * len(rest)))
    val_patients = set(rng.permutation(rest)[:n_val].tolist())

    df["split"] = np.where(
        df["subject_id"].isin(test_patients),
        TEST,
        np.where(df["subject_id"].isin(val_patients), VAL, TRAIN),
    )
    return df


# ---------------------------------------------------------------------------
# Images
# ---------------------------------------------------------------------------

def resize_shorter(img: Image.Image, size: int) -> Image.Image:
    w, h = img.size
    scale = size / min(w, h)
    new = (max(1, round(w * scale)), max(1, round(h * scale)))
    return img.resize(new, Image.BICUBIC)


def _convert_one(task: tuple[str, str, int]) -> tuple[str, Optional[str]]:
    """JPG -> grayscale PNG with shorter side `size`. Returns (dst, error or None)."""
    src, dst, size = task
    try:
        with Image.open(src) as img:
            # Let the JPEG decoder downscale by a power of two first (much faster on 3k-px films).
            img.draft("L", (2 * size, 2 * size))
            img = img.convert("L")
            out = resize_shorter(img, size)
        tmp = f"{dst}.tmp{os.getpid()}"
        out.save(tmp, format="PNG", compress_level=3)
        os.replace(tmp, dst)
        return dst, None
    except Exception as exc:  # noqa: BLE001 - any unreadable file is recorded, not fatal
        return dst, f"{type(exc).__name__}: {exc}"


def write_images(manifest: pd.DataFrame, root: Path, out_dir: Path, size: int, workers: int) -> dict:
    """Write images/{study_id}.png for every manifest row that lacks one. Returns failures."""
    img_dir = out_dir / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    marker = img_dir / ".image_size"
    if marker.exists() and int(marker.read_text().strip()) != size:
        raise SystemExit(
            f"{img_dir} holds images with shorter side {marker.read_text().strip()} px, not {size}. "
            "Use the same --image-size, or delete that directory first."
        )
    marker.write_text(str(size))
    tasks = []
    for row in manifest.itertuples(index=False):
        dst = img_dir / f"{row.study_id}.png"
        if dst.exists() and dst.stat().st_size > 0:
            continue
        tasks.append((str(root / row.jpg), str(dst), size))
    logger.info("Images: %d to write, %d already present", len(tasks), len(manifest) - len(tasks))
    failures = {}
    if not tasks:
        return failures

    done = 0
    report_every = max(1000, len(tasks) // 20)
    if workers <= 1:
        results = map(_convert_one, tasks)
        for dst, err in results:
            done += 1
            if err:
                failures[dst] = err
            if done % report_every == 0:
                logger.info("  %d / %d", done, len(tasks))
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            for dst, err in pool.map(_convert_one, tasks, chunksize=16):
                done += 1
                if err:
                    failures[dst] = err
                if done % report_every == 0:
                    logger.info("  %d / %d", done, len(tasks))
    return failures


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------

def read_rad_dino_list(source: str) -> Optional[pd.DataFrame]:
    """MIMIC rows of RAD-DINO's training_images.csv (dicom_id, subject_id), or None if skipped."""
    if source == "skip":
        return None
    if source == "auto":
        from huggingface_hub import hf_hub_download

        path = hf_hub_download("microsoft/rad-dino", "training_images.csv")
    else:
        path = source
    df = pd.read_csv(path, dtype=str)
    mimic = df[df["rel_image_path"].str.startswith("mimic-cxr")].copy()
    mimic["subject_id"] = mimic["rel_image_path"].str.extract(r"/p(\d{8})/", expand=False).astype("Int64")
    return mimic.rename(columns={"image_id": "dicom_id"})[["dicom_id", "subject_id"]]


def rad_dino_overlap(manifest: pd.DataFrame, rad: pd.DataFrame) -> dict:
    """How many chosen images / patients of each test set RAD-DINO saw in pretraining."""
    dicoms = set(rad["dicom_id"])
    subjects = set(rad["subject_id"].dropna().astype(np.int64))
    out = {"n_mimic_images_in_list": int(len(rad))}
    groups = {
        "test_clean": manifest["in_test_clean"],
        "test_v1": manifest["in_test_v1"],
        "train": manifest["split"] == TRAIN,
        "val": manifest["split"] == VAL,
    }
    for name, mask in groups.items():
        sub = manifest[mask]
        out[name] = {
            "images": int(len(sub)),
            "images_in_list": int(sub["dicom_id"].isin(dicoms).sum()),
            "patients": int(sub["subject_id"].nunique()),
            "patients_in_list": int(sub["subject_id"].drop_duplicates().isin(subjects).sum()),
        }
    return out


def v1_lateral_share(h5_path: Path, meta: pd.DataFrame) -> dict:
    """Share of lateral views among the images in a v1 images.h5 (v1 picked views by filename)."""
    import h5py

    if h5_path.is_dir():
        h5_path = h5_path / "images.h5"
    views = meta.set_index("dicom_id")["ViewPosition"]
    counts: dict[str, int] = {}
    with h5py.File(h5_path, "r") as f:
        group = f["metadata"]
        for key in group:
            raw = group[key][()]
            info = json.loads(raw.decode() if isinstance(raw, bytes) else str(raw))
            dicom = Path(info.get("image_path", "")).stem
            view = views.get(dicom, "not found")
            view = "missing" if not isinstance(view, str) else view
            counts[view] = counts.get(view, 0) + 1
    total = sum(counts.values())
    lateral = sum(v for k, v in counts.items() if k not in ("PA", "AP", "not found", "missing"))
    return {
        "file": str(h5_path),
        "images": total,
        "views": dict(sorted(counts.items(), key=lambda kv: -kv[1])),
        "non_frontal_share": round(lateral / total, 4) if total else None,
    }


def split_overlaps(manifest: pd.DataFrame) -> dict:
    patients = {s: set(manifest.loc[manifest["split"] == s, "subject_id"]) for s in (TRAIN, VAL, TEST)}
    return {
        "train/val": len(patients[TRAIN] & patients[VAL]),
        "train/test": len(patients[TRAIN] & patients[TEST]),
        "val/test": len(patients[VAL] & patients[TEST]),
    }


def measurable_sets(manifest: pd.DataFrame, min_count: int) -> dict:
    out = {}
    for test_set, flag in (("test_clean", "in_test_clean"), ("test_v1", "in_test_v1")):
        raw = label_matrix(manifest[manifest[flag]])
        for metric in ("v2", "v1"):
            out[f"{test_set}:{metric}"] = measurable(raw, metric, min_count) if len(raw) else []
    return out


# ---------------------------------------------------------------------------
# Audit
# ---------------------------------------------------------------------------

def build_audit(manifest: pd.DataFrame, drops: dict, extras: dict, min_count: int) -> dict:
    groups = {
        "train": manifest["split"] == TRAIN,
        "val": manifest["split"] == VAL,
        "test (all)": manifest["split"] == TEST,
        "test_clean": manifest["in_test_clean"],
        "test_v1": manifest["in_test_v1"],
    }
    sizes = {
        name: {"studies": int(mask.sum()), "patients": int(manifest.loc[mask, "subject_id"].nunique())}
        for name, mask in groups.items()
    }
    counts = {}
    for name, mask in groups.items():
        raw = label_matrix(manifest[mask])
        table = label_counts(raw, "v2")
        table["uncertain"] = (raw == -1.0).sum(axis=0)
        table["blank"] = np.isnan(raw).sum(axis=0)
        counts[name] = table.drop(columns="ignored").set_index("label").astype(int).to_dict(orient="index")
    measurable_lists = measurable_sets(manifest, min_count)
    n_clean = len(measurable_lists["test_clean:v2"])
    return {
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "commit": git_commit(),
        "sizes": sizes,
        "views": manifest["view"].value_counts().to_dict(),
        "sources": manifest["source"].value_counts().to_dict(),
        "label_counts": counts,
        "min_count": min_count,
        "measurable": measurable_lists,
        "patient_overlap": split_overlaps(manifest),
        "drops": drops,
        "gate_g0": {
            "measurable_test_clean": n_clean,
            "required": G0_MIN_CLASSES,
            "pass": n_clean >= G0_MIN_CLASSES,
        },
        **extras,
    }


def audit_markdown(audit: dict) -> str:
    lines = [
        "# v2 data audit",
        "",
        f"Created {audit['created']} at commit `{audit['commit']}`. Label policy: 1 positive; "
        "0 or blank negative; -1 ignored (v1 metric: blank ignored too).",
        "",
        "## Splits",
        "",
        "| Split | Studies | Patients |",
        "|---|---|---|",
    ]
    for name, s in audit["sizes"].items():
        lines.append(f"| {name} | {s['studies']:,} | {s['patients']:,} |")
    ov = audit["patient_overlap"]
    ok = all(v == 0 for v in ov.values())
    lines += [
        "",
        f"Patient overlap between train, val and test: {ov} -> **{'OK (0)' if ok else 'FAIL'}**",
        "",
        f"Views chosen: {audit['views']}. Sources: {audit['sources']}.",
        "",
        "## Dropped studies",
        "",
    ]
    for reason, n in audit["drops"].items():
        lines.append(f"- {reason}: {n:,}")
    lines += ["", "## Measurable classes (>= {} positives and negatives)".format(audit["min_count"]), ""]
    for key, labels in audit["measurable"].items():
        lines.append(f"- **{key}** ({len(labels)}): {', '.join(labels) if labels else '(none)'}")
    g0 = audit["gate_g0"]
    verdict = "PASS" if g0["pass"] else "FAIL: re-run with --extend-test-clean (§3.1 fallback)"
    lines += [
        "",
        f"**Gate G0:** test_clean has {g0['measurable_test_clean']} measurable classes "
        f"(need {g0['required']}) -> **{verdict}**",
        "",
        "## Label counts (v2 policy)",
        "",
    ]
    for name, table in audit["label_counts"].items():
        lines += [f"### {name}", "", "| Label | Pos | Neg | Uncertain (ignored) | Blank (counted negative) |",
                  "|---|---|---|---|---|"]
        for label, c in table.items():
            lines.append(f"| {label} | {c['pos']:,} | {c['neg']:,} | {c['uncertain']:,} | {c['blank']:,} |")
        lines.append("")
    rad = audit.get("rad_dino")
    lines += ["## RAD-DINO pretraining overlap", ""]
    if rad is None:
        lines.append("Not checked (--rad-dino-list skip).")
    elif "error" in rad:
        lines.append(f"**NOT CHECKED:** {rad['error']}")
    else:
        clean = rad["test_clean"]
        status = "OK (0)" if clean["images_in_list"] == 0 and clean["patients_in_list"] == 0 else "FAIL"
        lines += [
            f"RAD-DINO's list has {rad['n_mimic_images_in_list']:,} MIMIC-CXR images.",
            "",
            "| Set | Images | In list | Patients | In list |",
            "|---|---|---|---|---|",
        ]
        for name in ("test_clean", "test_v1", "train", "val"):
            r = rad[name]
            lines.append(f"| {name} | {r['images']:,} | {r['images_in_list']:,} | {r['patients']:,} | {r['patients_in_list']:,} |")
        lines += ["", f"test_clean must not overlap: **{status}**. test_v1 overlap is expected (§3.1)."]
    cov = audit.get("v1_cohort_coverage")
    if cov:
        lines += [
            "",
            "## v1 cohort coverage",
            "",
            f"v1's cohort holds {cov['in_v1_cohort']:,} of MIMIC-CXR's {cov['abnormal_studies']:,} abnormal studies "
            f"({cov['share']:.1%}). Share acquired before 10:00: {cov['before_10am_share_cohort']} in the cohort vs "
            f"{cov['before_10am_share_not_in_cohort']} outside it. A much lower share inside means v1's time "
            "parsing dropped early studies (see `cohort_coverage` in v2/prepare_data.py).",
        ]
    lateral = audit.get("v1_lateral_share")
    if lateral:
        lines += [
            "",
            "## v1 training images by view",
            "",
            f"{lateral['file']}: {lateral['images']:,} images; views {lateral['views']}; "
            f"non-frontal share **{lateral['non_frontal_share']}**.",
        ]
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def cohort_coverage(cohort: pd.DataFrame, chexpert: pd.DataFrame, frontal: pd.DataFrame) -> dict:
    """
    How much of MIMIC-CXR's abnormal studies v1's cohort kept, and whether it kept
    early-morning studies as often as the rest. v1 parsed StudyTime with str[:6], which
    fails if a time before 10:00 is stored without its leading zero, and its ED linking
    drops studies without a datetime for patients who had any ED stay.
    """
    abnormal = abnormal_studies(chexpert)
    hour = frontal.set_index("study_id")["study_datetime"].dt.hour

    def early_share(ids: pd.Series) -> Optional[float]:
        h = hour.reindex(ids).dropna()
        return round(float((h < 10).mean()), 4) if len(h) else None

    in_cohort = abnormal["study_id"].isin(cohort["study_id"])
    return {
        "abnormal_studies": int(len(abnormal)),
        "in_v1_cohort": int(in_cohort.sum()),
        "share": round(float(in_cohort.mean()), 4) if len(abnormal) else None,
        "before_10am_share_cohort": early_share(abnormal.loc[in_cohort, "study_id"]),
        "before_10am_share_not_in_cohort": early_share(abnormal.loc[~in_cohort, "study_id"]),
    }


def build_manifest(args: argparse.Namespace, root: Path) -> tuple[pd.DataFrame, dict, pd.DataFrame, dict]:
    meta, split, chexpert = read_mimic_tables(root)
    logger.info("MIMIC-CXR: %d images, %d studies with CheXpert labels", len(meta), len(chexpert))
    drops: dict[str, int] = {}
    info: dict = {}

    if args.cohort_dir:
        population, v1_val_studies = read_cohort(Path(args.cohort_dir))
        source = "v1 cohort"
    else:
        population, v1_val_studies = abnormal_studies(chexpert), set()
        source = "all abnormal studies"
        logger.warning("No --cohort-dir: population = all abnormal studies; test_v1 will be empty")
    logger.info("Population (%s): %d studies", source, len(population))

    labels = chexpert.drop_duplicates("study_id")[["study_id", *LABELS]]
    frontal = select_frontal(meta)
    if args.cohort_dir:
        info["v1_cohort_coverage"] = cohort_coverage(population, chexpert, frontal)

    df = population.merge(labels, on="study_id", how="left", indicator=True)
    drops["no CheXpert row"] = int((df["_merge"] == "left_only").sum())
    df = df[df["_merge"] == "both"].drop(columns="_merge")
    df = df.merge(frontal.drop(columns="subject_id"), on="study_id", how="left")
    drops["no frontal (PA/AP) image"] = int(df["dicom_id"].isna().sum())
    df = df[df["dicom_id"].notna()].copy()
    df["source"] = "cohort"

    official = official_split_by_patient(split)
    unknown = int((~df["subject_id"].isin(official.index)).sum())
    if unknown:
        logger.warning("%d studies belong to patients missing from the official split file", unknown)

    if args.extend_test_clean:
        test_patients = set(official.index[official.isin(OFFICIAL_TEST)])
        extra = abnormal_studies(chexpert)
        extra = extra[extra["subject_id"].isin(test_patients) & ~extra["study_id"].isin(df["study_id"])]
        extra = extra.merge(labels, on="study_id").merge(frontal.drop(columns="subject_id"), on="study_id")
        extra["source"] = "test_clean extension"
        logger.info("--extend-test-clean: +%d studies of official validate/test patients", len(extra))
        df = pd.concat([df, extra], ignore_index=True)

    if args.limit:
        rng = np.random.RandomState(args.seed)
        patients = rng.permutation(np.array(sorted(df["subject_id"].unique())))
        sizes = df.groupby("subject_id").size().reindex(patients).cumsum()
        keep = set(sizes.index[sizes <= args.limit]) or {patients[0]}
        df = df[df["subject_id"].isin(keep)].copy()
        logger.info("--limit %d: kept %d studies of %d patients", args.limit, len(df), len(keep))

    df = assign_splits(df, official, v1_val_studies, args.val_frac, args.seed)
    df["jpg"] = [
        str(jpg_path(Path("."), s, t, d)) for s, t, d in zip(df["subject_id"], df["study_id"], df["dicom_id"])
    ]
    df["image"] = "images/" + df["study_id"].astype(str) + ".png"
    columns = ["study_id", "subject_id", "dicom_id", "view", "study_datetime", "official_split", "split",
               "in_test_clean", "in_test_v1", "source", "jpg", "image", *LABELS]
    df = df[columns].sort_values("study_id").reset_index(drop=True)
    df["study_id"] = df["study_id"].astype(np.int64)
    df["subject_id"] = df["subject_id"].astype(np.int64)
    return df, drops, meta, info


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mimic-cxr-jpg", help="MIMIC-CXR-JPG root (default: MIMIC_CXR_JPG_PATH)")
    parser.add_argument("--cohort-dir", help="v1 cohort dir with anomalous_{train,val}.parquet (default: "
                        "COHORT_DIR; without one, every abnormal study is used and test_v1 is empty)")
    parser.add_argument("--out", default=None, help="Output dir (default: $OUTPUT_PATH/v2 or output/v2)")
    parser.add_argument("--image-size", type=int, default=518, help="Shorter side of the saved PNGs")
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument("--limit", type=int, default=0, help="Keep about N studies (whole patients), for rehearsals")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--val-frac", type=float, default=0.1)
    parser.add_argument("--min-count", type=int, default=MIN_COUNT, help="Measurable-class threshold (§3.2)")
    parser.add_argument("--rad-dino-list", default="auto",
                        help="'auto' (download from Hugging Face), a local training_images.csv, or 'skip'")
    parser.add_argument("--v1-hdf5", help="v1 images.h5 (or its directory): report its share of lateral views")
    parser.add_argument("--extend-test-clean", action="store_true",
                        help="§3.1 fallback: add every abnormal frontal study of official validate/test patients")
    parser.add_argument("--skip-images", action="store_true", help="Manifest and audit only")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)
    setup_logging(args.verbose)

    root = resolve_mimic_root(args.mimic_cxr_jpg)  # also loads .env
    args.cohort_dir = args.cohort_dir or os.environ.get("COHORT_DIR") or None
    out_dir = Path(args.out) if args.out else default_out_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    if not (root / "files").is_dir():
        raise SystemExit(f"{root}/files not found: is --mimic-cxr-jpg the 2.1.0 directory?")

    manifest, drops, meta, info = build_manifest(args, root)

    extras: dict = {"population": "v1 cohort" if args.cohort_dir else "all abnormal studies",
                    "image_size": args.image_size, **info}
    if not args.skip_images:
        failures = write_images(manifest, root, out_dir, args.image_size, args.workers)
        if failures:
            bad = {Path(p).stem for p in failures}
            # Record-level details go to a local file only, never to the console (see v2/README.md §10).
            report = out_dir / "unreadable_images.csv"
            pd.DataFrame({"study_id": [Path(p).stem for p in failures], "error": list(failures.values())}).to_csv(
                report, index=False)
            kinds = pd.Series([e.split(":")[0] for e in failures.values()]).value_counts().to_dict()
            logger.warning("%d images missing or unreadable %s; details in %s", len(failures), kinds, report)
            manifest = manifest[~manifest["study_id"].astype(str).isin(bad)].reset_index(drop=True)
        drops["image missing or unreadable"] = len(failures)
        free_gb = shutil.disk_usage(out_dir).free / 1e9
        extras["disk_free_gb_after"] = round(free_gb, 1)

    if args.rad_dino_list != "skip":
        try:
            rad = read_rad_dino_list(args.rad_dino_list)
            extras["rad_dino"] = rad_dino_overlap(manifest, rad)
        except Exception as exc:  # noqa: BLE001 - offline machines still get an audit
            logger.warning("RAD-DINO overlap check failed: %s", exc)
            extras["rad_dino"] = {"error": f"{type(exc).__name__}: {exc}"}
    else:
        extras["rad_dino"] = None
    if args.v1_hdf5:
        extras["v1_lateral_share"] = v1_lateral_share(Path(args.v1_hdf5), meta)

    manifest.to_parquet(out_dir / "manifest.parquet", index=False)
    audit = build_audit(manifest, drops, extras, args.min_count)

    measurable_path = out_dir / "measurable_classes.json"
    new = {"min_count": args.min_count, "commit": audit["commit"], "created": audit["created"], **audit["measurable"]}
    if measurable_path.exists():
        old = json.loads(measurable_path.read_text())
        changed = [k for k in audit["measurable"] if old.get(k) != new[k]]
        if changed:
            logger.warning("Measurable classes changed for %s (they should be frozen after M0)", changed)
    measurable_path.write_text(json.dumps(new, indent=2))
    (out_dir / "audit.json").write_text(json.dumps(audit, indent=2, default=str))
    (out_dir / "audit.md").write_text(audit_markdown(audit))

    logger.info("Manifest: %d studies -> %s", len(manifest), out_dir / "manifest.parquet")
    for name, s in audit["sizes"].items():
        logger.info("  %-11s %6d studies  %6d patients", name, s["studies"], s["patients"])
    logger.info("Measurable classes in test_clean: %d (G0 %s)", audit["gate_g0"]["measurable_test_clean"],
                "PASS" if audit["gate_g0"]["pass"] else "FAIL")
    logger.info("Audit: %s", out_dir / "audit.md")


if __name__ == "__main__":
    main()
