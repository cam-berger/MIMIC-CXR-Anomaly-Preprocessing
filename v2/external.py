"""
M7: external validation (docs/V2_PLAN.md §6 M7). Nothing is trained here.

    python -m v2.external radiologist --bench output/v2_bench                      # M7a
    python -m v2.external chexpert --chexpert DIR --bench output/v2_bench --out output/v2_chexpert   # M7b

M7a scores the benchmark's predictions on the official test studies that a
radiologist labeled (mimic-cxr-2.1.0-test-set-labeled.csv in MIMIC-CXR-JPG
2.1.0), with those labels and with the labeler's on the same studies, and
writes BENCH_OUT/reports/m7a_radiologist.md.

M7b makes the CheXpert test set its own v2 output directory: one frontal image
per study, resized like the benchmark's, labels from groundtruth.csv. It stops
if RAD-DINO's training list holds a test patient. It then applies the
benchmark's probe (refit at the C each label chose, which must reproduce the
benchmark's own predictions) and fine-tunes (their best checkpoints, checked
against their benchmark predictions), and writes reports/m7b_chexpert.md with
gate G7. --steps runs part of it (prepare, apply, report). Only aggregates are
logged.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from .common import (
    LABELS,
    MIN_COUNT,
    TEST,
    TRAIN,
    VAL,
    amp_dtype,
    git_commit,
    label_matrix,
    load_dotenv,
    load_manifest,
    load_measurable,
    measurable,
    pick_device,
    read_predictions,
    setup_logging,
    write_predictions,
)
from .evaluate import N_BOOT, _get, compare_labels, compare_runs, evaluate_run, evaluate_set, fmt, fmt_ci, split_scheme

logger = logging.getLogger("v2.external")

PREFIX = "rad-dino_518"
CANDIDATES = ("probe", "ft_s0", "ft_attn_s0")  # §6 M7: the primary image model P is the best of these by val
STACK, STACK_IMAGE = "ft_attn_s0_stack_indication_bert", "ft_attn_s0"  # M6's adopted image + indication system
EXTERNAL = "external"  # split_scheme of an external test set's output dir

# --- M7a ------------------------------------------------------------------------------------------------------
RADIOLOGIST_FILE = "mimic-cxr-2.1.0-test-set-labeled.csv"
OLD_NAMES = {"Airspace Opacity": "Lung Opacity"}  # CheXpert's first name for the category, in that file

# --- M7b ------------------------------------------------------------------------------------------------------
FIVE = ("Atelectasis", "Cardiomegaly", "Consolidation", "Edema", "Pleural Effusion")
# Tiu et al., Nat. Biomed. Eng. 2022: Table 1 (CheXzero, AUROC [95% CI]) and Table 2 (means).
CHEXZERO = {"Atelectasis": (0.816, 0.777, 0.852), "Cardiomegaly": (0.906, 0.876, 0.930),
            "Consolidation": (0.892, 0.823, 0.947), "Edema": (0.897, 0.864, 0.928),
            "Pleural Effusion": (0.932, 0.906, 0.955)}
CHEXZERO_MEAN = 0.889
IN_DOMAIN = {"DenseNet-121": 0.902, "DAM": 0.931}  # trained on CheXpert's own labels: context only
RED_FLAG_MEAN = 0.95
REPRODUCE_TOL = 1e-6
RELOAD_MIN_CORR = 0.99
STUDY_RE = re.compile(r"patient(\d+)/study(\d+)")
IMAGE_RE = re.compile(r"patient(\d+)[/_]study(\d+)[/_]view(\d+)_(frontal|lateral)\.(?:jpe?g|png)$", re.IGNORECASE)
MANIFEST_COLUMNS = ["study_id", "subject_id", "dicom_id", "view", "study_datetime", "official_split", "split",
                    "in_test_clean", "in_test_v1", "source", "jpg", "image", *LABELS]


def _env_dir(var: str, default: str) -> Path:
    load_dotenv()
    return Path(os.environ[var]) if os.environ.get(var) else Path(os.environ.get("OUTPUT_PATH", "output")) / default


def _exists(out_dir: Path, run: str) -> bool:
    return (Path(out_dir) / "predictions" / f"{run}.parquet").exists()


def val_score(out_dir: Path, run: str, manifest: pd.DataFrame, frozen: dict) -> float:
    """A run's val macro AUROC (v2 policy), the point estimate only; nothing is written."""
    pred, _ = read_predictions(Path(out_dir) / "predictions" / f"{run}.parquet")
    res = evaluate_set(pred, manifest, frozen, "val", "v2", n_boot=1)
    return res["macro_auroc"] if res and np.isfinite(res["macro_auroc"]) else float("nan")


def pick_primary(bench: Path, candidates: list) -> tuple[Optional[str], dict]:
    """§6 M7: of the candidate image runs, the one with the best benchmark val score; and every score."""
    manifest, frozen = load_manifest(bench), load_measurable(bench)
    scores = {run: val_score(bench, run, manifest, frozen) for run in candidates if _exists(bench, run)}
    finite = {run: s for run, s in scores.items() if np.isfinite(s)}
    return (max(finite, key=finite.get) if finite else None), scores


# ---------------------------------------------------------------------------
# M7a: a radiologist's labels on the official test split
# ---------------------------------------------------------------------------

def read_radiologist(path: Path) -> tuple[pd.DataFrame, list]:
    """The radiologist's labels for our 12 classes (1 / 0 / -1 / blank) by study_id; and the columns renamed."""
    df = pd.read_csv(path)
    renamed = [old for old in OLD_NAMES if old in df.columns and OLD_NAMES[old] not in df.columns]
    df = df.rename(columns={old: OLD_NAMES[old] for old in renamed})
    missing = [c for c in ("study_id", *LABELS) if c not in df.columns]
    if missing:
        raise SystemExit(f"{Path(path).name} has no column(s) {missing}; its columns are {list(df.columns)}")
    ids = pd.to_numeric(df["study_id"].astype(str).str.lstrip("s"), errors="coerce")
    if ids.isna().any():
        raise SystemExit(f"{Path(path).name}: {int(ids.isna().sum())} study ids aren't numbers")
    values = df[list(LABELS)].apply(pd.to_numeric, errors="coerce")
    bad = set(np.unique(values.to_numpy(np.float64))) - {1.0, 0.0, -1.0}
    bad = {v for v in bad if not np.isnan(v)}
    if bad or (values.isna() & df[list(LABELS)].notna()).any(axis=None):
        raise SystemExit(f"{Path(path).name}: label values other than 1, 0, -1 and blank")
    out = values.astype(np.float64)
    out.insert(0, "study_id", ids.astype(np.int64))
    if out["study_id"].duplicated().any():
        raise SystemExit(f"{Path(path).name}: {int(out['study_id'].duplicated().sum())} study ids appear twice")
    return out, renamed


def radiologist_manifests(bench: Path, rad: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """
    The benchmark's manifest twice, with test_clean cut to the radiologist's studies: once with the labeler's
    labels, once with the radiologist's.
    """
    m = load_manifest(bench)
    test = (m["split"] == TEST) & m["in_test_clean"]
    keep = test & m["study_id"].isin(rad["study_id"])
    labeler = m.assign(in_test_clean=keep.to_numpy())
    reader = labeler.copy()
    theirs = rad.set_index("study_id").reindex(labeler.loc[keep, "study_id"])
    for c in LABELS:
        reader.loc[keep, c] = theirs[c].to_numpy()
    info = {"radiologist_studies": int(len(rad)), "in_official_test": int(keep.sum()),
            "patients": int(m.loc[keep, "subject_id"].nunique()),
            "not_in_benchmark_test": int((~rad["study_id"].isin(m.loc[test, "study_id"])).sum())}
    return labeler, reader, info


def joint_classes(labeler: pd.DataFrame, reader: pd.DataFrame, min_count: int = MIN_COUNT) -> dict:
    """§6 M7a: per policy, the classes measurable under both sets of labels on the radiologist's studies."""
    out = {}
    for metric in ("v2", "u0"):
        a = set(measurable(label_matrix(labeler[labeler["in_test_clean"]]), metric, min_count))
        b = set(measurable(label_matrix(reader[reader["in_test_clean"]]), metric, min_count))
        out[f"test_clean:{metric}"] = [c for c in LABELS if c in a and c in b]
    return out


def radiologist_report(bench: Path, labels_path: Path, prefix: str = PREFIX, candidates: Optional[list] = None,
                       stack: Optional[str] = None, stack_image: Optional[str] = None, n_boot: int = N_BOOT,
                       min_count: int = MIN_COUNT) -> dict:
    bench = Path(bench)
    if split_scheme(bench) != "official":
        raise SystemExit(f"{bench} isn't M5's benchmark directory (official split): pass --bench")
    rad, renamed = read_radiologist(Path(labels_path))
    labeler, reader, info = radiologist_manifests(bench, rad)
    if info["in_official_test"] == 0:
        raise SystemExit("none of the radiologist's studies is in the benchmark's official test split")
    frozen = joint_classes(labeler, reader, min_count)
    candidates = candidates or [f"{prefix}_{c}" for c in CANDIDATES]
    stack, stack_image = stack or f"{prefix}_{STACK}", stack_image or f"{prefix}_{STACK_IMAGE}"
    primary, val = pick_primary(bench, candidates)
    fixed = [run for run in dict.fromkeys((primary, stack)) if run and _exists(bench, run)]
    logger.info("M7a: %d radiologist studies in the official test split (%d patients); P = %s",
                info["in_official_test"], info["patients"], primary)

    lines = [f"# M7a: a radiologist's labels on the official test split ({prefix})", "",
             f"{info['in_official_test']:,} of the radiologist's {info['radiologist_studies']:,} studies are in the "
             f"benchmark's official test split ({info['patients']:,} patients)"
             + (f"; {info['not_in_benchmark_test']} are not (no frontal image, or not in the split)"
                if info["not_in_benchmark_test"] else "") + ". The radiologist read the same reports as the "
             "labeler, so these numbers measure the labeler's errors, not disagreement between report and image "
             "(docs/V2_PLAN.md §6 M7).", "",
             "Classes measurable under both sets of labels: v2 policy " + (", ".join(frozen["test_clean:v2"]) or "none")
             + "; positive vs rest " + (", ".join(frozen["test_clean:u0"]) or "none") + "."]
    if renamed:
        lines += ["", "The radiologist's file calls one category " + ", ".join(f"\"{r}\"" for r in renamed)
                  + "; it is read as " + ", ".join(OLD_NAMES[r] for r in renamed) + ", the labeler's later name."]

    lines += ["", "## The primary image model, chosen on the benchmark's val", "", "| Run | val (v2 policy) |",
              "|---|---|"]
    lines += [f"| `{run}`{' **(P)**' if run == primary else ''} | {fmt(s)} |" for run, s in val.items()]
    if not val:
        lines.append("| - | - |")

    cmp = {}
    lines += ["", "## The radiologist's labels minus the labeler's (paired, same studies)", "",
              "| Run | Policy | Labeler | Radiologist | Radiologist minus labeler |", "|---|---|---|---|---|"]
    for run in fixed:
        pred, _ = read_predictions(bench / "predictions" / f"{run}.parquet")
        for metric, name in (("v2", "v2"), ("u0", "positive vs rest")):
            if not frozen[f"test_clean:{metric}"]:
                continue
            lab = evaluate_set(pred, labeler, frozen, "test_clean", metric, n_boot)
            rdr = evaluate_set(pred, reader, frozen, "test_clean", metric, n_boot)
            c = compare_labels(pred, reader, labeler, frozen, "test_clean", metric, n_boot)
            cmp[(run, metric)] = c
            lines.append(f"| `{run}` | {name} | {fmt_ci(lab['macro_auroc'], lab['macro_auroc_ci'])} | "
                         f"{fmt_ci(rdr['macro_auroc'], rdr['macro_auroc_ci'])} | {fmt_ci(c['diff'], c['diff_ci'])} |")

    gain = {}
    lines += ["", f"## The text gain with each set of labels: `{stack}` minus `{stack_image}`", "",
              "| Policy | With the labeler's labels | With the radiologist's |", "|---|---|---|"]
    if _exists(bench, stack) and _exists(bench, stack_image):
        for metric, name in (("v2", "v2"), ("u0", "positive vs rest")):
            if not frozen[f"test_clean:{metric}"]:
                continue
            cells = []
            for which, man in (("labeler", labeler), ("radiologist", reader)):
                g = compare_runs(bench, stack, stack_image, "test_clean", metric, n_boot, write=False,
                                 manifest=man, frozen=frozen)
                gain[(which, metric)] = g
                cells.append(fmt_ci(g["diff"], g["diff_ci"]))
            lines.append(f"| {name} | " + " | ".join(cells) + " |")
    else:
        lines.append("| - | not run: needs both runs | |")

    if primary and (primary, "v2") in cmp:
        c = cmp[(primary, "v2")]
        Yl = label_matrix(labeler[labeler["in_test_clean"]])
        Yr = label_matrix(reader[reader["in_test_clean"]])
        lines += ["", f"## Per class: `{primary}`, v2 policy (Cardiomegaly first)", "",
                  "| Class | Positives: labeler / radiologist | Labeler | Radiologist | Radiologist minus labeler |",
                  "|---|---|---|---|---|"]
        for pc in sorted(c["per_class"], key=lambda p: (p["label"] != "Cardiomegaly", LABELS.index(p["label"]))):
            k = LABELS.index(pc["label"])
            lines.append(f"| {pc['label']} | {int((Yl[:, k] == 1).sum())} / {int((Yr[:, k] == 1).sum())} | "
                         f"{fmt(pc['b'])} | {fmt(pc['a'])} | {fmt_ci(pc['diff'], pc['diff_ci'])} |")

    lines += ["", "## Every benchmark run with the radiologist's labels (no inference)", "",
              "| Run | v2 policy | Positive vs rest |", "|---|---|---|"]
    for path in sorted((bench / "predictions").glob("*.parquet")):
        if path.stem.endswith("_smoke"):
            continue
        pred, _ = read_predictions(path)
        cells = []
        for metric in ("v2", "u0"):
            res = evaluate_set(pred, reader, frozen, "test_clean", metric, n_boot) \
                if frozen[f"test_clean:{metric}"] else None
            cells.append(fmt_ci(res["macro_auroc"], res["macro_auroc_ci"]) if res else "-")
        lines.append(f"| `{path.stem}` | " + " | ".join(cells) + " |")
    lines += ["", f"Power (§6 M7): {info['in_official_test']:,} studies from {info['patients']:,} patients give "
              "intervals of roughly ±0.02-0.03, so this can show a large effect, not confirm a small one."]
    out = bench / "reports" / "m7a_radiologist.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n")
    return {"path": out, "primary": primary, "val": val, "info": info, "frozen": frozen, "renamed": renamed,
            "label_effect": cmp, "text_gain": gain}


# ---------------------------------------------------------------------------
# M7b: the CheXpert test set
# ---------------------------------------------------------------------------

def chexpert_study_id(patient: int, study: int) -> int:
    return int(patient) * 1000 + int(study)


def find_one(root: Path, name: str) -> Optional[Path]:
    hits = sorted(Path(root).rglob(name))
    if len(hits) > 1:
        raise SystemExit(f"{len(hits)} files named {name} under {root}: keep one")
    return hits[0] if hits else None


def read_chexpert_labels(path: Path) -> pd.DataFrame:
    """
    CheXpert's test labels (binary) for our 12 classes, keyed by study_id; subject_id is the patient number.
    groundtruth.csv has one row per study (Study); CheXlocalize's test_labels.csv has one per image (Path),
    and a study's images must carry the same labels.
    """
    df = pd.read_csv(path, dtype={"Study": str, "Path": str})
    key = "Study" if "Study" in df.columns else ("Path" if "Path" in df.columns else None)
    missing = ([] if key else ["Study"]) + [c for c in LABELS if c not in df.columns]
    if missing:
        raise SystemExit(f"{Path(path).name} has no column(s) {missing}; its columns are {list(df.columns)}")
    keys = df[key].str.extract(STUDY_RE)
    if keys.isna().any(axis=None):
        raise SystemExit(f"{Path(path).name}: {int(keys.isna().any(axis=1).sum())} Study values name no patient and study")
    values = df[list(LABELS)].apply(pd.to_numeric, errors="coerce")
    if values.isna().any(axis=None) or not set(np.unique(values.to_numpy())) <= {0.0, 1.0}:
        raise SystemExit(f"{Path(path).name}: the test labels must all be 0 or 1")
    out = values.astype(np.float64)
    out.insert(0, "subject_id", keys[0].astype(np.int64).to_numpy())
    out.insert(0, "study_id", [chexpert_study_id(p, s) for p, s in zip(keys[0], keys[1])])
    if key == "Path":  # one row per image: collapse to one per study
        by_study = out.groupby("study_id", sort=False)
        if (by_study[list(LABELS)].nunique() > 1).any(axis=None):
            raise SystemExit(f"{Path(path).name}: the images of one study carry different labels")
        out = by_study.first().reset_index()[["study_id", "subject_id", *LABELS]]
    if out["study_id"].duplicated().any():
        raise SystemExit(f"{Path(path).name}: {int(out['study_id'].duplicated().sum())} studies appear twice")
    return out.reset_index(drop=True)


def label_files_agree(a: pd.DataFrame, b: pd.DataFrame) -> dict:
    both = a.merge(b, on="study_id", suffixes=("", "_other"))
    differ = int(sum((both[c] != both[f"{c}_other"]).sum() for c in LABELS))
    return {"studies_in_both": int(len(both)), "only_in_groundtruth": int(len(a) - len(both)),
            "only_in_other": int(len(b) - len(both)), "differing_labels": differ,
            "agree": bool(differ == 0 and len(both) == len(a) == len(b))}


def find_images(root: Path, studies: set) -> tuple[pd.DataFrame, dict]:
    """
    The frontal image of each labeled study: the frontal view with the lowest view number (§6 M7). Files of
    other studies (the validation set's, say) are ignored; a copy under a test/ folder wins over one elsewhere.
    """
    rows = []
    for p in Path(root).rglob("*"):
        rel = p.relative_to(root).as_posix()
        m = IMAGE_RE.search(rel)
        if not m or not p.is_file():
            continue
        sid = chexpert_study_id(m.group(1), m.group(2))
        if sid in studies:
            rows.append({"study_id": sid, "view_number": int(m.group(3)), "view": m.group(4).lower(), "rel": rel,
                         "under_test": "/test/" in f"/{rel}"})
    df = pd.DataFrame(rows, columns=["study_id", "view_number", "view", "rel", "under_test"])
    frontal = df[df["view"] == "frontal"].sort_values(["study_id", "view_number", "under_test", "rel"],
                                                      ascending=[True, True, False, True])
    pick = frontal.drop_duplicates("study_id").reset_index(drop=True)
    counts = {"image_files": int(len(df)), "frontal_files": int(len(frontal)),
              "lateral_files": int((df["view"] == "lateral").sum()),
              "duplicate_frontal_files": int(frontal.duplicated(["study_id", "view_number"]).sum()),
              "studies_with_a_frontal_image": int(len(pick))}
    return pick, counts


def rad_dino_exposure(source: str, patients: set) -> Optional[dict]:
    """CheXpert rows of RAD-DINO's training_images.csv, and how many come from these test patients (§6 M7)."""
    if source == "skip":
        return None
    if source == "auto":
        from huggingface_hub import hf_hub_download

        path = hf_hub_download("microsoft/rad-dino", "training_images.csv")
    else:
        path = source
    df = pd.read_csv(path, dtype=str)
    chex = df[df["rel_image_path"].str.startswith("Chexpert/")]
    ids = chex["rel_image_path"].str.extract(r"^Chexpert/p(\d+)_s", expand=False)
    if ids.isna().any():
        raise SystemExit(f"can't check RAD-DINO's list: {int(ids.isna().sum())} of its CheXpert rows name no patient")
    ids = ids.astype(np.int64)
    hit = ids.isin(patients)
    return {"chexpert_images": int(len(chex)), "chexpert_patients": int(ids.nunique()),
            "test_images_in_list": int(hit.sum()), "test_patients_in_list": int(ids[hit].nunique())}


def guard_external(out_dir: Path, bench: Path) -> None:
    if Path(out_dir).resolve() == Path(bench).resolve():
        raise SystemExit("--out must differ from the benchmark directory")
    if (Path(out_dir) / "audit.json").exists() and split_scheme(out_dir) != EXTERNAL:
        raise SystemExit(f"{out_dir} holds another v2 directory: give the CheXpert test set its own --out")


def prepare_chexpert(root: Path, out_dir: Path, bench: Path, rad_dino_list: str = "auto", workers: int = 4,
                     image_size: Optional[int] = None, image_format: Optional[str] = None,
                     min_count: int = MIN_COUNT, max_unreadable: float = 0.005) -> dict:
    """§6 M7b: manifest, images, measurable classes and audit for the CheXpert test set, in its own directory."""
    from .prepare_data import measurable_sets, write_images

    root, out_dir, bench = Path(root), Path(out_dir), Path(bench)
    guard_external(out_dir, bench)
    out_dir.mkdir(parents=True, exist_ok=True)
    bench_audit = json.loads((bench / "audit.json").read_text()) if (bench / "audit.json").exists() else {}
    size = image_size or int(bench_audit.get("image_size", 518))
    ext = image_format or bench_audit.get("image_format", "jpg")

    gt = find_one(root, "groundtruth.csv")
    if gt is None:
        raise SystemExit(f"no groundtruth.csv under {root}: download it from github.com/rajpurkarlab/"
                         "cheXpert-test-set-labels (§6 M7)")
    labels = read_chexpert_labels(gt)
    other = find_one(root, "test_labels.csv")
    agreement = label_files_agree(labels, read_chexpert_labels(other)) if other else None
    if agreement and not agreement["agree"]:
        raise SystemExit(f"groundtruth.csv and test_labels.csv disagree: {agreement}")
    pick, counts = find_images(root, set(labels["study_id"]))
    if counts["image_files"] == 0:
        raise SystemExit(f"no CheXpert test images under {root} for the {len(labels)} studies in groundtruth.csv: "
                         "download them first (§6 M7)")
    exposure = rad_dino_exposure(rad_dino_list, set(labels["subject_id"]))
    if exposure and exposure["test_patients_in_list"]:
        raise SystemExit(f"RED FLAG: RAD-DINO's training list holds {exposure['test_images_in_list']} images of "
                         f"{exposure['test_patients_in_list']} CheXpert test patients: stop (§6 M7)")
    if exposure is None:
        logger.warning("RAD-DINO's training list was not checked (--rad-dino-list skip)")

    df = labels.merge(pick[["study_id", "view_number", "rel"]], on="study_id", how="inner")
    drops = {"no frontal image": int(len(labels) - len(df))}
    df["dicom_id"] = "view" + df["view_number"].astype(str) + "_frontal"
    df["view"], df["study_datetime"], df["official_split"], df["split"] = "frontal", pd.NaT, "test", TEST
    df["in_test_clean"], df["in_test_v1"], df["source"], df["jpg"] = True, False, "chexpert test", df["rel"]
    df["image"] = "images/" + df["study_id"].astype(str) + "." + ext
    manifest = df[MANIFEST_COLUMNS].sort_values("study_id").reset_index(drop=True)
    failures = write_images(manifest, root, out_dir, size, workers)
    if failures:
        limit = max(5, int(max_unreadable * len(manifest)))
        kinds = pd.Series([e.split(":")[0] for e in failures.values()]).value_counts().to_dict()
        if len(failures) > limit:
            raise SystemExit(f"{len(failures)} of {len(manifest)} images failed to read {kinds}: check the download")
        logger.warning("%d images missing or unreadable %s: dropped", len(failures), kinds)
        bad = {Path(p).stem for p in failures}
        manifest = manifest[~manifest["study_id"].astype(str).isin(bad)].reset_index(drop=True)
    drops["image missing or unreadable"] = len(failures)
    manifest.to_parquet(out_dir / "manifest.parquet", index=False)

    created = datetime.now(timezone.utc).isoformat(timespec="seconds")
    frozen = measurable_sets(manifest, min_count)
    (out_dir / "measurable_classes.json").write_text(json.dumps(
        {"min_count": min_count, "commit": git_commit(), "created": created, **frozen}, indent=2))
    Y = label_matrix(manifest)
    audit = {"created": created, "commit": git_commit(), "split_scheme": EXTERNAL, "population": "CheXpert test set",
             "test_set_name": "CheXpert test", "image_size": size, "image_format": ext,
             "sizes": {"test_clean": {"studies": int(len(manifest)), "patients": int(manifest["subject_id"].nunique())}},
             "labeled_studies": int(len(labels)), "images": counts, "drops": drops,
             "positives": {c: int((Y[:, k] == 1).sum()) for k, c in enumerate(LABELS)},
             "measurable": frozen, "rad_dino": exposure, "label_files": agreement}
    (out_dir / "audit.json").write_text(json.dumps(audit, indent=2, default=str))
    (out_dir / "audit.md").write_text(external_audit_md(audit))
    logger.info("CheXpert test set: %d studies with a frontal image of %d labeled -> %s", len(manifest), len(labels),
                out_dir / "manifest.parquet")
    return audit


def external_audit_md(audit: dict) -> str:
    rad = audit["rad_dino"]
    lines = ["# Audit: the CheXpert test set (M7b)", "",
             f"{audit['sizes']['test_clean']['studies']:,} studies from {audit['sizes']['test_clean']['patients']:,} "
             f"patients, of {audit['labeled_studies']:,} in groundtruth.csv; one frontal image each, shorter side "
             f"{audit['image_size']} px ({audit['image_format']}).", "",
             "| Check | Result |", "|---|---|",
             f"| Image files found (frontal / lateral) | {audit['images']['frontal_files']:,} / "
             f"{audit['images']['lateral_files']:,} |",
             f"| Duplicate frontal files | {audit['images']['duplicate_frontal_files']} |"]
    lines += [f"| Dropped: {k} | {v} |" for k, v in audit["drops"].items()]
    lines.append("| RAD-DINO's training list | " + ("not checked" if rad is None else
                 f"{rad['chexpert_images']:,} CheXpert images from {rad['chexpert_patients']:,} patients; "
                 f"{rad['test_images_in_list']} from test patients (must be 0)") + " |")
    agreement = audit["label_files"]
    lines.append("| test_labels.csv | " + ("absent" if agreement is None else
                 ("agrees with groundtruth.csv" if agreement["agree"] else f"disagrees: {agreement}")) + " |")
    lines += ["", "| Class | Positives | Measurable (v2) |", "|---|---|---|"]
    lines += [f"| {c} | {n} | {'yes' if c in audit['measurable']['test_clean:v2'] else ''} |"
              for c, n in audit["positives"].items()]
    return "\n".join(lines) + "\n"


def apply_probe(bench: Path, out_dir: Path, run: str, workers: int, device: str, n_jobs: int) -> dict:
    """Refit the benchmark probe at each label's chosen C, check it reproduces, then score the external images."""
    from .embed import embeddings_path, load_npz
    from .embed import main as embed_main
    from .probe import DEFAULT_CS, fit_probe, split_xy

    pred, meta = read_predictions(Path(bench) / "predictions" / f"{run}.parquet")
    encoder, size = meta["encoder"], int(meta["size"])
    manifest = load_manifest(bench)
    ids, emb, _ = load_npz(embeddings_path(bench, encoder, size))
    parts = split_xy(manifest, ids, emb)
    del emb
    (tr_rows, X_tr), (va_rows, X_va), (te_rows, X_te) = parts[TRAIN], parts[VAL], parts[TEST]
    if meta.get("n_train") and int(meta["n_train"]) != len(tr_rows):
        raise SystemExit(f"{run} was fit on {meta['n_train']} training studies, not all {len(tr_rows)}: can't refit it")
    chosen = {label: info.get("C") for label, info in meta["per_label"].items()}
    probe = fit_probe(X_tr, label_matrix(tr_rows), X_va, label_matrix(va_rows), meta.get("Cs") or DEFAULT_CS,
                      n_jobs, chosen=chosen)
    rows = pd.concat([va_rows, te_rows], ignore_index=True)
    refit = probe.predict(np.concatenate([X_va, X_te]))
    stored = pred.set_index("study_id").loc[rows["study_id"], list(LABELS)].to_numpy(np.float64)
    diff = float(np.max(np.abs(refit - stored))) if len(rows) else float("nan")
    if not diff <= REPRODUCE_TOL:
        raise SystemExit(f"RED FLAG: the refit {run} differs from its benchmark predictions by up to {diff:.2e}: "
                         "stop (§6 M7)")
    logger.info("%s refit at its chosen C: largest difference from its benchmark predictions %.1e", run, diff)
    del X_tr, X_va, X_te, parts

    embed_main(["--model", encoder, "--size", str(size), "--out", str(out_dir), "--workers", str(workers),
                "--device", device])
    ext_ids, ext_emb, _ = load_npz(embeddings_path(out_dir, encoder, size))
    ext = load_manifest(out_dir)
    pos = pd.Series(np.arange(len(ext_ids)), index=ext_ids).reindex(ext["study_id"])
    if pos.isna().any():
        raise SystemExit(f"{int(pos.isna().sum())} external studies have no embedding: re-run")
    scores = probe.predict(ext_emb[pos.to_numpy(np.int64)])
    write_predictions(out_dir, run, ext["study_id"], ext["split"], scores, {
        "kind": "probe", "milestone": "M7", "source_run": run, "source_dir": str(bench), "encoder": encoder,
        "size": size, "reproduced_max_diff": diff, "per_label": probe.summary(), "commit": git_commit()})
    return {"run": run, "kind": "probe", "reproduced_max_diff": diff}


def apply_finetune(bench: Path, out_dir: Path, run: str, workers: int, device_name: str, batch_size: int = 32,
                   check_studies: int = 64) -> dict:
    """A benchmark fine-tune from its best checkpoint: checked on benchmark test studies, then the external set."""
    import torch
    from torch.utils.data import DataLoader

    from .data import ImageDataset, image_paths, loader_kwargs
    from .encoders import load_encoder
    from .finetune import AttentionClassifier, Classifier, predict

    path = Path(bench) / "runs" / run / "best.pt"
    if not path.exists():
        raise SystemExit(f"{path} not found: bring the fine-tune's best checkpoint home first (v2/README.md §3)")
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    device = pick_device(device_name)
    dtype = amp_dtype(device, "auto")
    enc = load_encoder(cfg["model"], int(cfg["size"]), device)
    if cfg.get("head") == "attention":
        model = AttentionClassifier(enc, int(cfg.get("grid", 12)), np.zeros(enc.token_dim), np.ones(enc.token_dim))
    else:
        model = Classifier(enc)
    model.load_state_dict(ckpt["model"])
    model.to(device)
    tf, kw = enc.eval_transform(), loader_kwargs(workers, device, persistent=False)

    def logits_for(directory: Path, rows: pd.DataFrame) -> np.ndarray:
        dl = DataLoader(ImageDataset(image_paths(directory, rows), tf), batch_size=batch_size, shuffle=False, **kw)
        return predict(model, dl, device, dtype)[0]

    pred, _ = read_predictions(Path(bench) / "predictions" / f"{run}.parquet")
    bm = load_manifest(bench)
    sample = bm[(bm["split"] == TEST) & bm["study_id"].isin(pred["study_id"])].sort_values("study_id")
    sample = sample.head(check_studies)
    got = logits_for(Path(bench), sample)
    want = pred.set_index("study_id").loc[sample["study_id"], list(LABELS)].to_numpy(np.float64)
    corr = float(np.corrcoef(got.ravel(), want.ravel())[0, 1])
    if not corr >= RELOAD_MIN_CORR:
        raise SystemExit(f"RED FLAG: {run} reloaded from {path.name} doesn't give back its benchmark predictions "
                         f"(correlation {corr:.3f} on {len(sample)} test studies): stop")
    logger.info("%s reloaded: correlation %.4f with its benchmark predictions on %d test studies", run, corr,
                len(sample))
    ext = load_manifest(out_dir)
    logits = logits_for(Path(out_dir), ext)
    write_predictions(out_dir, run, ext["study_id"], ext["split"], logits, {
        "kind": "finetune", "milestone": "M7", "head": cfg.get("head", "mean"), "source_run": run,
        "source_dir": str(bench), "best_epoch": ckpt.get("epoch"),
        "reload_check": {"studies": int(len(sample)), "correlation": corr,
                         "max_abs_diff": float(np.max(np.abs(got - want)))},
        **enc.describe(), "commit": git_commit()})
    return {"run": run, "kind": "finetune", "reload_correlation": corr}


def apply_models(bench: Path, out_dir: Path, runs: list, workers: int, device: str, n_jobs: int) -> list:
    done = []
    for run in runs:
        if not _exists(bench, run):
            raise SystemExit(f"{run} has no predictions in {bench}: is --bench M5's benchmark directory?")
        kind = read_predictions(Path(bench) / "predictions" / f"{run}.parquet")[1].get("kind")
        if kind == "probe":
            done.append(apply_probe(bench, out_dir, run, workers, device, n_jobs))
        elif kind == "finetune":
            done.append(apply_finetune(bench, out_dir, run, workers, device))
        else:
            raise SystemExit(f"{run} is a {kind} run: M7b applies probes and fine-tunes only")
    return done


def chexpert_report(bench: Path, out_dir: Path, candidates: list, n_boot: int = N_BOOT) -> dict:
    from .lineup import verdict

    bench, out_dir = Path(bench), Path(out_dir)
    audit = json.loads((out_dir / "audit.json").read_text())
    ext = load_manifest(out_dir)
    five = {"test_clean:v2": list(FIVE), "test_clean:u0": list(FIVE)}
    primary, val = pick_primary(bench, candidates)
    applied = [run for run in candidates if _exists(out_dir, run)]
    res, meta = {}, {}
    for run in applied:
        pred, meta[run] = read_predictions(out_dir / "predictions" / f"{run}.parquet")
        res[run] = {"five": evaluate_set(pred, ext, five, "test_clean", "v2", n_boot),
                    "measurable": _get(evaluate_run(out_dir, run, n_boot), "test_clean", "v2")}

    flags = []
    if audit.get("rad_dino") is None:
        flags.append("RAD-DINO's training list was not checked (prepare ran with --rad-dino-list skip)")
    flags += [f"`{run}`'s five-finding mean is {fmt(r['five']['macro_auroc'])}, above {RED_FLAG_MEAN}: check how "
              "labels and images are joined" for run, r in res.items() if r["five"]["macro_auroc"] > RED_FLAG_MEAN]

    gate = {"primary": primary, "verdict": "not run: P has no predictions on the CheXpert test set", "ci": None}
    if primary in res:
        p = res[primary]["five"]
        side = verdict(CHEXZERO_MEAN, p["macro_auroc_ci"])
        gate = {"primary": primary, "mean": p["macro_auroc"], "ci": p["macro_auroc_ci"], "chexzero_is": side,
                "verdict": ("FAIL: CheXzero transfers better" if side == "**above ours**" else
                            "PASS: at least level with CheXzero" if side in ("level", "below ours") else "n/a")}

    n = audit["sizes"]["test_clean"]
    lines = [f"# M7b: the CheXpert test set ({n['studies']:,} studies)", "",
             f"External and image only (docs/V2_PLAN.md §6 M7): {n['studies']:,} Stanford studies from "
             f"{n['patients']:,} patients, one frontal image each; ground truth from the majority vote of 5 "
             "radiologists. Nothing was trained or chosen on this set.", "", "## Red flags (read these first)", ""]
    lines += [f"- **RED FLAG:** {f}" for f in flags] or ["- none"]

    rad = audit.get("rad_dino")
    lines += ["", "## Checks", ""]
    if rad:
        lines.append(f"- RAD-DINO's training list: {rad['chexpert_images']:,} CheXpert images from "
                     f"{rad['chexpert_patients']:,} patients, {rad['test_images_in_list']} of them from this test set's "
                     "patients.")
    for run in applied:
        m = meta[run]
        if m.get("kind") == "probe":
            lines.append(f"- `{run}` refit at each label's chosen C gives back its benchmark predictions: largest "
                         f"difference {m['reproduced_max_diff']:.1e} (allowed {REPRODUCE_TOL:.0e}).")
        else:
            rc = m["reload_check"]
            lines.append(f"- `{run}` reloaded from its best checkpoint: correlation {rc['correlation']:.4f} with its "
                         f"benchmark predictions on {rc['studies']} test studies.")

    lines += ["", "## Gate G7: against CheXzero", "",
              "The primary model P is the candidate with the best benchmark val score, fixed before this set was "
              "scored.", "", "| Run | Benchmark val (v2 policy) | Five findings on CheXpert |", "|---|---|---|"]
    for run, s in val.items():
        five_cell = fmt_ci(res[run]["five"]["macro_auroc"], res[run]["five"]["macro_auroc_ci"]) if run in res else "-"
        lines.append(f"| `{run}`{' **(P)**' if run == primary else ''} | {fmt(s)} | {five_cell} |")
    if gate["ci"]:
        lines += ["", f"**G7** (`{primary}`'s five-finding mean, {fmt_ci(gate['mean'], gate['ci'])}, against "
                  f"CheXzero's {CHEXZERO_MEAN}): CheXzero is {gate['chexzero_is'].strip('*')} -> **{gate['verdict']}**."]
    else:
        lines += ["", f"**G7**: {gate['verdict']}"]

    Y = label_matrix(ext)
    lines += ["", "## The five findings, per class (AUROC [95% CI])", "",
              "| Class | Positives | " + " | ".join(f"`{r}`" for r in applied) + " | CheXzero |",
              "|---|---|" + "---|" * (len(applied) + 1)]
    for label in FIVE:
        k = LABELS.index(label)
        cells = [fmt_ci(res[r]["five"]["per_class"][k]["auroc"], res[r]["five"]["per_class"][k]["auroc_ci"])
                 for r in applied]
        z = CHEXZERO[label]
        lines.append(f"| {label} | {int((Y[:, k] == 1).sum())} | " + " | ".join(cells) + f" | {fmt_ci(z[0], z[1:])} |")
    lines.append("| **Mean** | | " + " | ".join(fmt_ci(res[r]["five"]["macro_auroc"], res[r]["five"]["macro_auroc_ci"])
                                           for r in applied) + f" | {CHEXZERO_MEAN} |")
    lines += ["", "In-domain, for context only (trained on CheXpert's own labels): " + ", ".join(
        f"{name} {v}" for name, v in IN_DOMAIN.items()) + " (Tiu et al., Nat. Biomed. Eng. 2022, Table 2)."]

    lines += ["", "## Our 12 classes where measurable here (secondary)", "", "| Run | Classes | Macro AUROC |",
              "|---|---|---|"]
    for run in applied:
        r = res[run]["measurable"]
        lines.append(f"| `{run}` | {len(r['classes'])} | {fmt_ci(r['macro_auroc'], r['macro_auroc_ci'])} |" if r else
                     f"| `{run}` | - | - |")

    if primary and _exists(bench, primary):
        pred_b, _ = read_predictions(bench / "predictions" / f"{primary}.parquet")
        bm = load_manifest(bench)
        mimic = {metric: evaluate_set(pred_b, bm, five, "test_clean", metric, n_boot) for metric in ("v2", "u0")}
        here = fmt_ci(res[primary]["five"]["macro_auroc"], res[primary]["five"]["macro_auroc_ci"]) \
            if primary in res else "-"
        lines += ["", "## The transfer gap (descriptive: different labels and patients)", "",
                  f"`{primary}`'s five-finding mean on MIMIC's official test: v2 policy "
                  f"{fmt_ci(mimic['v2']['macro_auroc'], mimic['v2']['macro_auroc_ci'])}, positive vs rest "
                  f"{fmt_ci(mimic['u0']['macro_auroc'], mimic['u0']['macro_auroc_ci'])}; on the CheXpert test set {here}."]
    lines += ["", "Caveat (§6 M7): RAD-DINO pretrained on CheXpert's training and validation images from this "
              "hospital, without labels and without this test set, so M7b tests how heads trained on MIMIC's labels "
              "transfer with an encoder that knows the domain. CheXzero's encoder didn't."]
    out = out_dir / "reports" / "m7b_chexpert.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n")
    return {"path": out, "gate": gate, "flags": flags, "primary": primary, "val": val, "results": res}


# ---------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------

def main(argv: Optional[list[str]] = None) -> dict:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    for name, text in (("radiologist", "M7a: the radiologist's labels on the official test split"),
                       ("chexpert", "M7b: the CheXpert test set")):
        p = sub.add_parser(name, help=text, description=text)
        p.add_argument("--bench", help="M5's benchmark dir (default: BENCH_OUT, else $OUTPUT_PATH/v2_bench)")
        p.add_argument("--prefix", default=PREFIX, help="the benchmark's image run prefix")
        p.add_argument("--candidates", help="comma-separated image runs P is chosen from (default: the prefix's "
                                            + ", ".join(CANDIDATES) + ")")
        p.add_argument("--n-boot", type=int, default=N_BOOT)
        p.add_argument("--min-count", type=int, default=MIN_COUNT, help="Measurable-class threshold (§3.2)")
        p.add_argument("-v", "--verbose", action="store_true")
        if name == "radiologist":
            p.add_argument("--labels", help=f"the radiologist's labels (default: MIMIC_CXR_JPG_PATH/{RADIOLOGIST_FILE})")
            p.add_argument("--stack", help=f"the image + indication run (default: the prefix's {STACK})")
            p.add_argument("--stack-image", help=f"its image run (default: the prefix's {STACK_IMAGE})")
        else:
            p.add_argument("--chexpert", help="folder with the test images and groundtruth.csv "
                                              "(default: CHEXPERT_TEST_PATH)")
            p.add_argument("--out", help="its own output dir (default: EXT_OUT, else $OUTPUT_PATH/v2_chexpert)")
            p.add_argument("--steps", default="prepare,apply,report")
            p.add_argument("--rad-dino-list", default="auto",
                           help="'auto' (download from Hugging Face), a local training_images.csv, or 'skip'")
            p.add_argument("--image-size", type=int, help="default: the benchmark's")
            p.add_argument("--image-format", choices=("png", "jpg"), help="default: the benchmark's")
            p.add_argument("--device", default="auto", help="auto | cuda | mps | cpu")
            p.add_argument("--workers", type=int, default=min(4, os.cpu_count() or 1))
            p.add_argument("--jobs", type=int, default=min(12, os.cpu_count() or 1))
    args = parser.parse_args(argv)
    setup_logging(args.verbose)
    bench = Path(args.bench) if args.bench else _env_dir("BENCH_OUT", "v2_bench")
    candidates = [c.strip() for c in args.candidates.split(",")] if args.candidates else \
        [f"{args.prefix}_{c}" for c in CANDIDATES]

    if args.command == "radiologist":
        load_dotenv()
        labels = args.labels or (Path(os.environ["MIMIC_CXR_JPG_PATH"]) / RADIOLOGIST_FILE
                                 if os.environ.get("MIMIC_CXR_JPG_PATH") else None)
        if not labels or not Path(labels).exists():
            raise SystemExit(f"the radiologist's labels not found: pass --labels or set MIMIC_CXR_JPG_PATH "
                             f"({RADIOLOGIST_FILE} is in MIMIC-CXR-JPG 2.1.0)")
        out = radiologist_report(bench, Path(labels), args.prefix, candidates, args.stack, args.stack_image,
                                 args.n_boot, args.min_count)
        for (run, metric), c in out["label_effect"].items():
            logger.info("%s (%s): radiologist minus labeler %s", run, metric, fmt_ci(c["diff"], c["diff_ci"]))
        logger.info("Report: %s", out["path"])
        return out

    load_dotenv()
    root = Path(args.chexpert) if args.chexpert else (Path(os.environ["CHEXPERT_TEST_PATH"])
                                                      if os.environ.get("CHEXPERT_TEST_PATH") else None)
    out_dir = Path(args.out) if args.out else _env_dir("EXT_OUT", "v2_chexpert")
    steps = {s.strip() for s in args.steps.split(",")}
    unknown = steps - {"prepare", "apply", "report"}
    if unknown:
        raise SystemExit(f"Unknown steps {sorted(unknown)}: choose from prepare, apply, report")
    result: dict = {}
    if "prepare" in steps:
        if root is None:
            raise SystemExit("the CheXpert test set's folder is unknown: pass --chexpert or set CHEXPERT_TEST_PATH")
        result["audit"] = prepare_chexpert(root, out_dir, bench, args.rad_dino_list, args.workers, args.image_size,
                                           args.image_format, args.min_count)
    if "apply" in steps:
        result["applied"] = apply_models(bench, out_dir, candidates, args.workers, args.device, args.jobs)
    if "report" in steps:
        result["report"] = chexpert_report(bench, out_dir, candidates, args.n_boot)
        for flag in result["report"]["flags"]:
            logger.warning("RED FLAG: %s", flag)
        logger.info("G7: %s; report: %s", result["report"]["gate"]["verdict"], result["report"]["path"])
    return result


if __name__ == "__main__":
    main()
