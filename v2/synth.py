"""
A synthetic MIMIC-CXR-JPG tree, for tests and the wiring check (v2.smoke).

Same file names, columns and folder layout as the real dataset, plus v1-style
cohort files, but with small images that carry a planted signal: for each
positive label k, cell k of a 3x4 grid on the *frontal* image is brightened.
Lateral images carry no signal. If a pipeline picks a lateral view, or
misaligns images and labels, the planted signal disappears and AUROC falls to
about 0.5, which is what the wiring check looks for.

For M4 it also writes free-text reports (mimic-cxr-reports.zip), MIMIC-IV-ED
stays and triage, and MIMIC-IV patients. Indications and chief complaints carry
a weak planted signal. Every report's findings, impression and wet read carry a
perfect one: the marker leak{study_id} and the names of its positive labels.
That text must never reach an M4 input for its own study.

    python -m v2.synth --out /tmp/fake_mimic --patients 300
"""

from __future__ import annotations

import argparse
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from PIL import Image

from .common import LABELS, jpg_path

# Real column order of mimic-cxr-2.0.0-chexpert.csv.gz.
CHEXPERT_COLUMNS = (
    "Atelectasis", "Cardiomegaly", "Consolidation", "Edema",
    "Enlarged Cardiomediastinum", "Fracture", "Lung Lesion", "Lung Opacity",
    "No Finding", "Pleural Effusion", "Pleural Other", "Pneumonia",
    "Pneumothorax", "Support Devices",
)
METADATA_COLUMNS = (
    "dicom_id", "subject_id", "study_id", "PerformedProcedureStepDescription",
    "ViewPosition", "Rows", "Columns", "StudyDate", "StudyTime",
    "ProcedureCodeSequence_CodeMeaning", "ViewCodeSequence_CodeMeaning",
    "PatientOrientationCodeSequence_CodeMeaning",
)
GRID = (3, 4)  # rows x cols: one cell per label
# Positive rate per label in abnormal studies: common ones high, rare ones low.
POS_RATE = (0.35, 0.35, 0.15, 0.3, 0.12, 0.08, 0.1, 0.45, 0.4, 0.08, 0.15, 0.12)

# Study layouts: list of views per study, with their probabilities.
LAYOUTS = (
    (("PA", "LATERAL"), 0.40),
    (("AP",), 0.25),
    (("AP", "LL"), 0.10),
    (("PA", "AP", "LATERAL"), 0.08),   # PA must win over AP
    (("AP", "AP"), 0.07),              # tie: lowest dicom_id wins
    (("LATERAL",), 0.05),              # no frontal: dropped
    (("PA",), 0.05),
)
VIEW_MEANING = {"PA": "postero-anterior", "AP": "antero-posterior", "LATERAL": "lateral", "LL": "left lateral"}

# M4 text: label -> (phrase, rate when positive, rate otherwise), planted in indications and chief complaints.
HINTS = {
    "Pneumonia": ("fever and productive cough, eval for pneumonia", 0.6, 0.1),
    "Edema": ("worsening dyspnea, history of chf", 0.5, 0.1),
    "Pneumothorax": ("chest tube check", 0.5, 0.05),
}
NEUTRAL_HINTS = ("chest pain", "routine preoperative film", "line placement", "weakness", "fall from standing")
COMPLAINTS = {"Pneumonia": ("Fever", 0.5, 0.1), "Edema": ("Dyspnea", 0.4, 0.1)}
NEUTRAL_COMPLAINTS = ("Chest pain", "Abd pain", "Fall", "Weakness", "Cough")
# Report layouts seen in MIMIC-CXR, including the ones that could leak findings into an indication.
REPORT_STYLES = (("standard", 0.5), ("wet_read", 0.15), ("history", 0.1), ("inline", 0.1),
                 ("headerless_findings", 0.08), ("no_indication", 0.07))


@dataclass
class SynthPaths:
    mimic_root: Path
    cohort_dir: Optional[Path]
    rad_dino_list: Path
    v1_val_subjects: set
    official: dict  # subject_id -> official split
    reports: Optional[Path] = None      # mimic-cxr-reports.zip
    mimic_iv_ed: Optional[Path] = None  # MIMIC-IV-ED root (ed/edstays, ed/triage)
    mimic_iv: Optional[Path] = None     # MIMIC-IV root (hosp/patients)


def _dicom_id(rng: np.random.Generator) -> str:
    return "-".join(f"{int(x):08x}" for x in rng.integers(0, 2**32, size=5))


def _render(rng: np.random.Generator, labels: np.ndarray, view: str, size: tuple[int, int]) -> np.ndarray:
    h, w = size
    img = rng.normal(90.0, 15.0, size=(h, w))
    img += np.linspace(0, 30, h)[:, None]  # mild vertical gradient, like a real film
    if view in ("PA", "AP"):
        ch, cw = h // GRID[0], w // GRID[1]
        for k, value in enumerate(labels):
            if value == 1.0:
                r, c = divmod(k, GRID[1])
                img[r * ch:(r + 1) * ch, c * cw:(c + 1) * cw] += 70.0
    else:
        img = img[:, ::-1] * 0.85  # laterals: no planted signal
    return np.clip(img, 0, 255).astype(np.uint8)


def _study_labels(rng: np.random.Generator) -> tuple[np.ndarray, float]:
    """12 raw pathology values (1 / 0 / -1 / NaN) and the No Finding value."""
    if rng.random() < 0.3:  # normal study
        raw = np.where(rng.random(12) < 0.2, 0.0, np.nan)
        return raw, 1.0
    raw = np.full(12, np.nan)
    for k in range(12):
        u = rng.random()
        if u < POS_RATE[k]:
            raw[k] = 1.0
        elif u < POS_RATE[k] + 0.06:
            raw[k] = -1.0
        elif u < POS_RATE[k] + 0.25:
            raw[k] = 0.0
    if not (raw == 1.0).any():
        raw[int(rng.integers(0, 12))] = 1.0
    return raw, np.nan


def make_dataset(
    root: Path,
    n_patients: int = 120,
    seed: int = 0,
    image_size: tuple[int, int] = (120, 100),
    with_cohort: bool = True,
) -> SynthPaths:
    """Write a synthetic MIMIC-CXR-JPG tree (and v1 cohort files) under `root`."""
    rng = np.random.default_rng(seed)
    root = Path(root)
    mimic = root / "mimic-cxr-jpg" / "2.1.0"
    mimic.mkdir(parents=True, exist_ok=True)

    layouts = [layout for layout, _ in LAYOUTS]
    probs = np.array([p for _, p in LAYOUTS])
    probs = probs / probs.sum()

    meta_rows, split_rows, label_rows, cohort_rows, studies = [], [], [], [], []
    official = {}
    study_counter = 50_000_000
    for i in range(n_patients):
        subject_id = 10_000_000 + i * 7919 % 9_000_000
        u = rng.random()
        official[subject_id] = "train" if u < 0.7 else ("validate" if u < 0.8 else "test")
        n_studies = int(rng.integers(1, 5))
        for s in range(n_studies):
            study_counter += int(rng.integers(1, 5000))
            study_id = study_counter
            raw, no_finding = _study_labels(rng)
            studies.append((subject_id, study_id, raw))
            views = layouts[int(rng.choice(len(layouts), p=probs))]
            date = 21500101 + int(rng.integers(0, 28)) + 100 * int(rng.integers(0, 12)) + 10000 * s
            # MIMIC stores StudyTime as a float, so early times lose their leading zeros.
            hhmmss = int(rng.integers(0, 24)) * 10000 + int(rng.integers(0, 60)) * 100 + int(rng.integers(0, 60))
            time = float(f"{hhmmss}.{int(rng.integers(0, 1000)):03d}")
            dicoms = []
            for view in views:
                dicom = _dicom_id(rng)
                dicoms.append((dicom, view))
                h, w = image_size if view in ("PA", "AP") else image_size[::-1]
                meta_rows.append({
                    "dicom_id": dicom, "subject_id": subject_id, "study_id": study_id,
                    "PerformedProcedureStepDescription": "CHEST (PA AND LAT)" if "PA" in views else "CHEST (PORTABLE AP)",
                    "ViewPosition": view, "Rows": h, "Columns": w,
                    "StudyDate": date, "StudyTime": time,
                    "ProcedureCodeSequence_CodeMeaning": "CHEST (PA AND LAT)",
                    "ViewCodeSequence_CodeMeaning": VIEW_MEANING[view],
                    "PatientOrientationCodeSequence_CodeMeaning": "Erect" if view == "PA" else "Recumbent",
                })
                split_rows.append({"dicom_id": dicom, "study_id": study_id, "subject_id": subject_id,
                                   "split": official[subject_id]})
                path = jpg_path(mimic, subject_id, study_id, dicom)
                path.parent.mkdir(parents=True, exist_ok=True)
                Image.fromarray(_render(rng, raw, view, (h, w))).save(path, quality=95)

            row = {"subject_id": subject_id, "study_id": study_id}
            row.update({name: np.nan for name in CHEXPERT_COLUMNS})
            row.update({label: raw[k] for k, label in enumerate(LABELS)})
            row["No Finding"] = no_finding
            row["Support Devices"] = 1.0 if rng.random() < 0.2 else np.nan
            if rng.random() > 0.02:  # a few studies have no CheXpert row, like the real data
                label_rows.append(row)
                # v1's cohort misses some abnormal studies (its ED-linking step drops them).
                if (raw == 1.0).any() and rng.random() < 0.85:
                    cohort_rows.append((row, date, time))

    # One frontal JPG that exists in the metadata but not on disk.
    frontal = [r for r in meta_rows if r["ViewPosition"] in ("PA", "AP")]
    missing = frontal[len(frontal) // 2]
    jpg_path(mimic, missing["subject_id"], missing["study_id"], missing["dicom_id"]).unlink()

    pd.DataFrame(meta_rows, columns=list(METADATA_COLUMNS)).to_csv(mimic / "mimic-cxr-2.0.0-metadata.csv.gz", index=False)
    pd.DataFrame(split_rows).to_csv(mimic / "mimic-cxr-2.0.0-split.csv.gz", index=False)
    pd.DataFrame(label_rows, columns=["subject_id", "study_id", *CHEXPERT_COLUMNS]).to_csv(
        mimic / "mimic-cxr-2.0.0-chexpert.csv.gz", index=False
    )

    # A RAD-DINO-style training list: every official-train MIMIC image, plus other datasets.
    train_dicoms = [r for r in split_rows if r["split"] == "train"]
    rad_rows = [{"image_id": f"image-{k:08d}", "rel_image_path": f"Chexpert/image-{k:08d}.png"} for k in range(5)]
    for r in train_dicoms:
        sid = str(r["subject_id"])
        rad_rows.append({
            "image_id": r["dicom_id"],
            "rel_image_path": f"mimic-cxr-2.0.0_png_518/files/p{sid[:2]}/p{sid}/s{r['study_id']}/{r['dicom_id']}.png",
        })
    rad_list = root / "rad_dino_training_images.csv"
    pd.DataFrame(rad_rows).to_csv(rad_list, index=False)

    # Text for M4 comes from its own generator, so the images and labels above do not change.
    trng = np.random.default_rng(seed + 1000)
    reports = _write_reports(root / "mimic-cxr-reports.zip", studies, trng)

    cohort_dir, v1_val_subjects, ed_root, iv_root = None, set(), None, None
    if with_cohort:
        cohort_dir = root / "cohorts"
        cohort_dir.mkdir(parents=True, exist_ok=True)
        cohort = _cohort_frame(rng, cohort_rows)
        subjects = np.array(sorted(cohort["subject_id"].unique()))
        v1_val_subjects = set(rng.choice(subjects, size=max(1, int(0.15 * len(subjects))), replace=False).tolist())
        is_val = cohort["subject_id"].isin(v1_val_subjects)
        cohort.to_parquet(cohort_dir / "anomalous_full.parquet", index=False)
        cohort[~is_val].to_parquet(cohort_dir / "anomalous_train.parquet", index=False)
        cohort[is_val].to_parquet(cohort_dir / "anomalous_val.parquet", index=False)
        ed_root = _write_ed_tables(root / "mimic-iv-ed" / "2.2", cohort, trng)
        iv_root = _write_patients(root / "mimiciv" / "3.1", cohort)

    return SynthPaths(mimic, cohort_dir, rad_list, v1_val_subjects, official, reports, ed_root, iv_root)


def _pick(rng: np.random.Generator, raw: np.ndarray, table: dict, neutral: tuple) -> str:
    """Planted phrases for the study's labels, or a neutral one."""
    picks = [phrase for label, (phrase, pos, neg) in table.items()
             if rng.random() < (pos if raw[LABELS.index(label)] == 1.0 else neg)]
    return ", ".join(picks) if picks else neutral[int(rng.integers(len(neutral)))]


def _report(rng: np.random.Generator, study_id: int, raw: np.ndarray) -> str:
    leak = f"leak{study_id} " + " ".join(LABELS[k].lower() for k in range(len(LABELS)) if raw[k] == 1.0)
    indication = "___M with " + _pick(rng, raw, HINTS, NEUTRAL_HINTS)
    probs = np.array([p for _, p in REPORT_STYLES])
    style = REPORT_STYLES[int(rng.choice(len(REPORT_STYLES), p=probs / probs.sum()))][0]
    title = "                                 FINAL REPORT\n"
    if style == "inline":
        return (f" CHEST RADIOGRAPH PERFORMED ON ___  COMPARISON: ___. CLINICAL HISTORY: {indication}. "
                f"FINDINGS: {leak}. IMPRESSION: {leak}.\n")
    if style == "headerless_findings":
        return f"{title} INDICATION:  {indication}\n\n {leak}. The lungs are clear.\n"
    text = title + " EXAMINATION:  CHEST (PA AND LAT)\n\n"
    if style != "no_indication":
        text += f" {'HISTORY' if style == 'history' else 'INDICATION'}:  {indication}\n\n"
    text += (" TECHNIQUE:  Chest PA and lateral\n\n COMPARISON:  ___\n\n"
             f" FINDINGS:\n\n {leak}. The cardiomediastinal silhouette is stable.\n\n IMPRESSION:\n\n {leak}.\n")
    if style == "wet_read":
        text = f" WET READ: ___ ___ 10:12 PM\n   {leak}\n {'_' * 78}\n" + text
    return text


def _write_reports(path: Path, studies: list, rng: np.random.Generator) -> Path:
    """One report per study, as in mimic-cxr-reports.zip; about 2% are missing."""
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zf:
        for subject_id, study_id, raw in studies:
            text = _report(rng, study_id, raw)
            if rng.random() < 0.02:
                continue
            sid = str(subject_id)
            zf.writestr(f"files/p{sid[:2]}/p{sid}/s{study_id}.txt", text)
    return path


def _write_ed_tables(root: Path, cohort: pd.DataFrame, rng: np.random.Generator) -> Path:
    """MIMIC-IV-ED edstays and triage for the cohort's ED stays, plus decoys that must never be linked."""
    stays = cohort.dropna(subset=["stay_id"]).copy()
    stays["chiefcomplaint"] = [_pick(rng, r, COMPLAINTS, NEUTRAL_COMPLAINTS) for r in stays[list(LABELS)].to_numpy()]
    stays.loc[stays["intime"] > stays["study_datetime"], "chiefcomplaint"] = "later visit"   # arrival after the X-ray
    stale = stays.iloc[::10].copy()                                                       # ended long before it
    stale["stay_id"] = stale["stay_id"] + 50_000_000
    stale["intime"] = stale["study_datetime"] - pd.to_timedelta(rng.uniform(30, 60, len(stale)), unit="h")
    stale["outtime"] = stale["intime"] + pd.Timedelta(hours=6)
    stale["chiefcomplaint"] = "stale visit"
    stays = pd.concat([stays, stale], ignore_index=True)
    stays["stay_id"] = stays["stay_id"].astype(np.int64)
    (root / "ed").mkdir(parents=True, exist_ok=True)
    fmt = "%Y-%m-%d %H:%M:%S"
    edstays = pd.DataFrame({
        "subject_id": stays["subject_id"], "hadm_id": np.nan, "stay_id": stays["stay_id"],
        "intime": stays["intime"].dt.strftime(fmt), "outtime": stays["outtime"].dt.strftime(fmt),
        "gender": stays["gender"], "race": "UNKNOWN", "arrival_transport": "WALK IN", "disposition": "ADMITTED",
    })
    triage = pd.DataFrame({"subject_id": stays["subject_id"], "stay_id": stays["stay_id"],
                           **{c: stays[f"triage_{c}"] for c in ("temperature", "heartrate", "resprate", "o2sat",
                                                                 "sbp", "dbp")},
                           "pain": "3", "acuity": stays["triage_acuity"], "chiefcomplaint": stays["chiefcomplaint"]})
    edstays.to_csv(root / "ed" / "edstays.csv.gz", index=False)
    triage.to_csv(root / "ed" / "triage.csv.gz", index=False)
    return root


def _write_patients(root: Path, cohort: pd.DataFrame) -> Path:
    """MIMIC-IV hosp/patients for the cohort's patients (anchor year 2150, like the synthetic study dates)."""
    (root / "hosp").mkdir(parents=True, exist_ok=True)
    patients = cohort.drop_duplicates("subject_id")[["subject_id", "gender", "anchor_age"]].assign(
        anchor_year=2150, anchor_year_group="2017 - 2019", dod=np.nan)
    patients.to_csv(root / "hosp" / "patients.csv.gz", index=False)
    return root


def _cohort_frame(rng: np.random.Generator, rows: list) -> pd.DataFrame:
    """v1-style cohort rows: labels, demographics, ED stay times and triage vitals."""
    out = []
    for row, date, time in rows:
        study_dt = pd.to_datetime(f"{date}{int(time):06d}", format="%Y%m%d%H%M%S")
        raw = np.array([row[label] for label in LABELS], dtype=float)
        rec = dict(row)
        rec["study_datetime"] = study_dt
        rec["gender"] = "M" if rng.random() < 0.5 else "F"
        rec["anchor_age"] = int(rng.integers(18, 91))
        if rng.random() < 0.7:
            # Mostly arrival before the X-ray; some after (must be excluded by fuse).
            offset_h = rng.uniform(0.5, 6.0) if rng.random() < 0.85 else -rng.uniform(0.5, 6.0)
            rec["stay_id"] = int(rng.integers(30_000_000, 39_999_999))
            rec["intime"] = study_dt - pd.Timedelta(hours=offset_h)
            rec["outtime"] = rec["intime"] + pd.Timedelta(hours=8)
            # Planted vitals signal: fever with pneumonia, hypoxia with edema.
            rec["triage_temperature"] = 98.2 + 2.5 * (raw[10] == 1.0) + rng.normal(0, 0.8)
            rec["triage_heartrate"] = 85 + 12 * (raw[8] == 1.0) + rng.normal(0, 10)
            rec["triage_resprate"] = 18 + rng.normal(0, 2)
            rec["triage_o2sat"] = 97 - 5 * (raw[3] == 1.0) + rng.normal(0, 1.5)
            rec["triage_sbp"] = 130 + rng.normal(0, 15)
            rec["triage_dbp"] = 75 + rng.normal(0, 10)
            rec["triage_acuity"] = float(rng.integers(1, 6))
            if rng.random() < 0.1:
                rec["triage_o2sat"] = np.nan
        else:
            for col in ("stay_id", "intime", "outtime", "triage_temperature", "triage_heartrate",
                        "triage_resprate", "triage_o2sat", "triage_sbp", "triage_dbp", "triage_acuity"):
                rec[col] = pd.NaT if col in ("intime", "outtime") else np.nan
        out.append(rec)
    return pd.DataFrame(out)


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", required=True, help="Directory to create the synthetic tree in")
    parser.add_argument("--patients", type=int, default=120)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    paths = make_dataset(Path(args.out), n_patients=args.patients, seed=args.seed)
    print(f"MIMIC-CXR-JPG root: {paths.mimic_root}")
    print(f"v1 cohort files:    {paths.cohort_dir}")
    print(f"RAD-DINO list:      {paths.rad_dino_list}")
    print(f"Reports (M4):       {paths.reports}")
    print(f"MIMIC-IV-ED (M4):   {paths.mimic_iv_ed}")
    print(f"MIMIC-IV (M4):      {paths.mimic_iv}")


if __name__ == "__main__":
    main()
