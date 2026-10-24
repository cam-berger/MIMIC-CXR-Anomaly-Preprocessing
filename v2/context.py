"""
M4 inputs: what was written before each X-ray (docs/ANALYSIS.md, M4).

    python -m v2.context --reports /path/to/mimic-cxr-reports.zip --mimic-iv-ed /path/to/mimic-iv-ed/2.2

For every manifest study:

    indication       the INDICATION / HISTORY / CLINICAL INFORMATION / REASON FOR EXAM
                     sections of the study's own report, and nothing else from it: the
                     labels come from the rest
    chief_complaint  MIMIC-IV-ED triage chief complaint of the latest ED stay that began
                     at or before the X-ray and at most 24 h before it
    prior_report     FINDINGS and IMPRESSION of the patient's latest MIMIC-CXR study taken
                     at least 24 h before (reports carry no sign time; 24 h makes it near
                     certain the report existed)
    vitals           triage vitals and acuity of that ED stay, sex, and age (with --mimic-iv)
    prior labels     (M4b) the 14 CheXpert values of that same earlier study, from
                     mimic-cxr-2.0.0-chexpert.csv.gz: the labeler's reading of the prior report

Writes context.parquet, which holds record-level text: keep it local and never
print it. context_audit.md and context_audit.json hold counts only.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from .common import CHEXPERT_COLUMNS, TRAIN, VAL, default_out_dir, git_commit, load_dotenv, load_manifest, mimic_csv
from .common import setup_logging
from .prepare_data import study_datetime

logger = logging.getLogger("v2.context")

TEXT_SOURCES = ("indication", "chief_complaint", "prior_report")
TRIAGE = ("temperature", "heartrate", "resprate", "o2sat", "sbp", "dbp", "acuity")
ED_WINDOW = pd.Timedelta(hours=24)       # chief complaint: ED arrival at most this long before the X-ray
PRIOR_MIN_GAP = pd.Timedelta(hours=24)   # prior report: that study at least this long before the X-ray
INDICATION_MAX_CHARS = 400
PRIOR_MAX_CHARS = 4000
PRIOR_LABEL_COLUMNS = tuple(f"prior_label::{name}" for name in CHEXPERT_COLUMNS)

# ---------------------------------------------------------------------------
# Report sections
# ---------------------------------------------------------------------------

# A section header: an upper-case name and a colon at the start of a line.
_LINE_HEADER = re.compile(r"(?m)^[ \t]*([A-Z][A-Z /&(),'-]{1,60}?)[ \t]*:")
# Headers that also start a section mid-line: older reports run their sections together.
_INLINE_HEADER = re.compile(
    r"\b(FINDINGS?|IMPRESSIONS?|CONCLUSIONS?|WET READ|ADDENDUM|RECOMMENDATIONS?|NOTIFICATION|COMPARISONS?|"
    r"TECHNIQUE|EXAMINATION|INDICATIONS?|CLINICAL HISTORY|HISTORY|CLINICAL INFORMATION|"
    r"REASON FOR EXAM(?:INATION)?)[ \t]*:"
)
# Words that never belong in an indication: the text is cut where one appears.
_LEAK_GUARD = re.compile(r"(?i)\b(?:findings?|impressions?|conclusions?|wet read|addendum)\s*:|\bfinal report\b")
_PRE_IMAGING = ("INDICATION", "HISTORY", "REASON FOR", "CLINICAL INFORMATION")
_POST_IMAGING = ("FINDING", "IMPRESSION", "CONCLUSION", "COMPARISON", "TECHNIQUE", "WET READ", "ADDENDUM", "RECOMMEND")


def report_sections(text: str) -> list[tuple[str, str]]:
    """(header, body) pairs in order. Text before the first header (titles, a bare WET READ) is dropped."""
    marks: dict[int, tuple[int, str]] = {}
    for rx in (_LINE_HEADER, _INLINE_HEADER):
        for m in rx.finditer(text):
            end, start = m.end(), m.start(1)
            if end not in marks or start < marks[end][0]:  # the longest name wins ("CLINICAL HISTORY" over "HISTORY")
                marks[end] = (start, m.group(1))
    ordered = sorted((start, end, name) for end, (start, name) in marks.items())
    kept, last_end = [], -1
    for start, end, name in ordered:
        if start >= last_end:
            kept.append((start, end, name))
            last_end = end
    return [(" ".join(name.split()), text[end:kept[i + 1][0] if i + 1 < len(kept) else len(text)])
            for i, (start, end, name) in enumerate(kept)]


def is_pre_imaging(header: str) -> bool:
    return any(k in header for k in _PRE_IMAGING) and not any(k in header for k in _POST_IMAGING)


def _first_paragraph(body: str) -> tuple[str, bool]:
    """The first block of non-blank lines, and whether anything followed it."""
    lines, rest = [], False
    for line in body.splitlines():
        if line.strip():
            if rest:
                return " ".join(lines), True
            lines.append(line.strip())
        elif lines:
            rest = True
    return " ".join(lines), False


def indication_text(report: str) -> tuple[str, dict]:
    """
    The pre-imaging sections of a report, each cut at its first blank line (a report whose findings
    follow without a header must not leak them), at any post-imaging word, and at INDICATION_MAX_CHARS.
    """
    cuts = {"paragraph": False, "guard": False, "length": False}
    parts = []
    for header, body in report_sections(report):
        if not is_pre_imaging(header):
            continue
        text, more = _first_paragraph(body)
        cuts["paragraph"] |= more
        m = _LEAK_GUARD.search(text)
        if m:
            text, cuts["guard"] = text[:m.start()], True
        parts.append(text.strip())
    text = " ".join(" ".join(parts).split())
    if len(text) > INDICATION_MAX_CHARS:
        text, cuts["length"] = text[:INDICATION_MAX_CHARS], True
    return text, cuts


def prior_text(report: str) -> str:
    """
    FINDINGS and IMPRESSION of an earlier report. A report with neither header is used whole, up to any
    ADDENDUM, which may have been written after the current X-ray.
    """
    parts = [" ".join(body.split()) for header, body in report_sections(report)
             if any(k in header for k in ("FINDING", "IMPRESSION", "CONCLUSION"))]
    text = " ".join(p for p in parts if p) or " ".join(re.split(r"(?i)\baddendum\b", report)[0].split())
    return text[:PRIOR_MAX_CHARS]


class Reports:
    """MIMIC-CXR free-text reports: mimic-cxr-reports.zip, or the unpacked files/pXX/pXXXXXXXX/sXXXXXXXX.txt."""

    _NAME = re.compile(r"(?:^|/)p\d+/s(\d+)\.txt$")

    def __init__(self, source: Path):
        self.source = Path(source)
        self._zip: Optional[zipfile.ZipFile] = None
        if self.source.is_file():
            self._zip = zipfile.ZipFile(self.source)
            self._members = {int(m.group(1)): name for name in self._zip.namelist()
                             if (m := self._NAME.search(name))}
        else:
            self._files = self.source / "files" if (self.source / "files").is_dir() else self.source

    def get(self, subject_id: int, study_id: int) -> Optional[str]:
        if self._zip is not None:
            name = self._members.get(int(study_id))
            return None if name is None else self._zip.read(name).decode("utf-8", errors="replace")
        sid = str(int(subject_id))
        path = self._files / f"p{sid[:2]}" / f"p{sid}" / f"s{int(study_id)}.txt"
        return path.read_text(errors="replace") if path.is_file() else None

    def __len__(self) -> int:
        """Reports in the zip; for a folder, 1 if any report exists (counting 227k files is slow on USB drives)."""
        if self._zip is not None:
            return len(self._members)
        return int(next(self._files.glob("p*/p*/s*.txt"), None) is not None)

    def close(self) -> None:
        if self._zip is not None:
            self._zip.close()


# ---------------------------------------------------------------------------
# Linking studies to ED stays and to earlier studies
# ---------------------------------------------------------------------------

def find_csv(root: Path, name: str) -> Optional[Path]:
    """`{name}.csv.gz` or `.csv` in root/ed, root/hosp or root."""
    for folder in (root / "ed", root / "hosp", root):
        for suffix in (".csv.gz", ".csv"):
            if (folder / f"{name}{suffix}").exists():
                return folder / f"{name}{suffix}"
    return None


def link_ed_stays(studies: pd.DataFrame, edstays: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Per study, the latest ED stay that began at or before the X-ray and at most ED_WINDOW before it."""
    cand = studies[["study_id", "subject_id", "study_datetime"]].merge(
        edstays[["subject_id", "stay_id", "intime"]], on="subject_id")
    delta = cand["study_datetime"] - cand["intime"]
    ok = delta.notna() & (delta >= pd.Timedelta(0)) & (delta <= ED_WINDOW)
    best = (cand[ok].assign(ed_hours_before=delta[ok].dt.total_seconds() / 3600)
            .sort_values(["study_id", "intime"]).drop_duplicates("study_id", keep="last"))
    after = set(cand.loc[(delta < pd.Timedelta(0)) & (delta >= -ED_WINDOW), "study_id"]) - set(best["study_id"])
    early = set(cand.loc[delta > ED_WINDOW, "study_id"]) - set(best["study_id"]) - after
    stats = {"linked": len(best), "only_stays_after_the_xray": len(after), "only_stays_over_24h_before": len(early)}
    return best[["study_id", "stay_id", "ed_hours_before"]], stats


def link_priors(studies: pd.DataFrame, all_studies: pd.DataFrame) -> pd.DataFrame:
    """Per study, the same patient's latest study taken at least PRIOR_MIN_GAP before it."""
    left = studies.loc[studies["study_datetime"].notna(), ["study_id", "subject_id", "study_datetime"]].copy()
    left["cutoff"] = left["study_datetime"] - PRIOR_MIN_GAP
    right = all_studies.dropna(subset=["prior_datetime"]).sort_values("prior_datetime")
    linked = pd.merge_asof(left.sort_values("cutoff"), right, left_on="cutoff", right_on="prior_datetime",
                           by="subject_id", direction="backward")
    linked = linked.dropna(subset=["prior_study_id"])
    linked["prior_study_id"] = linked["prior_study_id"].astype(np.int64)
    linked["prior_days"] = (linked["study_datetime"] - linked["prior_datetime"]).dt.total_seconds() / 86400
    return linked[["study_id", "prior_study_id", "prior_days"]]


def add_prior_labels(ctx: pd.DataFrame, chexpert: pd.DataFrame) -> pd.DataFrame:
    """The prior study's 14 CheXpert values (1 / 0 / -1 / NaN) as prior_label::{name}, and whether it has a row."""
    labels = chexpert[["study_id", *CHEXPERT_COLUMNS]].drop_duplicates("study_id")
    labels = labels.rename(columns={"study_id": "prior_study_id", **dict(zip(CHEXPERT_COLUMNS, PRIOR_LABEL_COLUMNS))})
    labels["prior_study_id"] = labels["prior_study_id"].astype("Int64")
    labels["prior_labels_found"] = True
    out = ctx.drop(columns=[*PRIOR_LABEL_COLUMNS, "prior_labels_found"]).merge(labels, on="prior_study_id", how="left")
    out["prior_labels_found"] = out["prior_labels_found"].astype("boolean").fillna(False).astype(bool)
    return out


def all_study_times(mimic_root: Path, subjects: set) -> pd.DataFrame:
    """Every MIMIC-CXR study of these patients with its earliest image time."""
    meta = pd.read_csv(mimic_csv(mimic_root, "metadata"), usecols=["subject_id", "study_id", "StudyDate", "StudyTime"],
                       dtype={"subject_id": np.int64, "study_id": np.int64})
    meta = meta[meta["subject_id"].isin(subjects)]
    meta["prior_datetime"] = study_datetime(meta["StudyDate"], meta["StudyTime"])
    out = meta.groupby("study_id").agg(subject_id=("subject_id", "first"), prior_datetime=("prior_datetime", "min"))
    return out.reset_index().rename(columns={"study_id": "prior_study_id"})


# ---------------------------------------------------------------------------
# Building the context table
# ---------------------------------------------------------------------------

def build_context(manifest: pd.DataFrame, reports: Optional[Reports], mimic_root: Optional[Path],
                  ed_root: Optional[Path], mimic_iv: Optional[Path]) -> tuple[pd.DataFrame, dict]:
    ctx = manifest[["study_id", "subject_id", "split", "in_test_clean", "in_test_v1", "study_datetime"]].copy()
    ctx = ctx.reset_index(drop=True)
    for col in TEXT_SOURCES:
        ctx[col] = ""
    ctx["prior_study_id"] = pd.array([pd.NA] * len(ctx), dtype="Int64")
    ctx["prior_days"] = np.nan
    for col in PRIOR_LABEL_COLUMNS:
        ctx[col] = np.nan
    ctx["prior_labels_found"] = False
    ctx["ed_hours_before"] = np.nan
    for col in (*TRIAGE, "sex_male", "age"):
        ctx[col] = np.nan
    stats: dict = {"sources": {"reports": reports is not None, "mimic_iv_ed": ed_root is not None,
                               "mimic_iv_patients": mimic_iv is not None, "mimic_cxr_metadata": mimic_root is not None}}
    subjects = set(ctx["subject_id"])

    if reports is not None:
        found, cuts = 0, {"paragraph": 0, "guard": 0, "length": 0}
        texts = []
        for subject_id, study_id in zip(ctx["subject_id"], ctx["study_id"]):
            report = reports.get(subject_id, study_id)
            if report is None:
                texts.append("")
                continue
            found += 1
            text, cut = indication_text(report)
            for k in cuts:
                cuts[k] += int(cut[k])
            texts.append(text)
        ctx["indication"] = texts
        stats["indication"] = {"reports_found": found, "reports_missing": len(ctx) - found,
                               "with_indication": int((ctx["indication"] != "").sum()),
                               "cut_at_blank_line": cuts["paragraph"], "cut_at_post_imaging_word": cuts["guard"],
                               "cut_at_length": cuts["length"]}

    if mimic_root is not None:
        priors = link_priors(ctx, all_study_times(mimic_root, subjects))
        ctx = ctx.drop(columns=["prior_study_id", "prior_days"]).merge(priors, on="study_id", how="left")
        ctx["prior_study_id"] = ctx["prior_study_id"].astype("Int64")
        stats["prior"] = {"with_earlier_study": int(ctx["prior_study_id"].notna().sum())}
        if reports is not None:
            prior_texts, prior_found = [], 0
            for subject_id, prior_id in zip(ctx["subject_id"], ctx["prior_study_id"]):
                report = None if pd.isna(prior_id) else reports.get(subject_id, int(prior_id))
                prior_found += report is not None
                prior_texts.append("" if report is None else prior_text(report))
            ctx["prior_report"] = prior_texts
            stats["prior"]["report_found"] = int(prior_found)
        labels_path = mimic_csv(mimic_root, "chexpert")
        if labels_path.exists():
            chexpert = pd.read_csv(labels_path, usecols=["study_id", *CHEXPERT_COLUMNS], dtype={"study_id": np.int64})
            ctx = add_prior_labels(ctx, chexpert)
            stats["prior"]["labels_found"] = int(ctx["prior_labels_found"].sum())
        stats["prior"]["median_days_before"] = _median(ctx["prior_days"])

    if ed_root is not None:
        edstays_path, triage_path = find_csv(ed_root, "edstays"), find_csv(ed_root, "triage")
        if edstays_path is None or triage_path is None:
            raise SystemExit(f"{ed_root}: need ed/edstays.csv.gz and ed/triage.csv.gz (MIMIC-IV-ED 2.2)")
        edstays = pd.read_csv(edstays_path, usecols=["subject_id", "stay_id", "intime", "gender"])
        edstays = edstays[edstays["subject_id"].isin(subjects)]
        edstays["intime"] = pd.to_datetime(edstays["intime"], errors="coerce")
        linked, ed_stats = link_ed_stays(ctx, edstays)
        triage = pd.read_csv(triage_path, usecols=["stay_id", *TRIAGE, "chiefcomplaint"])
        linked = linked.merge(triage.drop_duplicates("stay_id"), on="stay_id", how="left")
        ctx = ctx.drop(columns=["ed_hours_before", *TRIAGE]).merge(
            linked.drop(columns="stay_id"), on="study_id", how="left")
        ctx["chief_complaint"] = ctx.pop("chiefcomplaint").fillna("").astype(str).str.strip()
        for col in TRIAGE:
            ctx[col] = pd.to_numeric(ctx[col], errors="coerce")
        sex = edstays.drop_duplicates("subject_id").set_index("subject_id")["gender"]
        ctx["sex_male"] = ctx["subject_id"].map(sex).map({"M": 1.0, "F": 0.0})
        stats["ed"] = {**ed_stats, "with_chief_complaint": int((ctx["chief_complaint"] != "").sum()),
                       "median_hours_before": _median(ctx["ed_hours_before"])}

    if mimic_iv is not None:
        patients_path = find_csv(mimic_iv, "patients")
        if patients_path is None:
            raise SystemExit(f"{mimic_iv}: need hosp/patients.csv.gz (MIMIC-IV)")
        patients = pd.read_csv(patients_path, usecols=["subject_id", "gender", "anchor_age", "anchor_year"])
        patients = patients[patients["subject_id"].isin(subjects)].drop_duplicates("subject_id").set_index("subject_id")
        year = ctx["study_datetime"].dt.year
        ctx["age"] = (ctx["subject_id"].map(patients["anchor_age"]) + year - ctx["subject_id"].map(patients["anchor_year"]))
        ctx["sex_male"] = ctx["subject_id"].map(patients["gender"]).map({"M": 1.0, "F": 0.0}).fillna(ctx["sex_male"])

    keep = ["study_id", "subject_id", "split", "in_test_clean", "in_test_v1", *TEXT_SOURCES, "prior_study_id",
            "prior_days", "ed_hours_before", *TRIAGE, "sex_male", "age", *PRIOR_LABEL_COLUMNS, "prior_labels_found"]
    return ctx[keep], stats


def _median(values: pd.Series) -> Optional[float]:
    v = pd.to_numeric(values, errors="coerce").dropna()
    return round(float(v.median()), 1) if len(v) else None


# ---------------------------------------------------------------------------
# Audit (counts only)
# ---------------------------------------------------------------------------

def build_audit(ctx: pd.DataFrame, stats: dict) -> dict:
    groups = {"train": ctx["split"] == TRAIN, "val": ctx["split"] == VAL,
              "test_clean": ctx["in_test_clean"].astype(bool), "test_v1": ctx["in_test_v1"].astype(bool)}
    has = {src: ctx[src].fillna("").str.strip().ne("") for src in TEXT_SOURCES}
    has["vitals"] = ctx[list(TRIAGE)].notna().any(axis=1)
    has["prior_labels"] = ctx["prior_labels_found"].astype(bool)
    coverage = {name: {"studies": int(m.sum()), **{k: round(float(v[m].mean()), 3) if m.any() else None
                                                   for k, v in has.items()}}
                for name, m in groups.items()}
    words = {src: _median(ctx.loc[has[src], src].str.split().str.len()) for src in TEXT_SOURCES}
    return {
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "commit": git_commit(),
        "rules": {"ed_window_hours": ED_WINDOW.total_seconds() / 3600,
                  "prior_min_gap_hours": PRIOR_MIN_GAP.total_seconds() / 3600,
                  "indication_max_chars": INDICATION_MAX_CHARS},
        "studies": len(ctx),
        "coverage": coverage,
        "median_words": words,
        **stats,
    }


def audit_markdown(audit: dict) -> str:
    cols = ["studies", *TEXT_SOURCES, "vitals", "prior_labels"]
    lines = ["# M4 context audit", "",
             "Pre-imaging inputs per study (docs/ANALYSIS.md, M4). Counts only; the text stays in context.parquet.", "",
             f"Sources found: {', '.join(k for k, v in audit['sources'].items() if v) or 'none'}. "
             f"Missing: {', '.join(k for k, v in audit['sources'].items() if not v) or 'none'}.", "",
             "## Coverage (share of studies with each input)", "",
             "| Set | " + " | ".join(cols) + " |", "|" + "---|" * (len(cols) + 1)]
    for name, row in audit["coverage"].items():
        cells = [str(row["studies"])] + ["-" if row[c] is None else f"{row[c]:.1%}" for c in cols[1:]]
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    lines += ["", f"Median words when present: {audit['median_words']}", ""]
    for key in ("indication", "prior", "ed"):
        if key in audit:
            lines.append(f"- **{key}**: " + ", ".join(f"{k} {v}" for k, v in audit[key].items()))
    lines += ["", f"Rules: {audit['rules']}", ""]
    return "\n".join(lines)


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default=None, help="v2 output dir (default: $OUTPUT_PATH/v2 or output/v2)")
    parser.add_argument("--reports", help="mimic-cxr-reports.zip or its unpacked folder (default: MIMIC_CXR_REPORTS_PATH)")
    parser.add_argument("--mimic-iv-ed", help="MIMIC-IV-ED 2.2 root (default: MIMIC_IV_ED_PATH)")
    parser.add_argument("--mimic-iv", help="MIMIC-IV root, for age (default: MIMIC_IV_PATH; optional)")
    parser.add_argument("--mimic-cxr-jpg", help="MIMIC-CXR-JPG root, for earlier studies (default: MIMIC_CXR_JPG_PATH)")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)
    setup_logging(args.verbose)
    load_dotenv()  # the data paths live in .env even when --out is given
    out_dir = Path(args.out) if args.out else default_out_dir()

    def path_of(value: Optional[str], env: str) -> Optional[Path]:
        value = value or os.environ.get(env)
        if not value:
            logger.warning("No %s: the inputs that need it stay empty", env)
            return None
        path = Path(value).expanduser()
        if not path.exists():
            raise SystemExit(f"{env} points to {path}, which does not exist")
        return path

    report_path = path_of(args.reports, "MIMIC_CXR_REPORTS_PATH")
    ed_root = path_of(args.mimic_iv_ed, "MIMIC_IV_ED_PATH")
    mimic_iv = path_of(args.mimic_iv, "MIMIC_IV_PATH")
    mimic_root = path_of(args.mimic_cxr_jpg, "MIMIC_CXR_JPG_PATH")
    if mimic_iv is not None and find_csv(mimic_iv, "patients") is None:
        logger.warning("%s has no hosp/patients.csv.gz: age stays empty", mimic_iv)
        mimic_iv = None

    manifest = load_manifest(out_dir)
    reports = Reports(report_path) if report_path else None
    try:
        ctx, stats = build_context(manifest, reports, mimic_root, ed_root, mimic_iv)
    finally:
        if reports is not None:
            reports.close()
    ctx.to_parquet(out_dir / "context.parquet", index=False)
    audit = build_audit(ctx, stats)
    (out_dir / "context_audit.json").write_text(json.dumps(audit, indent=2, default=str))
    (out_dir / "context_audit.md").write_text(audit_markdown(audit))
    cov = audit["coverage"]["train"]
    logger.info("Context for %d studies; train coverage: %s", len(ctx),
                ", ".join(f"{k} {v:.0%}" for k, v in cov.items() if k != "studies" and v is not None))
    logger.info("Audit: %s (record-level text stays in %s)", out_dir / "context_audit.md", out_dir / "context.parquet")


if __name__ == "__main__":
    main()
