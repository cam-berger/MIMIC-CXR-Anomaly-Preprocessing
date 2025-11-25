"""
Cross-dataset subject linker.

This module handles the complex task of linking records across MIMIC datasets.

Key Linking Relationships:
=========================

1. SUBJECT-LEVEL (subject_id):
   - Consistent across ALL MIMIC datasets
   - Links: MIMIC-CXR ↔ MIMIC-IV ↔ MIMIC-IV-ED ↔ CXR-PRO

2. CXR to ED (temporal matching):
   - Match study_datetime (from CXR) to intime/outtime (from ED)
   - Typical window: CXR within ±24 hours of ED visit

3. ED to Hospital Admission (hadm_id):
   - ED stays MAY result in hospital admission
   - hadm_id links: MIMIC-IV-ED.edstays → MIMIC-IV.admissions
   - NOTE: hadm_id is NULL for ED visits without admission

4. CXR to Reports (study_id):
   - Direct link: MIMIC-CXR.study_id → CXR-PRO.study_id
   - One report per study (may have multiple images)

Usage Example:
=============
```python
linker = DatasetLinker()

# Link CXR studies to ED stays
linked = linker.link_cxr_to_ed(
    cxr_studies=cxr_loader.get_normal_studies(),
    ed_stays=ed_loader.edstays,
    time_window_hours=24
)

# Add hospital admission info (where available)
linked = linker.add_admission_info(linked, iv_loader.admissions)

# Add reports
linked = linker.add_reports(linked, cxr_pro_loader)
```
"""

import logging
from typing import Optional

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class DatasetLinker:
    """Links records across MIMIC datasets."""

    def link_cxr_to_ed(
        self,
        cxr_studies: pd.DataFrame,
        cxr_metadata: pd.DataFrame,
        ed_stays: pd.DataFrame,
        time_window_hours: int = 24,
    ) -> pd.DataFrame:
        """
        Link CXR studies to ED stays via temporal matching.

        A CXR is matched to an ED stay if study_datetime falls within:
        [ed_intime - time_window_hours, ed_outtime + time_window_hours]

        Args:
            cxr_studies: CheXpert labels DataFrame (subject_id, study_id)
            cxr_metadata: Metadata DataFrame (study_id, study_datetime)
            ed_stays: ED stays DataFrame (subject_id, stay_id, intime, outtime)
            time_window_hours: Time window for matching (default: 24)

        Returns:
            DataFrame with linked records containing:
            - subject_id, study_id (from CXR)
            - stay_id, hadm_id, intime, outtime, disposition (from ED)
            - study_datetime, time_to_cxr_hours (computed)
        """
        logger.info(f"Linking {len(cxr_studies):,} CXR studies to ED stays...")

        # Get study datetime (aggregate to study level if needed)
        study_dt = (
            cxr_metadata.groupby("study_id")
            .agg({"subject_id": "first", "study_datetime": "first"})
            .reset_index()
        )

        # Merge study datetime with CXR labels
        cxr_with_dt = cxr_studies.merge(
            study_dt[["study_id", "study_datetime"]],
            on="study_id",
            how="left",
        )

        # Remove studies without datetime
        cxr_with_dt = cxr_with_dt.dropna(subset=["study_datetime"])
        logger.info(f"CXR studies with valid datetime: {len(cxr_with_dt):,}")

        # Prepare ED stays
        ed_for_merge = ed_stays[
            ["subject_id", "stay_id", "hadm_id", "intime", "outtime", "disposition"]
        ].copy()

        # Compute expanded time window
        time_delta = pd.Timedelta(hours=time_window_hours)
        ed_for_merge["window_start"] = ed_for_merge["intime"] - time_delta
        ed_for_merge["window_end"] = ed_for_merge["outtime"] + time_delta

        # Merge on subject_id first (many-to-many)
        merged = cxr_with_dt.merge(ed_for_merge, on="subject_id", how="inner")
        logger.info(f"After subject merge: {len(merged):,} candidate pairs")

        # Filter by temporal window
        matched = merged[
            (merged["study_datetime"] >= merged["window_start"]) &
            (merged["study_datetime"] <= merged["window_end"])
        ].copy()

        # Compute time from ED arrival to CXR
        matched["time_to_cxr_hours"] = (
            (matched["study_datetime"] - matched["intime"])
            .dt.total_seconds() / 3600
        )

        # Keep only the closest ED stay per study (in case of multiple matches)
        matched["time_to_cxr_abs"] = matched["time_to_cxr_hours"].abs()
        matched = matched.sort_values("time_to_cxr_abs")
        matched = matched.drop_duplicates(subset=["study_id"], keep="first")
        matched = matched.drop(columns=["time_to_cxr_abs", "window_start", "window_end"])

        logger.info(f"Linked {len(matched):,} CXR studies to ED stays")
        return matched

    def add_admission_info(
        self,
        linked_df: pd.DataFrame,
        admissions: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Add hospital admission information where available.

        Many ED visits don't result in admission, so hadm_id may be NULL.
        This adds admission details for those that do have hadm_id.

        Args:
            linked_df: Previously linked DataFrame with hadm_id column
            admissions: MIMIC-IV admissions DataFrame

        Returns:
            DataFrame with added admission columns
        """
        logger.info("Adding hospital admission info...")

        # Select relevant admission columns
        adm_cols = admissions[[
            "hadm_id", "admittime", "dischtime", "deathtime",
            "admission_type", "discharge_location",
        ]].copy()

        # Merge (left join to keep all linked records)
        result = linked_df.merge(adm_cols, on="hadm_id", how="left")

        has_admission = result["admittime"].notna().sum()
        logger.info(f"Records with hospital admission: {has_admission:,} / {len(result):,}")

        return result

    def add_demographics(
        self,
        linked_df: pd.DataFrame,
        patients: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Add patient demographics (age, gender).

        Args:
            linked_df: Previously linked DataFrame with subject_id
            patients: MIMIC-IV patients DataFrame

        Returns:
            DataFrame with added demographic columns
        """
        logger.info("Adding patient demographics...")

        demo_cols = patients[["subject_id", "gender", "anchor_age", "dod"]].copy()

        result = linked_df.merge(demo_cols, on="subject_id", how="left")
        return result

    def add_reports(
        self,
        linked_df: pd.DataFrame,
        cxr_pro_loader: "CXRPROLoader",
    ) -> pd.DataFrame:
        """
        Add radiology reports from CXR-PRO.

        Args:
            linked_df: Previously linked DataFrame with study_id
            cxr_pro_loader: CXRPROLoader instance

        Returns:
            DataFrame with added 'report' column
        """
        logger.info("Adding radiology reports...")

        study_ids = set(linked_df["study_id"])
        reports = cxr_pro_loader.get_reports_for_studies(study_ids)

        result = linked_df.merge(
            reports[["study_id", "report"]],
            on="study_id",
            how="left",
        )

        has_report = result["report"].notna().sum()
        logger.info(f"Records with reports: {has_report:,} / {len(result):,}")

        return result

    def add_ed_triage(
        self,
        linked_df: pd.DataFrame,
        ed_triage: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Add ED triage vitals.

        Args:
            linked_df: Previously linked DataFrame with stay_id
            ed_triage: MIMIC-IV-ED triage DataFrame

        Returns:
            DataFrame with added triage columns
        """
        logger.info("Adding ED triage vitals...")

        triage_cols = ed_triage[[
            "stay_id", "temperature", "heartrate", "resprate",
            "o2sat", "sbp", "dbp", "pain", "acuity", "chiefcomplaint",
        ]].copy()

        # Rename to indicate these are triage values
        triage_cols = triage_cols.rename(columns={
            "temperature": "triage_temperature",
            "heartrate": "triage_heartrate",
            "resprate": "triage_resprate",
            "o2sat": "triage_o2sat",
            "sbp": "triage_sbp",
            "dbp": "triage_dbp",
            "pain": "triage_pain",
            "acuity": "triage_acuity",
            "chiefcomplaint": "triage_chiefcomplaint",
        })

        result = linked_df.merge(triage_cols, on="stay_id", how="left")
        return result

    def add_ed_diagnoses(
        self,
        linked_df: pd.DataFrame,
        ed_diagnosis: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Add ED diagnoses as comma-separated ICD codes.

        Args:
            linked_df: Previously linked DataFrame with stay_id
            ed_diagnosis: MIMIC-IV-ED diagnosis DataFrame

        Returns:
            DataFrame with added 'ed_diagnoses' column
        """
        logger.info("Adding ED diagnoses...")

        # Aggregate diagnoses per stay
        diag_agg = (
            ed_diagnosis.groupby("stay_id")["icd_code"]
            .apply(lambda x: ",".join(x.dropna().astype(str)))
            .reset_index()
            .rename(columns={"icd_code": "ed_diagnoses"})
        )

        result = linked_df.merge(diag_agg, on="stay_id", how="left")
        return result

    def add_hospital_diagnoses(
        self,
        linked_df: pd.DataFrame,
        diagnoses: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Add hospital diagnoses as comma-separated ICD codes.

        Only applies to records with hadm_id (admitted patients).

        Args:
            linked_df: Previously linked DataFrame with hadm_id
            diagnoses: MIMIC-IV diagnoses_icd DataFrame

        Returns:
            DataFrame with added 'hospital_diagnoses' column
        """
        logger.info("Adding hospital diagnoses...")

        # Aggregate diagnoses per admission
        diag_agg = (
            diagnoses.groupby("hadm_id")["icd_code"]
            .apply(lambda x: ",".join(x.dropna().astype(str)))
            .reset_index()
            .rename(columns={"icd_code": "hospital_diagnoses"})
        )

        result = linked_df.merge(diag_agg, on="hadm_id", how="left")
        return result

    def add_procedures(
        self,
        linked_df: pd.DataFrame,
        procedures: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Add procedures as comma-separated ICD codes.

        Only applies to records with hadm_id (admitted patients).

        Args:
            linked_df: Previously linked DataFrame with hadm_id
            procedures: MIMIC-IV procedures_icd DataFrame

        Returns:
            DataFrame with added 'procedures' column
        """
        logger.info("Adding procedures...")

        # Aggregate procedures per admission
        proc_agg = (
            procedures.groupby("hadm_id")["icd_code"]
            .apply(lambda x: ",".join(x.dropna().astype(str)))
            .reset_index()
            .rename(columns={"icd_code": "procedures"})
        )

        result = linked_df.merge(proc_agg, on="hadm_id", how="left")
        return result

    def build_complete_cohort(
        self,
        cxr_studies: pd.DataFrame,
        cxr_loader: "MIMICCXRLoader",
        ed_loader: "MIMICIVEDLoader",
        iv_loader: "MIMICIVLoader",
        cxr_pro_loader: "CXRPROLoader",
        time_window_hours: int = 24,
    ) -> pd.DataFrame:
        """
        Build a complete cohort with all features linked.

        This is a convenience method that runs all linking steps.

        Args:
            cxr_studies: CheXpert labels for selected studies
            cxr_loader: MIMICCXRLoader instance
            ed_loader: MIMICIVEDLoader instance
            iv_loader: MIMICIVLoader instance
            cxr_pro_loader: CXRPROLoader instance
            time_window_hours: Temporal matching window

        Returns:
            Complete cohort DataFrame with all features
        """
        logger.info("Building complete cohort with all features...")

        # Step 1: Link CXR to ED
        cohort = self.link_cxr_to_ed(
            cxr_studies=cxr_studies,
            cxr_metadata=cxr_loader.metadata,
            ed_stays=ed_loader.edstays,
            time_window_hours=time_window_hours,
        )

        # Step 2: Add demographics
        cohort = self.add_demographics(cohort, iv_loader.patients)

        # Step 3: Add admission info (for those admitted)
        cohort = self.add_admission_info(cohort, iv_loader.admissions)

        # Step 4: Add ED triage
        cohort = self.add_ed_triage(cohort, ed_loader.triage)

        # Step 5: Add ED diagnoses
        cohort = self.add_ed_diagnoses(cohort, ed_loader.diagnosis)

        # Step 6: Add hospital diagnoses (where available)
        cohort = self.add_hospital_diagnoses(cohort, iv_loader.diagnoses)

        # Step 7: Add procedures (where available)
        cohort = self.add_procedures(cohort, iv_loader.procedures)

        # Step 8: Add radiology reports
        cohort = self.add_reports(cohort, cxr_pro_loader)

        logger.info(f"Complete cohort: {len(cohort):,} records")
        return cohort
