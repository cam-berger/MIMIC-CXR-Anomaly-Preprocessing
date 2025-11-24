"""
Structured data preprocessing for labs, vitals, and demographics.

Extracts and processes:
- Demographics: age, gender
- ED triage vitals: temperature, HR, RR, O2sat, BP, pain, acuity
- Lab values: within temporal window of CXR study
- Diagnoses: ED and hospital ICD codes
- Procedures: ICD procedure codes

Output Parquet schema:
- study_id, subject_id (keys)
- age, gender (demographics)
- triage_* columns (ED triage vitals)
- lab_* columns (aggregated lab values)
- has_* columns (boolean flags for data availability)
"""

import logging
from pathlib import Path
from typing import Optional
from concurrent.futures import ProcessPoolExecutor, as_completed

import pandas as pd
import numpy as np
from tqdm import tqdm

from ..config import Settings, get_settings
from ..datasets import MIMICIVLoader, MIMICIVEDLoader

logger = logging.getLogger(__name__)


class StructuredPreprocessor:
    """Process structured clinical data (labs, vitals, demographics)."""

    def __init__(self, settings: Optional[Settings] = None):
        """Initialize preprocessor."""
        self.settings = settings or get_settings()
        self.config = self.settings.preprocessing

        self._iv_loader = None
        self._ed_loader = None

    @property
    def iv_loader(self) -> MIMICIVLoader:
        """MIMIC-IV loader (lazy initialization)."""
        if self._iv_loader is None:
            self._iv_loader = MIMICIVLoader(self.settings.paths)
        return self._iv_loader

    @property
    def ed_loader(self) -> MIMICIVEDLoader:
        """MIMIC-IV-ED loader (lazy initialization)."""
        if self._ed_loader is None:
            self._ed_loader = MIMICIVEDLoader(self.settings.paths)
        return self._ed_loader

    def process_cohort(
        self,
        cohort: pd.DataFrame,
        output_path: Path,
    ) -> pd.DataFrame:
        """
        Process structured data for entire cohort.

        Args:
            cohort: Cohort DataFrame (must have subject_id, study_id, study_datetime)
            output_path: Output parquet file path

        Returns:
            Processed structured data DataFrame
        """
        logger.info(f"Processing structured data for {len(cohort):,} samples...")

        # Extract required columns from cohort
        required = ["subject_id", "study_id"]
        missing = [c for c in required if c not in cohort.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        # Initialize result DataFrame with cohort keys
        result = cohort[["subject_id", "study_id"]].copy()

        # 1. Process demographics (from cohort if available, else load)
        logger.info("Processing demographics...")
        result = self._add_demographics(result, cohort)

        # 2. Process triage vitals (from cohort if available)
        logger.info("Processing triage vitals...")
        result = self._add_triage_vitals(result, cohort)

        # 3. Process ED vitals (aggregated)
        if "stay_id" in cohort.columns:
            logger.info("Processing ED vital signs...")
            result = self._add_ed_vitals(result, cohort)

        # 4. Process labs (requires streaming for large data)
        if "hadm_id" in cohort.columns or "stay_id" in cohort.columns:
            logger.info("Processing lab values...")
            result = self._add_labs(result, cohort)

        # 5. Add diagnosis and procedure flags
        logger.info("Processing diagnoses and procedures...")
        result = self._add_diagnosis_flags(result, cohort)

        # 6. Add data availability flags
        result = self._add_availability_flags(result)

        # Save to parquet
        output_path.parent.mkdir(parents=True, exist_ok=True)
        result.to_parquet(output_path, index=False)
        logger.info(f"Saved structured data to {output_path}")

        return result

    def _add_demographics(
        self,
        result: pd.DataFrame,
        cohort: pd.DataFrame,
    ) -> pd.DataFrame:
        """Add demographic features."""
        # Check if already in cohort
        if "anchor_age" in cohort.columns:
            result["age"] = cohort["anchor_age"].values
        if "gender" in cohort.columns:
            result["gender"] = cohort["gender"].values
            # Encode gender as numeric
            result["gender_M"] = (result["gender"] == "M").astype(np.int8)

        return result

    def _add_triage_vitals(
        self,
        result: pd.DataFrame,
        cohort: pd.DataFrame,
    ) -> pd.DataFrame:
        """Add triage vital signs."""
        triage_cols = [
            "triage_temperature", "triage_heartrate", "triage_resprate",
            "triage_o2sat", "triage_sbp", "triage_dbp", "triage_pain", "triage_acuity",
        ]

        for col in triage_cols:
            if col in cohort.columns:
                result[col] = cohort[col].values

        return result

    def _add_ed_vitals(
        self,
        result: pd.DataFrame,
        cohort: pd.DataFrame,
    ) -> pd.DataFrame:
        """Add aggregated ED vital signs."""
        stay_ids = set(cohort["stay_id"].dropna().astype(int))

        if not stay_ids:
            return result

        # Get vital sign summary
        vitals_summary = self.ed_loader.get_vitals_summary(stay_ids)

        if vitals_summary.empty:
            return result

        # Create mapping from stay_id to study_id
        cohort_mapping = cohort[["study_id", "stay_id"]].dropna().copy()
        cohort_mapping["stay_id"] = cohort_mapping["stay_id"].astype(int)

        # Merge vitals with mapping
        vitals_with_study = vitals_summary.merge(cohort_mapping, on="stay_id", how="inner")

        # Merge with result
        result = result.merge(
            vitals_with_study.drop(columns=["stay_id"]),
            on="study_id",
            how="left",
        )

        return result

    def _add_labs(
        self,
        result: pd.DataFrame,
        cohort: pd.DataFrame,
    ) -> pd.DataFrame:
        """Add aggregated lab values within temporal window."""
        # Get subjects that need labs
        subject_ids = set(cohort["subject_id"])

        # Stream lab events and aggregate
        lab_agg = self._aggregate_labs_for_cohort(cohort, subject_ids)

        if lab_agg.empty:
            return result

        # Merge with result
        result = result.merge(lab_agg, on="study_id", how="left")

        return result

    def _aggregate_labs_for_cohort(
        self,
        cohort: pd.DataFrame,
        subject_ids: set[int],
    ) -> pd.DataFrame:
        """
        Aggregate lab values for cohort within temporal windows.

        Uses streaming to handle large labevents table efficiently.
        """
        # Get all relevant item IDs
        all_itemids = set()
        for lab_ids in self.iv_loader.PRIORITY_LAB_IDS.values():
            all_itemids.update(lab_ids)

        # Prepare cohort reference times
        if "study_datetime" not in cohort.columns:
            logger.warning("No study_datetime in cohort, skipping labs")
            return pd.DataFrame()

        cohort_times = cohort[["subject_id", "study_id", "study_datetime"]].copy()
        cohort_times = cohort_times.dropna(subset=["study_datetime"])

        if cohort_times.empty:
            return pd.DataFrame()

        # Time window
        before_hours = self.config.labs_time_window_before_hours
        after_hours = self.config.labs_time_window_after_hours

        # Stream labs and match to studies
        matched_labs = []

        for chunk in self.iv_loader.iter_labevents(
            subject_ids=subject_ids,
            itemids=all_itemids,
            chunksize=500_000,
        ):
            # Merge with cohort times
            merged = chunk.merge(cohort_times, on="subject_id", how="inner")

            # Filter by time window
            time_diff = (merged["charttime"] - merged["study_datetime"]).dt.total_seconds() / 3600
            in_window = (time_diff >= -before_hours) & (time_diff <= after_hours)
            matched = merged[in_window]

            if not matched.empty:
                matched_labs.append(matched[["study_id", "itemid", "valuenum"]])

        if not matched_labs:
            return pd.DataFrame()

        all_matched = pd.concat(matched_labs, ignore_index=True)

        # Map itemid to lab type
        itemid_to_lab = {}
        for lab_type, itemids in self.iv_loader.PRIORITY_LAB_IDS.items():
            for itemid in itemids:
                itemid_to_lab[itemid] = lab_type

        all_matched["lab_type"] = all_matched["itemid"].map(itemid_to_lab)

        # Aggregate per study and lab type
        agg = (
            all_matched.groupby(["study_id", "lab_type"])["valuenum"]
            .agg(["mean", "min", "max", "std", "count"])
            .reset_index()
        )

        # Pivot to wide format
        result_dfs = []
        for stat in ["mean", "min", "max", "count"]:
            pivot = agg.pivot(index="study_id", columns="lab_type", values=stat)
            pivot.columns = [f"lab_{col}_{stat}" for col in pivot.columns]
            result_dfs.append(pivot)

        if result_dfs:
            lab_agg = pd.concat(result_dfs, axis=1).reset_index()
            return lab_agg

        return pd.DataFrame()

    def _add_diagnosis_flags(
        self,
        result: pd.DataFrame,
        cohort: pd.DataFrame,
    ) -> pd.DataFrame:
        """Add diagnosis-based feature flags."""
        # Count diagnoses
        if "ed_diagnoses" in cohort.columns:
            result["num_ed_diagnoses"] = cohort["ed_diagnoses"].apply(
                lambda x: len(str(x).split(",")) if pd.notna(x) and x else 0
            )

        if "hospital_diagnoses" in cohort.columns:
            result["num_hospital_diagnoses"] = cohort["hospital_diagnoses"].apply(
                lambda x: len(str(x).split(",")) if pd.notna(x) and x else 0
            )

        if "procedures" in cohort.columns:
            result["num_procedures"] = cohort["procedures"].apply(
                lambda x: len(str(x).split(",")) if pd.notna(x) and x else 0
            )

        return result

    def _add_availability_flags(self, result: pd.DataFrame) -> pd.DataFrame:
        """Add boolean flags for data availability."""
        # Check for triage data
        triage_cols = [c for c in result.columns if c.startswith("triage_")]
        if triage_cols:
            result["has_triage"] = result[triage_cols].notna().any(axis=1)

        # Check for lab data
        lab_cols = [c for c in result.columns if c.startswith("lab_")]
        if lab_cols:
            result["has_labs"] = result[lab_cols].notna().any(axis=1)

        # Check for ED vitals
        vital_cols = [c for c in result.columns if any(
            c.startswith(f"{v}_") for v in ["temperature", "heartrate", "resprate", "o2sat", "sbp", "dbp"]
        )]
        if vital_cols:
            result["has_ed_vitals"] = result[vital_cols].notna().any(axis=1)

        return result
