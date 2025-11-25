"""
Cohort builder for normal and anomalous CXR studies.

Normal Cohort (for unsupervised pretraining):
- CheXpert "No Finding" = 1.0
- All pathology labels != 1.0
- ED visit with disposition: discharged home
- No critical diagnoses (sepsis, pneumonia, MI, etc.)
- Age >= 18

Anomalous Cohort (for classification fine-tuning):
- At least one pathology label = 1.0
- Matched to ED/admission where possible
- Age >= 18
"""

import logging
from typing import Optional

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from ..config import Settings, get_settings
from ..datasets import (
    MIMICIVLoader,
    MIMICIVEDLoader,
    MIMICCXRLoader,
    CXRPROLoader,
    DatasetLinker,
)

logger = logging.getLogger(__name__)


class CohortBuilder:
    """Builds normal and anomalous cohorts with all required features."""

    def __init__(self, settings: Optional[Settings] = None):
        """
        Initialize cohort builder.

        Args:
            settings: Configuration settings (uses defaults if None)
        """
        self.settings = settings or get_settings()
        self.config = self.settings.cohort

        # Initialize loaders (lazy loading)
        self._iv_loader = None
        self._ed_loader = None
        self._cxr_loader = None
        self._cxr_pro_loader = None
        self._linker = None

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

    @property
    def cxr_loader(self) -> MIMICCXRLoader:
        """MIMIC-CXR loader (lazy initialization)."""
        if self._cxr_loader is None:
            self._cxr_loader = MIMICCXRLoader(self.settings.paths)
        return self._cxr_loader

    @property
    def cxr_pro_loader(self) -> CXRPROLoader:
        """CXR-PRO loader (lazy initialization)."""
        if self._cxr_pro_loader is None:
            self._cxr_pro_loader = CXRPROLoader(self.settings.paths)
        return self._cxr_pro_loader

    @property
    def linker(self) -> DatasetLinker:
        """Dataset linker (lazy initialization)."""
        if self._linker is None:
            self._linker = DatasetLinker()
        return self._linker

    def build_normal_cohort(
        self,
        require_ed_match: bool = True,
        filter_dispositions: bool = True,
        filter_diagnoses: bool = True,
        include_reports: bool = True,
    ) -> pd.DataFrame:
        """
        Build cohort of normal (healthy) CXR studies.

        Filtering steps:
        1. CheXpert "No Finding" = 1.0 AND all pathologies != 1.0
        2. Match to ED stay (temporal window)
        3. Filter by ED disposition (discharged home)
        4. Exclude critical diagnoses
        5. Filter by age (>= 18)
        6. Add all features (demographics, vitals, reports, etc.)

        Args:
            require_ed_match: Require temporal match to ED stay
            filter_dispositions: Filter by acceptable dispositions
            filter_diagnoses: Exclude critical diagnoses
            include_reports: Include radiology reports

        Returns:
            Complete normal cohort DataFrame
        """
        logger.info("Building normal cohort...")

        # Step 1: Get normal CXR studies
        normal_studies = self.cxr_loader.get_normal_studies(strict=True)
        logger.info(f"Step 1 - Normal CXR studies: {len(normal_studies):,}")

        # Step 2: Link to ED stays
        if require_ed_match:
            cohort = self.linker.link_cxr_to_ed(
                cxr_studies=normal_studies,
                cxr_metadata=self.cxr_loader.metadata,
                ed_stays=self.ed_loader.edstays,
                time_window_hours=self.config.ed_cxr_time_window_hours,
            )
            logger.info(f"Step 2 - After ED matching: {len(cohort):,}")
        else:
            # Just get study datetimes without ED matching
            cohort = self._add_study_datetime(normal_studies)

        if len(cohort) == 0:
            logger.warning("No records after ED matching!")
            return pd.DataFrame()

        # Step 3: Filter by ED disposition (discharged home)
        if filter_dispositions and "disposition" in cohort.columns:
            before = len(cohort)
            cohort = cohort[
                cohort["disposition"].isin(self.config.acceptable_dispositions)
            ]
            logger.info(f"Step 3 - After disposition filter: {len(cohort):,} ({len(cohort)/before*100:.1f}%)")

        # Step 4: Exclude critical diagnoses
        if filter_diagnoses and "stay_id" in cohort.columns:
            stay_ids = set(cohort["stay_id"])
            excluded_stays = self.ed_loader.has_excluded_diagnosis(
                stay_ids,
                self.config.excluded_diagnosis_patterns,
            )

            # Also check hospital diagnoses if hadm_id available
            hadm_ids = set(cohort["hadm_id"].dropna().astype(int))
            if hadm_ids:
                excluded_hadm = self.iv_loader.has_excluded_diagnosis(
                    hadm_ids,
                    self.config.excluded_diagnosis_patterns,
                )
                # Find stay_ids corresponding to excluded hadm_ids
                excluded_from_hadm = set(
                    cohort[cohort["hadm_id"].isin(excluded_hadm)]["stay_id"]
                )
                excluded_stays.update(excluded_from_hadm)

            before = len(cohort)
            cohort = cohort[~cohort["stay_id"].isin(excluded_stays)]
            logger.info(f"Step 4 - After diagnosis filter: {len(cohort):,} ({len(cohort)/before*100:.1f}%)")

        # Step 5: Add demographics and filter by age
        cohort = self.linker.add_demographics(cohort, self.iv_loader.patients)
        before = len(cohort)
        cohort = cohort[cohort["anchor_age"] >= self.config.min_age]
        logger.info(f"Step 5 - After age filter (>={self.config.min_age}): {len(cohort):,}")

        # Step 6: Add all features
        cohort = self._add_all_features(cohort, include_reports=include_reports)

        logger.info(f"Final normal cohort: {len(cohort):,} records")
        return cohort

    def build_anomalous_cohort(
        self,
        pathologies: Optional[list[str]] = None,
        require_ed_match: bool = False,  # More permissive for anomalous
        include_reports: bool = True,
    ) -> pd.DataFrame:
        """
        Build cohort of abnormal CXR studies.

        Args:
            pathologies: List of pathology labels to include (all if None)
            require_ed_match: Require temporal match to ED stay
            include_reports: Include radiology reports

        Returns:
            Complete anomalous cohort DataFrame
        """
        logger.info("Building anomalous cohort...")

        # Step 1: Get abnormal CXR studies
        abnormal_studies = self.cxr_loader.get_abnormal_studies(pathologies)
        logger.info(f"Step 1 - Abnormal CXR studies: {len(abnormal_studies):,}")

        # Step 2: Link to ED stays (optional for anomalous cohort)
        if require_ed_match:
            cohort = self.linker.link_cxr_to_ed(
                cxr_studies=abnormal_studies,
                cxr_metadata=self.cxr_loader.metadata,
                ed_stays=self.ed_loader.edstays,
                time_window_hours=self.config.ed_cxr_time_window_hours,
            )
            logger.info(f"Step 2 - After ED matching: {len(cohort):,}")
        else:
            # Just add study datetime
            cohort = self._add_study_datetime(abnormal_studies)
            logger.info(f"Step 2 - Added study datetime: {len(cohort):,}")

        if len(cohort) == 0:
            logger.warning("No records in anomalous cohort!")
            return pd.DataFrame()

        # Step 3: Add demographics and filter by age
        cohort = self.linker.add_demographics(cohort, self.iv_loader.patients)
        before = len(cohort)
        cohort = cohort[cohort["anchor_age"] >= self.config.min_age]
        logger.info(f"Step 3 - After age filter: {len(cohort):,}")

        # Step 4: Add features (where available)
        # For anomalous cohort, try to link to ED but don't require it
        if not require_ed_match:
            cohort = self._try_ed_linking(cohort)

        cohort = self._add_all_features(cohort, include_reports=include_reports)

        logger.info(f"Final anomalous cohort: {len(cohort):,} records")
        return cohort

    def _add_study_datetime(self, studies: pd.DataFrame) -> pd.DataFrame:
        """Add study_datetime from metadata."""
        study_dt = (
            self.cxr_loader.metadata
            .groupby("study_id")
            .agg({"study_datetime": "first"})
            .reset_index()
        )
        return studies.merge(study_dt, on="study_id", how="left")

    def _try_ed_linking(self, cohort: pd.DataFrame) -> pd.DataFrame:
        """Try to link to ED stays without filtering out non-matches."""
        if "stay_id" in cohort.columns:
            return cohort  # Already linked

        logger.info("Attempting ED linking for anomalous cohort...")

        # Get potential ED stays for subjects
        subject_ids = set(cohort["subject_id"])
        ed_stays = self.ed_loader.get_stays_for_subjects(subject_ids)

        if ed_stays.empty:
            logger.info("No ED matches found")
            return cohort

        # Try temporal matching
        time_window = self.config.ed_cxr_time_window_hours
        time_delta = pd.Timedelta(hours=time_window)

        ed_for_merge = ed_stays[[
            "subject_id", "stay_id", "hadm_id", "intime", "outtime", "disposition"
        ]].copy()
        ed_for_merge["window_start"] = ed_for_merge["intime"] - time_delta
        ed_for_merge["window_end"] = ed_for_merge["outtime"] + time_delta

        # Merge and filter
        merged = cohort.merge(ed_for_merge, on="subject_id", how="left")

        # For rows with ED data, check temporal match
        has_ed = merged["intime"].notna()
        in_window = (
            (merged["study_datetime"] >= merged["window_start"]) &
            (merged["study_datetime"] <= merged["window_end"])
        )

        # Keep matches or rows without ED data
        keep_mask = ~has_ed | in_window
        result = merged[keep_mask].copy()

        # For non-matches, clear ED columns
        result.loc[~in_window, ["stay_id", "hadm_id", "intime", "outtime", "disposition"]] = np.nan

        # Clean up
        result = result.drop(columns=["window_start", "window_end"], errors="ignore")

        # Deduplicate (keep first match per study)
        result = result.drop_duplicates(subset=["study_id"], keep="first")

        matched = result["stay_id"].notna().sum()
        logger.info(f"ED matched: {matched:,} / {len(result):,}")

        return result

    def _add_all_features(
        self,
        cohort: pd.DataFrame,
        include_reports: bool = True,
    ) -> pd.DataFrame:
        """Add all available features to cohort."""

        # Add admission info if hadm_id available
        if "hadm_id" in cohort.columns:
            cohort = self.linker.add_admission_info(cohort, self.iv_loader.admissions)
            cohort = self.linker.add_hospital_diagnoses(cohort, self.iv_loader.diagnoses)
            cohort = self.linker.add_procedures(cohort, self.iv_loader.procedures)

        # Add ED features if stay_id available
        if "stay_id" in cohort.columns:
            cohort = self.linker.add_ed_triage(cohort, self.ed_loader.triage)
            cohort = self.linker.add_ed_diagnoses(cohort, self.ed_loader.diagnosis)

        # Add radiology reports
        if include_reports:
            cohort = self.linker.add_reports(cohort, self.cxr_pro_loader)

        return cohort

    def split_cohort(
        self,
        cohort: pd.DataFrame,
        test_size: Optional[float] = None,
        stratify_by: Optional[str] = "gender",
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split cohort into train and validation sets.

        Uses subject-level splitting to prevent data leakage.

        Args:
            cohort: Full cohort DataFrame
            test_size: Validation fraction (uses config if None)
            stratify_by: Column to stratify by (None for no stratification)

        Returns:
            (train_cohort, val_cohort)
        """
        test_size = test_size or self.config.validation_fraction

        # Get unique subjects
        if stratify_by and stratify_by in cohort.columns:
            subject_info = cohort.groupby("subject_id")[stratify_by].first().reset_index()
            train_subjects, val_subjects = train_test_split(
                subject_info["subject_id"],
                test_size=test_size,
                stratify=subject_info[stratify_by],
                random_state=self.config.random_seed,
            )
        else:
            subjects = cohort["subject_id"].unique()
            train_subjects, val_subjects = train_test_split(
                subjects,
                test_size=test_size,
                random_state=self.config.random_seed,
            )

        train_cohort = cohort[cohort["subject_id"].isin(train_subjects)].copy()
        val_cohort = cohort[cohort["subject_id"].isin(val_subjects)].copy()

        logger.info(
            f"Split: {len(train_cohort):,} train ({len(train_subjects):,} subjects), "
            f"{len(val_cohort):,} val ({len(val_subjects):,} subjects)"
        )

        return train_cohort, val_cohort

    def save_cohort(
        self,
        cohort: pd.DataFrame,
        name: str,
        split: Optional[str] = None,
    ) -> None:
        """
        Save cohort to parquet file.

        Args:
            cohort: Cohort DataFrame to save
            name: Cohort name (e.g., "normal", "anomalous")
            split: Split name (e.g., "train", "val") or None for full cohort
        """
        output_dir = self.settings.paths.cohort_dir
        output_dir.mkdir(parents=True, exist_ok=True)

        if split:
            filename = f"{name}_{split}.parquet"
        else:
            filename = f"{name}_full.parquet"

        output_path = output_dir / filename
        cohort.to_parquet(output_path, index=False)
        logger.info(f"Saved cohort to {output_path}")

    def build_and_save_all(self) -> dict[str, pd.DataFrame]:
        """
        Build and save both cohorts with train/val splits.

        Returns:
            Dictionary with all cohort DataFrames
        """
        results = {}

        # Build normal cohort
        logger.info("=" * 60)
        logger.info("BUILDING NORMAL COHORT")
        logger.info("=" * 60)
        normal = self.build_normal_cohort()
        normal_train, normal_val = self.split_cohort(normal)

        self.save_cohort(normal, "normal")
        self.save_cohort(normal_train, "normal", "train")
        self.save_cohort(normal_val, "normal", "val")

        results["normal_full"] = normal
        results["normal_train"] = normal_train
        results["normal_val"] = normal_val

        # Build anomalous cohort
        logger.info("=" * 60)
        logger.info("BUILDING ANOMALOUS COHORT")
        logger.info("=" * 60)
        anomalous = self.build_anomalous_cohort()
        anomalous_train, anomalous_val = self.split_cohort(anomalous)

        self.save_cohort(anomalous, "anomalous")
        self.save_cohort(anomalous_train, "anomalous", "train")
        self.save_cohort(anomalous_val, "anomalous", "val")

        results["anomalous_full"] = anomalous
        results["anomalous_train"] = anomalous_train
        results["anomalous_val"] = anomalous_val

        # Summary
        logger.info("=" * 60)
        logger.info("COHORT SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Normal cohort: {len(normal):,} total ({len(normal_train):,} train, {len(normal_val):,} val)")
        logger.info(f"Anomalous cohort: {len(anomalous):,} total ({len(anomalous_train):,} train, {len(anomalous_val):,} val)")

        return results
