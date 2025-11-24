"""
CXR-PRO v1.0.0 data loader.

CXR-PRO contains radiology report impressions from MIMIC-CXR with
prior study references removed (important for de-identification).

Files:
- mimic_train_impressions.csv: Training set impressions (~371k)
- mimic_test_impressions.csv: Test set impressions (~2k)

Columns:
- subject_id: Patient identifier (links to MIMIC)
- study_id: Radiology study (links to MIMIC-CXR)
- dicom_id: Specific image (multiple per study possible)
- report: Impression section text (prior refs removed)

NOTE: Each row is per-image (dicom_id), not per-study.
      Multiple rows may exist per study if it has multiple images.
"""

import logging
from pathlib import Path
from typing import Optional

import pandas as pd
import numpy as np

from ..config import get_settings

logger = logging.getLogger(__name__)


class CXRPROLoader:
    """Loader for CXR-PRO radiology report impressions."""

    def __init__(self, paths: Optional["DataPaths"] = None):
        """Initialize loader with paths."""
        self.paths = paths or get_settings().paths
        self._train = None
        self._test = None

    @property
    def train(self) -> pd.DataFrame:
        """Load training impressions (cached)."""
        if self._train is None:
            logger.info(f"Loading CXR-PRO train from {self.paths.cxr_pro_train}")
            self._train = self._load_impressions(self.paths.cxr_pro_train)
            logger.info(f"Loaded {len(self._train):,} train impressions")
        return self._train

    @property
    def test(self) -> pd.DataFrame:
        """Load test impressions (cached)."""
        if self._test is None:
            logger.info(f"Loading CXR-PRO test from {self.paths.cxr_pro_test}")
            self._test = self._load_impressions(self.paths.cxr_pro_test)
            logger.info(f"Loaded {len(self._test):,} test impressions")
        return self._test

    def _load_impressions(self, path: Path) -> pd.DataFrame:
        """Load impressions file with proper dtypes."""
        return pd.read_csv(
            path,
            dtype={
                "subject_id": np.int32,
                "study_id": np.int64,
                "dicom_id": str,
                "report": str,
            },
        )

    @property
    def all_impressions(self) -> pd.DataFrame:
        """Get all impressions (train + test combined)."""
        return pd.concat([self.train, self.test], ignore_index=True)

    def get_reports_for_studies(self, study_ids: set[int]) -> pd.DataFrame:
        """
        Get unique reports for specific studies.

        Since CXR-PRO has one row per dicom_id but report is at study level,
        this deduplicates to return one report per study.

        Returns DataFrame with: subject_id, study_id, report
        """
        all_reports = self.all_impressions
        filtered = all_reports[all_reports["study_id"].isin(study_ids)]

        # Deduplicate to one report per study
        return (
            filtered.groupby("study_id")
            .agg({
                "subject_id": "first",
                "report": "first",
            })
            .reset_index()
        )

    def get_reports_for_subjects(self, subject_ids: set[int]) -> pd.DataFrame:
        """
        Get all unique reports for specific subjects.

        Returns DataFrame with: subject_id, study_id, report
        """
        all_reports = self.all_impressions
        filtered = all_reports[all_reports["subject_id"].isin(subject_ids)]

        # Deduplicate to one report per study
        return (
            filtered.groupby("study_id")
            .agg({
                "subject_id": "first",
                "report": "first",
            })
            .reset_index()
        )

    def get_report(self, study_id: int) -> Optional[str]:
        """Get report text for a single study."""
        all_reports = self.all_impressions
        match = all_reports[all_reports["study_id"] == study_id]
        if match.empty:
            return None
        return match.iloc[0]["report"]
