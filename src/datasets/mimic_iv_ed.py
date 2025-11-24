"""
MIMIC-IV-ED v2.2 data loader.

Tables loaded:
- edstays: ED encounters (stay_id, intime, outtime, disposition)
- diagnosis: ED diagnoses (ICD codes at ED level)
- triage: Initial triage vitals
- vitalsign: Vital signs during ED stay

Linking:
- subject_id -> All tables
- stay_id -> All tables (unique ED encounter)
- hadm_id -> Links to MIMIC-IV admissions (when ED visit leads to admission)

NOTE: Not all ED visits result in hospital admission.
      hadm_id is NULL for ED visits without subsequent admission.
"""

import logging
from pathlib import Path
from typing import Optional

import pandas as pd
import numpy as np

from ..config import get_settings

logger = logging.getLogger(__name__)


class MIMICIVEDLoader:
    """Efficient loader for MIMIC-IV-ED emergency department data."""

    # Valid ED dispositions and their categories
    DISCHARGE_DISPOSITIONS = {
        "HOME",
        "DISCHARGED",
        "LEFT WITHOUT BEING SEEN",
        "LEFT AGAINST MEDICAL ADVICE",
        "OTHER",
    }

    ADMISSION_DISPOSITIONS = {
        "ADMITTED",
    }

    ADVERSE_DISPOSITIONS = {
        "EXPIRED",
        "ELOPED",
    }

    def __init__(self, paths: Optional["DataPaths"] = None):
        """Initialize loader with paths."""
        self.paths = paths or get_settings().paths
        self._edstays = None
        self._diagnosis = None
        self._triage = None
        self._vitalsign = None

    @property
    def edstays(self) -> pd.DataFrame:
        """Load ED stays table (cached)."""
        if self._edstays is None:
            logger.info(f"Loading ED stays from {self.paths.ed_stays}")
            self._edstays = pd.read_csv(
                self.paths.ed_stays,
                dtype={
                    "subject_id": np.int32,
                    "hadm_id": "Int32",  # nullable - not all ED visits lead to admission
                    "stay_id": np.int32,
                    "disposition": "category",
                    "arrival_transport": "category",
                    "race": "category",
                    "gender": "category",
                },
                parse_dates=["intime", "outtime"],
            )
            logger.info(f"Loaded {len(self._edstays):,} ED stays")
        return self._edstays

    @property
    def diagnosis(self) -> pd.DataFrame:
        """Load ED diagnosis table (cached)."""
        if self._diagnosis is None:
            logger.info(f"Loading ED diagnoses from {self.paths.ed_diagnosis}")
            self._diagnosis = pd.read_csv(
                self.paths.ed_diagnosis,
                dtype={
                    "subject_id": np.int32,
                    "stay_id": np.int32,
                    "seq_num": np.int16,
                    "icd_code": str,
                    "icd_version": np.int8,
                    "icd_title": str,
                },
            )
            logger.info(f"Loaded {len(self._diagnosis):,} ED diagnosis records")
        return self._diagnosis

    @property
    def triage(self) -> pd.DataFrame:
        """Load triage table (cached)."""
        if self._triage is None:
            logger.info(f"Loading triage from {self.paths.ed_triage}")
            self._triage = pd.read_csv(
                self.paths.ed_triage,
                dtype={
                    "subject_id": np.int32,
                    "stay_id": np.int32,
                    "temperature": np.float32,
                    "heartrate": np.float32,
                    "resprate": np.float32,
                    "o2sat": np.float32,
                    "sbp": np.float32,
                    "dbp": np.float32,
                    "pain": str,
                    "acuity": "Int8",  # nullable
                    "chiefcomplaint": str,
                },
            )
            logger.info(f"Loaded {len(self._triage):,} triage records")
        return self._triage

    @property
    def vitalsign(self) -> pd.DataFrame:
        """Load vital signs table (cached)."""
        if self._vitalsign is None:
            logger.info(f"Loading vital signs from {self.paths.ed_vitalsign}")
            self._vitalsign = pd.read_csv(
                self.paths.ed_vitalsign,
                dtype={
                    "subject_id": np.int32,
                    "stay_id": np.int32,
                    "temperature": np.float32,
                    "heartrate": np.float32,
                    "resprate": np.float32,
                    "o2sat": np.float32,
                    "sbp": np.float32,
                    "dbp": np.float32,
                    "rhythm": "category",
                    "pain": str,
                },
                parse_dates=["charttime"],
            )
            logger.info(f"Loaded {len(self._vitalsign):,} vital sign records")
        return self._vitalsign

    def get_discharged_stays(self) -> pd.DataFrame:
        """Get ED stays where patient was discharged (not admitted)."""
        edstays = self.edstays
        return edstays[edstays["disposition"].isin(self.DISCHARGE_DISPOSITIONS)].copy()

    def get_admitted_stays(self) -> pd.DataFrame:
        """Get ED stays that resulted in hospital admission."""
        edstays = self.edstays
        return edstays[edstays["disposition"].isin(self.ADMISSION_DISPOSITIONS)].copy()

    def get_stays_for_subjects(self, subject_ids: set[int]) -> pd.DataFrame:
        """Get all ED stays for specific subjects."""
        edstays = self.edstays
        return edstays[edstays["subject_id"].isin(subject_ids)].copy()

    def get_diagnoses_for_stays(self, stay_ids: set[int]) -> pd.DataFrame:
        """Get diagnoses for specific ED stays."""
        diagnosis = self.diagnosis
        return diagnosis[diagnosis["stay_id"].isin(stay_ids)].copy()

    def get_triage_for_stays(self, stay_ids: set[int]) -> pd.DataFrame:
        """Get triage info for specific ED stays."""
        triage = self.triage
        return triage[triage["stay_id"].isin(stay_ids)].copy()

    def get_vitals_for_stays(self, stay_ids: set[int]) -> pd.DataFrame:
        """Get all vital signs for specific ED stays."""
        vitalsign = self.vitalsign
        return vitalsign[vitalsign["stay_id"].isin(stay_ids)].copy()

    def has_excluded_diagnosis(
        self,
        stay_ids: set[int],
        excluded_patterns: tuple[str, ...],
    ) -> set[int]:
        """
        Find ED stays with excluded diagnoses.

        Args:
            stay_ids: Stays to check
            excluded_patterns: ICD code prefixes to exclude

        Returns:
            Set of stay_ids with excluded diagnoses
        """
        diagnoses = self.get_diagnoses_for_stays(stay_ids)
        excluded = set()

        for pattern in excluded_patterns:
            matches = diagnoses[diagnoses["icd_code"].str.startswith(pattern)]
            excluded.update(matches["stay_id"])

        return excluded

    def get_vitals_summary(self, stay_ids: set[int]) -> pd.DataFrame:
        """
        Get summary statistics of vitals for each ED stay.

        Returns DataFrame with columns like:
        - stay_id
        - temperature_mean, temperature_min, temperature_max
        - heartrate_mean, heartrate_min, heartrate_max
        - etc.
        """
        vitals = self.get_vitals_for_stays(stay_ids)

        if vitals.empty:
            return pd.DataFrame()

        vital_cols = ["temperature", "heartrate", "resprate", "o2sat", "sbp", "dbp"]

        agg_dict = {col: ["mean", "min", "max", "std", "count"] for col in vital_cols}

        summary = vitals.groupby("stay_id").agg(agg_dict)
        summary.columns = [f"{col}_{stat}" for col, stat in summary.columns]
        summary = summary.reset_index()

        return summary
