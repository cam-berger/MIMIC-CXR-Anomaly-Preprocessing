"""
MIMIC-IV v3.1 data loader.

Tables loaded:
- patients: Demographics (subject_id, gender, anchor_age, anchor_year, dod)
- admissions: Hospital admissions (hadm_id, admittime, dischtime, deathtime)
- diagnoses_icd: ICD diagnosis codes per admission
- procedures_icd: ICD procedure codes per admission
- labevents: Lab results (LARGE - loaded lazily)
- transfers: Unit transfers (for ICU detection)
- d_labitems: Lab item dictionary
- d_icd_diagnoses: ICD diagnosis dictionary
- d_icd_procedures: ICD procedure dictionary

Linking:
- subject_id -> All tables
- hadm_id -> admissions, diagnoses, procedures, transfers
"""

import logging
from pathlib import Path
from typing import Optional, Iterator

import pandas as pd
import numpy as np

from ..config import get_settings

logger = logging.getLogger(__name__)


class MIMICIVLoader:
    """Efficient loader for MIMIC-IV hospital data."""

    # Lab item IDs for priority labs (from d_labitems)
    PRIORITY_LAB_IDS = {
        "wbc": [51300, 51301],
        "hemoglobin": [51222, 50811],
        "hematocrit": [51221, 50810],
        "platelets": [51265],
        "glucose": [50931, 50809],
        "creatinine": [50912],
        "bun": [51006],
        "sodium": [50983, 50824],
        "potassium": [50971, 50822],
        "chloride": [50902, 50806],
        "bicarbonate": [50882, 50803],
        "calcium": [50893],
        "magnesium": [50960],
        "lactate": [50813],
        "troponin": [51002, 51003],
        "bnp": [50963],
        "procalcitonin": [50976],
    }

    # ICU care unit patterns
    ICU_CAREUNITS = {
        "Medical Intensive Care Unit (MICU)",
        "Surgical Intensive Care Unit (SICU)",
        "Cardiac Vascular Intensive Care Unit (CVICU)",
        "Coronary Care Unit (CCU)",
        "Neuro Intermediate",
        "Neuro Stepdown",
        "Neuro Surgical Intensive Care Unit (Neuro SICU)",
        "Medical/Surgical Intensive Care Unit (MICU/SICU)",
        "Trauma SICU (TSICU)",
    }

    def __init__(self, paths: Optional["DataPaths"] = None):
        """Initialize loader with paths."""
        self.paths = paths or get_settings().paths
        self._patients = None
        self._admissions = None
        self._diagnoses = None
        self._procedures = None
        self._transfers = None
        self._d_labitems = None
        self._d_icd_diagnoses = None
        self._d_icd_procedures = None

    @property
    def patients(self) -> pd.DataFrame:
        """Load patients table (cached)."""
        if self._patients is None:
            logger.info(f"Loading patients from {self.paths.iv_patients}")
            self._patients = pd.read_csv(
                self.paths.iv_patients,
                dtype={"subject_id": np.int32, "gender": "category"},
                parse_dates=["dod"],
            )
            logger.info(f"Loaded {len(self._patients):,} patients")
        return self._patients

    @property
    def admissions(self) -> pd.DataFrame:
        """Load admissions table (cached)."""
        if self._admissions is None:
            logger.info(f"Loading admissions from {self.paths.iv_admissions}")
            self._admissions = pd.read_csv(
                self.paths.iv_admissions,
                dtype={
                    "subject_id": np.int32,
                    "hadm_id": np.int32,
                    "admission_type": "category",
                    "admit_provider_id": str,
                    "admission_location": "category",
                    "discharge_location": "category",
                    "insurance": "category",
                    "language": "category",
                    "marital_status": "category",
                    "race": "category",
                },
                parse_dates=["admittime", "dischtime", "deathtime", "edregtime", "edouttime"],
            )
            logger.info(f"Loaded {len(self._admissions):,} admissions")
        return self._admissions

    @property
    def diagnoses(self) -> pd.DataFrame:
        """Load diagnoses_icd table (cached)."""
        if self._diagnoses is None:
            logger.info(f"Loading diagnoses from {self.paths.iv_diagnoses_icd}")
            self._diagnoses = pd.read_csv(
                self.paths.iv_diagnoses_icd,
                dtype={
                    "subject_id": np.int32,
                    "hadm_id": np.int32,
                    "seq_num": np.int16,
                    "icd_code": str,
                    "icd_version": np.int8,
                },
            )
            logger.info(f"Loaded {len(self._diagnoses):,} diagnosis records")
        return self._diagnoses

    @property
    def procedures(self) -> pd.DataFrame:
        """Load procedures_icd table (cached)."""
        if self._procedures is None:
            logger.info(f"Loading procedures from {self.paths.iv_procedures_icd}")
            self._procedures = pd.read_csv(
                self.paths.iv_procedures_icd,
                dtype={
                    "subject_id": np.int32,
                    "hadm_id": np.int32,
                    "seq_num": np.int16,
                    "icd_code": str,
                    "icd_version": np.int8,
                },
                parse_dates=["chartdate"],
            )
            logger.info(f"Loaded {len(self._procedures):,} procedure records")
        return self._procedures

    @property
    def transfers(self) -> pd.DataFrame:
        """Load transfers table for ICU detection (cached)."""
        if self._transfers is None:
            logger.info(f"Loading transfers from {self.paths.iv_transfers}")
            self._transfers = pd.read_csv(
                self.paths.iv_transfers,
                dtype={
                    "subject_id": np.int32,
                    "hadm_id": "Int32",  # nullable
                    "transfer_id": np.int32,
                    "eventtype": "category",
                    "careunit": "category",
                },
                parse_dates=["intime", "outtime"],
            )
            logger.info(f"Loaded {len(self._transfers):,} transfer records")
        return self._transfers

    @property
    def d_labitems(self) -> pd.DataFrame:
        """Load lab items dictionary (cached)."""
        if self._d_labitems is None:
            self._d_labitems = pd.read_csv(
                self.paths.iv_d_labitems,
                dtype={"itemid": np.int32, "label": str, "fluid": str, "category": str},
            )
        return self._d_labitems

    @property
    def d_icd_diagnoses(self) -> pd.DataFrame:
        """Load ICD diagnosis dictionary (cached)."""
        if self._d_icd_diagnoses is None:
            self._d_icd_diagnoses = pd.read_csv(
                self.paths.iv_d_icd_diagnoses,
                dtype={"icd_code": str, "icd_version": np.int8, "long_title": str},
            )
        return self._d_icd_diagnoses

    @property
    def d_icd_procedures(self) -> pd.DataFrame:
        """Load ICD procedure dictionary (cached)."""
        if self._d_icd_procedures is None:
            self._d_icd_procedures = pd.read_csv(
                self.paths.iv_d_icd_procedures,
                dtype={"icd_code": str, "icd_version": np.int8, "long_title": str},
            )
        return self._d_icd_procedures

    def iter_labevents(
        self,
        subject_ids: Optional[set[int]] = None,
        hadm_ids: Optional[set[int]] = None,
        itemids: Optional[set[int]] = None,
        chunksize: int = 500_000,
    ) -> Iterator[pd.DataFrame]:
        """
        Iterate over labevents in chunks (memory-efficient).

        Args:
            subject_ids: Filter to these subjects (optional)
            hadm_ids: Filter to these admissions (optional)
            itemids: Filter to these lab item IDs (optional)
            chunksize: Rows per chunk

        Yields:
            Filtered DataFrame chunks
        """
        logger.info(f"Streaming labevents from {self.paths.iv_labevents}")

        reader = pd.read_csv(
            self.paths.iv_labevents,
            dtype={
                "labevent_id": np.int64,
                "subject_id": np.int32,
                "hadm_id": "Int32",
                "specimen_id": np.int64,
                "itemid": np.int32,
                "value": str,
                "valuenum": np.float32,
                "valueuom": str,
                "ref_range_lower": np.float32,
                "ref_range_upper": np.float32,
                "flag": str,
                "priority": str,
                "comments": str,
            },
            parse_dates=["charttime", "storetime"],
            chunksize=chunksize,
        )

        for chunk in reader:
            # Apply filters
            if subject_ids is not None:
                chunk = chunk[chunk["subject_id"].isin(subject_ids)]
            if hadm_ids is not None:
                chunk = chunk[chunk["hadm_id"].isin(hadm_ids)]
            if itemids is not None:
                chunk = chunk[chunk["itemid"].isin(itemids)]

            if len(chunk) > 0:
                yield chunk

    def load_labs_for_subjects(
        self,
        subject_ids: set[int],
        lab_types: Optional[list[str]] = None,
    ) -> pd.DataFrame:
        """
        Load labs for specific subjects.

        Args:
            subject_ids: Set of subject IDs
            lab_types: List of lab types from PRIORITY_LAB_IDS keys

        Returns:
            DataFrame with lab results
        """
        # Get item IDs to filter
        if lab_types is None:
            lab_types = list(self.PRIORITY_LAB_IDS.keys())

        itemids = set()
        for lab_type in lab_types:
            itemids.update(self.PRIORITY_LAB_IDS.get(lab_type, []))

        # Stream and concatenate
        chunks = []
        for chunk in self.iter_labevents(
            subject_ids=subject_ids,
            itemids=itemids,
        ):
            chunks.append(chunk)

        if chunks:
            return pd.concat(chunks, ignore_index=True)
        return pd.DataFrame()

    def get_icu_admissions(self) -> set[int]:
        """Get hadm_ids that had ICU stays."""
        transfers = self.transfers
        icu_transfers = transfers[transfers["careunit"].isin(self.ICU_CAREUNITS)]
        return set(icu_transfers["hadm_id"].dropna().astype(int))

    def get_death_admissions(self) -> set[int]:
        """Get hadm_ids where patient died during admission."""
        admissions = self.admissions
        deaths = admissions[admissions["deathtime"].notna()]
        return set(deaths["hadm_id"])

    def get_patient_demographics(self, subject_ids: set[int]) -> pd.DataFrame:
        """
        Get demographics for specific subjects.

        Returns DataFrame with: subject_id, gender, anchor_age
        """
        patients = self.patients
        return patients[patients["subject_id"].isin(subject_ids)][
            ["subject_id", "gender", "anchor_age", "anchor_year", "dod"]
        ].copy()

    def get_diagnoses_for_admissions(self, hadm_ids: set[int]) -> pd.DataFrame:
        """Get all diagnoses for specific admissions."""
        diagnoses = self.diagnoses
        return diagnoses[diagnoses["hadm_id"].isin(hadm_ids)].copy()

    def get_procedures_for_admissions(self, hadm_ids: set[int]) -> pd.DataFrame:
        """Get all procedures for specific admissions."""
        procedures = self.procedures
        return procedures[procedures["hadm_id"].isin(hadm_ids)].copy()

    def has_excluded_diagnosis(
        self,
        hadm_ids: set[int],
        excluded_patterns: tuple[str, ...],
    ) -> set[int]:
        """
        Find admissions with excluded diagnoses.

        Args:
            hadm_ids: Admissions to check
            excluded_patterns: ICD code prefixes to exclude

        Returns:
            Set of hadm_ids with excluded diagnoses
        """
        diagnoses = self.get_diagnoses_for_admissions(hadm_ids)
        excluded = set()

        for pattern in excluded_patterns:
            matches = diagnoses[diagnoses["icd_code"].str.startswith(pattern)]
            excluded.update(matches["hadm_id"])

        return excluded
