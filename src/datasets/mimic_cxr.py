"""
MIMIC-CXR-JPG v2.1.0 data loader.

Files loaded:
- mimic-cxr-2.0.0-chexpert.csv.gz: CheXpert automated labels (14 conditions)
- mimic-cxr-2.0.0-metadata.csv.gz: Study metadata (datetime, view, etc.)
- mimic-cxr-2.0.0-split.csv.gz: Official train/val/test splits

Image structure:
    files/p{subject_prefix}/p{subject_id}/s{study_id}/{dicom_id}.jpg

Linking:
- subject_id: Patient identifier (links to MIMIC-IV)
- study_id: Radiology study (can have multiple images)
- dicom_id: Individual image within study
- study_datetime: For temporal matching with ED/admission

CheXpert Labels (14 conditions):
- Negative (0.0): Explicitly ruled out
- Positive (1.0): Explicitly present
- Uncertain (-1.0): Mentioned with uncertainty
- Blank (NaN): Not mentioned

For "normal" CXR filtering: No Finding = 1.0 AND all pathologies != 1.0
"""

import logging
from pathlib import Path
from typing import Optional

import pandas as pd
import numpy as np

from ..config import get_settings

logger = logging.getLogger(__name__)


class MIMICCXRLoader:
    """Efficient loader for MIMIC-CXR-JPG chest X-ray data."""

    # CheXpert pathology labels (excluding "No Finding" and support devices)
    PATHOLOGY_LABELS = [
        "Atelectasis",
        "Cardiomegaly",
        "Consolidation",
        "Edema",
        "Enlarged Cardiomediastinum",
        "Fracture",
        "Lung Lesion",
        "Lung Opacity",
        "Pleural Effusion",
        "Pleural Other",
        "Pneumonia",
        "Pneumothorax",
    ]

    # Support device label (not a pathology)
    SUPPORT_DEVICES_LABEL = "Support Devices"

    # No Finding label (indicates normal study)
    NO_FINDING_LABEL = "No Finding"

    def __init__(self, paths: Optional["DataPaths"] = None):
        """Initialize loader with paths."""
        self.paths = paths or get_settings().paths
        self._chexpert = None
        self._metadata = None
        self._split = None

    @property
    def chexpert(self) -> pd.DataFrame:
        """Load CheXpert labels (cached)."""
        if self._chexpert is None:
            logger.info(f"Loading CheXpert labels from {self.paths.cxr_chexpert}")
            self._chexpert = pd.read_csv(
                self.paths.cxr_chexpert,
                dtype={
                    "subject_id": np.int32,
                    "study_id": np.int64,
                    **{label: np.float32 for label in self.PATHOLOGY_LABELS},
                    self.NO_FINDING_LABEL: np.float32,
                    self.SUPPORT_DEVICES_LABEL: np.float32,
                },
            )
            logger.info(f"Loaded {len(self._chexpert):,} CheXpert records")
        return self._chexpert

    @property
    def metadata(self) -> pd.DataFrame:
        """Load study metadata (cached)."""
        if self._metadata is None:
            logger.info(f"Loading metadata from {self.paths.cxr_metadata}")
            self._metadata = pd.read_csv(
                self.paths.cxr_metadata,
                dtype={
                    "dicom_id": str,
                    "subject_id": np.int32,
                    "study_id": np.int64,
                    "ViewPosition": "category",
                    "Rows": "Int32",
                    "Columns": "Int32",
                    "StudyDate": str,
                    "StudyTime": str,
                    "ProcedureCodeSequence_CodeMeaning": str,
                    "ViewCodeSequence_CodeMeaning": str,
                    "PatientOrientationCodeSequence_CodeMeaning": str,
                },
            )
            # Parse study datetime
            self._metadata["study_datetime"] = pd.to_datetime(
                self._metadata["StudyDate"] + " " + self._metadata["StudyTime"].str[:6],
                format="%Y%m%d %H%M%S",
                errors="coerce",
            )
            logger.info(f"Loaded {len(self._metadata):,} metadata records")
        return self._metadata

    @property
    def split(self) -> pd.DataFrame:
        """Load official train/val/test split (cached)."""
        if self._split is None:
            logger.info(f"Loading split from {self.paths.cxr_split}")
            self._split = pd.read_csv(
                self.paths.cxr_split,
                dtype={
                    "dicom_id": str,
                    "subject_id": np.int32,
                    "study_id": np.int64,
                    "split": "category",
                },
            )
            logger.info(f"Loaded {len(self._split):,} split records")
        return self._split

    def get_normal_studies(self, strict: bool = True) -> pd.DataFrame:
        """
        Get studies with no pathological findings.

        Args:
            strict: If True, require "No Finding" = 1.0 AND all pathologies != 1.0
                   If False, just require "No Finding" = 1.0

        Returns:
            DataFrame with normal study records
        """
        chexpert = self.chexpert

        # Base filter: No Finding = 1.0
        normal_mask = chexpert[self.NO_FINDING_LABEL] == 1.0

        if strict:
            # Additional filter: no pathology is positive (1.0)
            for label in self.PATHOLOGY_LABELS:
                normal_mask &= chexpert[label] != 1.0

        normal = chexpert[normal_mask].copy()
        logger.info(f"Found {len(normal):,} normal studies (strict={strict})")
        return normal

    def get_abnormal_studies(self, pathologies: Optional[list[str]] = None) -> pd.DataFrame:
        """
        Get studies with pathological findings.

        Args:
            pathologies: List of pathology labels to require (any match).
                        If None, returns studies where any pathology = 1.0

        Returns:
            DataFrame with abnormal study records
        """
        chexpert = self.chexpert

        if pathologies is None:
            pathologies = self.PATHOLOGY_LABELS

        # Any specified pathology is positive
        abnormal_mask = pd.Series(False, index=chexpert.index)
        for label in pathologies:
            if label in chexpert.columns:
                abnormal_mask |= chexpert[label] == 1.0

        abnormal = chexpert[abnormal_mask].copy()
        logger.info(f"Found {len(abnormal):,} abnormal studies")
        return abnormal

    def get_studies_for_subjects(self, subject_ids: set[int]) -> pd.DataFrame:
        """Get all CheXpert labels for specific subjects."""
        chexpert = self.chexpert
        return chexpert[chexpert["subject_id"].isin(subject_ids)].copy()

    def get_metadata_for_studies(self, study_ids: set[int]) -> pd.DataFrame:
        """Get metadata for specific studies."""
        metadata = self.metadata
        return metadata[metadata["study_id"].isin(study_ids)].copy()

    def get_study_datetime(self, study_ids: set[int]) -> pd.DataFrame:
        """
        Get study datetime for specific studies.

        Returns DataFrame with: study_id, subject_id, study_datetime
        """
        metadata = self.get_metadata_for_studies(study_ids)

        # Get one datetime per study (there may be multiple images)
        return (
            metadata.groupby("study_id")
            .agg({
                "subject_id": "first",
                "study_datetime": "first",
            })
            .reset_index()
        )

    def get_image_path(self, subject_id: int, study_id: int, dicom_id: str) -> Path:
        """
        Get path to a specific image.

        Path structure: files/p{subject_prefix}/p{subject_id}/s{study_id}/{dicom_id}.jpg
        """
        subject_prefix = str(subject_id)[:2]
        return (
            self.paths.cxr_images
            / f"p{subject_prefix}"
            / f"p{subject_id}"
            / f"s{study_id}"
            / f"{dicom_id}.jpg"
        )

    def get_study_images(self, subject_id: int, study_id: int) -> list[Path]:
        """
        Get all image paths for a study.

        Returns list of paths to all images in the study directory.
        """
        subject_prefix = str(subject_id)[:2]
        study_dir = (
            self.paths.cxr_images
            / f"p{subject_prefix}"
            / f"p{subject_id}"
            / f"s{study_id}"
        )

        if not study_dir.exists():
            return []

        return sorted(study_dir.glob("*.jpg"))

    def get_frontal_image(self, subject_id: int, study_id: int) -> Optional[Path]:
        """
        Get path to frontal view image (PA or AP) for a study.

        Returns None if no frontal view available.
        """
        metadata = self.metadata
        study_meta = metadata[
            (metadata["subject_id"] == subject_id) &
            (metadata["study_id"] == study_id)
        ]

        # Prefer PA over AP
        for view in ["PA", "AP"]:
            frontal = study_meta[study_meta["ViewPosition"] == view]
            if not frontal.empty:
                dicom_id = frontal.iloc[0]["dicom_id"]
                return self.get_image_path(subject_id, study_id, dicom_id)

        return None

    def get_study_summary(self) -> pd.DataFrame:
        """
        Get summary of studies with labels and metadata merged.

        Returns DataFrame with:
        - subject_id, study_id
        - All CheXpert labels
        - study_datetime, ViewPosition (from first image)
        - num_images (count of images per study)
        """
        # Merge chexpert with aggregated metadata
        chexpert = self.chexpert
        metadata = self.metadata

        # Aggregate metadata per study
        meta_agg = (
            metadata.groupby("study_id")
            .agg({
                "study_datetime": "first",
                "ViewPosition": lambda x: ",".join(x.dropna().unique()),
                "dicom_id": "count",
            })
            .rename(columns={"dicom_id": "num_images"})
            .reset_index()
        )

        # Merge
        summary = chexpert.merge(meta_agg, on="study_id", how="left")
        return summary
