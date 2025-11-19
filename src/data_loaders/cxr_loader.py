"""
Data loader for MIMIC-CXR-JPG dataset.
"""
import pandas as pd
import logging
from pathlib import Path
from typing import Optional, Tuple

logger = logging.getLogger(__name__)


class CXRDataLoader:
    """Loader for MIMIC-CXR-JPG data files."""

    def __init__(self, paths_config):
        self.paths = paths_config

    def load_chexpert_labels(self) -> pd.DataFrame:
        """
        Load CheXpert labels from MIMIC-CXR.

        Returns:
            DataFrame with columns: subject_id, study_id, and pathology labels
        """
        logger.info(f"Loading CheXpert labels from {self.paths.CXR_CHEXPERT}")

        try:
            df = pd.read_csv(
                self.paths.CXR_CHEXPERT,
                compression='gzip',
                dtype={
                    'subject_id': 'int32',
                    'study_id': 'int32'
                }
            )
            logger.info(f"Loaded {len(df):,} studies with CheXpert labels")
            logger.info(f"Columns: {df.columns.tolist()}")

            return df

        except Exception as e:
            logger.error(f"Error loading CheXpert labels: {e}")
            raise

    def load_metadata(self) -> pd.DataFrame:
        """
        Load MIMIC-CXR metadata including study dates and view positions.

        Returns:
            DataFrame with metadata for each DICOM image
        """
        logger.info(f"Loading CXR metadata from {self.paths.CXR_METADATA}")

        try:
            df = pd.read_csv(
                self.paths.CXR_METADATA,
                compression='gzip',
                dtype={
                    'subject_id': 'int32',
                    'study_id': 'int32',
                    'dicom_id': 'str'
                }
            )

            # Convert StudyDate and StudyTime to datetime
            if 'StudyDate' in df.columns and 'StudyTime' in df.columns:
                df['study_datetime'] = pd.to_datetime(
                    df['StudyDate'].astype(str) + ' ' + df['StudyTime'].astype(str),
                    format='%Y%m%d %H%M%S.%f',
                    errors='coerce'
                )

            logger.info(f"Loaded metadata for {len(df):,} images")

            return df

        except Exception as e:
            logger.error(f"Error loading metadata: {e}")
            raise

    def load_negbio_labels(self) -> pd.DataFrame:
        """
        Load NegBio labels (alternative to CheXpert).

        Returns:
            DataFrame with NegBio labels
        """
        logger.info(f"Loading NegBio labels from {self.paths.CXR_NEGBIO}")

        try:
            df = pd.read_csv(
                self.paths.CXR_NEGBIO,
                compression='gzip',
                dtype={
                    'subject_id': 'int32',
                    'study_id': 'int32'
                }
            )
            logger.info(f"Loaded {len(df):,} studies with NegBio labels")

            return df

        except Exception as e:
            logger.error(f"Error loading NegBio labels: {e}")
            raise

    def load_split_info(self) -> pd.DataFrame:
        """
        Load train/validation/test split information.

        Returns:
            DataFrame with split assignments
        """
        logger.info(f"Loading split info from {self.paths.CXR_SPLIT}")

        try:
            df = pd.read_csv(
                self.paths.CXR_SPLIT,
                compression='gzip',
                dtype={
                    'subject_id': 'int32',
                    'study_id': 'int32',
                    'split': 'str'
                }
            )
            logger.info(f"Loaded split info for {len(df):,} studies")
            logger.info(f"Split distribution:\n{df['split'].value_counts()}")

            return df

        except Exception as e:
            logger.error(f"Error loading split info: {e}")
            raise

    def get_study_images_path(self, subject_id: int, study_id: int) -> Optional[Path]:
        """
        Get the path to a specific study's images directory.

        Args:
            subject_id: Patient subject ID
            study_id: Study ID

        Returns:
            Path to study directory if it exists, None otherwise
        """
        # MIMIC-CXR directory structure: files/p{subject_id[:2]}/p{subject_id}/s{study_id}/
        subject_dir = f"p{str(subject_id)[:2]}"
        patient_dir = f"p{subject_id}"
        study_dir = f"s{study_id}"

        study_path = self.paths.CXR_IMAGES / subject_dir / patient_dir / study_dir

        return study_path if study_path.exists() else None

    def load_all_cxr_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load all necessary CXR data for filtering.

        Returns:
            Tuple of (labels_df, metadata_df)
        """
        labels = self.load_chexpert_labels()
        metadata = self.load_metadata()

        return labels, metadata
