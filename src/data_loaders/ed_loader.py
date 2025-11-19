"""
Data loader for MIMIC-IV-ED dataset.
"""
import pandas as pd
import logging
from typing import Optional

logger = logging.getLogger(__name__)


class EDDataLoader:
    """Loader for MIMIC-IV-ED data files."""

    def __init__(self, paths_config):
        self.paths = paths_config

    def load_ed_stays(self) -> pd.DataFrame:
        """
        Load ED stays data including disposition.

        Returns:
            DataFrame with ED stay information
        """
        logger.info(f"Loading ED stays from {self.paths.ED_STAYS}")

        try:
            df = pd.read_csv(
                self.paths.ED_STAYS,
                dtype={
                    'subject_id': 'int32',
                    'hadm_id': 'float64',  # Can be null
                    'stay_id': 'int32',
                    'gender': 'str',
                    'race': 'str',
                    'arrival_transport': 'str',
                    'disposition': 'str'
                },
                parse_dates=['intime', 'outtime']
            )

            logger.info(f"Loaded {len(df):,} ED stays")
            logger.info(f"Disposition distribution:\n{df['disposition'].value_counts()}")

            return df

        except Exception as e:
            logger.error(f"Error loading ED stays: {e}")
            raise

    def load_ed_diagnosis(self) -> pd.DataFrame:
        """
        Load ED diagnosis codes.

        Returns:
            DataFrame with ED diagnoses
        """
        logger.info(f"Loading ED diagnoses from {self.paths.ED_DIAGNOSIS}")

        try:
            df = pd.read_csv(
                self.paths.ED_DIAGNOSIS,
                dtype={
                    'subject_id': 'int32',
                    'stay_id': 'int32',
                    'seq_num': 'int16',
                    'icd_code': 'str',
                    'icd_version': 'int8',
                    'icd_title': 'str'
                }
            )

            logger.info(f"Loaded {len(df):,} ED diagnosis records")

            return df

        except Exception as e:
            logger.error(f"Error loading ED diagnoses: {e}")
            raise

    def load_ed_triage(self) -> pd.DataFrame:
        """
        Load ED triage data including vitals and chief complaint.

        Returns:
            DataFrame with triage information
        """
        logger.info(f"Loading ED triage from {self.paths.ED_TRIAGE}")

        try:
            df = pd.read_csv(
                self.paths.ED_TRIAGE,
                dtype={
                    'subject_id': 'int32',
                    'stay_id': 'int32',
                    'temperature': 'float32',
                    'heartrate': 'float32',
                    'resprate': 'float32',
                    'o2sat': 'float32',
                    'sbp': 'float32',
                    'dbp': 'float32',
                    'pain': 'str',
                    'acuity': 'float32',
                    'chiefcomplaint': 'str'
                }
            )

            logger.info(f"Loaded {len(df):,} ED triage records")

            return df

        except Exception as e:
            logger.error(f"Error loading ED triage: {e}")
            raise

    def load_ed_vitalsign(self) -> pd.DataFrame:
        """
        Load ED vital signs (continuous monitoring).

        Returns:
            DataFrame with vital signs
        """
        logger.info(f"Loading ED vital signs from {self.paths.ED_VITALSIGN}")

        try:
            df = pd.read_csv(
                self.paths.ED_VITALSIGN,
                dtype={
                    'subject_id': 'int32',
                    'stay_id': 'int32',
                    'temperature': 'float32',
                    'heartrate': 'float32',
                    'resprate': 'float32',
                    'o2sat': 'float32',
                    'sbp': 'float32',
                    'dbp': 'float32',
                    'rhythm': 'str',
                    'pain': 'str'
                },
                parse_dates=['charttime']
            )

            logger.info(f"Loaded {len(df):,} ED vital sign records")

            return df

        except Exception as e:
            logger.error(f"Error loading ED vital signs: {e}")
            raise

    def load_all_ed_data(self) -> tuple:
        """
        Load all necessary ED data for filtering.

        Returns:
            Tuple of (stays_df, diagnosis_df, triage_df)
        """
        stays = self.load_ed_stays()
        diagnosis = self.load_ed_diagnosis()
        triage = self.load_ed_triage()

        return stays, diagnosis, triage
