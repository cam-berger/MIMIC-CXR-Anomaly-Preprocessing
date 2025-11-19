"""
Data loader for MIMIC-IV dataset.
"""
import pandas as pd
import logging
from typing import Optional

logger = logging.getLogger(__name__)


class IVDataLoader:
    """Loader for MIMIC-IV data files."""

    def __init__(self, paths_config):
        self.paths = paths_config

    def load_patients(self) -> pd.DataFrame:
        """
        Load patient demographics.

        Returns:
            DataFrame with patient information
        """
        logger.info(f"Loading patients from {self.paths.IV_PATIENTS}")

        try:
            df = pd.read_csv(
                self.paths.IV_PATIENTS,
                dtype={
                    'subject_id': 'int32',
                    'gender': 'str',
                    'anchor_age': 'int16',
                    'anchor_year': 'int16',
                    'anchor_year_group': 'str'
                },
                parse_dates=['dod']
            )

            logger.info(f"Loaded {len(df):,} patients")

            return df

        except Exception as e:
            logger.error(f"Error loading patients: {e}")
            raise

    def load_admissions(self) -> pd.DataFrame:
        """
        Load hospital admissions data.

        Returns:
            DataFrame with admission information
        """
        logger.info(f"Loading admissions from {self.paths.IV_ADMISSIONS}")

        try:
            df = pd.read_csv(
                self.paths.IV_ADMISSIONS,
                dtype={
                    'subject_id': 'int32',
                    'hadm_id': 'int32',
                    'admission_type': 'str',
                    'admission_location': 'str',
                    'discharge_location': 'str',
                    'insurance': 'str',
                    'language': 'str',
                    'marital_status': 'str',
                    'race': 'str',
                    'hospital_expire_flag': 'int8'
                },
                parse_dates=['admittime', 'dischtime', 'deathtime', 'edregtime', 'edouttime']
            )

            logger.info(f"Loaded {len(df):,} admissions")

            return df

        except Exception as e:
            logger.error(f"Error loading admissions: {e}")
            raise

    def load_diagnoses(self, chunk_size: Optional[int] = None) -> pd.DataFrame:
        """
        Load ICD diagnosis codes.

        Args:
            chunk_size: If specified, read file in chunks

        Returns:
            DataFrame with diagnosis codes
        """
        logger.info(f"Loading diagnoses from {self.paths.IV_DIAGNOSES}")

        try:
            if chunk_size:
                chunks = []
                for chunk in pd.read_csv(
                    self.paths.IV_DIAGNOSES,
                    dtype={
                        'subject_id': 'int32',
                        'hadm_id': 'int32',
                        'seq_num': 'int16',
                        'icd_code': 'str',
                        'icd_version': 'int8'
                    },
                    chunksize=chunk_size
                ):
                    chunks.append(chunk)
                df = pd.concat(chunks, ignore_index=True)
            else:
                df = pd.read_csv(
                    self.paths.IV_DIAGNOSES,
                    dtype={
                        'subject_id': 'int32',
                        'hadm_id': 'int32',
                        'seq_num': 'int16',
                        'icd_code': 'str',
                        'icd_version': 'int8'
                    }
                )

            logger.info(f"Loaded {len(df):,} diagnosis records")

            return df

        except Exception as e:
            logger.error(f"Error loading diagnoses: {e}")
            raise

    def load_transfers(self) -> pd.DataFrame:
        """
        Load patient transfer events (including ICU admissions).

        Returns:
            DataFrame with transfer information
        """
        logger.info(f"Loading transfers from {self.paths.IV_TRANSFERS}")

        try:
            df = pd.read_csv(
                self.paths.IV_TRANSFERS,
                dtype={
                    'subject_id': 'int32',
                    'hadm_id': 'float64',  # Can be null
                    'transfer_id': 'int32',
                    'eventtype': 'str',
                    'careunit': 'str'
                },
                parse_dates=['intime', 'outtime']
            )

            logger.info(f"Loaded {len(df):,} transfer records")

            return df

        except Exception as e:
            logger.error(f"Error loading transfers: {e}")
            raise

    def load_all_iv_data(self) -> tuple:
        """
        Load all necessary MIMIC-IV data for filtering.

        Returns:
            Tuple of (patients_df, admissions_df, transfers_df)
        """
        patients = self.load_patients()
        admissions = self.load_admissions()
        transfers = self.load_transfers()

        return patients, admissions, transfers
