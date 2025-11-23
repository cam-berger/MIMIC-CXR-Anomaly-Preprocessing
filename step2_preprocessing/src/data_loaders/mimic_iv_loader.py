"""
Data loader for MIMIC-IV dataset with credential checks.

Handles optional data sources that may require additional credentialing:
- MIMIC-IV-Note: Clinical notes (requires additional credentialing)
- Medications: Medication administration records
- Procedures: Procedure data
"""
import pandas as pd
import logging
from pathlib import Path
from typing import Dict, Optional, Tuple, List

logger = logging.getLogger(__name__)


class MIMICIVLoader:
    """
    Loader for MIMIC-IV dataset with support for conditional data sources.

    Automatically detects availability of optional data sources that may
    require additional credentialing.
    """

    def __init__(self, mimic_iv_base_path: Path):
        """
        Initialize MIMIC-IV loader.

        Args:
            mimic_iv_base_path: Path to MIMIC-IV base directory
                               (e.g., /path/to/physionet.org/files/mimiciv/3.1/)
        """
        self.base_path = Path(mimic_iv_base_path)
        self.hosp_path = self.base_path / "hosp"
        self.icu_path = self.base_path / "icu"

        # Optional paths (may not exist)
        self.note_path = self.base_path / "note"

        logger.info(f"Initialized MIMIC-IV loader from: {self.base_path}")

    def check_credential_availability(self) -> Dict[str, bool]:
        """
        Check availability of credential-restricted data sources.

        Returns:
            Dictionary mapping data source names to availability status:
                - 'notes': MIMIC-IV-Note available
                - 'medications': Medication data available
                - 'procedures': Procedure data available
        """
        logger.info("Checking MIMIC-IV data source availability...")

        availability = {}

        # Check for MIMIC-IV-Note
        note_discharge_path = self.note_path / "discharge.csv.gz"
        note_radiology_path = self.note_path / "radiology.csv.gz"
        availability['notes_discharge'] = note_discharge_path.exists()
        availability['notes_radiology'] = note_radiology_path.exists()
        availability['notes'] = availability['notes_discharge'] or availability['notes_radiology']

        # Check for medications
        prescriptions_path = self.hosp_path / "prescriptions.csv.gz"
        poe_path = self.hosp_path / "poe.csv.gz"  # Provider order entry
        availability['prescriptions'] = prescriptions_path.exists()
        availability['poe'] = poe_path.exists()
        availability['medications'] = availability['prescriptions'] or availability['poe']

        # Check for procedures
        procedures_icd_path = self.hosp_path / "procedures_icd.csv.gz"
        availability['procedures'] = procedures_icd_path.exists()

        # Log results
        logger.info("Data source availability:")
        logger.info(f"  MIMIC-IV-Note:")
        logger.info(f"    Discharge summaries: {availability['notes_discharge']}")
        logger.info(f"    Radiology reports: {availability['notes_radiology']}")
        logger.info(f"  Medications:")
        logger.info(f"    Prescriptions: {availability['prescriptions']}")
        logger.info(f"    Provider orders: {availability['poe']}")
        logger.info(f"  Procedures (ICD): {availability['procedures']}")

        return availability

    def load_discharge_notes(
        self,
        subject_ids: Optional[List[int]] = None,
        hadm_ids: Optional[List[int]] = None
    ) -> Optional[pd.DataFrame]:
        """
        Load discharge summaries from MIMIC-IV-Note (if available).

        Args:
            subject_ids: Optional list of subject IDs to filter
            hadm_ids: Optional list of hospital admission IDs to filter

        Returns:
            DataFrame with discharge notes or None if not available
        """
        note_path = self.note_path / "discharge.csv.gz"

        if not note_path.exists():
            logger.warning(f"Discharge notes not available at: {note_path}")
            logger.warning("  MIMIC-IV-Note requires additional credentialing")
            return None

        logger.info(f"Loading discharge notes from: {note_path}")

        try:
            # Load with dtypes
            df = pd.read_csv(
                note_path,
                compression='gzip',
                dtype={
                    'note_id': 'str',
                    'subject_id': 'int32',
                    'hadm_id': 'float64',  # Can be null
                    'note_type': 'str',
                    'note_seq': 'int32',
                    'charttime': 'str'
                }
            )

            # Filter if requested
            if subject_ids is not None:
                df = df[df['subject_id'].isin(subject_ids)]

            if hadm_ids is not None:
                df = df[df['hadm_id'].isin(hadm_ids)]

            logger.info(f"  Loaded {len(df):,} discharge notes")

            return df

        except Exception as e:
            logger.error(f"Error loading discharge notes: {e}")
            return None

    def load_radiology_notes(
        self,
        subject_ids: Optional[List[int]] = None,
        hadm_ids: Optional[List[int]] = None
    ) -> Optional[pd.DataFrame]:
        """
        Load radiology reports from MIMIC-IV-Note (if available).

        Note: CXR-PRO is preferred for radiology reports as it removes
        prior references. This is provided as a fallback.

        Args:
            subject_ids: Optional list of subject IDs to filter
            hadm_ids: Optional list of hospital admission IDs to filter

        Returns:
            DataFrame with radiology notes or None if not available
        """
        note_path = self.note_path / "radiology.csv.gz"

        if not note_path.exists():
            logger.warning(f"Radiology notes not available at: {note_path}")
            logger.warning("  MIMIC-IV-Note requires additional credentialing")
            logger.info("  Using CXR-PRO as alternative for radiology reports")
            return None

        logger.info(f"Loading radiology notes from: {note_path}")

        try:
            df = pd.read_csv(
                note_path,
                compression='gzip',
                dtype={
                    'note_id': 'str',
                    'subject_id': 'int32',
                    'hadm_id': 'float64',
                    'note_type': 'str',
                    'note_seq': 'int32',
                    'charttime': 'str'
                }
            )

            # Filter if requested
            if subject_ids is not None:
                df = df[df['subject_id'].isin(subject_ids)]

            if hadm_ids is not None:
                df = df[df['hadm_id'].isin(hadm_ids)]

            logger.info(f"  Loaded {len(df):,} radiology notes")

            return df

        except Exception as e:
            logger.error(f"Error loading radiology notes: {e}")
            return None

    def load_prescriptions(
        self,
        subject_ids: Optional[List[int]] = None,
        hadm_ids: Optional[List[int]] = None
    ) -> Optional[pd.DataFrame]:
        """
        Load medication prescriptions.

        Args:
            subject_ids: Optional list of subject IDs to filter
            hadm_ids: Optional list of hospital admission IDs to filter

        Returns:
            DataFrame with prescriptions or None if not available
        """
        prescriptions_path = self.hosp_path / "prescriptions.csv.gz"

        if not prescriptions_path.exists():
            logger.warning(f"Prescriptions not available at: {prescriptions_path}")
            return None

        logger.info(f"Loading prescriptions from: {prescriptions_path}")

        try:
            # Read in chunks for memory efficiency
            chunks = []
            for chunk in pd.read_csv(
                prescriptions_path,
                compression='gzip',
                dtype={
                    'subject_id': 'int32',
                    'hadm_id': 'float64',
                    'pharmacy_id': 'int32',
                    'poe_id': 'str',
                    'poe_seq': 'int32'
                },
                chunksize=100000
            ):
                # Filter chunk if requested
                if subject_ids is not None:
                    chunk = chunk[chunk['subject_id'].isin(subject_ids)]

                if hadm_ids is not None:
                    chunk = chunk[chunk['hadm_id'].isin(hadm_ids)]

                if len(chunk) > 0:
                    chunks.append(chunk)

            if not chunks:
                logger.info("  No prescriptions found for specified filters")
                return pd.DataFrame()

            df = pd.concat(chunks, ignore_index=True)
            logger.info(f"  Loaded {len(df):,} prescriptions")

            return df

        except Exception as e:
            logger.error(f"Error loading prescriptions: {e}")
            return None

    def load_procedures_icd(
        self,
        subject_ids: Optional[List[int]] = None,
        hadm_ids: Optional[List[int]] = None
    ) -> Optional[pd.DataFrame]:
        """
        Load procedure data with ICD codes.

        Args:
            subject_ids: Optional list of subject IDs to filter
            hadm_ids: Optional list of hospital admission IDs to filter

        Returns:
            DataFrame with procedures or None if not available
        """
        procedures_path = self.hosp_path / "procedures_icd.csv.gz"

        if not procedures_path.exists():
            logger.warning(f"Procedures not available at: {procedures_path}")
            return None

        logger.info(f"Loading procedures from: {procedures_path}")

        try:
            df = pd.read_csv(
                procedures_path,
                compression='gzip',
                dtype={
                    'subject_id': 'int32',
                    'hadm_id': 'int32',
                    'seq_num': 'int32',
                    'chartdate': 'str',
                    'icd_code': 'str',
                    'icd_version': 'int8'
                }
            )

            # Filter if requested
            if subject_ids is not None:
                df = df[df['subject_id'].isin(subject_ids)]

            if hadm_ids is not None:
                df = df[df['hadm_id'].isin(hadm_ids)]

            logger.info(f"  Loaded {len(df):,} procedures")

            return df

        except Exception as e:
            logger.error(f"Error loading procedures: {e}")
            return None

    def get_data_sources_for_config(self, config: Dict) -> Dict[str, bool]:
        """
        Resolve data source availability based on config settings.

        Handles "auto" detection mode specified in config.

        Args:
            config: Configuration dictionary with precompilation.data_sources section

        Returns:
            Dictionary mapping data source names to whether they should be used
        """
        data_sources_config = config.get('precompilation', {}).get('data_sources', {})
        availability = self.check_credential_availability()

        resolved = {}

        # Resolve notes
        notes_setting = data_sources_config.get('mimic_iv_note', 'auto')
        if notes_setting == 'auto':
            resolved['mimic_iv_note'] = availability['notes']
        elif notes_setting is True:
            if not availability['notes']:
                logger.error("MIMIC-IV-Note required but not available!")
                raise FileNotFoundError("MIMIC-IV-Note data not found")
            resolved['mimic_iv_note'] = True
        else:
            resolved['mimic_iv_note'] = False

        # Resolve medications
        med_setting = data_sources_config.get('mimic_iv_med', 'auto')
        if med_setting == 'auto':
            resolved['mimic_iv_med'] = availability['medications']
        elif med_setting is True:
            if not availability['medications']:
                logger.error("Medication data required but not available!")
                raise FileNotFoundError("Medication data not found")
            resolved['mimic_iv_med'] = True
        else:
            resolved['mimic_iv_med'] = False

        logger.info("Resolved data sources:")
        logger.info(f"  MIMIC-IV-Note: {resolved['mimic_iv_note']}")
        logger.info(f"  MIMIC-IV Medications: {resolved['mimic_iv_med']}")

        return resolved
