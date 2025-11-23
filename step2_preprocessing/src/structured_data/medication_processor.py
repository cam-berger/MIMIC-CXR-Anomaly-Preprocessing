"""
Medication processor for temporal medication feature extraction.

Extracts medication administration data using temporal aggregation,
following the same pattern as vitals and labs.
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from datetime import datetime, timedelta
from pathlib import Path
import logging
import sys

# Add parent directory to path for base imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from base.processor import StructuredProcessor

logger = logging.getLogger(__name__)


class MedicationProcessor(StructuredProcessor):
    """
    Extract medication features with temporal awareness.

    Extracts medications from MIMIC-IV prescriptions data within a temporal
    window around the CXR study datetime, similar to vitals and labs extraction.
    """

    MISSING_TOKEN = "NOT_DONE"

    # Priority medication categories for chest X-ray interpretation
    PRIORITY_MED_CATEGORIES = [
        'antibiotics',  # Infection treatment
        'diuretics',  # Fluid management (relevant for pulmonary edema)
        'bronchodilators',  # Respiratory conditions
        'corticosteroids',  # Inflammation/COPD
        'anticoagulants',  # Relevant for PE
        'vasopressors',  # Shock/critical illness
        'antihypertensives',  # Cardiac conditions
        'opioids'  # Pain management (relevant for respiratory depression)
    ]

    # Medication name mappings (generic drug names to categories)
    MED_CATEGORY_MAPPING = {
        # Antibiotics
        'vancomycin': 'antibiotics',
        'cefepime': 'antibiotics',
        'piperacillin': 'antibiotics',
        'tazobactam': 'antibiotics',
        'meropenem': 'antibiotics',
        'azithromycin': 'antibiotics',
        'levofloxacin': 'antibiotics',
        'ceftriaxone': 'antibiotics',

        # Diuretics
        'furosemide': 'diuretics',
        'lasix': 'diuretics',
        'bumetanide': 'diuretics',
        'torsemide': 'diuretics',
        'hydrochlorothiazide': 'diuretics',

        # Bronchodilators
        'albuterol': 'bronchodilators',
        'ipratropium': 'bronchodilators',
        'tiotropium': 'bronchodilators',

        # Corticosteroids
        'prednisone': 'corticosteroids',
        'methylprednisolone': 'corticosteroids',
        'dexamethasone': 'corticosteroids',
        'hydrocortisone': 'corticosteroids',

        # Anticoagulants
        'heparin': 'anticoagulants',
        'enoxaparin': 'anticoagulants',
        'warfarin': 'anticoagulants',
        'apixaban': 'anticoagulants',

        # Vasopressors
        'norepinephrine': 'vasopressors',
        'epinephrine': 'vasopressors',
        'vasopressin': 'vasopressors',
        'dopamine': 'vasopressors',

        # Antihypertensives
        'metoprolol': 'antihypertensives',
        'lisinopril': 'antihypertensives',
        'amlodipine': 'antihypertensives',
        'losartan': 'antihypertensives',

        # Opioids
        'fentanyl': 'opioids',
        'morphine': 'opioids',
        'hydromorphone': 'opioids',
        'oxycodone': 'opioids'
    }

    def __init__(self, config: Dict, mimic_iv_loader):
        """
        Initialize medication processor.

        Args:
            config: Configuration dictionary
            mimic_iv_loader: Instance of MIMICIVLoader for data access
        """
        super().__init__(config)
        self.mimic_iv_loader = mimic_iv_loader

        # Load configuration
        self.temporal_window_before = config.get('precompilation', {}).get(
            'temporal_window', {}
        ).get('before_study_hours', 48)
        self.temporal_window_after = config.get('precompilation', {}).get(
            'temporal_window', {}
        ).get('after_study_hours', 24)

        self.prescriptions_cache = None  # Lazy loading

        logger.info(f"Initialized MedicationProcessor")
        logger.info(f"  Temporal window: -{self.temporal_window_before}h to +{self.temporal_window_after}h")
        logger.info(f"  Priority categories: {len(self.PRIORITY_MED_CATEGORIES)}")

    def validate_config(self) -> None:
        """Validate medication processing configuration"""
        # Check if medication data is enabled
        enabled = self.get_config_value(
            'precompilation', 'data_sources', 'mimic_iv_med',
            default=False
        )

        if enabled and enabled != 'auto':
            # Verify data availability
            availability = self.mimic_iv_loader.check_credential_availability()
            if not availability.get('medications', False):
                raise FileNotFoundError("Medication data required but not available")

    def process(
        self,
        subject_id: int,
        hadm_id: Optional[int],
        study_datetime: datetime,
        **kwargs
    ) -> Optional[Dict]:
        """
        Process medications (implements BaseProcessor.process).

        Args:
            subject_id: Patient ID
            hadm_id: Hospital admission ID
            study_datetime: CXR study datetime

        Returns:
            Dictionary of medication features or None if processing fails
        """
        try:
            return self.extract_medication_features(subject_id, hadm_id, study_datetime)
        except Exception as e:
            self._handle_error(e, f"subject {subject_id}")
            return None

    def extract_medication_features(
        self,
        subject_id: int,
        hadm_id: Optional[int],
        study_datetime: datetime
    ) -> Dict:
        """
        Extract medication features for a patient case.

        Args:
            subject_id: Patient ID
            hadm_id: Hospital admission ID (may be None)
            study_datetime: CXR study datetime

        Returns:
            Dictionary of medication features with temporal metadata
        """
        features = {}

        # If no hospital admission, all medications are NOT_DONE
        if hadm_id is None or pd.isna(hadm_id):
            for category in self.PRIORITY_MED_CATEGORIES:
                features[f"med_{category}"] = self._create_missing_feature(category)
            return features

        # Load prescriptions for this patient (cached)
        if self.prescriptions_cache is None:
            self._load_prescriptions()

        # Filter to this admission
        hadm_id = int(hadm_id)
        admission_meds = self.prescriptions_cache[
            self.prescriptions_cache['hadm_id'] == hadm_id
        ]

        if len(admission_meds) == 0:
            # No medications for this admission
            for category in self.PRIORITY_MED_CATEGORIES:
                features[f"med_{category}"] = self._create_missing_feature(category)
            return features

        # Parse timestamps
        if not isinstance(study_datetime, pd.Timestamp):
            study_datetime = pd.Timestamp(study_datetime)

        # Define temporal window
        window_start = study_datetime - timedelta(hours=self.temporal_window_before)
        window_end = study_datetime + timedelta(hours=self.temporal_window_after)

        # Convert starttime and stoptime to timestamps
        if 'starttime' in admission_meds.columns:
            admission_meds = admission_meds.copy()
            admission_meds['starttime'] = pd.to_datetime(
                admission_meds['starttime'],
                errors='coerce'
            )

        # Filter to temporal window (medications started within window)
        if 'starttime' in admission_meds.columns:
            temporal_meds = admission_meds[
                (admission_meds['starttime'] >= window_start) &
                (admission_meds['starttime'] <= window_end)
            ]
        else:
            # If no timestamps, use all medications from admission
            temporal_meds = admission_meds

        # Categorize medications
        for category in self.PRIORITY_MED_CATEGORIES:
            category_meds = self._filter_by_category(temporal_meds, category)

            if len(category_meds) > 0:
                features[f"med_{category}"] = {
                    'present': True,
                    'count': len(category_meds),
                    'medications': category_meds['drug'].head(5).tolist() if 'drug' in category_meds.columns else []
                }
            else:
                features[f"med_{category}"] = self._create_missing_feature(category)

        return features

    def _load_prescriptions(self):
        """Load prescriptions data (cached for efficiency)"""
        logger.info("Loading MIMIC-IV prescriptions data...")

        prescriptions = self.mimic_iv_loader.load_prescriptions()

        if prescriptions is None or len(prescriptions) == 0:
            logger.warning("No prescription data available")
            self.prescriptions_cache = pd.DataFrame()
        else:
            self.prescriptions_cache = prescriptions
            logger.info(f"  Cached {len(self.prescriptions_cache):,} prescriptions")

    def _filter_by_category(self, meds_df: pd.DataFrame, category: str) -> pd.DataFrame:
        """
        Filter medications by category based on drug names.

        Args:
            meds_df: DataFrame of medications
            category: Medication category (e.g., 'antibiotics')

        Returns:
            Filtered DataFrame containing only medications in this category
        """
        if len(meds_df) == 0 or 'drug' not in meds_df.columns:
            return pd.DataFrame()

        # Get drug names for this category
        category_drugs = [
            drug for drug, cat in self.MED_CATEGORY_MAPPING.items()
            if cat == category
        ]

        # Filter by partial string match (case-insensitive)
        mask = meds_df['drug'].str.lower().str.contains(
            '|'.join(category_drugs),
            case=False,
            na=False
        )

        return meds_df[mask]

    def _create_missing_feature(self, category: str) -> Dict:
        """
        Create a feature dictionary for missing medication data.

        Args:
            category: Medication category name

        Returns:
            Dictionary with MISSING_TOKEN indicator
        """
        return {
            'present': False,
            'count': 0,
            'value': self.MISSING_TOKEN
        }

    def get_feature_names(self) -> List[str]:
        """
        Get list of medication feature names.

        Returns:
            List of feature names (e.g., ['med_antibiotics', 'med_diuretics', ...])
        """
        return [f"med_{category}" for category in self.PRIORITY_MED_CATEGORIES]
