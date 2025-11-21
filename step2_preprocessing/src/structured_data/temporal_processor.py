"""
Temporal structured data processor for MIMIC-IV labs and vitals.
Handles missing values with NOT_DONE token and creates time-aware features.
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from pathlib import Path
import logging
import sys

# Add parent directory to path for base imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from base.processor import StructuredProcessor

logger = logging.getLogger(__name__)


class TemporalFeatureExtractor(StructuredProcessor):
    """
    Extract structured clinical data with temporal awareness.
    Implements missing value tokens and time-series features as specified in Step 2.
    """

    MISSING_TOKEN = "NOT_DONE"

    def __init__(self, config: Dict, paths):
        super().__init__(config)
        self.paths = paths

        # Load configuration
        self.priority_labs = config['structured']['priority_labs']
        self.priority_vitals = config['structured']['priority_vitals']
        self.temporal_enabled = config['structured']['temporal_features']['enabled']
        self.encoding_method = config['structured']['encoding_method']

        # Load data once for efficiency
        self._load_mimic_data()

        logger.info(f"Initialized TemporalFeatureExtractor")
        logger.info(f"  Labs to extract: {len(self.priority_labs)}")
        logger.info(f"  Vitals to extract: {len(self.priority_vitals)}")
        logger.info(f"  Temporal features: {self.temporal_enabled}")
        logger.info(f"  Encoding method: {self.encoding_method}")

    def validate_config(self) -> None:
        """Validate structured data configuration"""
        # Check required keys
        self.get_config_value('structured', 'priority_labs', required=True)
        self.get_config_value('structured', 'priority_vitals', required=True)
        self.get_config_value('structured', 'temporal_features', 'enabled', required=True)
        self.get_config_value('structured', 'encoding_method', required=True)

        # Validate encoding method
        valid_methods = ['aggregated', 'sequential']
        encoding = self.get_config_value('structured', 'encoding_method')
        if encoding not in valid_methods:
            raise ValueError(f"encoding_method must be one of {valid_methods}, got '{encoding}'")

    def process(self, subject_id: int, hadm_id: Optional[int],
                ed_intime: datetime, study_datetime: datetime, **kwargs) -> Optional[Dict]:
        """Process structured data (implements BaseProcessor.process)"""
        try:
            return self.extract_features(subject_id, hadm_id, ed_intime, study_datetime)
        except Exception as e:
            self._handle_error(e, f"subject {subject_id}")
            return None

    def _load_mimic_data(self):
        """Load MIMIC-IV data tables"""
        logger.info("Loading MIMIC-IV data tables...")

        # ED vitals (smaller file)
        try:
            self.ed_vitals = pd.read_csv(
                self.paths.get_vital_signs_path(),
                parse_dates=['charttime']
            )
            logger.info(f"  Loaded {len(self.ed_vitals):,} ED vital records")
        except Exception as e:
            logger.warning(f"Could not load ED vitals: {e}")
            self.ed_vitals = pd.DataFrame()

        # ED triage
        try:
            self.ed_triage = pd.read_csv(self.paths.get_triage_path())
            logger.info(f"  Loaded {len(self.ed_triage):,} ED triage records")
        except Exception as e:
            logger.warning(f"Could not load ED triage: {e}")
            self.ed_triage = pd.DataFrame()

        # Lab events - this is very large, we'll load in chunks as needed
        self.lab_events_path = self.paths.get_lab_events_path()
        logger.info(f"  Lab events will be loaded on-demand from {self.lab_events_path}")

    def extract_features(
        self,
        subject_id: int,
        hadm_id: Optional[int],
        ed_intime: datetime,
        study_datetime: datetime
    ) -> Dict:
        """
        Extract all structured features for a patient case.

        Args:
            subject_id: Patient ID
            hadm_id: Hospital admission ID (may be None for ED-only visits)
            ed_intime: ED admission time
            study_datetime: CXR study datetime

        Returns:
            Dictionary of features with temporal metadata
        """
        features = {}

        # Extract vitals
        vital_features = self._extract_vitals(subject_id, ed_intime, study_datetime)
        features.update(vital_features)

        # Extract labs (if hospital admission exists)
        if hadm_id is not None and not pd.isna(hadm_id):
            lab_features = self._extract_labs(int(hadm_id), study_datetime)
            features.update(lab_features)
        else:
            # All labs are NOT_DONE
            for lab in self.priority_labs:
                features[f"lab_{lab}"] = self._create_missing_feature(lab)

        return features

    def _extract_vitals(
        self,
        subject_id: int,
        ed_intime: datetime,
        study_datetime: datetime
    ) -> Dict:
        """Extract vital signs from ED data"""
        features = {}

        # Ensure ed_intime and study_datetime are pandas Timestamps
        if not isinstance(ed_intime, pd.Timestamp):
            ed_intime = pd.Timestamp(ed_intime)
        if not isinstance(study_datetime, pd.Timestamp):
            study_datetime = pd.Timestamp(study_datetime)

        # Get ED vitals for this patient
        patient_vitals = self.ed_vitals[
            self.ed_vitals['subject_id'] == subject_id
        ].copy()

        if len(patient_vitals) == 0:
            # No vitals recorded - use triage if available
            patient_triage = self.ed_triage[
                self.ed_triage['subject_id'] == subject_id
            ]

            if len(patient_triage) > 0:
                # Use triage vitals (single measurement)
                triage_row = patient_triage.iloc[0]
                for vital in self.priority_vitals:
                    if vital in triage_row and not pd.isna(triage_row[vital]):
                        features[f"vital_{vital}"] = self._create_single_measurement_feature(
                            vital,
                            float(triage_row[vital]),
                            time_since_admission=0.0  # Triage is at admission
                        )
                    else:
                        features[f"vital_{vital}"] = self._create_missing_feature(vital)
            else:
                # No vitals at all
                for vital in self.priority_vitals:
                    features[f"vital_{vital}"] = self._create_missing_feature(vital)

            return features

        # Filter vitals before study
        time_diff = patient_vitals['charttime'] - ed_intime
        patient_vitals['time_since_admission'] = time_diff.dt.total_seconds() / 3600  # Hours

        pre_study_vitals = patient_vitals[
            patient_vitals['charttime'] <= study_datetime
        ].copy()

        # Extract each priority vital
        for vital in self.priority_vitals:
            if vital not in pre_study_vitals.columns:
                features[f"vital_{vital}"] = self._create_missing_feature(vital)
                continue

            vital_measurements = pre_study_vitals[
                pre_study_vitals[vital].notna()
            ][[vital, 'time_since_admission', 'charttime']].copy()

            if len(vital_measurements) == 0:
                features[f"vital_{vital}"] = self._create_missing_feature(vital)
            else:
                features[f"vital_{vital}"] = self._create_temporal_feature(
                    vital, vital_measurements
                )

        return features

    def _extract_labs(self, hadm_id: int, study_datetime: datetime) -> Dict:
        """Extract laboratory results"""
        features = {}

        # Load relevant lab events (chunk by chunk to save memory)
        try:
            # Read only rows for this admission
            lab_df = pd.read_csv(
                self.lab_events_path,
                chunksize=100000,
                parse_dates=['charttime']
            )

            patient_labs = []
            for chunk in lab_df:
                patient_chunk = chunk[chunk['hadm_id'] == hadm_id]
                if len(patient_chunk) > 0:
                    patient_labs.append(patient_chunk)

            if len(patient_labs) > 0:
                patient_labs = pd.concat(patient_labs, ignore_index=True)
            else:
                patient_labs = pd.DataFrame()

        except Exception as e:
            logger.warning(f"Error loading labs for hadm_id {hadm_id}: {e}")
            patient_labs = pd.DataFrame()

        if len(patient_labs) == 0:
            # No labs available
            for lab in self.priority_labs:
                features[f"lab_{lab}"] = self._create_missing_feature(lab)
            return features

        # Filter labs before study
        pre_study_labs = patient_labs[
            patient_labs['charttime'] <= study_datetime
        ].copy()

        # Extract each priority lab
        for lab in self.priority_labs:
            # Find matching lab items (case-insensitive match on label)
            lab_measurements = pre_study_labs[
                pre_study_labs['label'].str.lower().str.contains(lab.lower(), na=False)
            ]

            if len(lab_measurements) == 0:
                features[f"lab_{lab}"] = self._create_missing_feature(lab)
            else:
                features[f"lab_{lab}"] = self._create_temporal_feature(
                    lab, lab_measurements[['valuenum', 'charttime']]
                )

        return features

    def _create_temporal_feature(
        self,
        name: str,
        measurements: pd.DataFrame
    ) -> Dict:
        """
        Create temporal feature representation for a sequence of measurements.

        Implements both sequential and aggregated encoding methods.
        """
        measurements = measurements.sort_values('charttime')

        if self.encoding_method == 'aggregated':
            return self._create_aggregated_features(name, measurements)
        elif self.encoding_method == 'sequential':
            return self._create_sequential_features(name, measurements)
        else:
            raise ValueError(f"Unknown encoding method: {self.encoding_method}")

    def _create_aggregated_features(self, name: str, measurements: pd.DataFrame) -> Dict:
        """
        Create aggregated temporal features (Step 2 specification).

        Returns summary statistics and temporal patterns.
        """
        value_col = measurements.columns[0]  # First column is the value
        time_col = measurements.columns[1]   # Second column is time

        values = measurements[value_col].values
        times = pd.to_datetime(measurements[time_col])

        # Convert times to hours since first measurement
        first_time = times.iloc[0]
        hours_since_first = [(pd.Timestamp(t) - first_time).total_seconds() / 3600 for t in times]

        features = {
            'is_missing': False,
            'measurement_count': len(measurements),

            # Last recorded value (most recent)
            'last_value': float(values[-1]),

            # First recorded value
            'first_value': float(values[0]),

            # Trend (slope from first to last)
            'trend_slope': float(values[-1] - values[0]) / max(hours_since_first[-1], 1.0) if len(values) > 1 else 0.0,

            # Summary statistics
            'mean_value': float(np.mean(values)),
            'std_value': float(np.std(values)) if len(values) > 1 else 0.0,
            'min_value': float(np.min(values)),
            'max_value': float(np.max(values)),

            # Temporal metadata
            'time_span_hours': float(hours_since_first[-1]) if len(hours_since_first) > 1 else 0.0,
            'avg_time_between_measurements': float(np.mean(np.diff(hours_since_first))) if len(hours_since_first) > 1 else 0.0,
        }

        return features

    def _create_sequential_features(self, name: str, measurements: pd.DataFrame) -> Dict:
        """
        Create sequential temporal features (for RNN/Transformer encoding).

        Returns list of (value, time_metadata) tuples.
        """
        value_col = measurements.columns[0]
        time_col = measurements.columns[1]

        sequence = []
        first_time = pd.Timestamp(measurements[time_col].iloc[0])

        for idx, row in measurements.iterrows():
            time_since_first = (pd.Timestamp(row[time_col]) - first_time).total_seconds() / 3600

            measurement = {
                'value': float(row[value_col]),
                'time_since_start': float(time_since_first),
                'ordinal_index': len(sequence) + 1,
            }
            sequence.append(measurement)

        return {
            'is_missing': False,
            'sequence': sequence,
            'length': len(sequence)
        }

    def _create_single_measurement_feature(
        self,
        name: str,
        value: float,
        time_since_admission: float
    ) -> Dict:
        """Create feature for a single measurement (e.g., triage vital)"""
        return {
            'is_missing': False,
            'measurement_count': 1,
            'last_value': float(value),
            'first_value': float(value),
            'trend_slope': 0.0,
            'mean_value': float(value),
            'std_value': 0.0,
            'min_value': float(value),
            'max_value': float(value),
            'time_span_hours': 0.0,
            'avg_time_between_measurements': 0.0,
        }

    def _create_missing_feature(self, name: str) -> Dict:
        """
        Create feature representation for missing/NOT_DONE measurement.

        Uses special token instead of imputation.
        """
        if self.encoding_method == 'aggregated':
            return {
                'is_missing': True,
                'measurement_count': 0,
                'last_value': self.MISSING_TOKEN,
                'first_value': self.MISSING_TOKEN,
                'trend_slope': 0.0,
                'mean_value': 0.0,
                'std_value': 0.0,
                'min_value': 0.0,
                'max_value': 0.0,
                'time_span_hours': 0.0,
                'avg_time_between_measurements': 0.0,
            }
        else:  # sequential
            return {
                'is_missing': True,
                'sequence': [],
                'length': 0
            }
