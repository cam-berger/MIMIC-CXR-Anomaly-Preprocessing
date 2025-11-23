"""
DICOM Metadata Loader for MIMIC-CXR

Loads and aggregates study-level DICOM metadata from MIMIC-CXR-2.0.0-metadata.csv
to provide image acquisition context for preprocessing.

Available metadata:
- ViewPosition: PA, AP, LATERAL, LL (left lateral)
- PatientOrientation: Erect, Recumbent
- PerformedProcedureStepDescription: "CHEST (PA AND LAT)", "CHEST (PORTABLE AP)"
- Image dimensions: Rows, Columns

Purpose: Prevent model misclassifications due to acquisition technique
(e.g., AP portable showing enlarged cardiac silhouette vs. true cardiomegaly)
"""

import pandas as pd
import logging
from pathlib import Path
from typing import Dict, Optional, List

logger = logging.getLogger(__name__)


class DICOMMetadataLoader:
    """Load and aggregate study-level DICOM metadata from MIMIC-CXR"""

    def __init__(self, metadata_path: str):
        """
        Args:
            metadata_path: Path to mimic-cxr-2.0.0-metadata.csv or .csv.gz
        """
        self.metadata_path = Path(metadata_path)

        if not self.metadata_path.exists():
            raise FileNotFoundError(f"DICOM metadata not found: {metadata_path}")

        logger.info(f"Loading DICOM metadata from: {metadata_path}")

        # Load metadata CSV (handles .csv.gz automatically)
        self.metadata_df = pd.read_csv(metadata_path, low_memory=False)

        # Create study-level aggregation
        self._aggregate_study_metadata()

        logger.info(f"Loaded metadata for {len(self.study_metadata)} studies")

    def _aggregate_study_metadata(self):
        """
        Aggregate DICOM-level metadata to study level.

        Each study has 1-3 DICOM images (PA, LATERAL, etc.).
        We need study-level features combining all views.
        """
        # Group by study_id
        study_groups = self.metadata_df.groupby('study_id')

        study_records = []

        for study_id, group in study_groups:
            # Aggregate views in this study
            views = group['ViewPosition'].dropna().unique().tolist()
            orientations = group['PatientOrientationCodeSequence_CodeMeaning'].dropna().unique().tolist()
            procedures = group['PerformedProcedureStepDescription'].dropna().unique().tolist()

            # Get image dimensions (use first non-null)
            rows = group['Rows'].dropna()
            cols = group['Columns'].dropna()

            study_record = {
                'study_id': study_id,
                'num_dicoms': len(group),

                # Views available
                'views_list': ','.join(views) if views else '',
                'has_pa': 'PA' in views,
                'has_ap': 'AP' in views,
                'has_lateral': any(v in ['LATERAL', 'LL'] for v in views),

                # Patient orientation
                'orientations_list': ','.join(orientations) if orientations else '',
                'is_erect': 'Erect' in orientations,
                'is_recumbent': 'Recumbent' in orientations,
                'orientation_unknown': len(orientations) == 0,

                # Procedure type (portable indicator)
                'procedure_desc': procedures[0] if procedures else '',
                'is_portable': any('PORTABLE' in str(p).upper() for p in procedures),

                # Image dimensions (average if multiple)
                'avg_rows': rows.mean() if len(rows) > 0 else None,
                'avg_cols': cols.mean() if len(cols) > 0 else None,
            }

            study_records.append(study_record)

        # Convert to DataFrame for efficient lookup
        self.study_metadata = pd.DataFrame(study_records).set_index('study_id')

    def get_study_metadata(self, study_id: int) -> Optional[Dict]:
        """
        Get aggregated metadata for a study.

        Args:
            study_id: MIMIC-CXR study ID

        Returns:
            Dictionary of study-level metadata features, or None if not found
        """
        try:
            if study_id not in self.study_metadata.index:
                logger.warning(f"Study {study_id} not found in DICOM metadata")
                return None

            # Get row as dictionary
            metadata = self.study_metadata.loc[study_id].to_dict()

            return metadata

        except Exception as e:
            logger.error(f"Error loading metadata for study {study_id}: {e}")
            return None

    def get_metadata_features(self, study_id: int) -> Dict[str, float]:
        """
        Get metadata as model-ready features (all numeric).

        Args:
            study_id: MIMIC-CXR study ID

        Returns:
            Dictionary of numeric features ready for model input
        """
        metadata = self.get_study_metadata(study_id)

        if metadata is None:
            # Return default "unknown" features
            return self._default_features()

        # Convert to numeric features
        features = {
            # View position (one-hot encoded)
            'view_pa': 1.0 if metadata['has_pa'] else 0.0,
            'view_ap': 1.0 if metadata['has_ap'] else 0.0,
            'view_lateral': 1.0 if metadata['has_lateral'] else 0.0,

            # Patient orientation
            'orientation_erect': 1.0 if metadata['is_erect'] else 0.0,
            'orientation_recumbent': 1.0 if metadata['is_recumbent'] else 0.0,
            'orientation_unknown': 1.0 if metadata['orientation_unknown'] else 0.0,

            # Acquisition type
            'is_portable': 1.0 if metadata['is_portable'] else 0.0,

            # Image dimensions (normalized)
            'image_rows_normalized': self._normalize_dimension(metadata['avg_rows'], dim='rows'),
            'image_cols_normalized': self._normalize_dimension(metadata['avg_cols'], dim='cols'),

            # Number of views (indicates comprehensive study)
            'num_views': float(metadata['num_dicoms']),
        }

        return features

    def _default_features(self) -> Dict[str, float]:
        """Return default features when metadata is missing"""
        return {
            'view_pa': 0.0,
            'view_ap': 0.0,
            'view_lateral': 0.0,
            'orientation_erect': 0.0,
            'orientation_recumbent': 0.0,
            'orientation_unknown': 1.0,  # Mark as unknown
            'is_portable': 0.0,
            'image_rows_normalized': 0.0,
            'image_cols_normalized': 0.0,
            'num_views': 0.0,
        }

    def _normalize_dimension(self, value: Optional[float], dim: str) -> float:
        """
        Normalize image dimension to [0, 1] range.

        Typical MIMIC-CXR dimensions:
        - Rows: ~2000-3500 pixels
        - Cols: ~1500-3000 pixels
        """
        if value is None or pd.isna(value):
            return 0.0

        if dim == 'rows':
            # Normalize assuming range [1500, 3500]
            return (value - 1500) / 2000
        elif dim == 'cols':
            # Normalize assuming range [1500, 3000]
            return (value - 1500) / 1500
        else:
            return 0.0

    def get_coverage_stats(self) -> Dict:
        """Get statistics on metadata coverage"""
        total_studies = len(self.study_metadata)

        stats = {
            'total_studies': total_studies,
            'has_view_info': (self.study_metadata['views_list'] != '').sum(),
            'has_orientation': (~self.study_metadata['orientation_unknown']).sum(),
            'is_portable': self.study_metadata['is_portable'].sum(),
            'view_distribution': {
                'PA': self.study_metadata['has_pa'].sum(),
                'AP': self.study_metadata['has_ap'].sum(),
                'LATERAL': self.study_metadata['has_lateral'].sum(),
            },
            'orientation_distribution': {
                'Erect': self.study_metadata['is_erect'].sum(),
                'Recumbent': self.study_metadata['is_recumbent'].sum(),
                'Unknown': self.study_metadata['orientation_unknown'].sum(),
            }
        }

        return stats
