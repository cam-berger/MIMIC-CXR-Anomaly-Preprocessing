"""
Radiology report filtering based on CheXpert/NegBio labels.
"""
import pandas as pd
import numpy as np
import logging
from typing import Set, List

logger = logging.getLogger(__name__)


class RadiologyFilter:
    """Filter chest X-ray studies based on radiology report criteria."""

    def __init__(self, filter_config):
        self.config = filter_config

    def filter_normal_studies(self, labels_df: pd.DataFrame) -> pd.DataFrame:
        """
        Filter studies to identify those with "No Finding" and no pathologies.

        Criteria:
        1. "No Finding" label = 1.0
        2. All pathology labels are either blank, 0, -1, or NaN (not positive)

        Args:
            labels_df: DataFrame with CheXpert or NegBio labels

        Returns:
            Filtered DataFrame containing only normal studies
        """
        logger.info("Filtering for normal radiology studies...")
        logger.info(f"Starting with {len(labels_df):,} studies")

        df = labels_df.copy()

        # Step 1: Filter for "No Finding" = 1.0
        if 'No Finding' not in df.columns:
            logger.error("'No Finding' column not found in labels")
            raise ValueError("'No Finding' column is required")

        normal_mask = df['No Finding'] == self.config.no_finding_value
        logger.info(f"Studies with 'No Finding' = 1.0: {normal_mask.sum():,}")

        # Step 2: Ensure all pathology labels are not positive (not 1.0)
        pathology_labels = [label for label in self.config.exclude_pathology_labels
                           if label in df.columns]

        logger.info(f"Checking {len(pathology_labels)} pathology labels")

        for label in pathology_labels:
            # A label is considered abnormal if it equals 1.0
            # Values of 0, -1 (uncertain), or NaN are acceptable
            abnormal_mask = df[label] == 1.0
            normal_mask = normal_mask & ~abnormal_mask

            abnormal_count = abnormal_mask.sum()
            if abnormal_count > 0:
                logger.debug(f"  {label}: excluded {abnormal_count:,} studies")

        filtered_df = df[normal_mask].copy()
        logger.info(f"After filtering: {len(filtered_df):,} normal studies ({len(filtered_df)/len(df)*100:.2f}%)")

        # Log statistics about the filtered dataset
        self._log_filter_statistics(labels_df, filtered_df)

        return filtered_df

    def get_pathology_distribution(self, labels_df: pd.DataFrame) -> pd.DataFrame:
        """
        Get distribution of pathology labels.

        Args:
            labels_df: DataFrame with labels

        Returns:
            DataFrame with counts of each label value
        """
        pathology_labels = [label for label in self.config.exclude_pathology_labels
                           if label in labels_df.columns]

        distributions = []
        for label in pathology_labels:
            value_counts = labels_df[label].value_counts(dropna=False)
            distributions.append({
                'label': label,
                'positive': value_counts.get(1.0, 0),
                'negative': value_counts.get(0.0, 0),
                'uncertain': value_counts.get(-1.0, 0),
                'missing': labels_df[label].isna().sum()
            })

        return pd.DataFrame(distributions)

    def _log_filter_statistics(self, original_df: pd.DataFrame, filtered_df: pd.DataFrame):
        """Log detailed statistics about filtering."""
        logger.info("=" * 60)
        logger.info("Radiology Filtering Statistics")
        logger.info("=" * 60)
        logger.info(f"Original studies: {len(original_df):,}")
        logger.info(f"Normal studies: {len(filtered_df):,}")
        logger.info(f"Excluded studies: {len(original_df) - len(filtered_df):,}")
        logger.info(f"Retention rate: {len(filtered_df)/len(original_df)*100:.2f}%")

        # Check unique subjects and studies
        logger.info(f"Unique subjects in normal cohort: {filtered_df['subject_id'].nunique():,}")
        logger.info(f"Studies per subject (mean): {len(filtered_df)/filtered_df['subject_id'].nunique():.2f}")
        logger.info("=" * 60)

    def filter_by_view_position(self, metadata_df: pd.DataFrame,
                                view_positions: Set[str] = None) -> pd.DataFrame:
        """
        Filter studies by view position (e.g., PA, AP, LATERAL).

        Args:
            metadata_df: DataFrame with metadata
            view_positions: Set of acceptable view positions (default: {'PA', 'AP'})

        Returns:
            Filtered DataFrame
        """
        if view_positions is None:
            view_positions = {'PA', 'AP'}  # Frontal views only

        if 'ViewPosition' not in metadata_df.columns:
            logger.warning("ViewPosition column not found, skipping view filter")
            return metadata_df

        logger.info(f"Filtering for view positions: {view_positions}")
        filtered = metadata_df[metadata_df['ViewPosition'].isin(view_positions)].copy()
        logger.info(f"Retained {len(filtered):,} / {len(metadata_df):,} images")

        return filtered

    def exclude_support_devices(self, labels_df: pd.DataFrame,
                                allow_support_devices: bool = False) -> pd.DataFrame:
        """
        Optionally exclude studies with support devices.

        Args:
            labels_df: DataFrame with labels
            allow_support_devices: If False, exclude studies with support devices

        Returns:
            Filtered DataFrame
        """
        if allow_support_devices or 'Support Devices' not in labels_df.columns:
            return labels_df

        logger.info("Excluding studies with support devices")
        no_devices_mask = labels_df['Support Devices'] != 1.0
        filtered = labels_df[no_devices_mask].copy()
        logger.info(f"Excluded {len(labels_df) - len(filtered):,} studies with support devices")

        return filtered
