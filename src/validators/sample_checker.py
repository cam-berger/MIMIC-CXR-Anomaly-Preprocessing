"""
Sample checking tools for manual verification of normal cases.
"""
import pandas as pd
import numpy as np
import logging
from pathlib import Path
from typing import List, Dict

logger = logging.getLogger(__name__)


class SampleChecker:
    """Tools for sampling and manually checking filtered normal cases."""

    def __init__(self, paths_config):
        self.paths = paths_config

    def get_random_samples(self, cohort_df: pd.DataFrame,
                          n_samples: int = 100,
                          random_seed: int = 42) -> pd.DataFrame:
        """
        Get random sample of cases for manual review.

        Args:
            cohort_df: Cohort DataFrame
            n_samples: Number of samples to draw
            random_seed: Random seed for reproducibility

        Returns:
            Sample DataFrame
        """
        logger.info(f"Sampling {n_samples} cases for manual review...")

        if len(cohort_df) < n_samples:
            logger.warning(f"Cohort has only {len(cohort_df)} cases, returning all")
            return cohort_df

        sample = cohort_df.sample(n=n_samples, random_state=random_seed).copy()

        logger.info(f"Sampled {len(sample)} cases")

        return sample

    def get_stratified_samples(self, cohort_df: pd.DataFrame,
                              n_samples: int = 100,
                              stratify_by: str = 'gender') -> pd.DataFrame:
        """
        Get stratified sample for balanced manual review.

        Args:
            cohort_df: Cohort DataFrame
            n_samples: Total number of samples
            stratify_by: Column to stratify by

        Returns:
            Stratified sample DataFrame
        """
        logger.info(f"Getting stratified sample by {stratify_by}...")

        if stratify_by not in cohort_df.columns:
            logger.warning(f"Column {stratify_by} not found, using random sample")
            return self.get_random_samples(cohort_df, n_samples)

        # Calculate samples per stratum
        value_counts = cohort_df[stratify_by].value_counts()
        proportions = value_counts / len(cohort_df)

        samples = []
        for value, proportion in proportions.items():
            n_stratum = int(n_samples * proportion)
            if n_stratum > 0:
                stratum_df = cohort_df[cohort_df[stratify_by] == value]
                stratum_sample = stratum_df.sample(n=min(n_stratum, len(stratum_df)))
                samples.append(stratum_sample)
                logger.info(f"  {stratify_by}={value}: {len(stratum_sample)} samples")

        sample = pd.concat(samples, ignore_index=True)

        logger.info(f"Total stratified sample: {len(sample)} cases")

        return sample

    def create_review_csv(self, sample_df: pd.DataFrame,
                         output_path: Path) -> str:
        """
        Create a CSV file for manual review with key information.

        Args:
            sample_df: Sample DataFrame
            output_path: Path to save CSV

        Returns:
            Path to created CSV file
        """
        logger.info(f"Creating review CSV at {output_path}...")

        # Select key columns for review
        review_columns = [
            'subject_id', 'study_id', 'No Finding',
            'study_datetime', 'ed_disposition',
            'gender', 'anchor_age', 'time_to_cxr_hours'
        ]

        # Add pathology columns if present
        pathology_labels = [
            'Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema',
            'Enlarged Cardiomediastinum', 'Fracture', 'Lung Lesion',
            'Lung Opacity', 'Pleural Effusion', 'Pleural Other',
            'Pneumonia', 'Pneumothorax', 'Support Devices'
        ]

        for label in pathology_labels:
            if label in sample_df.columns:
                review_columns.append(label)

        # Create review DataFrame
        review_df = sample_df[
            [col for col in review_columns if col in sample_df.columns]
        ].copy()

        # Add review columns
        review_df['manual_review_normal'] = ''  # For reviewer to mark Y/N
        review_df['reviewer_notes'] = ''

        # Save to CSV
        review_df.to_csv(output_path, index=False)

        logger.info(f"Review CSV created with {len(review_df)} cases")
        logger.info(f"Columns: {list(review_df.columns)}")

        return str(output_path)

    def summarize_sample_characteristics(self, sample_df: pd.DataFrame) -> Dict:
        """
        Summarize characteristics of the sample.

        Args:
            sample_df: Sample DataFrame

        Returns:
            Dictionary with summary statistics
        """
        logger.info("\n" + "=" * 60)
        logger.info("Sample Characteristics Summary")
        logger.info("=" * 60)

        summary = {
            'total_cases': len(sample_df),
            'unique_subjects': sample_df['subject_id'].nunique(),
            'unique_studies': sample_df['study_id'].nunique()
        }

        # Gender distribution
        if 'gender' in sample_df.columns:
            summary['gender_distribution'] = sample_df['gender'].value_counts().to_dict()
            logger.info("\nGender Distribution:")
            for gender, count in summary['gender_distribution'].items():
                logger.info(f"  {gender}: {count} ({count/len(sample_df)*100:.1f}%)")

        # Age statistics
        if 'anchor_age' in sample_df.columns:
            summary['age_stats'] = {
                'mean': float(sample_df['anchor_age'].mean()),
                'median': float(sample_df['anchor_age'].median()),
                'min': int(sample_df['anchor_age'].min()),
                'max': int(sample_df['anchor_age'].max()),
                'std': float(sample_df['anchor_age'].std())
            }
            logger.info("\nAge Statistics:")
            logger.info(f"  Mean: {summary['age_stats']['mean']:.1f} years")
            logger.info(f"  Median: {summary['age_stats']['median']:.1f} years")
            logger.info(f"  Range: {summary['age_stats']['min']}-{summary['age_stats']['max']} years")
            logger.info(f"  Std Dev: {summary['age_stats']['std']:.1f} years")

        # Disposition distribution
        if 'ed_disposition' in sample_df.columns:
            summary['disposition_distribution'] = sample_df['ed_disposition'].value_counts().to_dict()
            logger.info("\nED Disposition Distribution:")
            for disp, count in summary['disposition_distribution'].items():
                logger.info(f"  {disp}: {count} ({count/len(sample_df)*100:.1f}%)")

        logger.info("=" * 60)

        return summary

    def find_edge_cases(self, cohort_df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """
        Identify edge cases that might warrant manual review.

        Args:
            cohort_df: Cohort DataFrame

        Returns:
            Dictionary of edge case DataFrames
        """
        logger.info("\n" + "=" * 60)
        logger.info("Identifying Edge Cases")
        logger.info("=" * 60)

        edge_cases = {}

        # Cases with very long time from ED admission to CXR
        if 'time_to_cxr_hours' in cohort_df.columns:
            long_wait = cohort_df[cohort_df['time_to_cxr_hours'] > 24].copy()
            if len(long_wait) > 0:
                edge_cases['long_wait_to_cxr'] = long_wait
                logger.info(f"Long wait to CXR (>24h): {len(long_wait)} cases")

        # Very young or very old patients
        if 'anchor_age' in cohort_df.columns:
            very_young = cohort_df[cohort_df['anchor_age'] < 25].copy()
            very_old = cohort_df[cohort_df['anchor_age'] > 85].copy()

            if len(very_young) > 0:
                edge_cases['very_young'] = very_young
                logger.info(f"Very young (<25): {len(very_young)} cases")

            if len(very_old) > 0:
                edge_cases['very_old'] = very_old
                logger.info(f"Very old (>85): {len(very_old)} cases")

        # Cases with uncertain labels (if any -1.0 values remain)
        pathology_labels = [
            'Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema',
            'Enlarged Cardiomediastinum', 'Fracture', 'Lung Lesion',
            'Lung Opacity', 'Pleural Effusion', 'Pleural Other',
            'Pneumonia', 'Pneumothorax', 'Support Devices'
        ]

        uncertain_mask = pd.Series([False] * len(cohort_df), index=cohort_df.index)
        for label in pathology_labels:
            if label in cohort_df.columns:
                uncertain_mask |= (cohort_df[label] == -1.0)

        uncertain_cases = cohort_df[uncertain_mask].copy()
        if len(uncertain_cases) > 0:
            edge_cases['uncertain_labels'] = uncertain_cases
            logger.info(f"Cases with uncertain labels: {len(uncertain_cases)}")

        logger.info("=" * 60)

        return edge_cases
