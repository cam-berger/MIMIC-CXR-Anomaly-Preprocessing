"""
Cohort builder for identifying and merging normal cases across datasets.
"""
import pandas as pd
import numpy as np
import logging
from typing import Tuple, Set
from datetime import timedelta

logger = logging.getLogger(__name__)


class NormalCohortBuilder:
    """Build cohort of normal cases by merging CXR, ED, and hospital data."""

    def __init__(self, filter_config, radiology_filter, clinical_filter):
        self.config = filter_config
        self.radiology_filter = radiology_filter
        self.clinical_filter = clinical_filter

    def build_normal_cohort(self,
                           cxr_labels: pd.DataFrame,
                           cxr_metadata: pd.DataFrame,
                           ed_stays: pd.DataFrame,
                           ed_diagnosis: pd.DataFrame,
                           patients: pd.DataFrame,
                           admissions: pd.DataFrame = None,
                           transfers: pd.DataFrame = None) -> pd.DataFrame:
        """
        Build a cohort of normal cases by applying all filters and merging datasets.

        Args:
            cxr_labels: CheXpert labels DataFrame
            cxr_metadata: CXR metadata DataFrame
            ed_stays: ED stays DataFrame
            ed_diagnosis: ED diagnosis DataFrame
            patients: Patient demographics DataFrame
            admissions: Hospital admissions DataFrame (optional)
            transfers: Transfer events DataFrame (optional)

        Returns:
            DataFrame with normal cases
        """
        logger.info("=" * 80)
        logger.info("Building Normal Cohort")
        logger.info("=" * 80)

        step_counts = {}
        original_count = len(cxr_labels)

        # Step 1: Filter radiology reports for normal findings
        logger.info("\nStep 1: Filtering radiology reports...")
        normal_cxr = self.radiology_filter.filter_normal_studies(cxr_labels)
        step_counts['After radiology filter'] = len(normal_cxr)

        # Step 2: Merge with metadata to get study datetime
        logger.info("\nStep 2: Merging with CXR metadata...")
        normal_cxr_with_metadata = self._merge_cxr_with_metadata(normal_cxr, cxr_metadata)
        step_counts['After metadata merge'] = len(normal_cxr_with_metadata)

        # Step 3: Filter patients by age
        logger.info("\nStep 3: Filtering patients by age...")
        eligible_patients = self.clinical_filter.filter_by_age(patients)
        normal_cxr_with_metadata = normal_cxr_with_metadata[
            normal_cxr_with_metadata['subject_id'].isin(eligible_patients['subject_id'])
        ].copy()
        step_counts['After age filter'] = len(normal_cxr_with_metadata)

        # Step 4: Filter ED stays by disposition
        logger.info("\nStep 4: Filtering ED stays by disposition...")
        normal_ed_stays = self.clinical_filter.filter_by_disposition(ed_stays)
        step_counts['Normal ED stays'] = len(normal_ed_stays)

        # Step 5: Exclude ED stays with critical diagnoses
        logger.info("\nStep 5: Excluding ED stays with critical diagnoses...")
        safe_stay_ids = self.clinical_filter.filter_by_diagnosis(
            ed_diagnosis,
            set(normal_ed_stays['stay_id'].unique())
        )
        normal_ed_stays = normal_ed_stays[normal_ed_stays['stay_id'].isin(safe_stay_ids)].copy()
        step_counts['After diagnosis filter'] = len(normal_ed_stays)

        # Step 6: Filter hospital admissions if provided
        if admissions is not None and transfers is not None:
            logger.info("\nStep 6: Filtering hospital admissions...")
            normal_ed_stays = self._filter_hospital_outcomes(
                normal_ed_stays, admissions, transfers
            )
            step_counts['After hospital outcome filter'] = len(normal_ed_stays)

        # Step 7: Match CXR studies with ED stays
        logger.info("\nStep 7: Matching CXR studies with ED stays...")
        matched_cohort = self._match_cxr_with_ed_stays(
            normal_cxr_with_metadata,
            normal_ed_stays
        )
        step_counts['After temporal matching'] = len(matched_cohort)

        # Step 8: Add patient demographics
        logger.info("\nStep 8: Adding patient demographics...")
        final_cohort = self._add_patient_demographics(matched_cohort, patients)

        # Log final summary
        summary = self.clinical_filter.get_filtering_summary(
            original_count, len(final_cohort), step_counts
        )
        logger.info(f"\n{summary}")

        # Log cohort characteristics
        self._log_cohort_characteristics(final_cohort)

        return final_cohort

    def _merge_cxr_with_metadata(self, labels_df: pd.DataFrame,
                                 metadata_df: pd.DataFrame) -> pd.DataFrame:
        """Merge CXR labels with metadata to get study datetime."""
        # Group metadata by study to get study-level information
        # (metadata is per-image, but we want per-study)
        grouped = metadata_df.groupby(['subject_id', 'study_id'])

        study_metadata = grouped.agg({
            'study_datetime': 'first',  # Use first image datetime
            'dicom_id': 'count'  # Number of images
        }).reset_index()

        # Get view positions separately (handle NaN values)
        view_positions = grouped['ViewPosition'].apply(
            lambda x: ', '.join(x.dropna().unique().astype(str))
        ).reset_index()

        # Merge them together
        study_metadata = study_metadata.merge(view_positions, on=['subject_id', 'study_id'])
        study_metadata.rename(columns={'dicom_id': 'num_images'}, inplace=True)

        merged = labels_df.merge(
            study_metadata,
            on=['subject_id', 'study_id'],
            how='left'
        )

        logger.info(f"Merged {len(merged):,} studies with metadata")
        logger.info(f"Studies with datetime: {merged['study_datetime'].notna().sum():,}")

        return merged

    def _filter_hospital_outcomes(self, ed_stays_df: pd.DataFrame,
                                  admissions_df: pd.DataFrame,
                                  transfers_df: pd.DataFrame) -> pd.DataFrame:
        """Filter ED stays based on subsequent hospital outcomes."""
        # Get hospital admission IDs from ED stays
        ed_with_hadm = ed_stays_df[ed_stays_df['hadm_id'].notna()].copy()

        if len(ed_with_hadm) == 0:
            logger.info("No ED stays have associated hospital admissions")
            return ed_stays_df

        hadm_ids = set(ed_with_hadm['hadm_id'].unique())

        # Filter admissions for critical outcomes
        safe_hadm_ids = self.clinical_filter.filter_by_hospital_admission_outcomes(
            admissions_df, transfers_df, hadm_ids
        )

        # Keep ED stays that either:
        # 1. Have no hospital admission (hadm_id is null), OR
        # 2. Have a safe hospital admission
        mask = ed_stays_df['hadm_id'].isna() | ed_stays_df['hadm_id'].isin(safe_hadm_ids)
        filtered = ed_stays_df[mask].copy()

        logger.info(f"Retained {len(filtered):,} / {len(ed_stays_df):,} ED stays after hospital outcome filter")

        return filtered

    def _match_cxr_with_ed_stays(self, cxr_df: pd.DataFrame,
                                 ed_stays_df: pd.DataFrame) -> pd.DataFrame:
        """
        Match CXR studies with ED stays based on subject_id and temporal proximity.

        A match is valid if:
        1. Same subject_id
        2. CXR datetime is during ED stay or within time window
        """
        logger.info("Matching CXR studies with ED stays...")

        matches = []
        time_window = timedelta(hours=self.config.time_window_hours)

        # Group by subject_id for efficiency
        for subject_id in cxr_df['subject_id'].unique():
            subject_cxrs = cxr_df[cxr_df['subject_id'] == subject_id]
            subject_ed_stays = ed_stays_df[ed_stays_df['subject_id'] == subject_id]

            if len(subject_ed_stays) == 0:
                continue

            for _, cxr_row in subject_cxrs.iterrows():
                cxr_datetime = cxr_row['study_datetime']

                if pd.isna(cxr_datetime):
                    continue

                # Find matching ED stays
                for _, ed_row in subject_ed_stays.iterrows():
                    ed_intime = ed_row['intime']
                    ed_outtime = ed_row['outtime']

                    if self.clinical_filter.check_temporal_consistency(
                        cxr_datetime, ed_intime, ed_outtime
                    ):
                        # Create matched record
                        matched_record = {
                            **cxr_row.to_dict(),
                            'stay_id': ed_row['stay_id'],
                            'hadm_id': ed_row['hadm_id'],
                            'ed_intime': ed_intime,
                            'ed_outtime': ed_outtime,
                            'ed_disposition': ed_row['disposition'],
                            'time_to_cxr_hours': (cxr_datetime - ed_intime).total_seconds() / 3600
                        }
                        matches.append(matched_record)
                        break  # Use first matching ED stay

        if len(matches) == 0:
            logger.warning("No matches found between CXR studies and ED stays!")
            return pd.DataFrame()

        matched_df = pd.DataFrame(matches)
        logger.info(f"Matched {len(matched_df):,} CXR studies with ED stays")
        logger.info(f"Unique subjects: {matched_df['subject_id'].nunique():,}")
        logger.info(f"Unique ED stays: {matched_df['stay_id'].nunique():,}")

        return matched_df

    def _add_patient_demographics(self, cohort_df: pd.DataFrame,
                                  patients_df: pd.DataFrame) -> pd.DataFrame:
        """Add patient demographic information to cohort."""
        merged = cohort_df.merge(
            patients_df[['subject_id', 'gender', 'anchor_age', 'anchor_year']],
            on='subject_id',
            how='left'
        )

        logger.info(f"Added demographics for {merged['gender'].notna().sum():,} cases")

        return merged

    def _log_cohort_characteristics(self, cohort_df: pd.DataFrame):
        """Log characteristics of the final cohort."""
        logger.info("\n" + "=" * 60)
        logger.info("Final Cohort Characteristics")
        logger.info("=" * 60)

        logger.info(f"Total cases: {len(cohort_df):,}")
        logger.info(f"Unique subjects: {cohort_df['subject_id'].nunique():,}")
        logger.info(f"Unique studies: {cohort_df['study_id'].nunique():,}")

        if 'gender' in cohort_df.columns:
            logger.info("\nGender distribution:")
            for gender, count in cohort_df['gender'].value_counts().items():
                logger.info(f"  {gender}: {count:,} ({count/len(cohort_df)*100:.1f}%)")

        if 'anchor_age' in cohort_df.columns:
            logger.info(f"\nAge statistics:")
            logger.info(f"  Mean: {cohort_df['anchor_age'].mean():.1f} years")
            logger.info(f"  Median: {cohort_df['anchor_age'].median():.1f} years")
            logger.info(f"  Range: {cohort_df['anchor_age'].min()}-{cohort_df['anchor_age'].max()} years")

        if 'time_to_cxr_hours' in cohort_df.columns:
            logger.info(f"\nTime from ED admission to CXR:")
            logger.info(f"  Mean: {cohort_df['time_to_cxr_hours'].mean():.2f} hours")
            logger.info(f"  Median: {cohort_df['time_to_cxr_hours'].median():.2f} hours")

        logger.info("=" * 60)

    def split_train_validation(self, cohort_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split cohort into training and validation sets.

        Args:
            cohort_df: Full cohort DataFrame

        Returns:
            Tuple of (train_df, val_df)
        """
        logger.info("\nSplitting cohort into train/validation sets...")

        # Shuffle and split
        shuffled = cohort_df.sample(frac=1, random_state=self.config.random_seed).reset_index(drop=True)
        n_val = int(len(shuffled) * self.config.validation_fraction)

        val_df = shuffled.iloc[:n_val].copy()
        train_df = shuffled.iloc[n_val:].copy()

        logger.info(f"Training set: {len(train_df):,} cases ({len(train_df)/len(cohort_df)*100:.1f}%)")
        logger.info(f"Validation set: {len(val_df):,} cases ({len(val_df)/len(cohort_df)*100:.1f}%)")

        return train_df, val_df
