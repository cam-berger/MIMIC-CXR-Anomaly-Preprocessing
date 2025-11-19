"""
Clinical context filtering based on ED/hospital outcomes.
"""
import pandas as pd
import numpy as np
import logging
from typing import Set, List
from datetime import timedelta

logger = logging.getLogger(__name__)


class ClinicalFilter:
    """Filter cases based on clinical context and outcomes."""

    def __init__(self, filter_config):
        self.config = filter_config

    def filter_by_disposition(self, ed_stays_df: pd.DataFrame) -> pd.DataFrame:
        """
        Filter ED stays by disposition (exclude admissions, deaths, etc.).

        Args:
            ed_stays_df: DataFrame with ED stay information

        Returns:
            Filtered DataFrame with only acceptable dispositions
        """
        logger.info("Filtering ED stays by disposition...")
        logger.info(f"Starting with {len(ed_stays_df):,} ED stays")

        # Log current disposition distribution
        logger.info("Disposition distribution (before filtering):")
        for disp, count in ed_stays_df['disposition'].value_counts().items():
            logger.info(f"  {disp}: {count:,}")

        # Filter for acceptable dispositions
        acceptable_mask = ed_stays_df['disposition'].isin(self.config.acceptable_dispositions)

        # Explicitly exclude certain dispositions
        excluded_mask = ed_stays_df['disposition'].isin(self.config.excluded_dispositions)

        # Combine filters: must be acceptable AND not excluded
        final_mask = acceptable_mask & ~excluded_mask

        filtered_df = ed_stays_df[final_mask].copy()

        logger.info(f"After filtering: {len(filtered_df):,} ED stays ({len(filtered_df)/len(ed_stays_df)*100:.2f}%)")
        logger.info("Disposition distribution (after filtering):")
        for disp, count in filtered_df['disposition'].value_counts().items():
            logger.info(f"  {disp}: {count:,}")

        return filtered_df

    def filter_by_diagnosis(self, ed_diagnosis_df: pd.DataFrame,
                           stay_ids: Set[int]) -> Set[int]:
        """
        Exclude stays with critical diagnoses.

        Args:
            ed_diagnosis_df: DataFrame with ED diagnoses
            stay_ids: Set of stay IDs to check

        Returns:
            Set of stay IDs without critical diagnoses
        """
        logger.info("Filtering out stays with critical diagnoses...")
        logger.info(f"Checking {len(stay_ids):,} stay IDs")

        # Filter diagnoses for relevant stays
        relevant_diagnoses = ed_diagnosis_df[ed_diagnosis_df['stay_id'].isin(stay_ids)].copy()
        logger.info(f"Found {len(relevant_diagnoses):,} diagnosis records")

        # Identify critical diagnoses
        critical_stays = set()

        for pattern in self.config.critical_diagnosis_patterns:
            # Match diagnosis codes that start with the pattern
            matches = relevant_diagnoses['icd_code'].str.startswith(pattern, na=False)
            critical_stay_ids = relevant_diagnoses.loc[matches, 'stay_id'].unique()

            if len(critical_stay_ids) > 0:
                logger.debug(f"Pattern '{pattern}': {len(critical_stay_ids)} stays")
                critical_stays.update(critical_stay_ids)

        # Remove critical stays
        safe_stays = stay_ids - critical_stays

        logger.info(f"Excluded {len(critical_stays):,} stays with critical diagnoses")
        logger.info(f"Remaining stays: {len(safe_stays):,}")

        # Log some examples of excluded diagnoses
        if critical_stays:
            sample_excluded = relevant_diagnoses[
                relevant_diagnoses['stay_id'].isin(list(critical_stays)[:10])
            ][['stay_id', 'icd_code', 'icd_title']].drop_duplicates()
            logger.debug(f"Sample excluded diagnoses:\n{sample_excluded.to_string()}")

        return safe_stays

    def filter_by_hospital_admission_outcomes(self,
                                             admissions_df: pd.DataFrame,
                                             transfers_df: pd.DataFrame,
                                             hadm_ids: Set[int]) -> Set[int]:
        """
        Exclude hospital admissions with ICU stays or death.

        Args:
            admissions_df: DataFrame with admission information
            transfers_df: DataFrame with transfer information
            hadm_ids: Set of hospital admission IDs to check

        Returns:
            Set of admission IDs with no critical outcomes
        """
        logger.info("Filtering hospital admissions for critical outcomes...")
        logger.info(f"Checking {len(hadm_ids):,} admissions")

        critical_admissions = set()

        # Filter for relevant admissions
        relevant_admissions = admissions_df[admissions_df['hadm_id'].isin(hadm_ids)].copy()

        # Exclude deaths in hospital
        deaths = relevant_admissions[relevant_admissions['hospital_expire_flag'] == 1]
        if len(deaths) > 0:
            logger.info(f"Excluding {len(deaths):,} admissions with in-hospital death")
            critical_admissions.update(deaths['hadm_id'].unique())

        # Exclude ICU admissions
        relevant_transfers = transfers_df[transfers_df['hadm_id'].isin(hadm_ids)].copy()

        # Identify ICU units (careunit contains 'ICU' or specific ICU names)
        icu_keywords = ['ICU', 'CCU', 'SICU', 'MICU', 'CSRU', 'TSICU', 'TRAUMA', 'Intensive Care']
        icu_mask = relevant_transfers['careunit'].str.contains('|'.join(icu_keywords), case=False, na=False)
        icu_admissions = relevant_transfers.loc[icu_mask, 'hadm_id'].unique()

        if len(icu_admissions) > 0:
            logger.info(f"Excluding {len(icu_admissions):,} admissions with ICU stay")
            critical_admissions.update(icu_admissions)

        safe_admissions = hadm_ids - critical_admissions

        logger.info(f"Excluded {len(critical_admissions):,} admissions with critical outcomes")
        logger.info(f"Remaining admissions: {len(safe_admissions):,}")

        return safe_admissions

    def filter_by_age(self, patients_df: pd.DataFrame) -> pd.DataFrame:
        """
        Filter patients by minimum age.

        Args:
            patients_df: DataFrame with patient information

        Returns:
            Filtered DataFrame
        """
        logger.info(f"Filtering patients by minimum age: {self.config.min_age}")
        logger.info(f"Starting with {len(patients_df):,} patients")

        filtered_df = patients_df[patients_df['anchor_age'] >= self.config.min_age].copy()

        logger.info(f"After filtering: {len(filtered_df):,} patients ({len(filtered_df)/len(patients_df)*100:.2f}%)")

        return filtered_df

    def check_temporal_consistency(self, cxr_datetime: pd.Timestamp,
                                  ed_intime: pd.Timestamp,
                                  ed_outtime: pd.Timestamp) -> bool:
        """
        Check if CXR was performed during or near the ED visit.

        Args:
            cxr_datetime: Datetime of CXR study
            ed_intime: ED admission time
            ed_outtime: ED discharge time

        Returns:
            True if temporally consistent
        """
        if pd.isna(cxr_datetime) or pd.isna(ed_intime) or pd.isna(ed_outtime):
            return False

        # CXR should be within the ED stay or within time window before/after
        time_window = timedelta(hours=self.config.time_window_hours)

        in_ed_stay = ed_intime <= cxr_datetime <= ed_outtime
        near_ed_visit = (ed_intime - time_window) <= cxr_datetime <= (ed_outtime + time_window)

        return in_ed_stay or near_ed_visit

    def get_filtering_summary(self, original_count: int, final_count: int,
                             step_counts: dict = None) -> str:
        """
        Generate a summary of filtering steps.

        Args:
            original_count: Original number of cases
            final_count: Final number of cases
            step_counts: Dictionary of intermediate counts

        Returns:
            Formatted summary string
        """
        summary = []
        summary.append("=" * 60)
        summary.append("Clinical Filtering Summary")
        summary.append("=" * 60)
        summary.append(f"Original cases: {original_count:,}")

        if step_counts:
            for step_name, count in step_counts.items():
                retention = count / original_count * 100 if original_count > 0 else 0
                summary.append(f"{step_name}: {count:,} ({retention:.2f}%)")

        summary.append(f"Final cases: {final_count:,}")
        retention = final_count / original_count * 100 if original_count > 0 else 0
        summary.append(f"Overall retention: {retention:.2f}%")
        summary.append("=" * 60)

        return "\n".join(summary)
