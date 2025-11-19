"""
Data validation and quality checking tools.
"""
import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Tuple

logger = logging.getLogger(__name__)


class DataValidator:
    """Validate data quality and consistency."""

    def __init__(self):
        pass

    def validate_cohort(self, cohort_df: pd.DataFrame) -> Dict[str, any]:
        """
        Perform comprehensive validation on the cohort.

        Args:
            cohort_df: Cohort DataFrame to validate

        Returns:
            Dictionary with validation results
        """
        logger.info("=" * 60)
        logger.info("Data Validation Report")
        logger.info("=" * 60)

        results = {}

        # Check for required columns
        results['required_columns'] = self._check_required_columns(cohort_df)

        # Check for missing values
        results['missing_values'] = self._check_missing_values(cohort_df)

        # Check for duplicates
        results['duplicates'] = self._check_duplicates(cohort_df)

        # Check data types
        results['data_types'] = self._check_data_types(cohort_df)

        # Check value ranges
        results['value_ranges'] = self._check_value_ranges(cohort_df)

        # Check temporal consistency
        results['temporal_consistency'] = self._check_temporal_consistency(cohort_df)

        # Overall validation status
        results['is_valid'] = all([
            results['required_columns']['status'] == 'PASS',
            results['duplicates']['status'] == 'PASS',
            results['data_types']['status'] == 'PASS'
        ])

        logger.info("\n" + "=" * 60)
        logger.info(f"Overall Validation: {'PASS' if results['is_valid'] else 'FAIL'}")
        logger.info("=" * 60)

        return results

    def _check_required_columns(self, df: pd.DataFrame) -> Dict:
        """Check if all required columns are present."""
        required_columns = [
            'subject_id', 'study_id', 'No Finding'
        ]

        missing_columns = [col for col in required_columns if col not in df.columns]

        status = 'PASS' if len(missing_columns) == 0 else 'FAIL'
        logger.info(f"\nRequired Columns: {status}")

        if missing_columns:
            logger.warning(f"  Missing columns: {missing_columns}")

        return {
            'status': status,
            'missing_columns': missing_columns
        }

    def _check_missing_values(self, df: pd.DataFrame) -> Dict:
        """Check for missing values in critical columns."""
        critical_columns = ['subject_id', 'study_id']

        missing_counts = {}
        for col in df.columns:
            n_missing = df[col].isna().sum()
            if n_missing > 0:
                missing_counts[col] = {
                    'count': int(n_missing),
                    'percentage': float(n_missing / len(df) * 100)
                }

        logger.info(f"\nMissing Values:")
        if missing_counts:
            for col, info in missing_counts.items():
                is_critical = col in critical_columns
                severity = "CRITICAL" if is_critical else "WARNING"
                logger.info(f"  [{severity}] {col}: {info['count']:,} ({info['percentage']:.2f}%)")
        else:
            logger.info("  No missing values found")

        return {
            'missing_counts': missing_counts,
            'status': 'FAIL' if any(col in critical_columns for col in missing_counts) else 'PASS'
        }

    def _check_duplicates(self, df: pd.DataFrame) -> Dict:
        """Check for duplicate records."""
        # Check for duplicate studies
        duplicate_studies = df.duplicated(subset=['study_id'], keep=False).sum()

        # Check for completely duplicate rows
        duplicate_rows = df.duplicated(keep=False).sum()

        status = 'PASS' if duplicate_studies == 0 else 'WARNING'

        logger.info(f"\nDuplicates: {status}")
        logger.info(f"  Duplicate study_ids: {duplicate_studies:,}")
        logger.info(f"  Duplicate rows: {duplicate_rows:,}")

        return {
            'status': status,
            'duplicate_studies': int(duplicate_studies),
            'duplicate_rows': int(duplicate_rows)
        }

    def _check_data_types(self, df: pd.DataFrame) -> Dict:
        """Check if data types are appropriate."""
        expected_types = {
            'subject_id': ['int32', 'int64'],
            'study_id': ['int32', 'int64'],
            'No Finding': ['float32', 'float64']
        }

        type_mismatches = []

        for col, expected in expected_types.items():
            if col in df.columns:
                actual = str(df[col].dtype)
                if actual not in expected:
                    type_mismatches.append({
                        'column': col,
                        'expected': expected,
                        'actual': actual
                    })

        status = 'PASS' if len(type_mismatches) == 0 else 'WARNING'

        logger.info(f"\nData Types: {status}")
        if type_mismatches:
            for mismatch in type_mismatches:
                logger.info(f"  {mismatch['column']}: expected {mismatch['expected']}, got {mismatch['actual']}")

        return {
            'status': status,
            'mismatches': type_mismatches
        }

    def _check_value_ranges(self, df: pd.DataFrame) -> Dict:
        """Check if values are within expected ranges."""
        issues = []

        # Check 'No Finding' is 1.0
        if 'No Finding' in df.columns:
            non_one_values = df[df['No Finding'] != 1.0]
            if len(non_one_values) > 0:
                issues.append(f"'No Finding' has {len(non_one_values):,} non-1.0 values")

        # Check age if present
        if 'anchor_age' in df.columns:
            min_age = df['anchor_age'].min()
            max_age = df['anchor_age'].max()
            logger.info(f"\nValue Ranges:")
            logger.info(f"  Age range: {min_age}-{max_age} years")
            if min_age < 0 or max_age > 120:
                issues.append(f"Suspicious age values: {min_age}-{max_age}")

        status = 'PASS' if len(issues) == 0 else 'WARNING'

        if issues:
            logger.info(f"\nValue Range Issues:")
            for issue in issues:
                logger.info(f"  {issue}")

        return {
            'status': status,
            'issues': issues
        }

    def _check_temporal_consistency(self, df: pd.DataFrame) -> Dict:
        """Check temporal consistency between dates."""
        issues = []

        if 'study_datetime' in df.columns and 'ed_intime' in df.columns:
            # CXR should not be way before ED admission
            df_temp = df.dropna(subset=['study_datetime', 'ed_intime'])
            time_diff = (df_temp['study_datetime'] - df_temp['ed_intime']).dt.total_seconds() / 3600

            early_cxr = (time_diff < -48).sum()  # More than 48 hours before ED
            late_cxr = (time_diff > 72).sum()     # More than 72 hours after ED admission

            if early_cxr > 0:
                issues.append(f"{early_cxr:,} CXRs more than 48h before ED admission")
            if late_cxr > 0:
                issues.append(f"{late_cxr:,} CXRs more than 72h after ED admission")

        status = 'PASS' if len(issues) == 0 else 'WARNING'

        if issues:
            logger.info(f"\nTemporal Consistency Issues:")
            for issue in issues:
                logger.info(f"  {issue}")

        return {
            'status': status,
            'issues': issues
        }

    def generate_validation_report(self, cohort_df: pd.DataFrame,
                                   output_path: str = None) -> str:
        """
        Generate a detailed validation report.

        Args:
            cohort_df: Cohort DataFrame
            output_path: Optional path to save report

        Returns:
            Report as string
        """
        results = self.validate_cohort(cohort_df)

        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append("COHORT VALIDATION REPORT")
        report_lines.append("=" * 80)
        report_lines.append(f"\nDataset Size: {len(cohort_df):,} records")
        report_lines.append(f"Unique Subjects: {cohort_df['subject_id'].nunique():,}")
        report_lines.append(f"Unique Studies: {cohort_df['study_id'].nunique():,}")

        report_lines.append(f"\n\nOVERALL STATUS: {'PASS' if results['is_valid'] else 'FAIL'}")

        report_lines.append("\n\nDETAILED RESULTS:")
        for check_name, check_results in results.items():
            if check_name != 'is_valid':
                report_lines.append(f"\n{check_name.upper()}: {check_results.get('status', 'N/A')}")

        report = "\n".join(report_lines)

        if output_path:
            with open(output_path, 'w') as f:
                f.write(report)
            logger.info(f"\nValidation report saved to: {output_path}")

        return report
