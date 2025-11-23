"""
Data loader for CXR-PRO dataset.

CXR-PRO is an adaptation of MIMIC-CXR that omits references to prior radiology reports.
Contains 374,139 free-text radiology reports with associated chest radiographs.

Dataset structure:
- mimic_train_impressions.csv: 371,951 training reports
- mimic_test_impressions.csv: 2,188 expert-edited test reports
- cxr.h5: Chest radiographs in HDF5 format (optional, we use MIMIC-CXR-JPG instead)

Reference: https://physionet.org/content/cxr-pro/1.0.0/
"""
import pandas as pd
import logging
from pathlib import Path
from typing import Dict, Optional, Tuple

logger = logging.getLogger(__name__)


class CXRProLoader:
    """
    Loader for CXR-PRO radiology report dataset.

    CXR-PRO provides radiology report impressions with references to priors removed,
    addressing hallucination issues in report generation models.
    """

    def __init__(self, cxr_pro_base_path: Path):
        """
        Initialize CXR-PRO loader.

        Args:
            cxr_pro_base_path: Path to CXR-PRO dataset directory
                              (e.g., /path/to/physionet.org/files/cxr-pro/1.0.0/)
        """
        self.base_path = Path(cxr_pro_base_path)
        self.train_impressions_path = self.base_path / "mimic_train_impressions.csv"
        self.test_impressions_path = self.base_path / "mimic_test_impressions.csv"
        self.cxr_h5_path = self.base_path / "cxr.h5"

        logger.info(f"Initialized CXR-PRO loader from: {self.base_path}")

    def validate_paths(self) -> Dict[str, bool]:
        """
        Validate that CXR-PRO dataset files exist.

        Returns:
            Dictionary mapping file names to existence status
        """
        paths_to_check = {
            'train_impressions': self.train_impressions_path,
            'test_impressions': self.test_impressions_path,
            'cxr_h5': self.cxr_h5_path
        }

        results = {}
        for name, path in paths_to_check.items():
            exists = path.exists()
            results[name] = exists
            if not exists and name != 'cxr_h5':  # h5 is optional
                logger.warning(f"  Missing: {name} at {path}")

        return results

    def load_train_impressions(self) -> pd.DataFrame:
        """
        Load training set radiology report impressions.

        Returns:
            DataFrame with columns:
                - dicom_id: DICOM image identifier
                - study_id: Study identifier (groups multiple images)
                - subject_id: Patient identifier
                - report: Radiology report impression text (prior references removed)
        """
        logger.info(f"Loading training impressions from: {self.train_impressions_path}")

        if not self.train_impressions_path.exists():
            raise FileNotFoundError(f"Training impressions not found: {self.train_impressions_path}")

        try:
            df = pd.read_csv(
                self.train_impressions_path,
                dtype={
                    'subject_id': 'int32',
                    'study_id': 'int32',
                    'dicom_id': 'str',
                    'report': 'str'
                }
            )

            logger.info(f"  Loaded {len(df):,} training reports")
            logger.info(f"  Columns: {list(df.columns)}")
            logger.info(f"  Unique studies: {df['study_id'].nunique():,}")
            logger.info(f"  Unique subjects: {df['subject_id'].nunique():,}")

            # Show sample report
            if len(df) > 0:
                sample_report = df.iloc[0]['report']
                logger.info(f"  Sample report: \"{sample_report[:100]}...\"")

            return df

        except Exception as e:
            logger.error(f"Error loading training impressions: {e}")
            raise

    def load_test_impressions(self) -> pd.DataFrame:
        """
        Load test set radiology report impressions (expert-edited).

        Returns:
            DataFrame with same structure as load_train_impressions()
        """
        logger.info(f"Loading test impressions from: {self.test_impressions_path}")

        if not self.test_impressions_path.exists():
            raise FileNotFoundError(f"Test impressions not found: {self.test_impressions_path}")

        try:
            df = pd.read_csv(
                self.test_impressions_path,
                dtype={
                    'subject_id': 'int32',
                    'study_id': 'int32',
                    'dicom_id': 'str',
                    'report': 'str'
                }
            )

            logger.info(f"  Loaded {len(df):,} test reports (expert-edited)")
            logger.info(f"  Unique studies: {df['study_id'].nunique():,}")

            return df

        except Exception as e:
            logger.error(f"Error loading test impressions: {e}")
            raise

    def load_all_impressions(self, include_test: bool = True) -> pd.DataFrame:
        """
        Load both training and test impressions combined.

        Args:
            include_test: Whether to include test set (default: True)

        Returns:
            Combined DataFrame of all impressions
        """
        train_df = self.load_train_impressions()

        if include_test:
            try:
                test_df = self.load_test_impressions()
                combined = pd.concat([train_df, test_df], ignore_index=True)
                logger.info(f"Combined train + test: {len(combined):,} total reports")
                return combined
            except FileNotFoundError:
                logger.warning("Test impressions not found, using train only")
                return train_df
        else:
            return train_df

    def aggregate_by_study(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Aggregate image-level reports to study-level reports.

        CXR-PRO has one row per image (dicom_id), but studies can have multiple images.
        This method aggregates to one report per study by taking the first report
        (reports are typically identical within a study).

        Args:
            df: DataFrame from load_train_impressions() or load_test_impressions()

        Returns:
            DataFrame with columns:
                - study_id: Study identifier
                - radiology_report: Report impression text
                - subject_id: Patient identifier
                - num_images: Count of images in this study
        """
        logger.info("Aggregating reports by study_id...")

        # Group by study_id and aggregate
        aggregated = df.groupby('study_id').agg({
            'report': 'first',  # Take first report (typically same within study)
            'subject_id': 'first',
            'dicom_id': 'count'  # Count images per study
        }).reset_index()

        # Rename columns for clarity
        aggregated.columns = ['study_id', 'radiology_report', 'subject_id', 'num_images']

        logger.info(f"  Aggregated {len(df):,} image-level reports → {len(aggregated):,} study-level reports")
        logger.info(f"  Average images per study: {aggregated['num_images'].mean():.2f}")
        logger.info(f"  Images per study distribution:")
        logger.info(f"    Min: {aggregated['num_images'].min()}")
        logger.info(f"    Median: {aggregated['num_images'].median():.0f}")
        logger.info(f"    Max: {aggregated['num_images'].max()}")

        return aggregated

    def join_with_cohort(
        self,
        cohort_df: pd.DataFrame,
        reports_df: Optional[pd.DataFrame] = None,
        include_test: bool = True
    ) -> pd.DataFrame:
        """
        Join radiology reports with an existing cohort DataFrame.

        Args:
            cohort_df: Cohort DataFrame with study_id column
            reports_df: Pre-loaded reports DataFrame (optional, will load if None)
            include_test: Whether to include test set reports (default: True)

        Returns:
            Cohort DataFrame with added radiology_report column
        """
        logger.info("Joining radiology reports with cohort...")
        logger.info(f"  Input cohort samples: {len(cohort_df)}")

        # Load and aggregate reports if not provided
        if reports_df is None:
            all_impressions = self.load_all_impressions(include_test=include_test)
            reports_df = self.aggregate_by_study(all_impressions)

        # Merge on study_id
        merged = cohort_df.merge(
            reports_df[['study_id', 'radiology_report', 'num_images']],
            on='study_id',
            how='left'
        )

        # Report join statistics
        reports_found = merged['radiology_report'].notna().sum()
        reports_missing = merged['radiology_report'].isna().sum()
        coverage = (reports_found / len(merged)) * 100

        logger.info(f"  Reports found: {reports_found}/{len(merged)} ({coverage:.1f}%)")
        logger.info(f"  Reports missing: {reports_missing}/{len(merged)} ({100-coverage:.1f}%)")

        if reports_missing > 0:
            missing_studies = merged[merged['radiology_report'].isna()]['study_id'].head(10).tolist()
            logger.warning(f"  Example studies without reports: {missing_studies}")

        # Show sample reports
        if reports_found > 0:
            logger.info("  Sample reports:")
            for idx, row in merged[merged['radiology_report'].notna()].head(2).iterrows():
                report_preview = row['radiology_report'][:80]
                logger.info(f"    Study {row['study_id']}: \"{report_preview}...\"")

        return merged

    def get_study_report(self, study_id: int, aggregated_df: pd.DataFrame) -> Optional[str]:
        """
        Get radiology report for a specific study.

        Args:
            study_id: Study identifier
            aggregated_df: DataFrame from aggregate_by_study()

        Returns:
            Report text or None if not found
        """
        matches = aggregated_df[aggregated_df['study_id'] == study_id]

        if len(matches) == 0:
            return None

        return matches.iloc[0]['radiology_report']

    def get_coverage_stats(self, cohort_df: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate coverage statistics for a cohort.

        Args:
            cohort_df: Cohort DataFrame with study_id column

        Returns:
            Dictionary with coverage statistics:
                - total_studies: Total studies in cohort
                - studies_with_reports: Studies that have CXR-PRO reports
                - coverage_percentage: Percentage coverage
        """
        all_impressions = self.load_all_impressions(include_test=True)
        aggregated = self.aggregate_by_study(all_impressions)

        total_studies = len(cohort_df)
        cohort_study_ids = set(cohort_df['study_id'])
        cxr_pro_study_ids = set(aggregated['study_id'])

        studies_with_reports = len(cohort_study_ids & cxr_pro_study_ids)
        coverage_percentage = (studies_with_reports / total_studies) * 100 if total_studies > 0 else 0

        return {
            'total_studies': total_studies,
            'studies_with_reports': studies_with_reports,
            'coverage_percentage': coverage_percentage
        }
