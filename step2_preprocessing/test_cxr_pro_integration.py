#!/usr/bin/env python3
"""
Test CXR-PRO loader integration with Step 1 cohort.

This script validates that:
1. CXR-PRO loader can successfully load radiology reports
2. Reports can be joined with Step 1 cohort
3. Coverage statistics are calculated correctly
"""
import sys
from pathlib import Path
import pandas as pd
import yaml
import logging

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from data_loaders.cxr_pro_loader import CXRProLoader

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_config():
    """Load configuration from config.yaml"""
    config_path = Path(__file__).parent / "config" / "config.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def test_cxr_pro_loader():
    """Test CXR-PRO loader functionality"""
    logger.info("=" * 80)
    logger.info("Testing CXR-PRO Loader")
    logger.info("=" * 80)

    # Load configuration
    config = load_config()
    cxr_pro_base = Path(config['data']['cxr_pro_base'])

    logger.info(f"\n1. Initializing CXR-PRO Loader")
    logger.info(f"   Base path: {cxr_pro_base}")

    # Initialize loader
    loader = CXRProLoader(cxr_pro_base)

    # Validate paths
    logger.info(f"\n2. Validating CXR-PRO paths")
    path_validation = loader.validate_paths()
    for name, exists in path_validation.items():
        status = "✓" if exists else "✗"
        logger.info(f"   {status} {name}: {exists}")

    if not all([path_validation['train_impressions'], path_validation['test_impressions']]):
        logger.error("   Required CXR-PRO files not found!")
        return False

    # Load impressions
    logger.info(f"\n3. Loading CXR-PRO impressions")
    all_impressions = loader.load_all_impressions(include_test=True)

    # Aggregate by study
    logger.info(f"\n4. Aggregating to study-level reports")
    aggregated = loader.aggregate_by_study(all_impressions)

    # Show sample reports
    logger.info(f"\n5. Sample radiology reports:")
    for idx in range(min(3, len(aggregated))):
        row = aggregated.iloc[idx]
        report_text = row['radiology_report'][:150]
        logger.info(f"   Study {row['study_id']}: \"{report_text}...\"")

    return aggregated


def test_cohort_integration(aggregated_reports):
    """Test integration with Step 1 cohort"""
    logger.info("\n" + "=" * 80)
    logger.info("Testing Cohort Integration")
    logger.info("=" * 80)

    config = load_config()

    # Test with validation cohort
    cohort_val_path = Path(__file__).parent.parent / config['data']['step1_cohort_val']

    if not cohort_val_path.exists():
        logger.warning(f"Validation cohort not found at: {cohort_val_path}")
        logger.info("Trying alternative path...")
        cohort_val_path = Path(__file__).parent / "cohorts" / "validation_subset_200.csv"

    if not cohort_val_path.exists():
        logger.error(f"Could not find validation cohort!")
        return False

    logger.info(f"\n1. Loading validation cohort")
    logger.info(f"   Path: {cohort_val_path}")
    cohort_df = pd.read_csv(cohort_val_path)
    logger.info(f"   Loaded {len(cohort_df)} samples")
    logger.info(f"   Columns: {list(cohort_df.columns)[:10]}...")  # Show first 10 columns

    # Join with reports
    logger.info(f"\n2. Joining radiology reports with cohort")
    loader = CXRProLoader(Path(config['data']['cxr_pro_base']))
    cohort_with_reports = loader.join_with_cohort(cohort_df, reports_df=aggregated_reports)

    # Analyze coverage
    logger.info(f"\n3. Coverage Analysis")
    total = len(cohort_with_reports)
    with_reports = cohort_with_reports['radiology_report'].notna().sum()
    without_reports = cohort_with_reports['radiology_report'].isna().sum()
    coverage_pct = (with_reports / total) * 100

    logger.info(f"   Total samples: {total}")
    logger.info(f"   With reports: {with_reports} ({coverage_pct:.1f}%)")
    logger.info(f"   Without reports: {without_reports} ({100-coverage_pct:.1f}%)")

    # Report length statistics
    if with_reports > 0:
        report_lengths = cohort_with_reports[cohort_with_reports['radiology_report'].notna()]['radiology_report'].str.len()
        logger.info(f"\n4. Report Length Statistics")
        logger.info(f"   Min length: {report_lengths.min()} characters")
        logger.info(f"   Mean length: {report_lengths.mean():.0f} characters")
        logger.info(f"   Median length: {report_lengths.median():.0f} characters")
        logger.info(f"   Max length: {report_lengths.max()} characters")

    # Show examples of studies with and without reports
    if with_reports > 0:
        logger.info(f"\n5. Sample Studies WITH Reports")
        samples = cohort_with_reports[cohort_with_reports['radiology_report'].notna()].head(2)
        for idx, row in samples.iterrows():
            logger.info(f"   Study {row['study_id']}:")
            logger.info(f"      Subject: {row['subject_id']}")
            logger.info(f"      Report: \"{row['radiology_report'][:100]}...\"")

    if without_reports > 0:
        logger.info(f"\n6. Sample Studies WITHOUT Reports")
        missing = cohort_with_reports[cohort_with_reports['radiology_report'].isna()].head(5)
        logger.info(f"   Missing study_ids: {missing['study_id'].tolist()}")

    return cohort_with_reports


def main():
    """Main test function"""
    try:
        # Test CXR-PRO loader
        aggregated_reports = test_cxr_pro_loader()

        if aggregated_reports is None:
            logger.error("CXR-PRO loader test failed!")
            return 1

        # Test cohort integration
        cohort_with_reports = test_cohort_integration(aggregated_reports)

        if cohort_with_reports is None:
            logger.error("Cohort integration test failed!")
            return 1

        logger.info("\n" + "=" * 80)
        logger.info("✓ All tests passed successfully!")
        logger.info("=" * 80)

        return 0

    except Exception as e:
        logger.error(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
