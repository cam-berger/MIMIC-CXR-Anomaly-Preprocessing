#!/usr/bin/env python3
"""
Main script for identifying and filtering normal cases from MIMIC datasets.

This script implements Step 1 of the MIMIC-CXR Anomaly Detection pipeline:
identifying "normal" cases based on radiology reports and clinical context.
"""
import argparse
import logging
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from config.paths import DataPaths
from config.config import FilterConfig, ProcessingConfig
from data_loaders.cxr_loader import CXRDataLoader
from data_loaders.ed_loader import EDDataLoader
from data_loaders.iv_loader import IVDataLoader
from filters.radiology_filter import RadiologyFilter
from filters.clinical_filter import ClinicalFilter
from mergers.cohort_builder import NormalCohortBuilder
from validators.data_validator import DataValidator
from validators.sample_checker import SampleChecker
from utils.logging_utils import setup_logging
from utils.data_utils import (
    save_cohort, optimize_dtypes, create_summary_statistics,
    export_summary_to_json, create_data_dictionary
)

logger = logging.getLogger(__name__)


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Identify and filter normal cases from MIMIC datasets (Step 1)'
    )

    parser.add_argument(
        '--output-dir',
        type=str,
        default='output',
        help='Output directory for results (default: output)'
    )

    parser.add_argument(
        '--no-hospital-filter',
        action='store_true',
        help='Skip hospital admission outcome filtering (faster but less strict)'
    )

    parser.add_argument(
        '--validation-samples',
        type=int,
        default=100,
        help='Number of samples for manual validation (default: 100)'
    )

    parser.add_argument(
        '--log-level',
        type=str,
        default='INFO',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        help='Logging level (default: INFO)'
    )

    parser.add_argument(
        '--skip-validation',
        action='store_true',
        help='Skip data validation step'
    )

    parser.add_argument(
        '--optimize-memory',
        action='store_true',
        help='Optimize memory usage (may be slower)'
    )

    return parser.parse_args()


def main():
    """Main execution function."""
    # Parse arguments
    args = parse_arguments()

    # Setup paths and configurations
    paths = DataPaths()
    filter_config = FilterConfig()
    processing_config = ProcessingConfig()

    # Setup logging
    output_dir = paths.get_output_dir(args.output_dir)
    log_dir = output_dir / "logs"
    log_level = getattr(logging, args.log_level)
    setup_logging(log_dir=log_dir, log_level=log_level)

    logger.info("=" * 80)
    logger.info("MIMIC Normal Cohort Identification - Step 1")
    logger.info("=" * 80)

    # Validate paths
    logger.info("\nValidating data paths...")
    path_validation = paths.validate_paths()
    for path_name, exists in path_validation.items():
        status = "✓" if exists else "✗"
        logger.info(f"  {status} {path_name}: {exists}")

    if not all(path_validation.values()):
        logger.error("Some required paths are missing. Please check your data locations.")
        return 1

    # Initialize data loaders
    logger.info("\n" + "=" * 80)
    logger.info("Step 1: Loading Data")
    logger.info("=" * 80)

    cxr_loader = CXRDataLoader(paths)
    ed_loader = EDDataLoader(paths)
    iv_loader = IVDataLoader(paths)

    # Load CXR data
    logger.info("\nLoading MIMIC-CXR data...")
    cxr_labels, cxr_metadata = cxr_loader.load_all_cxr_data()

    # Load ED data
    logger.info("\nLoading MIMIC-IV-ED data...")
    ed_stays, ed_diagnosis, ed_triage = ed_loader.load_all_ed_data()

    # Load MIMIC-IV data
    logger.info("\nLoading MIMIC-IV data...")
    patients, admissions, transfers = iv_loader.load_all_iv_data()

    # Initialize filters
    logger.info("\n" + "=" * 80)
    logger.info("Step 2: Filtering and Building Cohort")
    logger.info("=" * 80)

    radiology_filter = RadiologyFilter(filter_config)
    clinical_filter = ClinicalFilter(filter_config)

    # Build cohort
    cohort_builder = NormalCohortBuilder(
        filter_config,
        radiology_filter,
        clinical_filter
    )

    # Decide whether to include hospital filtering
    if args.no_hospital_filter:
        logger.info("\nSkipping hospital admission outcome filtering...")
        admissions = None
        transfers = None

    # Build the normal cohort
    normal_cohort = cohort_builder.build_normal_cohort(
        cxr_labels=cxr_labels,
        cxr_metadata=cxr_metadata,
        ed_stays=ed_stays,
        ed_diagnosis=ed_diagnosis,
        patients=patients,
        admissions=admissions,
        transfers=transfers
    )

    if len(normal_cohort) == 0:
        logger.error("No normal cases found! Check filtering criteria.")
        return 1

    # Optimize memory if requested
    if args.optimize_memory:
        logger.info("\nOptimizing memory usage...")
        normal_cohort = optimize_dtypes(normal_cohort)

    # Split into train/validation
    logger.info("\n" + "=" * 80)
    logger.info("Step 3: Splitting Train/Validation Sets")
    logger.info("=" * 80)

    train_cohort, val_cohort = cohort_builder.split_train_validation(normal_cohort)

    # Save cohorts
    logger.info("\n" + "=" * 80)
    logger.info("Step 4: Saving Results")
    logger.info("=" * 80)

    results_dir = output_dir / "cohorts"
    results_dir.mkdir(parents=True, exist_ok=True)

    # Save full cohort
    save_cohort(normal_cohort, results_dir / "normal_cohort_full.csv", format='csv')
    save_cohort(normal_cohort, results_dir / "normal_cohort_full.parquet", format='parquet')

    # Save train/val splits
    save_cohort(train_cohort, results_dir / "normal_cohort_train.csv", format='csv')
    save_cohort(val_cohort, results_dir / "normal_cohort_validation.csv", format='csv')

    # Data validation
    if not args.skip_validation:
        logger.info("\n" + "=" * 80)
        logger.info("Step 5: Data Validation")
        logger.info("=" * 80)

        validator = DataValidator()
        validation_report_path = output_dir / "reports" / "validation_report.txt"
        validation_report_path.parent.mkdir(parents=True, exist_ok=True)
        validator.generate_validation_report(normal_cohort, validation_report_path)

    # Create sample for manual review
    logger.info("\n" + "=" * 80)
    logger.info("Step 6: Creating Manual Review Sample")
    logger.info("=" * 80)

    sample_checker = SampleChecker(paths)

    # Get random sample
    review_sample = sample_checker.get_random_samples(
        normal_cohort,
        n_samples=args.validation_samples
    )

    # Create review CSV
    review_dir = output_dir / "manual_review"
    review_dir.mkdir(parents=True, exist_ok=True)
    review_csv_path = review_dir / "sample_for_review.csv"
    sample_checker.create_review_csv(review_sample, review_csv_path)

    # Summarize sample characteristics
    sample_checker.summarize_sample_characteristics(review_sample)

    # Find and save edge cases
    edge_cases = sample_checker.find_edge_cases(normal_cohort)
    for case_type, cases_df in edge_cases.items():
        edge_case_path = review_dir / f"edge_cases_{case_type}.csv"
        cases_df.to_csv(edge_case_path, index=False)
        logger.info(f"Saved {len(cases_df)} {case_type} cases to {edge_case_path}")

    # Generate summary statistics
    logger.info("\n" + "=" * 80)
    logger.info("Step 7: Generating Summary Statistics")
    logger.info("=" * 80)

    summary_stats = create_summary_statistics(normal_cohort)
    summary_path = output_dir / "reports" / "summary_statistics.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    export_summary_to_json(summary_stats, summary_path)

    # Create data dictionary
    data_dict_path = output_dir / "reports" / "data_dictionary.csv"
    create_data_dictionary(normal_cohort, data_dict_path)

    # Final summary
    logger.info("\n" + "=" * 80)
    logger.info("PROCESSING COMPLETE")
    logger.info("=" * 80)
    logger.info(f"\nResults saved to: {output_dir}")
    logger.info(f"  - Cohorts: {results_dir}")
    logger.info(f"  - Manual review: {review_dir}")
    logger.info(f"  - Reports: {output_dir / 'reports'}")
    logger.info(f"  - Logs: {log_dir}")

    logger.info(f"\nFinal Cohort Size:")
    logger.info(f"  - Total: {len(normal_cohort):,} cases")
    logger.info(f"  - Training: {len(train_cohort):,} cases")
    logger.info(f"  - Validation: {len(val_cohort):,} cases")

    logger.info("\nNext Steps:")
    logger.info("  1. Review the sample cases in manual_review/sample_for_review.csv")
    logger.info("  2. Check edge cases in manual_review/edge_cases_*.csv")
    logger.info("  3. Review validation report in reports/validation_report.txt")
    logger.info("  4. Proceed to Step 2 (Data Preprocessing) using the generated cohorts")

    logger.info("\n" + "=" * 80)

    return 0


if __name__ == "__main__":
    sys.exit(main())
