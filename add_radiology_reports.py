#!/usr/bin/env python3
"""
Add radiology reports from CXR-Pro to existing cohort CSV.

This script:
1. Loads existing cohort CSV (e.g., validation_subset_200.csv)
2. Loads CXR-Pro radiology report impressions
3. Joins on study_id to add radiology_report column
4. Saves updated cohort CSV

Usage:
    python add_radiology_reports.py --cohort step2_preprocessing/cohorts/validation_subset_200.csv \
                                     --cxr-pro /home/dev/Documents/Portfolio/MIMIC_Data/physionet.org/files/cxr-pro/1.0.0/mimic_train_impressions.csv \
                                     --output step2_preprocessing/cohorts/validation_subset_200_with_reports.csv
"""

import pandas as pd
import argparse
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_radiology_reports(cxr_pro_path: Path) -> pd.DataFrame:
    """Load and aggregate radiology reports by study_id"""
    logger.info(f"Loading radiology reports from: {cxr_pro_path}")

    # Load CXR-Pro impressions
    reports_df = pd.read_csv(cxr_pro_path)
    logger.info(f"  Loaded {len(reports_df):,} image-level reports")
    logger.info(f"  Columns: {list(reports_df.columns)}")

    # CXR-Pro has one row per image (dicom_id), but studies can have multiple images
    # Aggregate by study_id - take first report (usually identical within study)
    reports_agg = reports_df.groupby('study_id').agg({
        'report': 'first',  # Take first report
        'subject_id': 'first',
        'dicom_id': 'count'  # Count images per study
    }).reset_index()

    reports_agg.columns = ['study_id', 'radiology_report', 'subject_id', 'num_dicom_images']

    logger.info(f"  Aggregated to {len(reports_agg):,} study-level reports")
    logger.info(f"  Average images per study: {reports_agg['num_dicom_images'].mean():.2f}")

    return reports_agg


def add_reports_to_cohort(cohort_df: pd.DataFrame, reports_df: pd.DataFrame) -> pd.DataFrame:
    """Join radiology reports with cohort on study_id"""
    logger.info(f"Joining cohort with radiology reports...")
    logger.info(f"  Cohort samples: {len(cohort_df)}")

    # Merge on study_id
    merged = cohort_df.merge(
        reports_df[['study_id', 'radiology_report', 'num_dicom_images']],
        on='study_id',
        how='left'
    )

    # Check join success
    reports_found = merged['radiology_report'].notna().sum()
    reports_missing = merged['radiology_report'].isna().sum()

    logger.info(f"  Reports found: {reports_found}/{len(merged)} ({reports_found/len(merged)*100:.1f}%)")
    logger.info(f"  Reports missing: {reports_missing}/{len(merged)} ({reports_missing/len(merged)*100:.1f}%)")

    if reports_missing > 0:
        logger.warning(f"  Studies without reports:")
        missing_studies = merged[merged['radiology_report'].isna()]['study_id'].tolist()
        logger.warning(f"    {missing_studies[:10]}{'...' if len(missing_studies) > 10 else ''}")

    # Show sample reports
    logger.info(f"\n  Sample radiology reports:")
    for idx, row in merged[merged['radiology_report'].notna()].head(3).iterrows():
        logger.info(f"    Study {row['study_id']}: \"{row['radiology_report'][:80]}...\"")

    return merged


def main():
    parser = argparse.ArgumentParser(description='Add radiology reports to cohort CSV')
    parser.add_argument('--cohort', type=str, required=True,
                        help='Path to input cohort CSV')
    parser.add_argument('--cxr-pro', type=str, required=True,
                        help='Path to CXR-Pro impressions CSV')
    parser.add_argument('--output', type=str, required=True,
                        help='Path to output cohort CSV with reports')
    parser.add_argument('--overwrite', action='store_true',
                        help='Overwrite output file if it exists')

    args = parser.parse_args()

    cohort_path = Path(args.cohort)
    cxr_pro_path = Path(args.cxr_pro)
    output_path = Path(args.output)

    # Validate inputs
    if not cohort_path.exists():
        raise FileNotFoundError(f"Cohort file not found: {cohort_path}")
    if not cxr_pro_path.exists():
        raise FileNotFoundError(f"CXR-Pro file not found: {cxr_pro_path}")
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(f"Output file exists (use --overwrite to replace): {output_path}")

    logger.info("=" * 80)
    logger.info("Adding Radiology Reports to Cohort")
    logger.info("=" * 80)
    logger.info(f"Input cohort: {cohort_path}")
    logger.info(f"CXR-Pro data: {cxr_pro_path}")
    logger.info(f"Output file: {output_path}")
    logger.info("")

    # Load cohort
    logger.info(f"Loading cohort from: {cohort_path}")
    cohort_df = pd.read_csv(cohort_path)
    logger.info(f"  Loaded {len(cohort_df)} samples")
    logger.info(f"  Columns: {list(cohort_df.columns)}")
    logger.info("")

    # Load and aggregate reports
    reports_df = load_radiology_reports(cxr_pro_path)
    logger.info("")

    # Join
    merged_df = add_reports_to_cohort(cohort_df, reports_df)
    logger.info("")

    # Save
    logger.info(f"Saving updated cohort to: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    merged_df.to_csv(output_path, index=False)
    logger.info(f"  Saved {len(merged_df)} samples")
    logger.info(f"  File size: {output_path.stat().st_size / 1024:.1f} KB")
    logger.info("")

    # Summary
    logger.info("=" * 80)
    logger.info("Summary")
    logger.info("=" * 80)
    logger.info(f"✅ Successfully added radiology reports to cohort")
    logger.info(f"   - Input samples: {len(cohort_df)}")
    logger.info(f"   - Reports found: {merged_df['radiology_report'].notna().sum()}")
    logger.info(f"   - Reports missing: {merged_df['radiology_report'].isna().sum()}")
    logger.info(f"   - Output file: {output_path}")
    logger.info("")
    logger.info("Next steps:")
    logger.info("  1. Update multimodal_dataset.py to use 'radiology_report' column")
    logger.info("  2. Update extraction script to copy CXR-Pro data")
    logger.info("  3. Re-run preprocessing with updated cohort")
    logger.info("=" * 80)


if __name__ == '__main__':
    main()
