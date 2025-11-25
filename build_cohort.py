#!/usr/bin/env python3
"""
Build cohorts from MIMIC datasets.

This script generates:
1. Normal cohort (~33k): CXR studies with "No Finding" for unsupervised pretraining
2. Anomalous cohort (~200k): CXR studies with pathological findings

Usage:
    # Build both cohorts with train/val splits
    python build_cohort.py

    # Build only normal cohort
    python build_cohort.py --normal-only

    # Build only anomalous cohort
    python build_cohort.py --anomalous-only

    # Custom output directory
    python build_cohort.py --output-dir ./my_output

Environment Variables:
    MIMIC_CXR_JPG_PATH: Path to MIMIC-CXR-JPG dataset
    MIMIC_IV_PATH: Path to MIMIC-IV dataset
    MIMIC_IV_ED_PATH: Path to MIMIC-IV-ED dataset
    CXR_PRO_PATH: Path to CXR-PRO dataset
    OUTPUT_PATH: Default output directory
"""

import argparse
import logging
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.config import get_settings
from src.cohort import CohortBuilder
from src.utils import setup_logging


def main():
    parser = argparse.ArgumentParser(
        description="Build cohorts from MIMIC datasets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--normal-only",
        action="store_true",
        help="Build only the normal cohort",
    )
    parser.add_argument(
        "--anomalous-only",
        action="store_true",
        help="Build only the anomalous cohort",
    )
    parser.add_argument(
        "--no-ed-match",
        action="store_true",
        help="Don't require ED stay match (more samples, less clinical context)",
    )
    parser.add_argument(
        "--no-disposition-filter",
        action="store_true",
        help="Don't filter by disposition (include admitted patients)",
    )
    parser.add_argument(
        "--no-reports",
        action="store_true",
        help="Don't include radiology reports",
    )
    parser.add_argument(
        "--validation-fraction",
        type=float,
        default=0.15,
        help="Fraction of data for validation (default: 0.15)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Output directory for cohorts",
    )
    parser.add_argument(
        "--log-file",
        type=Path,
        help="Log file path",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Verbose output",
    )

    args = parser.parse_args()

    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    log_file = args.log_file or (args.output_dir / "build_cohort.log" if args.output_dir else None)
    logger = setup_logging(level=log_level, log_file=log_file)

    logger.info("=" * 60)
    logger.info("MIMIC-CXR Cohort Builder")
    logger.info("=" * 60)

    # Get settings
    settings = get_settings()

    # Override output path if specified
    if args.output_dir:
        settings.paths.output = args.output_dir
        settings.paths.cohort_dir.mkdir(parents=True, exist_ok=True)

    # Update validation fraction
    settings.cohort.validation_fraction = args.validation_fraction

    # Validate paths
    missing = settings.paths.validate()
    if missing:
        logger.error("Missing required data paths:")
        for m in missing:
            logger.error(f"  - {m}")
        logger.error("\nSet environment variables or check paths.")
        sys.exit(1)

    logger.info(f"Output directory: {settings.paths.cohort_dir}")

    # Build cohorts
    builder = CohortBuilder(settings)

    try:
        if args.normal_only:
            # Build only normal cohort
            logger.info("Building normal cohort only...")
            normal = builder.build_normal_cohort(
                require_ed_match=not args.no_ed_match,
                filter_dispositions=not args.no_disposition_filter,
                include_reports=not args.no_reports,
            )
            normal_train, normal_val = builder.split_cohort(normal)

            builder.save_cohort(normal, "normal")
            builder.save_cohort(normal_train, "normal", "train")
            builder.save_cohort(normal_val, "normal", "val")

            logger.info(f"Normal cohort: {len(normal):,} total")

        elif args.anomalous_only:
            # Build only anomalous cohort
            logger.info("Building anomalous cohort only...")
            anomalous = builder.build_anomalous_cohort(
                require_ed_match=not args.no_ed_match,
                include_reports=not args.no_reports,
            )
            anomalous_train, anomalous_val = builder.split_cohort(anomalous)

            builder.save_cohort(anomalous, "anomalous")
            builder.save_cohort(anomalous_train, "anomalous", "train")
            builder.save_cohort(anomalous_val, "anomalous", "val")

            logger.info(f"Anomalous cohort: {len(anomalous):,} total")

        else:
            # Build both cohorts
            results = builder.build_and_save_all()
            logger.info("Both cohorts built successfully!")

        logger.info("=" * 60)
        logger.info("COHORT BUILDING COMPLETE")
        logger.info("=" * 60)
        logger.info(f"Output saved to: {settings.paths.cohort_dir}")

    except Exception as e:
        logger.exception(f"Error building cohorts: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
