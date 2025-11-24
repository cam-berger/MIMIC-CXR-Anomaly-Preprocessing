#!/usr/bin/env python3
"""
Preprocess multimodal MIMIC-CXR data.

This script processes:
1. Images: CXR images -> HDF5 (chunked, compressed)
2. Structured: Labs, vitals, demographics -> Parquet
3. Text: Radiology reports -> Parquet (with optional Claude summarization)

Usage:
    # Process all cohorts
    python preprocess.py

    # Process specific cohort
    python preprocess.py --cohort output/cohorts/normal_train.parquet

    # Process only images
    python preprocess.py --images-only

    # Enable Claude summarization for text
    python preprocess.py --enable-summarization

    # Use more workers for parallel processing
    python preprocess.py --workers 8

Environment Variables:
    MIMIC_CXR_JPG_PATH: Path to MIMIC-CXR-JPG dataset
    MIMIC_IV_PATH: Path to MIMIC-IV dataset
    MIMIC_IV_ED_PATH: Path to MIMIC-IV-ED dataset
    CXR_PRO_PATH: Path to CXR-PRO dataset
    OUTPUT_PATH: Default output directory
    ANTHROPIC_API_KEY: API key for Claude (required for summarization)
"""

import argparse
import logging
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.config import get_settings
from src.preprocessing import PreprocessingPipeline
from src.utils import setup_logging


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess multimodal MIMIC-CXR data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--cohort",
        type=Path,
        help="Specific cohort file to process (processes all if not specified)",
    )
    parser.add_argument(
        "--images-only",
        action="store_true",
        help="Process only images",
    )
    parser.add_argument(
        "--structured-only",
        action="store_true",
        help="Process only structured data",
    )
    parser.add_argument(
        "--text-only",
        action="store_true",
        help="Process only text data",
    )
    parser.add_argument(
        "--enable-summarization",
        action="store_true",
        help="Enable Claude API for text summarization",
    )
    parser.add_argument(
        "--no-context",
        action="store_true",
        help="Disable clinical context in summarization (use report only)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of parallel workers for image processing (default: 4)",
    )
    parser.add_argument(
        "--cohorts-dir",
        type=Path,
        help="Directory containing cohort files",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Output directory for preprocessed data",
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
    logger = setup_logging(level=log_level, log_file=args.log_file)

    logger.info("=" * 60)
    logger.info("MIMIC-CXR Preprocessing Pipeline")
    logger.info("=" * 60)

    # Get settings
    settings = get_settings()

    # Override paths if specified
    if args.output_dir:
        settings.paths.output = args.output_dir
        settings.paths.preprocessed_dir.mkdir(parents=True, exist_ok=True)

    # Validate paths
    missing = settings.paths.validate()
    if missing:
        logger.error("Missing required data paths:")
        for m in missing:
            logger.error(f"  - {m}")
        logger.error("\nSet environment variables or check paths.")
        sys.exit(1)

    # Check for API key if summarization enabled
    if args.enable_summarization and not settings.anthropic_api_key:
        logger.error("ANTHROPIC_API_KEY environment variable required for summarization")
        sys.exit(1)

    # Determine what to process
    process_images = not (args.structured_only or args.text_only)
    process_structured = not (args.images_only or args.text_only)
    process_text = not (args.images_only or args.structured_only)

    logger.info(f"Processing: images={process_images}, structured={process_structured}, text={process_text}")
    logger.info(f"Workers: {args.workers}")
    logger.info(f"Summarization: {args.enable_summarization}")
    logger.info(f"Include clinical context: {not args.no_context}")

    # Initialize pipeline
    pipeline = PreprocessingPipeline(settings)

    try:
        if args.cohort:
            # Process specific cohort
            if not args.cohort.exists():
                logger.error(f"Cohort file not found: {args.cohort}")
                sys.exit(1)

            output_name = args.cohort.stem
            stats = pipeline.process_cohort(
                cohort_path=args.cohort,
                output_name=output_name,
                process_images=process_images,
                process_structured=process_structured,
                process_text=process_text,
                enable_summarization=args.enable_summarization,
                include_context=not args.no_context,
                num_workers=args.workers,
            )

        else:
            # Process all cohorts
            cohorts_dir = args.cohorts_dir or settings.paths.cohort_dir

            if not cohorts_dir.exists():
                logger.error(f"Cohorts directory not found: {cohorts_dir}")
                logger.error("Run build_cohort.py first to generate cohorts.")
                sys.exit(1)

            stats = pipeline.process_all_cohorts(
                cohorts_dir=cohorts_dir,
                process_images=process_images,
                process_structured=process_structured,
                process_text=process_text,
                enable_summarization=args.enable_summarization,
                include_context=not args.no_context,
                num_workers=args.workers,
            )

        logger.info("=" * 60)
        logger.info("PREPROCESSING COMPLETE")
        logger.info("=" * 60)
        logger.info(f"Output saved to: {settings.paths.preprocessed_dir}")

    except Exception as e:
        logger.exception(f"Error during preprocessing: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
