#!/usr/bin/env python3
"""
Main script for building pre-compiled aggregate multimodal dataset.

This script orchestrates the extraction and storage of all MIMIC data sources
into an efficient pre-compiled format optimized for both:
- Fast training iteration (memory-mapped HDF5 images)
- Data exploration/analysis (columnar Parquet structured/text)

Usage:
    # Single batch mode (local testing)
    python run_precompilation.py --split val --batch-size 0

    # Multi-batch mode (Lambda deployment)
    python run_precompilation.py --split train --batch-size 1000

    # Resume from checkpoint
    python run_precompilation.py --split train --resume
"""
import sys
from pathlib import Path
import argparse
import logging
import yaml
import os

# Add step2_preprocessing/src to path
sys.path.insert(0, str(Path(__file__).parent / "step2_preprocessing" / "src"))

from builders.aggregate_builder import AggregateDatasetBuilder

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('precompilation.log')
    ]
)
logger = logging.getLogger(__name__)


def load_config(config_path: Path) -> dict:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def get_anthropic_api_key() -> str:
    """Get Anthropic API key from environment"""
    api_key = os.getenv('ANTHROPIC_API_KEY')
    if not api_key:
        logger.warning("ANTHROPIC_API_KEY not found in environment")
        logger.warning("Text summarization will be disabled")
    return api_key


def validate_config(config: dict, args):
    """Validate configuration before building"""
    logger.info("Validating configuration...")

    # Check required paths
    required_paths = [
        ('mimic_cxr_base', config['data']['mimic_cxr_base']),
        ('mimic_iv_base', config['data']['mimic_iv_base']),
        ('mimic_ed_base', config['data']['mimic_ed_base']),
        ('cxr_pro_base', config['data']['cxr_pro_base'])
    ]

    all_exist = True
    for name, path in required_paths:
        path_obj = Path(path)
        if path_obj.exists():
            logger.info(f"  ✓ {name}: {path}")
        else:
            logger.error(f"  ✗ {name}: {path} (NOT FOUND)")
            all_exist = False

    if not all_exist:
        raise FileNotFoundError("Required data paths not found")

    # Check cohort file
    if args.split == 'train':
        cohort_key = 'step1_cohort_train'
    else:
        cohort_key = 'step1_cohort_val'

    cohort_path = Path(__file__).parent / config['data'][cohort_key]
    if not cohort_path.exists():
        raise FileNotFoundError(f"Cohort file not found: {cohort_path}")

    logger.info(f"  ✓ Cohort file: {cohort_path}")

    # Update config with batch size from args
    if args.batch_size is not None:
        config['precompilation']['batch_size'] = args.batch_size if args.batch_size > 0 else None
        logger.info(f"  Batch size set to: {config['precompilation']['batch_size'] or 'single batch'}")

    logger.info("Configuration validated successfully!")

    return cohort_path


def print_build_summary(stats: dict):
    """Print human-readable build summary"""
    print("\n" + "=" * 80)
    print("BUILD SUMMARY")
    print("=" * 80)
    print(f"Split: {stats['split']}")
    print(f"Total samples in cohort: {stats['total_samples']}")
    print(f"Successfully processed: {stats['processed_samples']}")
    print(f"Failed: {stats['failed_samples']}")
    print(f"Success rate: {stats['success_rate']:.1f}%")
    print(f"Number of batches: {stats['num_batches']}")
    print()

    manifest = stats['manifest']
    print("Batch Details:")
    print("-" * 80)
    print(f"{'Batch ID':<10} {'Samples':<10} {'HDF5 (MB)':<15} {'Parquet (MB)':<15}")
    print("-" * 80)

    total_hdf5_size = 0
    total_parquet_size = 0

    for batch in manifest['batches']:
        batch_id = batch['batch_id']
        num_samples = batch.get('num_samples', 0)
        hdf5_size = batch.get('hdf5_size_mb', 0)
        parquet_size = batch.get('parquet_size_mb', 0)

        print(f"{batch_id:<10} {num_samples:<10} {hdf5_size:<15.2f} {parquet_size:<15.2f}")

        total_hdf5_size += hdf5_size
        total_parquet_size += parquet_size

    print("-" * 80)
    print(f"{'TOTAL':<10} {manifest['total_samples']:<10} {total_hdf5_size:<15.2f} {total_parquet_size:<15.2f}")
    print()
    print(f"Total dataset size: {(total_hdf5_size + total_parquet_size):.2f} MB")
    print(f"Average per sample: {((total_hdf5_size + total_parquet_size) / manifest['total_samples']):.2f} MB")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description='Build pre-compiled aggregate multimodal dataset')

    parser.add_argument(
        '--config',
        type=str,
        default='step2_preprocessing/config/config.yaml',
        help='Path to configuration file'
    )

    parser.add_argument(
        '--split',
        type=str,
        choices=['train', 'val'],
        required=True,
        help='Dataset split to build'
    )

    parser.add_argument(
        '--batch-size',
        type=int,
        default=None,
        help='Batch size (0 or None = single batch for all samples, >0 = multi-batch)'
    )

    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Output directory (overrides config)'
    )

    parser.add_argument(
        '--resume',
        action='store_true',
        help='Resume from checkpoint if available'
    )

    parser.add_argument(
        '--skip-text',
        action='store_true',
        help='Skip text processing (for testing without API key)'
    )

    args = parser.parse_args()

    try:
        # Print header
        logger.info("=" * 80)
        logger.info("Pre-compiled Aggregate Dataset Builder")
        logger.info("=" * 80)
        logger.info(f"Split: {args.split}")
        logger.info(f"Batch size: {args.batch_size if args.batch_size else 'single batch'}")
        logger.info(f"Resume: {args.resume}")
        logger.info("")

        # Load configuration
        config_path = Path(args.config)
        logger.info(f"Loading configuration from: {config_path}")
        config = load_config(config_path)

        # Validate configuration
        cohort_path = validate_config(config, args)

        # Get API key (optional)
        api_key = None if args.skip_text else get_anthropic_api_key()

        # Override output directory if specified
        if args.output_dir:
            output_dir = Path(args.output_dir)
        else:
            output_dir = Path(config['precompilation']['output_dir'])

        logger.info(f"\nOutput directory: {output_dir}")

        # Initialize builder
        logger.info("\nInitializing aggregate dataset builder...")
        builder = AggregateDatasetBuilder(
            config=config,
            output_dir=output_dir,
            anthropic_api_key=api_key
        )

        # Build dataset
        logger.info("\nStarting dataset build...")
        stats = builder.build(
            cohort_path=cohort_path,
            split=args.split,
            resume_from_checkpoint=args.resume
        )

        # Print summary
        print_build_summary(stats)

        # Save summary to file
        summary_path = output_dir / args.split / "build_summary.txt"
        summary_path.parent.mkdir(parents=True, exist_ok=True)

        with open(summary_path, 'w') as f:
            f.write(f"Build Summary - {args.split}\n")
            f.write("=" * 80 + "\n")
            f.write(f"Total samples: {stats['total_samples']}\n")
            f.write(f"Processed: {stats['processed_samples']}\n")
            f.write(f"Failed: {stats['failed_samples']}\n")
            f.write(f"Success rate: {stats['success_rate']:.1f}%\n")
            f.write(f"Number of batches: {stats['num_batches']}\n")

        logger.info(f"\nBuild summary saved to: {summary_path}")

        logger.info("\n✓ Pre-compilation complete!")

        return 0

    except Exception as e:
        logger.error(f"\n✗ Build failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
