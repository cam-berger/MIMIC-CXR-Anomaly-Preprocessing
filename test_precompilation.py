#!/usr/bin/env python3
"""
Test script for pre-compilation infrastructure.

Tests:
1. Single-batch mode (small subset)
2. Multi-batch mode (if requested)
3. Dataset loading and iteration
4. Data integrity checks
"""
import sys
from pathlib import Path
import logging
import yaml
import shutil
import argparse

# Setup paths
sys.path.insert(0, str(Path(__file__).parent / "step2_preprocessing" / "src"))

from builders.aggregate_builder import AggregateDatasetBuilder
from integration.precompiled_dataset import PrecompiledMultimodalDataset, precompiled_collate_fn
from torch.utils.data import DataLoader

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_single_batch_build(config: dict, test_output_dir: Path, num_samples: int = 10):
    """
    Test single-batch mode on small subset.

    Args:
        config: Configuration dictionary
        test_output_dir: Output directory for test
        num_samples: Number of samples to process
    """
    logger.info("=" * 80)
    logger.info("TEST 1: Single-batch Build (Small Subset)")
    logger.info("=" * 80)

    # Create subset of validation cohort
    import pandas as pd

    cohort_path = Path("step2_preprocessing/cohorts/validation_subset_200.csv")
    if not cohort_path.exists():
        logger.error(f"Validation cohort not found: {cohort_path}")
        return False

    cohort = pd.read_csv(cohort_path)
    subset = cohort.head(num_samples)

    # Save subset
    subset_path = test_output_dir / "test_cohort_subset.csv"
    subset.to_csv(subset_path, index=False)

    logger.info(f"Created test subset: {num_samples} samples")

    # Build in single-batch mode
    config['precompilation']['batch_size'] = None  # Single batch
    config['precompilation']['enabled'] = True

    builder = AggregateDatasetBuilder(
        config=config,
        output_dir=test_output_dir / "precompiled",
        anthropic_api_key=None  # Skip text processing for speed
    )

    try:
        stats = builder.build(
            cohort_path=subset_path,
            split="test",
            resume_from_checkpoint=False
        )

        logger.info("\nBuild Statistics:")
        logger.info(f"  Total: {stats['total_samples']}")
        logger.info(f"  Processed: {stats['processed_samples']}")
        logger.info(f"  Failed: {stats['failed_samples']}")
        logger.info(f"  Success rate: {stats['success_rate']:.1f}%")

        if stats['success_rate'] < 50:
            logger.error("Success rate too low!")
            return False

        logger.info("\n✓ Single-batch build test PASSED")
        return True

    except Exception as e:
        logger.error(f"\n✗ Single-batch build test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_dataset_loading(test_output_dir: Path):
    """
    Test loading from pre-compiled dataset.

    Args:
        test_output_dir: Test output directory
    """
    logger.info("\n" + "=" * 80)
    logger.info("TEST 2: Dataset Loading")
    logger.info("=" * 80)

    dataset_dir = test_output_dir / "precompiled"

    try:
        # Initialize dataset
        dataset = PrecompiledMultimodalDataset(
            data_dir=dataset_dir,
            split="test",
            load_images=True,
            load_structured=True,
            load_text=False  # Skipped text in build
        )

        logger.info(f"Dataset initialized: {len(dataset)} samples")

        # Test single sample loading
        if len(dataset) > 0:
            sample = dataset[0]

            logger.info("\nSample 0:")
            logger.info(f"  Sample ID: {sample['sample_id']}")
            logger.info(f"  Subject ID: {sample['subject_id']}")
            logger.info(f"  Study ID: {sample['study_id']}")

            if 'image' in sample and sample['image'] is not None:
                logger.info(f"  Image shape: {sample['image'].shape}")
                logger.info(f"  Image dtype: {sample['image'].dtype}")
            else:
                logger.warning("  Image: None")

            if 'structured' in sample:
                logger.info(f"  Structured features: {len(sample['structured'])} features")
                # Show first few features
                for key in list(sample['structured'].keys())[:5]:
                    logger.info(f"    {key}: {sample['structured'][key]}")
            else:
                logger.warning("  Structured: None")

        # Test DataLoader
        logger.info("\nTesting DataLoader...")
        dataloader = DataLoader(
            dataset,
            batch_size=2,
            shuffle=False,
            collate_fn=precompiled_collate_fn,
            num_workers=0
        )

        batch = next(iter(dataloader))

        logger.info(f"  Batch size: {len(batch['sample_ids'])}")
        logger.info(f"  Sample IDs: {batch['sample_ids']}")

        if 'images' in batch:
            logger.info(f"  Images shape: {batch['images'].shape if hasattr(batch['images'], 'shape') else 'variable'}")

        # Close dataset
        dataset.close()

        logger.info("\n✓ Dataset loading test PASSED")
        return True

    except Exception as e:
        logger.error(f"\n✗ Dataset loading test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_multi_batch_build(config: dict, test_output_dir: Path, num_samples: int = 20, batch_size: int = 5):
    """
    Test multi-batch mode.

    Args:
        config: Configuration dictionary
        test_output_dir: Output directory for test
        num_samples: Number of samples to process
        batch_size: Batch size
    """
    logger.info("\n" + "=" * 80)
    logger.info("TEST 3: Multi-batch Build")
    logger.info("=" * 80)

    # Create subset
    import pandas as pd

    cohort_path = Path("step2_preprocessing/cohorts/validation_subset_200.csv")
    cohort = pd.read_csv(cohort_path)
    subset = cohort.head(num_samples)

    subset_path = test_output_dir / "test_cohort_multibatch.csv"
    subset.to_csv(subset_path, index=False)

    logger.info(f"Created test subset: {num_samples} samples")
    logger.info(f"Batch size: {batch_size}")
    logger.info(f"Expected batches: {(num_samples + batch_size - 1) // batch_size}")

    # Build in multi-batch mode
    config['precompilation']['batch_size'] = batch_size

    multibatch_dir = test_output_dir / "precompiled_multibatch"

    builder = AggregateDatasetBuilder(
        config=config,
        output_dir=multibatch_dir,
        anthropic_api_key=None
    )

    try:
        stats = builder.build(
            cohort_path=subset_path,
            split="test",
            resume_from_checkpoint=False
        )

        logger.info("\nBuild Statistics:")
        logger.info(f"  Processed: {stats['processed_samples']}")
        logger.info(f"  Batches: {stats['num_batches']}")

        expected_batches = (num_samples + batch_size - 1) // batch_size
        if stats['num_batches'] != expected_batches:
            logger.error(f"Expected {expected_batches} batches, got {stats['num_batches']}")
            return False

        # Test loading from multi-batch
        logger.info("\nTesting multi-batch dataset loading...")
        dataset = PrecompiledMultimodalDataset(
            data_dir=multibatch_dir,
            split="test",
            load_images=True,
            load_structured=True,
            load_text=False
        )

        logger.info(f"  Loaded {len(dataset)} samples from {stats['num_batches']} batches")

        # Test samples from different batches
        if len(dataset) >= 2:
            sample1 = dataset[0]  # From batch 0
            sample2 = dataset[batch_size]  # From batch 1

            logger.info(f"  Sample 0 (batch 0): {sample1['sample_id']}")
            logger.info(f"  Sample {batch_size} (batch 1): {sample2['sample_id']}")

        dataset.close()

        logger.info("\n✓ Multi-batch build test PASSED")
        return True

    except Exception as e:
        logger.error(f"\n✗ Multi-batch build test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(description='Test pre-compilation infrastructure')

    parser.add_argument(
        '--test',
        type=str,
        choices=['all', 'single', 'loading', 'multi'],
        default='all',
        help='Which tests to run'
    )

    parser.add_argument(
        '--num-samples',
        type=int,
        default=10,
        help='Number of samples for testing'
    )

    parser.add_argument(
        '--cleanup',
        action='store_true',
        help='Clean up test output directory after tests'
    )

    args = parser.parse_args()

    # Load config
    config_path = Path("step2_preprocessing/config/config.yaml")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Create test output directory
    test_output_dir = Path("./test_precompilation_output")
    test_output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Pre-compilation Infrastructure Test Suite")
    logger.info("=" * 80)
    logger.info(f"Test output dir: {test_output_dir}")
    logger.info(f"Running tests: {args.test}")
    logger.info("")

    # Run tests
    results = {}

    if args.test in ['all', 'single']:
        results['single'] = test_single_batch_build(config, test_output_dir, args.num_samples)

    if args.test in ['all', 'loading']:
        if 'single' in results and results['single']:
            results['loading'] = test_dataset_loading(test_output_dir)
        else:
            logger.warning("Skipping loading test (single-batch build failed)")

    if args.test in ['all', 'multi']:
        results['multi'] = test_multi_batch_build(config, test_output_dir, args.num_samples, batch_size=5)

    # Print summary
    logger.info("\n" + "=" * 80)
    logger.info("TEST SUMMARY")
    logger.info("=" * 80)

    all_passed = True
    for test_name, passed in results.items():
        status = "✓ PASSED" if passed else "✗ FAILED"
        logger.info(f"  {test_name}: {status}")

        if not passed:
            all_passed = False

    if all_passed:
        logger.info("\n✓ All tests PASSED!")
        return_code = 0
    else:
        logger.info("\n✗ Some tests FAILED")
        return_code = 1

    # Cleanup if requested
    if args.cleanup:
        logger.info(f"\nCleaning up test output: {test_output_dir}")
        shutil.rmtree(test_output_dir)

    return return_code


if __name__ == "__main__":
    sys.exit(main())
