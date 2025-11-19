#!/usr/bin/env python3
"""
Test processing a single sample to verify pipeline functionality.
"""
import argparse
import yaml
import logging
from pathlib import Path
import json

from config.paths import Step2Paths
from src.integration.multimodal_dataset import MultimodalMIMICDataset

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def format_size(num_bytes):
    """Format bytes as human-readable string"""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if num_bytes < 1024.0:
            return f"{num_bytes:.2f} {unit}"
        num_bytes /= 1024.0
    return f"{num_bytes:.2f} TB"


def test_single_sample(sample_idx: int = 0, split: str = 'train', api_key: str = None):
    """Test processing a single sample"""

    logger.info("="*80)
    logger.info("TESTING SINGLE SAMPLE PROCESSING")
    logger.info("="*80 + "\n")

    # Load config
    config_path = Path(__file__).parent / 'config' / 'config.yaml'
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    paths = Step2Paths(config_path)

    # Create dataset
    logger.info(f"Creating {split} dataset...")
    cohort_path = paths.step1_train if split == 'train' else paths.step1_val

    dataset = MultimodalMIMICDataset(
        cohort_csv_path=cohort_path,
        config=config,
        paths=paths,
        anthropic_api_key=api_key,
        split=split,
        load_images=True,
        load_structured=True,
        load_text=True
    )

    logger.info(f"Dataset size: {len(dataset)} samples\n")

    if sample_idx >= len(dataset):
        logger.error(f"Sample index {sample_idx} out of range (max: {len(dataset)-1})")
        return

    # Get sample info
    logger.info(f"Loading sample {sample_idx}...")
    sample_info = dataset.get_sample_info(sample_idx)
    logger.info("\nSample Metadata:")
    logger.info(f"  Subject ID: {sample_info['subject_id']}")
    logger.info(f"  Study ID: {sample_info['study_id']}")
    logger.info(f"  Study datetime: {sample_info['study_datetime']}")
    logger.info(f"  ED intime: {sample_info['ed_intime']}")
    logger.info(f"  View position: {sample_info['view_position']}")
    logger.info(f"  Has hospital admission: {sample_info['has_hospital_admission']}")

    # Load full sample
    logger.info("\nProcessing sample...")
    sample = dataset[sample_idx]

    # Analyze image
    logger.info("\n" + "-"*80)
    logger.info("IMAGE ANALYSIS")
    logger.info("-"*80)
    if 'image' in sample and sample['image'] is not None:
        img = sample['image']
        logger.info(f"  ✓ Image loaded successfully")
        logger.info(f"    - Shape: {tuple(img.shape)} (C, H, W)")
        logger.info(f"    - Size: {format_size(img.element_size() * img.nelement())}")
        logger.info(f"    - Dtype: {img.dtype}")
        logger.info(f"    - Range: [{img.min():.3f}, {img.max():.3f}]")
        logger.info(f"    - Mean: {img.mean():.3f}")
        logger.info(f"    - Std: {img.std():.3f}")
    else:
        logger.error("  ✗ Image failed to load")

    # Analyze structured data
    logger.info("\n" + "-"*80)
    logger.info("STRUCTURED DATA ANALYSIS")
    logger.info("-"*80)
    if 'structured' in sample and sample['structured'] is not None:
        struct = sample['structured']
        logger.info(f"  ✓ Structured data loaded successfully")
        logger.info(f"    - Total features: {len(struct)}")

        # Count vitals and labs
        vitals = [k for k in struct.keys() if k.startswith('vital_')]
        labs = [k for k in struct.keys() if k.startswith('lab_')]

        logger.info(f"    - Vitals: {len(vitals)}")
        logger.info(f"    - Labs: {len(labs)}")

        # Check for missing values
        missing = sum(1 for v in struct.values() if isinstance(v, dict) and v.get('is_missing', False))
        logger.info(f"    - Missing (NOT_DONE): {missing}/{len(struct)}")

        # Show example features
        logger.info("\n  Example vital sign (if available):")
        if vitals:
            vital_name = vitals[0]
            vital_data = struct[vital_name]
            logger.info(f"    {vital_name}:")
            if not vital_data.get('is_missing', False):
                logger.info(f"      - Last value: {vital_data.get('last_value', 'N/A')}")
                logger.info(f"      - Mean: {vital_data.get('mean_value', 'N/A')}")
                logger.info(f"      - Measurement count: {vital_data.get('measurement_count', 'N/A')}")
                logger.info(f"      - Trend slope: {vital_data.get('trend_slope', 'N/A')}")
            else:
                logger.info(f"      - Status: NOT_DONE")

        logger.info("\n  Example lab (if available):")
        if labs:
            lab_name = labs[0]
            lab_data = struct[lab_name]
            logger.info(f"    {lab_name}:")
            if not lab_data.get('is_missing', False):
                logger.info(f"      - Last value: {lab_data.get('last_value', 'N/A')}")
                logger.info(f"      - Range: [{lab_data.get('min_value', 'N/A')}, {lab_data.get('max_value', 'N/A')}]")
                logger.info(f"      - Measurement count: {lab_data.get('measurement_count', 'N/A')}")
            else:
                logger.info(f"      - Status: NOT_DONE")
    else:
        logger.error("  ✗ Structured data failed to load")

    # Analyze text
    logger.info("\n" + "-"*80)
    logger.info("TEXT ANALYSIS")
    logger.info("-"*80)
    if 'text' in sample and sample['text'] is not None:
        text = sample['text']
        logger.info(f"  ✓ Text processed successfully")
        logger.info(f"    - Summary length: {len(text['summary'])} characters")
        logger.info(f"    - Entities extracted: {text['num_entities']}")
        logger.info(f"    - Tokens: {text['tokens']['num_tokens']}")
        logger.info(f"    - Token truncated: {text['tokens']['is_truncated']}")

        if text['summary']:
            logger.info(f"\n  Summary preview:")
            preview = text['summary'][:200] + "..." if len(text['summary']) > 200 else text['summary']
            logger.info(f"    {preview}")

        if text.get('entities'):
            logger.info(f"\n  Sample entities:")
            for ent in text['entities'][:5]:
                logger.info(f"    - {ent}")
    else:
        logger.warning("  ⚠ Text data not available (may be expected if no clinical notes)")

    # Show errors
    logger.info("\n" + "-"*80)
    logger.info("ERRORS")
    logger.info("-"*80)
    if sample.get('errors'):
        logger.warning(f"  ⚠ {len(sample['errors'])} errors encountered:")
        for err in sample['errors']:
            logger.warning(f"    - {err}")
    else:
        logger.info("  ✓ No errors")

    # Save sample for inspection
    output_dir = Path(__file__).parent / 'test_output'
    output_dir.mkdir(exist_ok=True)

    sample_file = output_dir / f'sample_{sample_idx}.json'
    logger.info(f"\nSaving sample metadata to: {sample_file}")

    # Create JSON-serializable version
    sample_json = {
        'metadata': {k: str(v) for k, v in sample['metadata'].items()},
        'errors': sample.get('errors', []),
        'image_shape': tuple(sample['image'].shape) if 'image' in sample and sample['image'] is not None else None,
        'structured_feature_count': len(sample['structured']) if 'structured' in sample and sample['structured'] is not None else 0,
        'text_summary_length': len(sample['text']['summary']) if 'text' in sample and sample['text'] is not None else 0,
    }

    with open(sample_file, 'w') as f:
        json.dump(sample_json, f, indent=2)

    logger.info("\n" + "="*80)
    logger.info("TEST COMPLETE")
    logger.info("="*80)


def main():
    parser = argparse.ArgumentParser(description='Test single sample processing')
    parser.add_argument('--index', type=int, default=0, help='Sample index to test')
    parser.add_argument('--split', choices=['train', 'val'], default='train', help='Dataset split')
    parser.add_argument('--api-key', type=str, help='Anthropic API key')

    args = parser.parse_args()

    test_single_sample(args.index, args.split, args.api_key)


if __name__ == '__main__':
    main()
