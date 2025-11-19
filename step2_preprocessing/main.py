#!/usr/bin/env python3
"""
Step 2: Data Preprocessing and Feature Engineering
Main execution script for multimodal MIMIC-CXR preprocessing
"""
import argparse
import logging
import yaml
from pathlib import Path
import pandas as pd
import torch
from datetime import datetime
from tqdm import tqdm
import json
import os

from config.paths import Step2Paths
from src.integration.multimodal_dataset import MultimodalMIMICDataset


def setup_logging(log_level: str, log_file: Path):
    """Configure logging"""
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f'Invalid log level: {log_level}')

    logging.basicConfig(
        level=numeric_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )


def load_config(config_path: Path) -> dict:
    """Load configuration from YAML"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def process_dataset(
    dataset: MultimodalMIMICDataset,
    split: str,
    output_base: Path,
    save_frequency: int = 100
):
    """
    Process entire dataset and save outputs.

    Args:
        dataset: MultimodalMIMICDataset instance
        split: 'train' or 'val'
        output_base: Base output directory
        save_frequency: Save batch every N samples
    """
    logger = logging.getLogger(__name__)
    logger.info(f"\n{'='*60}")
    logger.info(f"Processing {split} dataset: {len(dataset)} samples")
    logger.info(f"{'='*60}\n")

    # Create output directories
    split_dir = output_base / split
    split_dir.mkdir(parents=True, exist_ok=True)

    image_dir = split_dir / "images"
    structured_dir = split_dir / "structured_features"
    text_dir = split_dir / "text_features"
    metadata_dir = split_dir / "metadata"

    for d in [image_dir, structured_dir, text_dir, metadata_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # Processing statistics
    stats = {
        'total_samples': len(dataset),
        'processed': 0,
        'failed': 0,
        'image_errors': 0,
        'structured_errors': 0,
        'text_errors': 0,
        'processing_times': []
    }

    # Process each sample
    failed_samples = []

    for idx in tqdm(range(len(dataset)), desc=f"Processing {split}"):
        start_time = datetime.now()

        try:
            # Load sample
            sample = dataset[idx]

            # Extract identifiers
            subject_id = sample['metadata']['subject_id']
            study_id = sample['metadata']['study_id']
            sample_key = f"s{subject_id}_study{study_id}"

            # Save image if present
            if 'image' in sample and sample['image'] is not None:
                image_path = image_dir / f"{sample_key}.pt"
                torch.save(sample['image'], image_path)
            else:
                stats['image_errors'] += 1

            # Save structured features if present
            if 'structured' in sample and sample['structured'] is not None:
                structured_path = structured_dir / f"{sample_key}.json"
                # Convert to JSON-serializable format
                structured_json = _convert_to_json_serializable(sample['structured'])
                with open(structured_path, 'w') as f:
                    json.dump(structured_json, f, indent=2)
            else:
                stats['structured_errors'] += 1

            # Save text features if present
            if 'text' in sample and sample['text'] is not None:
                text_path = text_dir / f"{sample_key}.pt"
                # Save tokens and summary
                text_data = {
                    'summary': sample['text']['summary'],
                    'tokens': sample['text']['tokens'],
                    'num_entities': sample['text']['num_entities'],
                    'entities': sample['text'].get('entities', [])
                }
                torch.save(text_data, text_path)
            else:
                stats['text_errors'] += 1

            # Save metadata
            metadata_path = metadata_dir / f"{sample_key}.json"
            metadata_json = _convert_to_json_serializable(sample['metadata'])
            metadata_json['errors'] = sample.get('errors', [])
            with open(metadata_path, 'w') as f:
                json.dump(metadata_json, f, indent=2)

            # Track processing time
            processing_time = (datetime.now() - start_time).total_seconds()
            stats['processing_times'].append(processing_time)
            stats['processed'] += 1

            # Log any errors
            if len(sample.get('errors', [])) > 0:
                logger.warning(f"Sample {sample_key} had errors: {sample['errors']}")

        except Exception as e:
            logger.error(f"Failed to process sample {idx}: {e}")
            stats['failed'] += 1
            failed_samples.append({
                'index': idx,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            })

        # Periodic progress report
        if (idx + 1) % save_frequency == 0:
            avg_time = sum(stats['processing_times'][-save_frequency:]) / save_frequency
            logger.info(f"Processed {idx + 1}/{len(dataset)} samples "
                       f"(avg time: {avg_time:.2f}s/sample)")

    # Calculate final statistics
    stats['avg_processing_time'] = sum(stats['processing_times']) / len(stats['processing_times']) if stats['processing_times'] else 0
    stats['image_success_rate'] = (stats['processed'] - stats['image_errors']) / stats['processed'] if stats['processed'] > 0 else 0
    stats['structured_success_rate'] = (stats['processed'] - stats['structured_errors']) / stats['processed'] if stats['processed'] > 0 else 0
    stats['text_success_rate'] = (stats['processed'] - stats['text_errors']) / stats['processed'] if stats['processed'] > 0 else 0

    # Save statistics
    stats_path = split_dir / "processing_stats.json"
    with open(stats_path, 'w') as f:
        json.dump({
            'statistics': {k: v for k, v in stats.items() if k != 'processing_times'},
            'failed_samples': failed_samples
        }, f, indent=2)

    # Log summary
    logger.info(f"\n{'='*60}")
    logger.info(f"{split.upper()} PROCESSING COMPLETE")
    logger.info(f"{'='*60}")
    logger.info(f"Total samples: {stats['total_samples']}")
    logger.info(f"Successfully processed: {stats['processed']}")
    logger.info(f"Failed: {stats['failed']}")
    logger.info(f"Image success rate: {stats['image_success_rate']:.1%}")
    logger.info(f"Structured success rate: {stats['structured_success_rate']:.1%}")
    logger.info(f"Text success rate: {stats['text_success_rate']:.1%}")
    logger.info(f"Average processing time: {stats['avg_processing_time']:.2f}s/sample")
    logger.info(f"Results saved to: {split_dir}")
    logger.info(f"{'='*60}\n")

    return stats


def _convert_to_json_serializable(obj):
    """Convert objects to JSON-serializable format"""
    if isinstance(obj, dict):
        return {k: _convert_to_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_convert_to_json_serializable(item) for item in obj]
    elif isinstance(obj, (pd.Timestamp, datetime)):
        return obj.isoformat()
    elif isinstance(obj, torch.Tensor):
        return f"<Tensor: shape={tuple(obj.shape)}>"
    elif isinstance(obj, (int, float, str, bool, type(None))):
        return obj
    else:
        return str(obj)


def main():
    parser = argparse.ArgumentParser(
        description='Step 2: MIMIC-CXR Multimodal Preprocessing'
    )

    parser.add_argument(
        '--config',
        type=Path,
        default=Path(__file__).parent / 'config' / 'config.yaml',
        help='Path to configuration file'
    )

    parser.add_argument(
        '--output-dir',
        type=Path,
        help='Override output directory from config'
    )

    parser.add_argument(
        '--anthropic-api-key',
        type=str,
        default=None,
        help='Anthropic API key for Claude (or set ANTHROPIC_API_KEY env var)'
    )

    parser.add_argument(
        '--skip-images',
        action='store_true',
        help='Skip image processing'
    )

    parser.add_argument(
        '--skip-structured',
        action='store_true',
        help='Skip structured data processing'
    )

    parser.add_argument(
        '--skip-text',
        action='store_true',
        help='Skip text processing'
    )

    parser.add_argument(
        '--train-only',
        action='store_true',
        help='Only process training set'
    )

    parser.add_argument(
        '--val-only',
        action='store_true',
        help='Only process validation set'
    )

    parser.add_argument(
        '--max-samples',
        type=int,
        default=None,
        help='Maximum number of samples to process per split (for testing)'
    )

    parser.add_argument(
        '--log-level',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        default='INFO',
        help='Logging level'
    )

    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)

    # Initialize paths
    paths = Step2Paths(args.config)

    # Override output directory if specified
    if args.output_dir:
        paths.output_base = args.output_dir
        paths._create_directories()

    # Setup logging
    log_file = paths.output_base / 'preprocessing.log'
    setup_logging(args.log_level, log_file)
    logger = logging.getLogger(__name__)

    logger.info("="*80)
    logger.info("STEP 2: MULTIMODAL PREPROCESSING")
    logger.info("="*80)
    logger.info(f"Configuration: {args.config}")
    logger.info(f"Output directory: {paths.output_base}")
    logger.info(f"Process images: {not args.skip_images}")
    logger.info(f"Process structured: {not args.skip_structured}")
    logger.info(f"Process text: {not args.skip_text}")
    logger.info("="*80 + "\n")

    # Get API key
    api_key = args.anthropic_api_key or os.environ.get('ANTHROPIC_API_KEY')
    if not args.skip_text and api_key is None:
        logger.warning("No Anthropic API key provided. Text summarization will use fallback.")

    # Process splits
    results = {}

    if not args.val_only:
        logger.info("\n" + "="*80)
        logger.info("PROCESSING TRAINING SET")
        logger.info("="*80 + "\n")

        # Create training dataset
        train_dataset = MultimodalMIMICDataset(
            cohort_csv_path=paths.step1_train,
            config=config,
            paths=paths,
            anthropic_api_key=api_key,
            split='train',
            load_images=not args.skip_images,
            load_structured=not args.skip_structured,
            load_text=not args.skip_text
        )

        # Limit samples if testing
        if args.max_samples:
            train_dataset.cohort = train_dataset.cohort.iloc[:args.max_samples]
            logger.info(f"Limited to {args.max_samples} training samples for testing")

        # Process training set
        train_stats = process_dataset(train_dataset, 'train', paths.output_base)
        results['train'] = train_stats

    if not args.train_only:
        logger.info("\n" + "="*80)
        logger.info("PROCESSING VALIDATION SET")
        logger.info("="*80 + "\n")

        # Create validation dataset
        val_dataset = MultimodalMIMICDataset(
            cohort_csv_path=paths.step1_val,
            config=config,
            paths=paths,
            anthropic_api_key=api_key,
            split='val',
            load_images=not args.skip_images,
            load_structured=not args.skip_structured,
            load_text=not args.skip_text
        )

        # Limit samples if testing
        if args.max_samples:
            val_dataset.cohort = val_dataset.cohort.iloc[:args.max_samples]
            logger.info(f"Limited to {args.max_samples} validation samples for testing")

        # Process validation set
        val_stats = process_dataset(val_dataset, 'val', paths.output_base)
        results['val'] = val_stats

    # Save overall summary
    summary_path = paths.output_base / 'preprocessing_summary.json'
    # Convert args to JSON-serializable dict
    args_dict = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    with open(summary_path, 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'config': str(args.config),
            'results': results,
            'arguments': args_dict
        }, f, indent=2)

    logger.info("\n" + "="*80)
    logger.info("PREPROCESSING COMPLETE")
    logger.info("="*80)
    logger.info(f"Summary saved to: {summary_path}")
    logger.info(f"All outputs in: {paths.output_base}")
    logger.info("="*80 + "\n")


if __name__ == '__main__':
    main()
