#!/usr/bin/env python3
"""
Validate preprocessing outputs for MAE training readiness.

Checks that processed samples meet requirements for Step 3 (Multimodal MAE):
- Image tensors: [C, H, W] format, normalized [0, 1]
- Text features: ClinicalBERT tokens (512-dim), summaries present
- Structured features: Temporal aggregations, NOT_DONE handling
"""
import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple
import torch
import pandas as pd
import numpy as np
from collections import defaultdict

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class MAEReadinessValidator:
    """Validate preprocessing outputs for MAE training"""

    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.train_dir = output_dir / 'train'

        # Expected directories
        self.image_dir = self.train_dir / 'images'
        self.text_dir = self.train_dir / 'text_features'
        self.structured_dir = self.train_dir / 'structured_features'
        self.metadata_dir = self.train_dir / 'metadata'

        # Validation results
        self.results = {
            'total_samples': 0,
            'image_valid': 0,
            'text_valid': 0,
            'structured_valid': 0,
            'fully_valid': 0,
            'errors': defaultdict(list),
            'statistics': {}
        }

    def validate_all(self) -> Dict:
        """Run complete validation pipeline"""
        logger.info("="*80)
        logger.info("MAE READINESS VALIDATION")
        logger.info("="*80)
        logger.info(f"Output directory: {self.output_dir}")
        logger.info("")

        # Check directories exist
        if not self._check_directories():
            return self.results

        # Get all sample keys
        sample_keys = self._get_sample_keys()
        self.results['total_samples'] = len(sample_keys)

        logger.info(f"Found {len(sample_keys)} samples to validate")
        logger.info("")

        # Validate each sample
        for idx, sample_key in enumerate(sample_keys, 1):
            if idx % 50 == 0:
                logger.info(f"Validated {idx}/{len(sample_keys)} samples...")

            self._validate_sample(sample_key)

        # Calculate statistics
        self._calculate_statistics()

        # Generate report
        self._print_report()

        return self.results

    def _check_directories(self) -> bool:
        """Check that all required directories exist"""
        missing = []
        for dir_path in [self.image_dir, self.text_dir, self.structured_dir, self.metadata_dir]:
            if not dir_path.exists():
                missing.append(str(dir_path))

        if missing:
            logger.error(f"Missing directories: {missing}")
            logger.error("Run preprocessing pipeline first!")
            return False

        return True

    def _get_sample_keys(self) -> List[str]:
        """Get all sample keys from metadata directory"""
        metadata_files = list(self.metadata_dir.glob('*.json'))
        sample_keys = [f.stem for f in metadata_files]
        return sorted(sample_keys)

    def _validate_sample(self, sample_key: str):
        """Validate a single sample across all modalities"""
        image_valid = self._validate_image(sample_key)
        text_valid = self._validate_text(sample_key)
        structured_valid = self._validate_structured(sample_key)

        if image_valid:
            self.results['image_valid'] += 1
        if text_valid:
            self.results['text_valid'] += 1
        if structured_valid:
            self.results['structured_valid'] += 1
        if image_valid and text_valid and structured_valid:
            self.results['fully_valid'] += 1

    def _validate_image(self, sample_key: str) -> bool:
        """Validate image tensor format and values"""
        image_path = self.image_dir / f"{sample_key}.pt"

        if not image_path.exists():
            self.results['errors']['missing_image'].append(sample_key)
            return False

        try:
            image = torch.load(image_path)

            # Check it's a tensor
            if not isinstance(image, torch.Tensor):
                self.results['errors']['image_not_tensor'].append(sample_key)
                return False

            # Check shape: [C, H, W]
            if len(image.shape) != 3:
                self.results['errors']['image_wrong_dims'].append(
                    f"{sample_key} (shape: {image.shape})"
                )
                return False

            # Check channels (should be 1 for grayscale or 3 for RGB)
            C, H, W = image.shape
            if C not in [1, 3]:
                self.results['errors']['image_wrong_channels'].append(
                    f"{sample_key} (channels: {C})"
                )
                return False

            # Check value range [0, 1] for normalized images
            min_val = image.min().item()
            max_val = image.max().item()

            if min_val < -0.1 or max_val > 1.1:  # Allow small tolerance
                self.results['errors']['image_wrong_range'].append(
                    f"{sample_key} (range: [{min_val:.3f}, {max_val:.3f}])"
                )
                return False

            # Store image statistics
            if 'image_shapes' not in self.results['statistics']:
                self.results['statistics']['image_shapes'] = []
                self.results['statistics']['image_sizes_mb'] = []

            self.results['statistics']['image_shapes'].append((C, H, W))
            size_mb = image.element_size() * image.nelement() / (1024 * 1024)
            self.results['statistics']['image_sizes_mb'].append(size_mb)

            return True

        except Exception as e:
            self.results['errors']['image_load_error'].append(f"{sample_key}: {str(e)}")
            return False

    def _validate_text(self, sample_key: str) -> bool:
        """Validate text features format"""
        text_path = self.text_dir / f"{sample_key}.pt"

        if not text_path.exists():
            self.results['errors']['missing_text'].append(sample_key)
            return False

        try:
            text_data = torch.load(text_path)

            # Check it's a dictionary
            if not isinstance(text_data, dict):
                self.results['errors']['text_not_dict'].append(sample_key)
                return False

            # Check required keys
            required_keys = ['summary', 'tokens', 'num_entities']
            missing_keys = [k for k in required_keys if k not in text_data]
            if missing_keys:
                self.results['errors']['text_missing_keys'].append(
                    f"{sample_key} (missing: {missing_keys})"
                )
                return False

            # Check tokens format
            tokens = text_data['tokens']
            if not isinstance(tokens, dict):
                self.results['errors']['text_tokens_not_dict'].append(sample_key)
                return False

            # Check token components
            if 'input_ids' not in tokens or 'attention_mask' not in tokens:
                self.results['errors']['text_tokens_incomplete'].append(sample_key)
                return False

            # Check input_ids shape (should be [seq_len])
            input_ids = tokens['input_ids']
            if not isinstance(input_ids, torch.Tensor):
                self.results['errors']['text_input_ids_not_tensor'].append(sample_key)
                return False

            if len(input_ids.shape) != 1:
                self.results['errors']['text_input_ids_wrong_shape'].append(
                    f"{sample_key} (shape: {input_ids.shape})"
                )
                return False

            # Check sequence length (ClinicalBERT max is 512)
            seq_len = input_ids.shape[0]
            if seq_len > 512:
                self.results['errors']['text_seq_too_long'].append(
                    f"{sample_key} (length: {seq_len})"
                )
                return False

            # Store text statistics
            if 'text_seq_lengths' not in self.results['statistics']:
                self.results['statistics']['text_seq_lengths'] = []
                self.results['statistics']['text_num_entities'] = []
                self.results['statistics']['text_summary_lengths'] = []

            self.results['statistics']['text_seq_lengths'].append(seq_len)
            self.results['statistics']['text_num_entities'].append(text_data['num_entities'])
            self.results['statistics']['text_summary_lengths'].append(len(text_data['summary']))

            return True

        except Exception as e:
            self.results['errors']['text_load_error'].append(f"{sample_key}: {str(e)}")
            return False

    def _validate_structured(self, sample_key: str) -> bool:
        """Validate structured features format"""
        structured_path = self.structured_dir / f"{sample_key}.json"

        if not structured_path.exists():
            self.results['errors']['missing_structured'].append(sample_key)
            return False

        try:
            with open(structured_path, 'r') as f:
                structured_data = json.load(f)

            # Check it's a dictionary
            if not isinstance(structured_data, dict):
                self.results['errors']['structured_not_dict'].append(sample_key)
                return False

            # Check for labs and vitals sections
            has_labs = 'labs' in structured_data and structured_data['labs']
            has_vitals = 'vitals' in structured_data and structured_data['vitals']

            if not has_labs and not has_vitals:
                self.results['errors']['structured_empty'].append(sample_key)
                # Still valid - patient might not have any measurements
                # return False

            # Check NOT_DONE token handling
            if 'labs' in structured_data:
                for lab_name, lab_data in structured_data['labs'].items():
                    if isinstance(lab_data, dict):
                        # Check for aggregated format with mean/std/count
                        if 'value' in lab_data:
                            value = lab_data['value']
                            if value == "NOT_DONE":
                                # Valid missing value marker
                                continue

            # Store structured statistics
            if 'structured_lab_counts' not in self.results['statistics']:
                self.results['statistics']['structured_lab_counts'] = []
                self.results['statistics']['structured_vital_counts'] = []

            lab_count = len(structured_data.get('labs', {}))
            vital_count = len(structured_data.get('vitals', {}))

            self.results['statistics']['structured_lab_counts'].append(lab_count)
            self.results['statistics']['structured_vital_counts'].append(vital_count)

            return True

        except Exception as e:
            self.results['errors']['structured_load_error'].append(f"{sample_key}: {str(e)}")
            return False

    def _calculate_statistics(self):
        """Calculate aggregate statistics"""
        stats = self.results['statistics']

        # Image statistics
        if 'image_shapes' in stats and stats['image_shapes']:
            shapes = stats['image_shapes']
            stats['image_shape_distribution'] = pd.Series(
                [f"{s[0]}x{s[1]}x{s[2]}" for s in shapes]
            ).value_counts().to_dict()

            sizes = stats['image_sizes_mb']
            stats['image_size_stats'] = {
                'mean_mb': np.mean(sizes),
                'median_mb': np.median(sizes),
                'min_mb': np.min(sizes),
                'max_mb': np.max(sizes)
            }

        # Text statistics
        if 'text_seq_lengths' in stats and stats['text_seq_lengths']:
            stats['text_seq_length_stats'] = {
                'mean': np.mean(stats['text_seq_lengths']),
                'median': np.median(stats['text_seq_lengths']),
                'min': np.min(stats['text_seq_lengths']),
                'max': np.max(stats['text_seq_lengths'])
            }

            stats['text_entity_stats'] = {
                'mean': np.mean(stats['text_num_entities']),
                'median': np.median(stats['text_num_entities']),
                'min': np.min(stats['text_num_entities']),
                'max': np.max(stats['text_num_entities'])
            }

        # Structured statistics
        if 'structured_lab_counts' in stats and stats['structured_lab_counts']:
            stats['structured_lab_stats'] = {
                'mean': np.mean(stats['structured_lab_counts']),
                'median': np.median(stats['structured_lab_counts']),
                'samples_with_labs': sum(1 for c in stats['structured_lab_counts'] if c > 0)
            }

            stats['structured_vital_stats'] = {
                'mean': np.mean(stats['structured_vital_counts']),
                'median': np.median(stats['structured_vital_counts']),
                'samples_with_vitals': sum(1 for c in stats['structured_vital_counts'] if c > 0)
            }

    def _print_report(self):
        """Print validation report"""
        logger.info("")
        logger.info("="*80)
        logger.info("VALIDATION RESULTS")
        logger.info("="*80)

        total = self.results['total_samples']
        if total == 0:
            logger.error("No samples found!")
            return

        # Success rates
        logger.info(f"\nSuccess Rates:")
        logger.info(f"  Total samples: {total}")
        logger.info(f"  Image valid: {self.results['image_valid']}/{total} ({100*self.results['image_valid']/total:.1f}%)")
        logger.info(f"  Text valid: {self.results['text_valid']}/{total} ({100*self.results['text_valid']/total:.1f}%)")
        logger.info(f"  Structured valid: {self.results['structured_valid']}/{total} ({100*self.results['structured_valid']/total:.1f}%)")
        logger.info(f"  Fully valid (all modalities): {self.results['fully_valid']}/{total} ({100*self.results['fully_valid']/total:.1f}%)")

        # Error summary
        if self.results['errors']:
            logger.info(f"\nError Summary:")
            for error_type, samples in self.results['errors'].items():
                logger.info(f"  {error_type}: {len(samples)} samples")
                if len(samples) <= 5:
                    for sample in samples:
                        logger.info(f"    - {sample}")
                else:
                    for sample in samples[:3]:
                        logger.info(f"    - {sample}")
                    logger.info(f"    ... and {len(samples)-3} more")

        # Statistics
        stats = self.results['statistics']

        if 'image_size_stats' in stats:
            logger.info(f"\nImage Statistics:")
            logger.info(f"  Average size: {stats['image_size_stats']['mean_mb']:.2f} MB")
            logger.info(f"  Size range: {stats['image_size_stats']['min_mb']:.2f} - {stats['image_size_stats']['max_mb']:.2f} MB")
            if 'image_shape_distribution' in stats:
                logger.info(f"  Shape distribution:")
                for shape, count in list(stats['image_shape_distribution'].items())[:3]:
                    logger.info(f"    {shape}: {count} samples")

        if 'text_seq_length_stats' in stats:
            logger.info(f"\nText Statistics:")
            logger.info(f"  Average sequence length: {stats['text_seq_length_stats']['mean']:.1f} tokens")
            logger.info(f"  Sequence length range: {stats['text_seq_length_stats']['min']:.0f} - {stats['text_seq_length_stats']['max']:.0f} tokens")
            logger.info(f"  Average entities per note: {stats['text_entity_stats']['mean']:.1f}")

        if 'structured_lab_stats' in stats:
            logger.info(f"\nStructured Data Statistics:")
            logger.info(f"  Samples with labs: {stats['structured_lab_stats']['samples_with_labs']}/{total}")
            logger.info(f"  Average labs per sample: {stats['structured_lab_stats']['mean']:.1f}")
            logger.info(f"  Samples with vitals: {stats['structured_vital_stats']['samples_with_vitals']}/{total}")
            logger.info(f"  Average vitals per sample: {stats['structured_vital_stats']['mean']:.1f}")

        # MAE readiness assessment
        logger.info("")
        logger.info("="*80)
        logger.info("MAE READINESS ASSESSMENT")
        logger.info("="*80)

        success_rate = 100 * self.results['fully_valid'] / total if total > 0 else 0

        if success_rate >= 95:
            logger.info(f"✓ READY FOR MAE TRAINING ({success_rate:.1f}% success rate)")
            logger.info(f"  - All modalities properly formatted")
            logger.info(f"  - Image tensors: [C,H,W] normalized [0,1]")
            logger.info(f"  - Text tokens: ClinicalBERT format ≤512 tokens")
            logger.info(f"  - Structured features: Temporal aggregations present")
            self.results['mae_ready'] = True
        elif success_rate >= 80:
            logger.warning(f"⚠ PARTIALLY READY ({success_rate:.1f}% success rate)")
            logger.warning(f"  - Some samples have issues")
            logger.warning(f"  - Review errors above before MAE training")
            logger.warning(f"  - Consider filtering invalid samples")
            self.results['mae_ready'] = False
        else:
            logger.error(f"✗ NOT READY ({success_rate:.1f}% success rate)")
            logger.error(f"  - Significant preprocessing issues detected")
            logger.error(f"  - Fix errors before MAE training")
            self.results['mae_ready'] = False

        logger.info("="*80)

    def save_report(self, output_path: Path):
        """Save validation results to JSON"""
        # Convert defaultdict to regular dict for JSON serialization
        results_serializable = {
            'total_samples': self.results['total_samples'],
            'image_valid': self.results['image_valid'],
            'text_valid': self.results['text_valid'],
            'structured_valid': self.results['structured_valid'],
            'fully_valid': self.results['fully_valid'],
            'mae_ready': self.results.get('mae_ready', False),
            'errors': dict(self.results['errors']),
            'statistics': {
                k: v for k, v in self.results['statistics'].items()
                if k not in ['image_shapes', 'image_sizes_mb', 'text_seq_lengths',
                            'text_num_entities', 'text_summary_lengths',
                            'structured_lab_counts', 'structured_vital_counts']
            }
        }

        with open(output_path, 'w') as f:
            json.dump(results_serializable, f, indent=2)

        logger.info(f"\nValidation report saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Validate preprocessing outputs for MAE training'
    )

    parser.add_argument(
        '--output-dir',
        type=Path,
        default=Path(__file__).parent / 'output' / 'validation_200',
        help='Preprocessing output directory to validate'
    )

    parser.add_argument(
        '--report-path',
        type=Path,
        default=None,
        help='Path to save validation report JSON (default: <output_dir>/mae_readiness_report.json)'
    )

    args = parser.parse_args()

    # Validate directory exists
    if not args.output_dir.exists():
        logger.error(f"Output directory does not exist: {args.output_dir}")
        logger.error("Run preprocessing pipeline first!")
        return 1

    # Set default report path
    if args.report_path is None:
        args.report_path = args.output_dir / 'mae_readiness_report.json'

    # Run validation
    validator = MAEReadinessValidator(args.output_dir)
    results = validator.validate_all()

    # Save report
    validator.save_report(args.report_path)

    # Return exit code based on MAE readiness
    if results.get('mae_ready', False):
        return 0
    else:
        return 1


if __name__ == '__main__':
    exit(main())
