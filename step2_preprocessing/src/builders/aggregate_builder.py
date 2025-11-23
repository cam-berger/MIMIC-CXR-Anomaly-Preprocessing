"""
Aggregate dataset builder for pre-compiled multimodal datasets.

Orchestrates the extraction and storage of all modalities (images, structured data, text)
into an efficient pre-compiled format for fast training and analysis.
"""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import logging
import json
from tqdm import tqdm
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from .hdf5_writer import HDF5Writer
from .parquet_writer import ParquetWriter
from data_loaders.cxr_pro_loader import CXRProLoader
from data_loaders.mimic_iv_loader import MIMICIVLoader
from image_processing.image_loader import FullResolutionImageLoader
from structured_data.temporal_processor import TemporalFeatureExtractor
from structured_data.medication_processor import MedicationProcessor
from text_processing.note_processor import ClinicalNoteProcessor

logger = logging.getLogger(__name__)


class AggregateDatasetBuilder:
    """
    Build pre-compiled aggregate multimodal dataset.

    Processes cohort samples and writes all modalities to efficient storage:
    - Images → HDF5 (chunked, compressed, memory-mapped)
    - Structured/Text → Parquet (columnar, queryable)
    - Metadata → JSON manifests with checksums
    """

    def __init__(
        self,
        config: Dict,
        output_dir: Path,
        anthropic_api_key: Optional[str] = None
    ):
        """
        Initialize aggregate dataset builder.

        Args:
            config: Configuration dictionary
            output_dir: Output directory for pre-compiled dataset
            anthropic_api_key: Optional API key for text processing
        """
        self.config = config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Get pre-compilation settings
        precomp_config = config.get('precompilation', {})
        self.batch_size = precomp_config.get('batch_size')  # None = single batch
        self.data_sources = precomp_config.get('data_sources', {})
        self.checkpoint_enabled = precomp_config.get('checkpoint', {}).get('enabled', True)
        self.checkpoint_dir = Path(precomp_config.get('checkpoint', {}).get('checkpoint_dir', './checkpoints'))

        # Initialize storage writers
        storage_config = precomp_config.get('storage', {})
        self.hdf5_writer = HDF5Writer(
            output_dir=self.output_dir,
            compression=storage_config.get('compression', 'gzip'),
            chunk_size=storage_config.get('chunk_size', 100)
        )
        self.parquet_writer = ParquetWriter(
            output_dir=self.output_dir,
            compression=storage_config.get('compression', 'gzip')
        )

        # Initialize data loaders
        self._init_data_loaders()

        # Initialize processors
        self._init_processors(anthropic_api_key)

        # Tracking
        self.processed_samples = 0
        self.failed_samples = 0
        self.current_batch = []

        logger.info(f"Initialized AggregateDatasetBuilder")
        logger.info(f"  Output dir: {self.output_dir}")
        logger.info(f"  Batch size: {self.batch_size or 'single batch (all samples)'}")
        logger.info(f"  Checkpoint enabled: {self.checkpoint_enabled}")

    def _init_data_loaders(self):
        """Initialize data loaders for all enabled sources"""
        logger.info("Initializing data loaders...")

        # CXR-PRO loader
        if self.data_sources.get('cxr_pro', True):
            cxr_pro_path = Path(self.config['data']['cxr_pro_base'])
            self.cxr_pro_loader = CXRProLoader(cxr_pro_path)
            logger.info("  ✓ CXR-PRO loader initialized")
        else:
            self.cxr_pro_loader = None
            logger.info("  ✗ CXR-PRO loader disabled")

        # MIMIC-IV loader (for notes/medications if available)
        mimic_iv_path = Path(self.config['data']['mimic_iv_base'])
        self.mimic_iv_loader = MIMICIVLoader(mimic_iv_path)

        # Check credential-restricted data availability
        resolved_sources = self.mimic_iv_loader.get_data_sources_for_config(self.config)
        self.use_mimic_notes = resolved_sources.get('mimic_iv_note', False)
        self.use_mimic_meds = resolved_sources.get('mimic_iv_med', False)

        logger.info(f"  MIMIC-IV Notes: {'enabled' if self.use_mimic_notes else 'disabled'}")
        logger.info(f"  MIMIC-IV Medications: {'enabled' if self.use_mimic_meds else 'disabled'}")

    def _init_processors(self, anthropic_api_key: Optional[str]):
        """Initialize data processors for all modalities"""
        logger.info("Initializing processors...")

        # Image processor
        try:
            self.image_processor = FullResolutionImageLoader(self.config)
            # Store CXR images base path for building study paths
            self.cxr_images_path = Path(self.config['data']['mimic_cxr_base']) / "files"
            logger.info("  ✓ Image processor initialized")
        except Exception as e:
            logger.error(f"  ✗ Image processor failed: {e}")
            self.image_processor = None
            self.cxr_images_path = None

        # Structured data processor (vitals + labs)
        try:
            from utils.config_loader import load_paths_config
            paths_config = load_paths_config(self.config)
            self.structured_processor = TemporalFeatureExtractor(self.config, paths_config)
            logger.info("  ✓ Structured data processor initialized")
        except Exception as e:
            logger.error(f"  ✗ Structured processor failed: {e}")
            self.structured_processor = None

        # Medication processor (if enabled)
        if self.use_mimic_meds:
            try:
                self.medication_processor = MedicationProcessor(self.config, self.mimic_iv_loader)
                logger.info("  ✓ Medication processor initialized")
            except Exception as e:
                logger.error(f"  ✗ Medication processor failed: {e}")
                self.medication_processor = None
        else:
            self.medication_processor = None

        # Text processor
        try:
            self.text_processor = ClinicalNoteProcessor(self.config, anthropic_api_key)
            logger.info("  ✓ Text processor initialized")
        except Exception as e:
            logger.error(f"  ✗ Text processor failed: {e}")
            self.text_processor = None

    def build(
        self,
        cohort_path: Path,
        split: str = "train",
        resume_from_checkpoint: bool = True
    ) -> Dict:
        """
        Build pre-compiled dataset from cohort.

        Args:
            cohort_path: Path to cohort CSV file
            split: Dataset split ("train" or "val")
            resume_from_checkpoint: Whether to resume from checkpoint if available

        Returns:
            Build statistics dictionary
        """
        logger.info("=" * 80)
        logger.info(f"Building Pre-compiled Dataset: {split}")
        logger.info("=" * 80)

        # Load cohort
        logger.info(f"\nLoading cohort from: {cohort_path}")
        cohort = pd.read_csv(cohort_path)
        logger.info(f"  Loaded {len(cohort)} samples")

        # Add CXR-PRO reports if not already present
        if 'radiology_report' not in cohort.columns and self.cxr_pro_loader:
            logger.info("\nAdding CXR-PRO radiology reports to cohort...")
            cohort = self.cxr_pro_loader.join_with_cohort(cohort)

        # Check for checkpoint
        start_idx = 0
        if resume_from_checkpoint and self.checkpoint_enabled:
            checkpoint = self._load_checkpoint(split)
            if checkpoint:
                start_idx = checkpoint['processed_samples']
                logger.info(f"\nResuming from checkpoint: {start_idx} samples already processed")

        # Process samples
        logger.info(f"\nProcessing {len(cohort) - start_idx} samples...")
        batch_id = start_idx // self.batch_size if self.batch_size else 0

        for idx in tqdm(range(start_idx, len(cohort)), desc=f"Building {split}"):
            row = cohort.iloc[idx]

            # Extract all modalities for this sample
            sample_data = self._process_sample(row)

            if sample_data:
                self.current_batch.append(sample_data)
                self.processed_samples += 1
            else:
                self.failed_samples += 1

            # Write batch when full (or at end)
            should_write_batch = False
            if self.batch_size:
                should_write_batch = len(self.current_batch) >= self.batch_size
            else:
                should_write_batch = (idx == len(cohort) - 1)  # Last sample

            if should_write_batch and len(self.current_batch) > 0:
                self._write_batch(batch_id, self.current_batch, split)
                batch_id += 1
                self.current_batch = []

            # Checkpoint
            if self.checkpoint_enabled and (self.processed_samples % 100 == 0):
                self._save_checkpoint(split, self.processed_samples)

        # Write any remaining samples
        if len(self.current_batch) > 0:
            self._write_batch(batch_id, self.current_batch, split)

        # Generate manifest
        manifest = self._generate_manifest(split)

        # Build statistics
        stats = {
            'split': split,
            'total_samples': len(cohort),
            'processed_samples': self.processed_samples,
            'failed_samples': self.failed_samples,
            'success_rate': self.processed_samples / len(cohort) * 100 if len(cohort) > 0 else 0,
            'num_batches': batch_id + 1,
            'manifest': manifest
        }

        logger.info("\n" + "=" * 80)
        logger.info("Build Complete!")
        logger.info("=" * 80)
        logger.info(f"  Total samples: {stats['total_samples']}")
        logger.info(f"  Processed: {stats['processed_samples']}")
        logger.info(f"  Failed: {stats['failed_samples']}")
        logger.info(f"  Success rate: {stats['success_rate']:.1f}%")
        logger.info(f"  Number of batches: {stats['num_batches']}")

        return stats

    def _process_sample(self, row: pd.Series) -> Optional[Dict]:
        """
        Process a single sample to extract all modalities.

        Args:
            row: Cohort DataFrame row

        Returns:
            Dictionary with all extracted features or None if failed
        """
        sample_id = f"study_{row['study_id']}"
        errors = []

        sample = {
            'sample_id': sample_id,
            'subject_id': int(row['subject_id']),
            'study_id': int(row['study_id']),
            'hadm_id': int(row['hadm_id']) if pd.notna(row.get('hadm_id')) else None,
            'study_datetime': row.get('study_datetime'),
            'ed_intime': row.get('ed_intime'),
        }

        # Extract image
        if self.image_processor and self.cxr_images_path:
            try:
                # Build MIMIC-CXR directory structure: files/p{subject_id[:2]}/p{subject_id}/s{study_id}/
                subject_id = str(int(row['subject_id']))
                study_id = str(int(row['study_id']))
                subject_dir = f"p{subject_id[:2]}"
                patient_dir = f"p{subject_id}"
                study_dir_name = f"s{study_id}"

                study_path = self.cxr_images_path / subject_dir / patient_dir / study_dir_name

                # Load all images from study directory
                image_dict = self.image_processor.load_study_images(study_path)

                if image_dict and len(image_dict) > 0:
                    # Use first image if multiple views
                    first_view = list(image_dict.values())[0]

                    # Convert PyTorch tensor to numpy array for HDF5 storage
                    import torch
                    if isinstance(first_view, torch.Tensor):
                        sample['image'] = first_view.numpy()
                    else:
                        sample['image'] = first_view
                else:
                    errors.append("Image loading failed")
                    sample['image'] = None
            except Exception as e:
                errors.append(f"Image error: {str(e)}")
                sample['image'] = None

        # Extract structured data
        if self.structured_processor:
            try:
                structured_features = self.structured_processor.extract_features(
                    subject_id=int(row['subject_id']),
                    hadm_id=int(row['hadm_id']) if pd.notna(row.get('hadm_id')) else None,
                    ed_intime=pd.Timestamp(row['ed_intime']),
                    study_datetime=pd.Timestamp(row['study_datetime'])
                )
                sample['structured'] = structured_features
            except Exception as e:
                errors.append(f"Structured error: {str(e)}")
                sample['structured'] = {}

        # Extract medications (if enabled)
        if self.medication_processor:
            try:
                med_features = self.medication_processor.extract_medication_features(
                    subject_id=int(row['subject_id']),
                    hadm_id=int(row['hadm_id']) if pd.notna(row.get('hadm_id')) else None,
                    study_datetime=pd.Timestamp(row['study_datetime'])
                )
                sample['structured'].update(med_features)
            except Exception as e:
                errors.append(f"Medication error: {str(e)}")

        # Extract text
        if self.text_processor:
            try:
                note_text = row.get('radiology_report', '')
                if pd.notna(note_text) and len(str(note_text).strip()) > 0:
                    text_result = self.text_processor.process_note(str(note_text))
                    sample['text'] = text_result
                else:
                    sample['text'] = self.text_processor._empty_note_result()
            except Exception as e:
                errors.append(f"Text error: {str(e)}")
                sample['text'] = {}

        sample['errors'] = errors

        # Only return sample if we have at least an image
        if sample.get('image') is not None:
            return sample
        else:
            return None

    def _write_batch(self, batch_id: int, batch_samples: List[Dict], split: str):
        """Write a batch of samples to storage"""
        logger.info(f"\nWriting batch {batch_id} ({len(batch_samples)} samples)...")

        # Write images to HDF5
        hdf5_samples = [
            {
                'sample_id': s['sample_id'],
                'image': s['image'],
                'metadata': {
                    'subject_id': s['subject_id'],
                    'study_id': s['study_id']
                }
            }
            for s in batch_samples if s.get('image') is not None
        ]

        if hdf5_samples:
            hdf5_path = self.hdf5_writer.write_batch(batch_id, hdf5_samples, split)
            logger.info(f"  HDF5: {hdf5_path}")

        # Write structured/text to Parquet
        parquet_path = self.parquet_writer.write_batch(batch_id, batch_samples, split)
        logger.info(f"  Parquet: {parquet_path}")

    def _save_checkpoint(self, split: str, processed_samples: int):
        """Save checkpoint for resuming"""
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            'split': split,
            'processed_samples': processed_samples,
            'timestamp': datetime.now().isoformat()
        }

        checkpoint_path = self.checkpoint_dir / f"checkpoint_{split}.json"
        with open(checkpoint_path, 'w') as f:
            json.dump(checkpoint, f, indent=2)

    def _load_checkpoint(self, split: str) -> Optional[Dict]:
        """Load checkpoint if exists"""
        checkpoint_path = self.checkpoint_dir / f"checkpoint_{split}.json"

        if checkpoint_path.exists():
            with open(checkpoint_path, 'r') as f:
                return json.load(f)

        return None

    def _generate_manifest(self, split: str) -> Dict:
        """Generate manifest file with batch information"""
        manifest_path = self.output_dir / split / "manifest.json"
        manifest_path.parent.mkdir(parents=True, exist_ok=True)

        # Find all batch directories
        batch_dirs = sorted((self.output_dir / split).glob("batch_*"))

        batches = []
        for batch_dir in batch_dirs:
            batch_id = int(batch_dir.name.split('_')[1])

            batch_info = {
                'batch_id': batch_id,
                'hdf5_path': str(batch_dir / "images.h5"),
                'parquet_path': str(batch_dir / "data.parquet"),
            }

            # Get file sizes
            hdf5_file = batch_dir / "images.h5"
            parquet_file = batch_dir / "data.parquet"

            if hdf5_file.exists():
                batch_info['hdf5_size_mb'] = hdf5_file.stat().st_size / 1024 / 1024

            if parquet_file.exists():
                batch_info['parquet_size_mb'] = parquet_file.stat().st_size / 1024 / 1024
                # Get sample count from Parquet
                df = pd.read_parquet(parquet_file)
                batch_info['num_samples'] = len(df)

            batches.append(batch_info)

        manifest = {
            'split': split,
            'num_batches': len(batches),
            'total_samples': sum(b.get('num_samples', 0) for b in batches),
            'batches': batches,
            'generated_at': datetime.now().isoformat()
        }

        with open(manifest_path, 'w') as f:
            json.dump(manifest, f, indent=2)

        logger.info(f"\nManifest written: {manifest_path}")

        return manifest
