"""
PyTorch Dataset for pre-compiled aggregate multimodal data.

Loads efficiently from pre-compiled HDF5 (images) and Parquet (structured/text) files.
Optimized for both:
- Fast training (memory-mapped loading)
- Analytical queries (SQL-like filtering)
"""
import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import logging
import json
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from builders.hdf5_writer import HDF5Reader
from builders.parquet_writer import ParquetReader

logger = logging.getLogger(__name__)


class PrecompiledMultimodalDataset(Dataset):
    """
    PyTorch Dataset for pre-compiled multimodal data.

    Features:
    - Memory-mapped HDF5 loading for images (lazy loading)
    - Fast Parquet loading for structured/text
    - Batch-aware indexing
    - Optional data filtering
    - Caching support
    """

    def __init__(
        self,
        data_dir: Path,
        split: str = "train",
        load_images: bool = True,
        load_structured: bool = True,
        load_text: bool = True,
        filter_expr: Optional[str] = None,
        cache_images: bool = False
    ):
        """
        Initialize pre-compiled dataset.

        Args:
            data_dir: Root directory of pre-compiled dataset
            split: Dataset split ("train" or "val")
            load_images: Whether to load images
            load_structured: Whether to load structured features
            load_text: Whether to load text features
            filter_expr: Optional pandas query expression for filtering
            cache_images: Whether to cache images in memory
        """
        self.data_dir = Path(data_dir)
        self.split = split
        self.load_images = load_images
        self.load_structured = load_structured
        self.load_text = load_text
        self.cache_images = cache_images

        self.split_dir = self.data_dir / split

        if not self.split_dir.exists():
            raise FileNotFoundError(f"Split directory not found: {self.split_dir}")

        logger.info(f"Initializing PrecompiledMultimodalDataset")
        logger.info(f"  Split: {split}")
        logger.info(f"  Data dir: {self.data_dir}")
        logger.info(f"  Load images: {load_images}")
        logger.info(f"  Load structured: {load_structured}")
        logger.info(f"  Load text: {load_text}")

        # Load manifest
        self.manifest = self._load_manifest()
        logger.info(f"  Found {self.manifest['num_batches']} batches")

        # Initialize readers
        self._init_readers()

        # Load metadata index
        self.index = self._load_index(filter_expr)
        logger.info(f"  Total samples: {len(self.index)}")

        # Image cache
        self.image_cache = {} if cache_images else None

    def _load_manifest(self) -> Dict:
        """Load manifest file"""
        manifest_path = self.split_dir / "manifest.json"

        if not manifest_path.exists():
            raise FileNotFoundError(f"Manifest not found: {manifest_path}")

        with open(manifest_path, 'r') as f:
            manifest = json.load(f)

        return manifest

    def _init_readers(self):
        """Initialize HDF5 and Parquet readers"""
        # Find all batch files
        hdf5_paths = []
        parquet_paths = []

        for batch in self.manifest['batches']:
            if self.load_images:
                hdf5_path = Path(batch['hdf5_path'])
                if hdf5_path.exists():
                    hdf5_paths.append(hdf5_path)

            parquet_path = Path(batch['parquet_path'])
            if parquet_path.exists():
                parquet_paths.append(parquet_path)

        # Initialize readers
        if self.load_images and hdf5_paths:
            self.hdf5_reader = HDF5Reader(hdf5_paths)
            logger.info(f"  Initialized HDF5 reader with {len(hdf5_paths)} files")
        else:
            self.hdf5_reader = None

        if parquet_paths:
            self.parquet_reader = ParquetReader(parquet_paths)
            logger.info(f"  Initialized Parquet reader with {len(parquet_paths)} files")
        else:
            self.parquet_reader = None
            raise ValueError("No Parquet files found")

    def _load_index(self, filter_expr: Optional[str] = None) -> pd.DataFrame:
        """
        Load sample index from Parquet files.

        Args:
            filter_expr: Optional pandas query expression for filtering

        Returns:
            DataFrame with sample metadata
        """
        # Load metadata columns only (fast)
        metadata_columns = [
            'sample_id', 'subject_id', 'study_id', 'hadm_id',
            'study_datetime', 'ed_intime', 'has_errors'
        ]

        index = self.parquet_reader.load_all(columns=metadata_columns)

        # Apply filter if specified
        if filter_expr:
            logger.info(f"  Applying filter: {filter_expr}")
            index = index.query(filter_expr)
            logger.info(f"  Filtered to {len(index)} samples")

        return index.reset_index(drop=True)

    def __len__(self) -> int:
        """Return number of samples"""
        return len(self.index)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get a single sample.

        Args:
            idx: Sample index

        Returns:
            Dictionary with loaded data for all enabled modalities
        """
        if idx < 0 or idx >= len(self.index):
            raise IndexError(f"Index {idx} out of range [0, {len(self.index)})")

        row = self.index.iloc[idx]
        sample_id = row['sample_id']

        sample = {
            'sample_id': sample_id,
            'subject_id': int(row['subject_id']),
            'study_id': int(row['study_id']),
            'idx': idx
        }

        # Load image
        if self.load_images and self.hdf5_reader:
            # Check cache first
            if self.image_cache is not None and sample_id in self.image_cache:
                image = self.image_cache[sample_id]
            else:
                image = self.hdf5_reader.load_image(sample_id)

                # Cache if enabled
                if self.image_cache is not None and image is not None:
                    self.image_cache[sample_id] = image

            if image is not None:
                # Convert to torch tensor
                sample['image'] = torch.from_numpy(image).float()
            else:
                sample['image'] = None

        # Load structured/text features from Parquet
        if self.load_structured or self.load_text:
            full_row = self.parquet_reader.load_sample(sample_id)

            if full_row is not None:
                # Extract structured features
                if self.load_structured:
                    structured_features = self._extract_structured_features(full_row)
                    sample['structured'] = structured_features

                # Extract text features
                if self.load_text:
                    text_features = self._extract_text_features(full_row)
                    sample['text'] = text_features

        return sample

    def _extract_structured_features(self, row: pd.Series) -> Dict:
        """Extract structured features from Parquet row"""
        features = {}

        # Get all columns that start with 'vital_' or 'lab_' or 'med_'
        for col in row.index:
            if col.startswith(('vital_', 'lab_', 'med_')):
                value = row[col]

                # Convert to appropriate type
                if pd.isna(value):
                    features[col] = None
                elif value == "NOT_DONE":
                    features[col] = "NOT_DONE"
                else:
                    # Try to convert to float
                    try:
                        features[col] = float(value)
                    except (ValueError, TypeError):
                        features[col] = value

        return features

    def _extract_text_features(self, row: pd.Series) -> Dict:
        """Extract text features from Parquet row"""
        features = {}

        # Text summary
        if 'text_summary' in row.index:
            features['summary'] = row['text_summary']

        # Entity count
        if 'text_entity_count' in row.index:
            features['entity_count'] = int(row['text_entity_count']) if pd.notna(row['text_entity_count']) else 0

        # Token IDs (stored as JSON string)
        if 'text_token_ids' in row.index and pd.notna(row['text_token_ids']):
            try:
                token_ids = json.loads(row['text_token_ids'])
                features['token_ids'] = torch.tensor(token_ids, dtype=torch.long)
            except (json.JSONDecodeError, TypeError):
                features['token_ids'] = None

        return features

    def get_sample_metadata(self, idx: int) -> Dict:
        """
        Get metadata for a sample without loading full data.

        Args:
            idx: Sample index

        Returns:
            Dictionary with metadata
        """
        row = self.index.iloc[idx]

        return {
            'sample_id': row['sample_id'],
            'subject_id': int(row['subject_id']),
            'study_id': int(row['study_id']),
            'hadm_id': int(row['hadm_id']) if pd.notna(row.get('hadm_id')) else None,
            'has_errors': bool(row.get('has_errors', False))
        }

    def filter(self, expr: str) -> 'PrecompiledMultimodalDataset':
        """
        Create a filtered view of the dataset.

        Args:
            expr: Pandas query expression

        Returns:
            New dataset instance with filtered index
        """
        filtered_dataset = PrecompiledMultimodalDataset.__new__(PrecompiledMultimodalDataset)

        # Copy attributes
        filtered_dataset.data_dir = self.data_dir
        filtered_dataset.split = self.split
        filtered_dataset.load_images = self.load_images
        filtered_dataset.load_structured = self.load_structured
        filtered_dataset.load_text = self.load_text
        filtered_dataset.cache_images = self.cache_images
        filtered_dataset.split_dir = self.split_dir
        filtered_dataset.manifest = self.manifest
        filtered_dataset.hdf5_reader = self.hdf5_reader
        filtered_dataset.parquet_reader = self.parquet_reader
        filtered_dataset.image_cache = self.image_cache

        # Apply filter to index
        filtered_dataset.index = self.index.query(expr).reset_index(drop=True)

        logger.info(f"Filtered dataset: {len(self.index)} -> {len(filtered_dataset.index)} samples")

        return filtered_dataset

    def get_statistics(self) -> Dict:
        """
        Get dataset statistics.

        Returns:
            Dictionary with statistics
        """
        stats = {
            'total_samples': len(self.index),
            'num_batches': self.manifest['num_batches'],
            'split': self.split,
        }

        # Error statistics
        if 'has_errors' in self.index.columns:
            stats['samples_with_errors'] = int(self.index['has_errors'].sum())
            stats['error_rate'] = float(stats['samples_with_errors'] / len(self.index) * 100)

        return stats

    def close(self):
        """Close file handles"""
        if self.hdf5_reader:
            self.hdf5_reader.close()


def precompiled_collate_fn(batch: List[Dict]) -> Dict[str, Any]:
    """
    Custom collate function for pre-compiled dataset.

    Handles variable-size data and missing modalities.

    Args:
        batch: List of samples from __getitem__

    Returns:
        Batched dictionary
    """
    collated = {
        'sample_ids': [s['sample_id'] for s in batch],
        'subject_ids': torch.tensor([s['subject_id'] for s in batch]),
        'study_ids': torch.tensor([s['study_id'] for s in batch]),
        'indices': torch.tensor([s['idx'] for s in batch])
    }

    # Batch images (if present)
    images = [s.get('image') for s in batch if s.get('image') is not None]
    if images:
        # Check if all images have same shape
        shapes = [img.shape for img in images]
        if len(set(shapes)) == 1:
            # All same shape - can stack
            collated['images'] = torch.stack(images)
        else:
            # Variable shapes - return list
            collated['images'] = images
            collated['image_shapes'] = shapes

    # Batch structured features
    if 'structured' in batch[0]:
        structured_features = {}
        for feature_name in batch[0]['structured'].keys():
            values = [s['structured'].get(feature_name) for s in batch]

            # Try to convert to tensor if all numeric
            if all(isinstance(v, (int, float)) for v in values):
                structured_features[feature_name] = torch.tensor(values, dtype=torch.float32)
            else:
                # Keep as list (e.g., for NOT_DONE tokens)
                structured_features[feature_name] = values

        collated['structured'] = structured_features

    # Batch text features
    if 'text' in batch[0]:
        collated['text'] = {
            'summaries': [s['text'].get('summary', '') for s in batch],
            'entity_counts': torch.tensor([s['text'].get('entity_count', 0) for s in batch])
        }

        # Batch token IDs (if present)
        token_ids = [s['text'].get('token_ids') for s in batch if s['text'].get('token_ids') is not None]
        if token_ids:
            # Pad to max length
            max_len = max(t.size(0) for t in token_ids)
            padded = torch.zeros((len(batch), max_len), dtype=torch.long)

            for i, tokens in enumerate(token_ids):
                padded[i, :tokens.size(0)] = tokens

            collated['text']['token_ids'] = padded

    return collated
