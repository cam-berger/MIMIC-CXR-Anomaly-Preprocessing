"""
Multimodal MIMIC dataset integration.
Combines images, structured data, and text into unified samples.
"""
import pandas as pd
import torch
from torch.utils.data import Dataset
from pathlib import Path
from typing import Dict, List, Optional, Any
import logging
from datetime import datetime

from ..image_processing.image_loader import FullResolutionImageLoader
from ..structured_data.temporal_processor import TemporalFeatureExtractor
from ..text_processing.note_processor import ClinicalNoteProcessor

logger = logging.getLogger(__name__)


class MultimodalMIMICDataset(Dataset):
    """
    PyTorch Dataset for multimodal MIMIC-CXR data.

    Integrates:
    1. Full-resolution chest X-ray images
    2. Temporal structured data (labs/vitals)
    3. Processed clinical notes (NER + Claude summaries)
    """

    def __init__(
        self,
        cohort_csv_path: Path,
        config: Dict,
        paths,
        anthropic_api_key: Optional[str] = None,
        split: str = 'train',
        load_images: bool = True,
        load_structured: bool = True,
        load_text: bool = True
    ):
        """
        Initialize multimodal dataset.

        Args:
            cohort_csv_path: Path to Step 1 cohort CSV (train or val)
            config: Configuration dict from config.yaml
            paths: Step2Paths object for data access
            anthropic_api_key: API key for Claude (optional)
            split: 'train' or 'val'
            load_images: Whether to load image modality
            load_structured: Whether to load structured data modality
            load_text: Whether to load text modality
        """
        self.config = config
        self.paths = paths
        self.split = split

        # Load cohort
        logger.info(f"Loading {split} cohort from {cohort_csv_path}")
        self.cohort = pd.read_csv(cohort_csv_path)
        logger.info(f"  Loaded {len(self.cohort)} samples")

        # Parse datetime columns
        datetime_cols = ['study_datetime', 'ed_intime', 'ed_outtime']
        for col in datetime_cols:
            if col in self.cohort.columns:
                self.cohort[col] = pd.to_datetime(self.cohort[col])

        # Initialize processors based on flags
        self.load_images = load_images
        self.load_structured = load_structured
        self.load_text = load_text

        if self.load_images:
            logger.info("Initializing image loader...")
            self.image_loader = FullResolutionImageLoader(config)
        else:
            self.image_loader = None

        if self.load_structured:
            logger.info("Initializing structured data processor...")
            self.structured_processor = TemporalFeatureExtractor(config, paths)
        else:
            self.structured_processor = None

        if self.load_text:
            logger.info("Initializing text processor...")
            self.text_processor = ClinicalNoteProcessor(config, anthropic_api_key)
        else:
            self.text_processor = None

        # Track processing errors
        self.error_log = []

        logger.info(f"Dataset initialized: {len(self)} samples")
        logger.info(f"  Images: {self.load_images}")
        logger.info(f"  Structured: {self.load_structured}")
        logger.info(f"  Text: {self.load_text}")

    def __len__(self) -> int:
        return len(self.cohort)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Load and process a single multimodal sample.

        Returns:
            Dict with keys:
                - 'image': torch.Tensor [C, H, W] (if load_images)
                - 'structured': Dict of temporal features (if load_structured)
                - 'text': Dict with summary and tokens (if load_text)
                - 'metadata': Dict with subject_id, study_id, etc.
                - 'errors': List of any processing errors
        """
        row = self.cohort.iloc[idx]
        sample = {'metadata': {}, 'errors': []}

        # Extract metadata
        sample['metadata'] = {
            'index': idx,
            'subject_id': int(row['subject_id']),
            'study_id': int(row['study_id']),
            'split': self.split,
            'study_datetime': row['study_datetime'],
            'view_position': row.get('ViewPosition', 'UNKNOWN'),
            'image_count': int(row.get('image_count', 1))
        }

        # Load image modality
        if self.load_images:
            sample['image'] = self._load_image(row, sample['errors'])

        # Load structured data modality
        if self.load_structured:
            sample['structured'] = self._load_structured(row, sample['errors'])

        # Load text modality
        if self.load_text:
            sample['text'] = self._load_text(row, sample['errors'])

        return sample

    def _load_image(self, row: pd.Series, errors: List[str]) -> Optional[torch.Tensor]:
        """Load full-resolution image(s) for the study"""
        try:
            subject_id = int(row['subject_id'])
            study_id = int(row['study_id'])

            # Get study directory
            study_dir = self.paths.get_image_path(subject_id, study_id)

            if not study_dir.exists():
                errors.append(f"Study directory not found: {study_dir}")
                return None

            # Load all images in study
            images = self.image_loader.load_study_images(study_dir)

            if len(images) == 0:
                errors.append(f"No images found in {study_dir}")
                return None

            # For now, return the first image (or could stack multiple views)
            # TODO: Handle multiple views more sophisticatedly
            first_image = list(images.values())[0]

            return first_image

        except Exception as e:
            errors.append(f"Image loading error: {str(e)}")
            logger.error(f"Error loading image for study {row['study_id']}: {e}")
            return None

    def _load_structured(self, row: pd.Series, errors: List[str]) -> Optional[Dict]:
        """Load temporal structured features (labs/vitals)"""
        try:
            subject_id = int(row['subject_id'])
            hadm_id = row.get('hadm_id')

            # Handle NaN hadm_id
            if pd.isna(hadm_id):
                hadm_id = None
            else:
                hadm_id = int(hadm_id)

            ed_intime = row['ed_intime']
            study_datetime = row['study_datetime']

            # Extract features
            features = self.structured_processor.extract_features(
                subject_id=subject_id,
                hadm_id=hadm_id,
                ed_intime=ed_intime,
                study_datetime=study_datetime
            )

            return features

        except Exception as e:
            errors.append(f"Structured data error: {str(e)}")
            logger.error(f"Error loading structured data for subject {row['subject_id']}: {e}")
            return None

    def _load_text(self, row: pd.Series, errors: List[str]) -> Optional[Dict]:
        """Load and process clinical notes"""
        try:
            # In Step 1, we don't have clinical notes loaded yet
            # This is a placeholder for when we integrate MIMIC-IV clinical notes

            # For now, return empty/placeholder
            # TODO: Load actual clinical notes from MIMIC-IV noteevents table
            note_text = row.get('clinical_note', '')

            if pd.isna(note_text) or len(str(note_text).strip()) == 0:
                # Return empty result
                return self.text_processor._empty_note_result()

            # Process note
            result = self.text_processor.process_note(str(note_text))

            return result

        except Exception as e:
            errors.append(f"Text processing error: {str(e)}")
            logger.error(f"Error processing text for subject {row['subject_id']}: {e}")
            return self.text_processor._empty_note_result() if self.text_processor else None

    def get_sample_info(self, idx: int) -> Dict:
        """Get metadata about a sample without loading full data"""
        row = self.cohort.iloc[idx]
        return {
            'subject_id': int(row['subject_id']),
            'study_id': int(row['study_id']),
            'study_datetime': row['study_datetime'],
            'ed_intime': row['ed_intime'],
            'view_position': row.get('ViewPosition', 'UNKNOWN'),
            'has_hospital_admission': not pd.isna(row.get('hadm_id'))
        }

    def get_error_summary(self) -> Dict[str, int]:
        """Get summary of processing errors"""
        error_types = {}
        for error in self.error_log:
            error_type = error.split(':')[0]
            error_types[error_type] = error_types.get(error_type, 0) + 1
        return error_types


def multimodal_collate_fn(batch: List[Dict]) -> Dict[str, Any]:
    """
    Custom collate function for multimodal data with variable sizes.

    Handles:
    - Variable image resolutions
    - Variable-length temporal sequences
    - Variable-length tokenized text

    Args:
        batch: List of samples from MultimodalMIMICDataset

    Returns:
        Batched dict with proper padding/stacking
    """
    collated = {
        'metadata': [],
        'errors': []
    }

    # Collect all modalities
    images = []
    structured = []
    text = []

    for sample in batch:
        collated['metadata'].append(sample['metadata'])
        collated['errors'].extend(sample.get('errors', []))

        if 'image' in sample and sample['image'] is not None:
            images.append(sample['image'])

        if 'structured' in sample and sample['structured'] is not None:
            structured.append(sample['structured'])

        if 'text' in sample and sample['text'] is not None:
            text.append(sample['text'])

    # Batch images (cannot stack if different sizes, return as list)
    if len(images) > 0:
        # Check if all images have same size
        shapes = [img.shape for img in images]
        if len(set(shapes)) == 1:
            # All same size - can stack
            collated['images'] = torch.stack(images)
        else:
            # Different sizes - return as list
            collated['images'] = images
            logger.debug(f"Variable image sizes in batch: {shapes}")
    else:
        collated['images'] = []

    # Batch structured data (keep as list of dicts for now)
    # TODO: Could implement smart batching/padding for sequential encoding
    collated['structured'] = structured

    # Batch text data
    if len(text) > 0:
        # Stack tokenized text with padding
        try:
            max_len = max(t['tokens']['num_tokens'] for t in text)

            input_ids_list = []
            attention_mask_list = []

            for t in text:
                input_ids = t['tokens']['input_ids']
                attention_mask = t['tokens']['attention_mask']

                # Pad if needed
                if len(input_ids) < max_len:
                    pad_len = max_len - len(input_ids)
                    input_ids = torch.cat([
                        input_ids,
                        torch.zeros(pad_len, dtype=input_ids.dtype)
                    ])
                    attention_mask = torch.cat([
                        attention_mask,
                        torch.zeros(pad_len, dtype=attention_mask.dtype)
                    ])

                input_ids_list.append(input_ids)
                attention_mask_list.append(attention_mask)

            collated['text'] = {
                'input_ids': torch.stack(input_ids_list),
                'attention_mask': torch.stack(attention_mask_list),
                'summaries': [t['summary'] for t in text],
                'num_entities': [t['num_entities'] for t in text]
            }
        except Exception as e:
            logger.warning(f"Error batching text data: {e}")
            collated['text'] = text
    else:
        collated['text'] = []

    return collated


class MultimodalDataLoader:
    """
    Wrapper for PyTorch DataLoader with multimodal-specific handling.
    """

    @staticmethod
    def create_dataloader(
        dataset: MultimodalMIMICDataset,
        batch_size: int = 4,
        shuffle: bool = True,
        num_workers: int = 4,
        pin_memory: bool = True
    ):
        """
        Create DataLoader with appropriate settings for multimodal data.

        Note: batch_size is small (default 4) due to full-resolution images
        consuming ~30MB each in memory.
        """
        from torch.utils.data import DataLoader

        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            collate_fn=multimodal_collate_fn
        )
