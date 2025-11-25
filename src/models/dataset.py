"""
PyTorch Dataset for loading preprocessed MIMIC-CXR images from HDF5.

Supports both HDF5 format (from image preprocessing pipeline) and
individual .pt files (from step2 preprocessing).
"""

import io
import json
import logging
from pathlib import Path
from typing import Optional, Callable, Union

import h5py
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
import torchvision.transforms.functional as TF

logger = logging.getLogger(__name__)


class MIMICCXRDataset(Dataset):
    """
    Dataset for loading preprocessed chest X-ray images from HDF5.

    The HDF5 file structure (from ImagePreprocessor):
        /images/{idx} - Image tensor [C, H, W]
        /metadata/{idx} - JSON metadata string
        /index - Parquet-encoded DataFrame with study_id -> idx mapping

    Args:
        hdf5_path: Path to HDF5 file containing preprocessed images
        transform: Optional transform to apply to images
        target_size: Target image size (H, W) for resizing, default (224, 224)
        normalize: Whether to apply ImageNet normalization
        return_metadata: Whether to return metadata dict with each sample
    """

    def __init__(
        self,
        hdf5_path: Union[str, Path],
        transform: Optional[Callable] = None,
        target_size: tuple[int, int] = (224, 224),
        normalize: bool = True,
        return_metadata: bool = False,
    ):
        self.hdf5_path = Path(hdf5_path)
        self.transform = transform
        self.target_size = target_size
        self.normalize = normalize
        self.return_metadata = return_metadata

        # Validate file exists
        if not self.hdf5_path.exists():
            raise FileNotFoundError(f"HDF5 file not found: {self.hdf5_path}")

        # Load index (mapping study_id -> idx)
        self._load_index()

        # Build default transform pipeline if none provided
        if self.transform is None:
            self.transform = self._build_default_transform()

        # Keep file handle closed until needed (for multiprocessing)
        self._hdf5_file = None

    def _load_index(self) -> None:
        """Load the index mapping from HDF5 file."""
        with h5py.File(self.hdf5_path, "r") as f:
            if "index" not in f:
                raise ValueError("HDF5 file missing 'index' dataset")

            index_bytes = f["index"][:]
            self.index_df = pd.read_parquet(io.BytesIO(bytes(index_bytes)))

        self.study_ids = self.index_df["study_id"].tolist()
        self.idx_to_study = {row["idx"]: row["study_id"] for _, row in self.index_df.iterrows()}
        self.study_to_idx = {row["study_id"]: row["idx"] for _, row in self.index_df.iterrows()}

        logger.info(f"Loaded index with {len(self.study_ids)} samples")

    def _build_default_transform(self) -> Callable:
        """Build default transform pipeline for MAE training."""
        transforms = [
            T.ToPILImage(),
            T.Resize(self.target_size),
        ]

        if self.normalize:
            # Convert grayscale to 3-channel for ImageNet pretrained models
            transforms.extend([
                T.Grayscale(num_output_channels=3),
                T.ToTensor(),
                # ImageNet normalization (commonly used even for grayscale)
                T.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                ),
            ])
        else:
            transforms.extend([
                T.Grayscale(num_output_channels=3),
                T.ToTensor(),
            ])

        return T.Compose(transforms)

    def _get_hdf5_file(self) -> h5py.File:
        """Get HDF5 file handle (lazy loading for multiprocessing)."""
        if self._hdf5_file is None:
            self._hdf5_file = h5py.File(self.hdf5_path, "r")
        return self._hdf5_file

    def __len__(self) -> int:
        return len(self.study_ids)

    def __getitem__(self, idx: int) -> Union[torch.Tensor, tuple[torch.Tensor, dict]]:
        """
        Get a sample by index.

        Args:
            idx: Sample index

        Returns:
            If return_metadata=False: image tensor [3, H, W]
            If return_metadata=True: (image tensor, metadata dict)
        """
        f = self._get_hdf5_file()

        # Get the HDF5 internal index
        row = self.index_df.iloc[idx]
        hdf5_idx = str(row["idx"])
        study_id = row["study_id"]
        subject_id = row["subject_id"]

        # Load image
        image = f["images"][hdf5_idx][:]  # Shape: [1, H, W]

        # Ensure correct shape and convert to float
        if image.ndim == 2:
            image = image[np.newaxis, ...]  # Add channel dim

        # Convert to torch tensor for transforms
        image = torch.from_numpy(image).float()

        # Apply transforms
        if self.transform is not None:
            # Transform expects [C, H, W] or [H, W]
            image = self.transform(image.squeeze(0))  # Remove channel for ToPILImage

        if self.return_metadata:
            # Load metadata if available
            metadata = {"study_id": study_id, "subject_id": subject_id}
            if "metadata" in f and hdf5_idx in f["metadata"]:
                meta_json = f["metadata"][hdf5_idx][()]
                if isinstance(meta_json, bytes):
                    meta_json = meta_json.decode("utf-8")
                metadata.update(json.loads(meta_json))
            return image, metadata

        return image

    def get_by_study_id(self, study_id: int) -> torch.Tensor:
        """Get sample by study_id."""
        if study_id not in self.study_to_idx:
            raise KeyError(f"Study ID {study_id} not found in dataset")

        idx = self.index_df[self.index_df["study_id"] == study_id].index[0]
        return self[idx]

    def __del__(self):
        """Clean up HDF5 file handle."""
        if self._hdf5_file is not None:
            self._hdf5_file.close()


class MIMICCXRHybridDataset(Dataset):
    """
    Dataset that loads from individual .pt files (step2 preprocessing format).

    Directory structure (from DATA_SCHEMA.md):
        {base_dir}/
        ├── images/
        │   ├── s{subject_id}_study{study_id}.pt
        ├── structured_features/
        │   ├── s{subject_id}_study{study_id}.json
        ├── text_features/
        │   ├── s{subject_id}_study{study_id}.pt
        └── metadata/
            ├── s{subject_id}_study{study_id}.json

    Args:
        base_dir: Base directory containing train/ or val/ subdirectory
        cohort_csv: Path to cohort CSV with subject_id, study_id columns
        split: 'train' or 'val'
        transform: Optional transform to apply to images
        target_size: Target image size (H, W) for resizing
        include_text: Whether to include text features
        include_structured: Whether to include structured features
    """

    def __init__(
        self,
        base_dir: Union[str, Path],
        cohort_csv: Union[str, Path],
        split: str = "train",
        transform: Optional[Callable] = None,
        target_size: tuple[int, int] = (224, 224),
        include_text: bool = False,
        include_structured: bool = False,
    ):
        self.base_dir = Path(base_dir) / split
        self.split = split
        self.transform = transform
        self.target_size = target_size
        self.include_text = include_text
        self.include_structured = include_structured

        # Load cohort
        self.cohort = pd.read_csv(cohort_csv)

        # Filter to samples that exist on disk
        self._filter_existing_samples()

        # Build default transform
        if self.transform is None:
            self.transform = self._build_default_transform()

        logger.info(f"Loaded {len(self)} samples from {split} split")

    def _filter_existing_samples(self) -> None:
        """Filter cohort to only samples that exist on disk."""
        valid_indices = []

        for idx, row in self.cohort.iterrows():
            sample_id = f"s{row['subject_id']}_study{row['study_id']}"
            image_path = self.base_dir / "images" / f"{sample_id}.pt"

            if image_path.exists():
                valid_indices.append(idx)

        original_len = len(self.cohort)
        self.cohort = self.cohort.loc[valid_indices].reset_index(drop=True)

        if len(self.cohort) < original_len:
            logger.warning(
                f"Filtered from {original_len} to {len(self.cohort)} samples "
                f"(missing {original_len - len(self.cohort)} image files)"
            )

    def _build_default_transform(self) -> Callable:
        """Build default transform pipeline."""
        return T.Compose([
            T.ToPILImage(),
            T.Resize(self.target_size),
            T.Grayscale(num_output_channels=3),
            T.ToTensor(),
            T.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])

    def __len__(self) -> int:
        return len(self.cohort)

    def __getitem__(self, idx: int) -> dict:
        """
        Get a sample by index.

        Returns:
            Dictionary with:
                - 'image': tensor [3, H, W]
                - 'study_id': int
                - 'subject_id': int
                - 'text_tokens': tensor [512] (if include_text)
                - 'attention_mask': tensor [512] (if include_text)
                - 'structured': dict (if include_structured)
        """
        row = self.cohort.iloc[idx]
        subject_id = int(row["subject_id"])
        study_id = int(row["study_id"])
        sample_id = f"s{subject_id}_study{study_id}"

        result = {
            "study_id": study_id,
            "subject_id": subject_id,
        }

        # Load image
        image_path = self.base_dir / "images" / f"{sample_id}.pt"
        image = torch.load(image_path, weights_only=True)

        # Handle different shapes
        if image.ndim == 2:
            image = image.unsqueeze(0)  # [H, W] -> [1, H, W]

        # Apply transform
        if self.transform is not None:
            image = self.transform(image.squeeze(0))

        result["image"] = image

        # Load text features if requested
        if self.include_text:
            text_path = self.base_dir / "text_features" / f"{sample_id}.pt"
            if text_path.exists():
                text_data = torch.load(text_path, weights_only=False)
                result["text_tokens"] = text_data["tokens"]["input_ids"]
                result["attention_mask"] = text_data["tokens"]["attention_mask"]
                result["summary"] = text_data.get("summary", "")
            else:
                # Empty placeholders
                result["text_tokens"] = torch.zeros(512, dtype=torch.long)
                result["attention_mask"] = torch.zeros(512, dtype=torch.long)
                result["summary"] = ""

        # Load structured features if requested
        if self.include_structured:
            struct_path = self.base_dir / "structured_features" / f"{sample_id}.json"
            if struct_path.exists():
                with open(struct_path, "r") as f:
                    result["structured"] = json.load(f)
            else:
                result["structured"] = {}

        return result


def get_mae_augmentations(
    target_size: tuple[int, int] = (224, 224),
    training: bool = True,
) -> Callable:
    """
    Get augmentation pipeline for MAE training.

    Based on medical_mae recommendations:
    - Moderate crop ranges (0.5-1.0) vs aggressive for natural images
    - Horizontal flip (anatomically valid)
    - Light rotation (up to 15 degrees)
    - Gaussian blur

    Args:
        target_size: Target image size (H, W)
        training: Whether this is for training (applies augmentations)

    Returns:
        Transform function
    """
    if training:
        return T.Compose([
            T.ToPILImage(),
            T.RandomResizedCrop(target_size, scale=(0.5, 1.0)),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomRotation(degrees=15),
            T.RandomApply([T.GaussianBlur(kernel_size=23)], p=0.5),
            T.Grayscale(num_output_channels=3),
            T.ToTensor(),
            T.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])
    else:
        return T.Compose([
            T.ToPILImage(),
            T.Resize(target_size),
            T.Grayscale(num_output_channels=3),
            T.ToTensor(),
            T.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])
