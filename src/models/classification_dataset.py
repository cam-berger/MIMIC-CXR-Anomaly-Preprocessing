"""
PyTorch Dataset for multimodal classification with CheXpert labels.

Extends MIMICCXRDataset to include:
- CheXpert pathology labels (12 classes)
- Label mask for handling uncertainty (-1.0 values)
- Tensorized structured features
- Support for training with masked loss
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

from ..datasets import MIMICCXRLoader

logger = logging.getLogger(__name__)


class MultimodalClassificationDataset(Dataset):
    """
    Dataset for multimodal classification with CheXpert labels.

    Combines:
    - Images from preprocessed HDF5 (MAE-compatible format)
    - Structured clinical features (demographics, vitals, labs)
    - Text tokens (ClinicalBERT tokenized summaries)
    - CheXpert pathology labels with uncertainty masking

    Label Handling:
    - Positive (1.0): Label = 1.0, Mask = 1.0
    - Negative (0.0): Label = 0.0, Mask = 1.0
    - Uncertain (-1.0): Label = 0.0, Mask = 0.0 (excluded from loss)
    - Missing (NaN): Label = 0.0, Mask = 0.0 (excluded from loss)

    Args:
        preprocessed_dir: Path to preprocessed cohort directory
        chexpert_csv: Path to mimic-cxr-2.0.0-chexpert.csv.gz
        transform: Optional transform to apply to images
        target_size: Target image size (H, W)
        normalize: Whether to apply ImageNet normalization
        augment: Whether to apply training augmentations
    """

    # CheXpert pathology labels (12 classes, excludes "No Finding" and "Support Devices")
    PATHOLOGY_LABELS = [
        "Atelectasis",
        "Cardiomegaly",
        "Consolidation",
        "Edema",
        "Enlarged Cardiomediastinum",
        "Fracture",
        "Lung Lesion",
        "Lung Opacity",
        "Pleural Effusion",
        "Pleural Other",
        "Pneumonia",
        "Pneumothorax",
    ]

    # Structured feature names (must match preprocessing output)
    STRUCTURED_FEATURES = [
        # Demographics
        "age",
        "gender_M",
        # Triage vitals
        "triage_temperature",
        "triage_heartrate",
        "triage_resprate",
        "triage_o2sat",
        "triage_sbp",
        "triage_dbp",
        "triage_acuity",
        # ED vitals (aggregated)
        "temperature_mean", "temperature_min", "temperature_max",
        "heartrate_mean", "heartrate_min", "heartrate_max",
        "resprate_mean", "resprate_min", "resprate_max",
        "o2sat_mean", "o2sat_min", "o2sat_max",
        "sbp_mean", "sbp_min", "sbp_max",
        "dbp_mean", "dbp_min", "dbp_max",
        # Labs
        "lab_bicarbonate_mean",
        "lab_bnp_mean",
        "lab_bun_mean",
        "lab_calcium_mean",
        "lab_chloride_mean",
        "lab_creatinine_mean",
        "lab_glucose_mean",
        "lab_hematocrit_mean",
        "lab_hemoglobin_mean",
        "lab_lactate_mean",
        "lab_magnesium_mean",
        "lab_platelets_mean",
        "lab_potassium_mean",
        "lab_procalcitonin_mean",
        "lab_sodium_mean",
        "lab_troponin_mean",
        "lab_wbc_mean",
    ]

    def __init__(
        self,
        preprocessed_dir: Union[str, Path],
        chexpert_csv: Union[str, Path],
        transform: Optional[Callable] = None,
        target_size: tuple[int, int] = (224, 224),
        normalize: bool = True,
        augment: bool = True,
    ):
        self.preprocessed_dir = Path(preprocessed_dir)
        self.chexpert_csv = Path(chexpert_csv)
        self.target_size = target_size
        self.normalize = normalize
        self.augment = augment

        # Paths
        self.hdf5_path = self.preprocessed_dir / "images.h5"
        self.structured_path = self.preprocessed_dir / "structured.parquet"
        self.text_path = self.preprocessed_dir / "text.parquet"

        # Validate paths
        if not self.hdf5_path.exists():
            raise FileNotFoundError(f"HDF5 file not found: {self.hdf5_path}")
        if not self.chexpert_csv.exists():
            raise FileNotFoundError(f"CheXpert CSV not found: {self.chexpert_csv}")

        # Load index from HDF5
        self._load_index()

        # Load CheXpert labels
        self._load_labels()

        # Load structured data
        self._load_structured()

        # Load text data
        self._load_text()

        # Build transform
        if transform is not None:
            self.transform = transform
        else:
            self.transform = self._build_transform(augment)

        # Lazy HDF5 file handle
        self._hdf5_file = None

        # Compute number of structured features
        self.num_structured_features = len(self.STRUCTURED_FEATURES)

        logger.info(
            f"Initialized MultimodalClassificationDataset: "
            f"{len(self)} samples, {len(self.PATHOLOGY_LABELS)} labels, "
            f"{self.num_structured_features} structured features"
        )

    def _load_index(self) -> None:
        """Load index mapping from HDF5."""
        with h5py.File(self.hdf5_path, "r") as f:
            if "index" not in f:
                raise ValueError("HDF5 file missing 'index' dataset")
            index_bytes = f["index"][:]
            self.index_df = pd.read_parquet(io.BytesIO(bytes(index_bytes)))

        self.study_ids = self.index_df["study_id"].tolist()
        logger.info(f"Loaded index with {len(self.study_ids)} images")

    def _load_labels(self) -> None:
        """Load CheXpert labels and create mask for uncertainty."""
        logger.info(f"Loading CheXpert labels from {self.chexpert_csv}")

        # Load CheXpert CSV
        chexpert_df = pd.read_csv(
            self.chexpert_csv,
            dtype={"subject_id": np.int32, "study_id": np.int64},
        )

        # Filter to only studies in our preprocessed set
        study_id_set = set(self.study_ids)
        chexpert_df = chexpert_df[chexpert_df["study_id"].isin(study_id_set)]
        chexpert_df = chexpert_df.set_index("study_id")

        # Extract labels and create mask
        # Labels: 1.0 (positive) -> 1, 0.0 (negative) -> 0, -1.0 (uncertain) -> 0, NaN -> 0
        # Mask: 1.0 (positive) -> 1, 0.0 (negative) -> 1, -1.0 (uncertain) -> 0, NaN -> 0

        labels_dict = {}
        masks_dict = {}

        for study_id in self.study_ids:
            if study_id in chexpert_df.index:
                row = chexpert_df.loc[study_id]
                labels = []
                masks = []

                for label_name in self.PATHOLOGY_LABELS:
                    val = row.get(label_name, np.nan)
                    if pd.isna(val):
                        # Missing -> exclude from loss
                        labels.append(0.0)
                        masks.append(0.0)
                    elif val == -1.0:
                        # Uncertain -> exclude from loss
                        labels.append(0.0)
                        masks.append(0.0)
                    elif val == 1.0:
                        # Positive
                        labels.append(1.0)
                        masks.append(1.0)
                    else:
                        # Negative (0.0)
                        labels.append(0.0)
                        masks.append(1.0)

                labels_dict[study_id] = torch.tensor(labels, dtype=torch.float32)
                masks_dict[study_id] = torch.tensor(masks, dtype=torch.float32)
            else:
                # No labels found -> all masked out
                labels_dict[study_id] = torch.zeros(len(self.PATHOLOGY_LABELS), dtype=torch.float32)
                masks_dict[study_id] = torch.zeros(len(self.PATHOLOGY_LABELS), dtype=torch.float32)

        self.labels = labels_dict
        self.label_masks = masks_dict

        # Log label statistics
        total_positive = sum((l.sum().item() for l in labels_dict.values()))
        total_valid = sum((m.sum().item() for m in masks_dict.values()))
        logger.info(
            f"Loaded labels: {len(labels_dict)} studies, "
            f"{total_positive:.0f} positive labels, {total_valid:.0f} valid labels"
        )

    def _load_structured(self) -> None:
        """Load structured features."""
        if self.structured_path.exists():
            self.structured_df = pd.read_parquet(self.structured_path)
            self.structured_df = self.structured_df.set_index("study_id")
            logger.info(f"Loaded structured data: {len(self.structured_df)} rows")
        else:
            self.structured_df = None
            logger.warning(f"Structured data not found: {self.structured_path}")

    def _load_text(self) -> None:
        """Load text features."""
        if self.text_path.exists():
            self.text_df = pd.read_parquet(self.text_path)
            self.text_df = self.text_df.set_index("study_id")
            logger.info(f"Loaded text data: {len(self.text_df)} rows")
        else:
            self.text_df = None
            logger.warning(f"Text data not found: {self.text_path}")

    def _build_transform(self, augment: bool) -> Callable:
        """Build image transform pipeline."""
        if augment:
            transforms = [
                T.ToPILImage(),
                T.CenterCrop(self.target_size),
                T.RandomHorizontalFlip(p=0.5),
                T.RandomRotation(degrees=15),
                T.RandomApply([T.GaussianBlur(kernel_size=23)], p=0.5),
                T.Grayscale(num_output_channels=3),
                T.ToTensor(),
            ]
        else:
            transforms = [
                T.ToPILImage(),
                T.CenterCrop(self.target_size),
                T.Grayscale(num_output_channels=3),
                T.ToTensor(),
            ]

        if self.normalize:
            transforms.append(
                T.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            )

        return T.Compose(transforms)

    def _get_hdf5_file(self) -> h5py.File:
        """Get HDF5 file handle (lazy loading)."""
        if self._hdf5_file is None:
            self._hdf5_file = h5py.File(self.hdf5_path, "r")
        return self._hdf5_file

    def __len__(self) -> int:
        return len(self.study_ids)

    def __getitem__(self, idx: int) -> dict:
        """
        Get a sample.

        Returns:
            dict with:
                - image: [3, H, W] tensor
                - text_tokens: [512] long tensor
                - attention_mask: [512] float tensor
                - structured: [num_features] float tensor
                - labels: [12] float tensor (multi-hot pathology labels)
                - label_mask: [12] float tensor (1.0 for valid, 0.0 for uncertain/missing)
                - study_id: int
                - subject_id: int
        """
        f = self._get_hdf5_file()

        # Get study info
        row = self.index_df.iloc[idx]
        hdf5_idx = str(int(row["idx"]))
        study_id = int(row["study_id"])
        subject_id = int(row["subject_id"])

        # Load image
        image = f["images"][hdf5_idx][:]
        if image.ndim == 2:
            image = image[np.newaxis, ...]
        image = torch.from_numpy(image).float()
        if self.transform is not None:
            image = self.transform(image.squeeze(0))

        # Get labels and mask
        labels = self.labels.get(study_id, torch.zeros(len(self.PATHOLOGY_LABELS)))
        label_mask = self.label_masks.get(study_id, torch.zeros(len(self.PATHOLOGY_LABELS)))

        # Get structured features
        structured = self._extract_structured_tensor(study_id)

        # Get text tokens
        text_tokens, attention_mask = self._extract_text_tokens(study_id)

        return {
            "image": image,
            "text_tokens": text_tokens,
            "attention_mask": attention_mask,
            "structured": structured,
            "labels": labels,
            "label_mask": label_mask,
            "study_id": study_id,
            "subject_id": subject_id,
        }

    def _extract_structured_tensor(self, study_id: int) -> torch.Tensor:
        """Extract structured features as a tensor."""
        features = []

        if self.structured_df is not None and study_id in self.structured_df.index:
            row = self.structured_df.loc[study_id]
            for feat_name in self.STRUCTURED_FEATURES:
                val = row.get(feat_name, np.nan)
                # Convert NaN to 0.0
                features.append(float(val) if pd.notna(val) else 0.0)
        else:
            features = [0.0] * len(self.STRUCTURED_FEATURES)

        return torch.tensor(features, dtype=torch.float32)

    def _extract_text_tokens(self, study_id: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Extract text tokens and attention mask."""
        max_length = 512

        if self.text_df is not None and study_id in self.text_df.index:
            row = self.text_df.loc[study_id]
            tokens_str = row.get("tokens", "")

            if tokens_str and isinstance(tokens_str, str):
                token_ids = [int(t) for t in tokens_str.split(",") if t.strip()]

                # Create attention mask (1 for real tokens, 0 for padding)
                actual_length = min(len(token_ids), max_length)

                # Pad or truncate
                if len(token_ids) < max_length:
                    padding_length = max_length - len(token_ids)
                    token_ids = token_ids + [0] * padding_length
                else:
                    token_ids = token_ids[:max_length]

                tokens = torch.tensor(token_ids, dtype=torch.long)
                attention_mask = torch.zeros(max_length, dtype=torch.float32)
                attention_mask[:actual_length] = 1.0

                return tokens, attention_mask

        # No text data
        return torch.zeros(max_length, dtype=torch.long), torch.zeros(max_length, dtype=torch.float32)

    def get_label_frequencies(self) -> dict[str, float]:
        """Get frequency of each pathology label (for class weighting)."""
        frequencies = {}
        total_valid = {}

        for label_idx, label_name in enumerate(self.PATHOLOGY_LABELS):
            pos_count = 0
            valid_count = 0

            for study_id in self.study_ids:
                if study_id in self.labels and study_id in self.label_masks:
                    mask_val = self.label_masks[study_id][label_idx].item()
                    if mask_val > 0:
                        valid_count += 1
                        if self.labels[study_id][label_idx].item() > 0:
                            pos_count += 1

            if valid_count > 0:
                frequencies[label_name] = pos_count / valid_count
            else:
                frequencies[label_name] = 0.0
            total_valid[label_name] = valid_count

        return frequencies

    def get_pos_weights(self) -> torch.Tensor:
        """Get positive class weights for BCEWithLogitsLoss."""
        frequencies = self.get_label_frequencies()
        weights = []

        for label_name in self.PATHOLOGY_LABELS:
            freq = frequencies[label_name]
            if freq > 0 and freq < 1:
                # Weight = (1 - freq) / freq to balance classes
                weight = (1 - freq) / freq
            else:
                weight = 1.0
            weights.append(weight)

        return torch.tensor(weights, dtype=torch.float32)

    def __del__(self):
        """Clean up HDF5 file handle."""
        if self._hdf5_file is not None:
            try:
                self._hdf5_file.close()
            except Exception:
                pass


def collate_multimodal(batch: list[dict]) -> dict:
    """
    Custom collate function for DataLoader.

    Stacks all tensors and preserves scalar metadata.
    """
    return {
        "image": torch.stack([b["image"] for b in batch]),
        "text_tokens": torch.stack([b["text_tokens"] for b in batch]),
        "attention_mask": torch.stack([b["attention_mask"] for b in batch]),
        "structured": torch.stack([b["structured"] for b in batch]),
        "labels": torch.stack([b["labels"] for b in batch]),
        "label_mask": torch.stack([b["label_mask"] for b in batch]),
        "study_id": torch.tensor([b["study_id"] for b in batch]),
        "subject_id": torch.tensor([b["subject_id"] for b in batch]),
    }
