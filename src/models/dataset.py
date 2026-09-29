"""
PyTorch Dataset for loading preprocessed MIMIC-CXR images from HDF5.

Supports the preprocessed data format as documented in PREPROCESSED_DATA_SCHEMA.md:
- images.h5: HDF5 file with images and index
- structured.parquet: Clinical features (demographics, vitals, labs)
- text.parquet: Radiology reports and summaries
"""

import io
import json
import logging
from pathlib import Path
from typing import Optional, Callable, Union, List

import h5py
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T

from .config import IMAGE_MODES

logger = logging.getLogger(__name__)

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def build_image_transform(
    target_size: tuple[int, int] = (224, 224),
    mode: str = "resize",
    training: bool = False,
    normalize: bool = True,
    crop_scale: Optional[tuple[float, float]] = None,
    horizontal_flip: bool = True,
    rotation_degrees: int = 15,
    gaussian_blur: bool = True,
) -> Callable:
    """
    Image transform used for training, validation and inference.

    Input: grayscale image [H, W] float32 in [0, 1] (native resolution, e.g. 3056x2544)
    Output: RGB tensor [3, target_size] (ImageNet-normalized if ``normalize``)

    Modes:
        resize: the whole radiograph is resized to ``target_size``, so lung
            apices, costophrenic angles and lateral fields stay in view. In
            training, ``crop_scale`` switches to RandomResizedCrop.
        center_crop: crops ``target_size`` pixels from the native-resolution
            image (~13% of the image at 1024, ~0.65% at 224). Only for
            reproducing models trained that way.

    Args:
        target_size: Output size (H, W)
        mode: One of IMAGE_MODES
        training: Apply random augmentations
        normalize: Apply ImageNet normalization
        crop_scale: RandomResizedCrop area range for training in "resize" mode
        horizontal_flip, rotation_degrees, gaussian_blur: Training augmentations
    """
    if mode not in IMAGE_MODES:
        raise ValueError(f"Unknown image mode {mode!r}; expected one of {IMAGE_MODES}")

    transforms = [T.ToPILImage()]
    if mode == "center_crop":
        transforms.append(T.CenterCrop(target_size))
    elif training and crop_scale is not None:
        transforms.append(T.RandomResizedCrop(target_size, scale=tuple(crop_scale), antialias=True))
    else:
        transforms.append(T.Resize(target_size, antialias=True))

    if training:
        if horizontal_flip:
            transforms.append(T.RandomHorizontalFlip(p=0.5))
        if rotation_degrees:
            transforms.append(T.RandomRotation(degrees=rotation_degrees))
        if gaussian_blur:
            transforms.append(T.RandomApply([T.GaussianBlur(kernel_size=23)], p=0.5))

    transforms += [
        T.Grayscale(num_output_channels=3),  # 1 channel -> 3 channels for ViT
        T.ToTensor(),
    ]
    if normalize:
        transforms.append(T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD))

    return T.Compose(transforms)


class MIMICCXRDataset(Dataset):
    """
    Dataset for loading preprocessed chest X-ray images from HDF5.

    Matches the preprocessed data schema:
        output/preprocessed/{cohort_name}/
        ├── images.h5              # HDF5 with /images/{idx}, /metadata/{idx}, /index
        ├── structured.parquet     # Clinical features
        ├── text.parquet           # Reports and summaries
        └── manifest.json          # Processing statistics

    Image format in HDF5:
        - Shape: [1, H, W] (grayscale with channel dim)
        - Dtype: float32
        - Normalization: Min-max scaled to [0, 1]
        - Resolution: Full resolution (variable, e.g., 2544x3056)

    Args:
        preprocessed_dir: Path to preprocessed cohort directory (e.g., output/preprocessed/normal_train)
        transform: Optional transform to apply to images
        target_size: Target image size (H, W) for resizing, default (224, 224)
        normalize: Whether to apply ImageNet normalization
        include_structured: Whether to load structured features
        include_text: Whether to load text features
        return_metadata: Whether to return full metadata dict
    """

    def __init__(
        self,
        preprocessed_dir: Union[str, Path],
        transform: Optional[Callable] = None,
        target_size: tuple[int, int] = (224, 224),
        normalize: bool = True,
        include_structured: bool = False,
        include_text: bool = False,
        return_metadata: bool = False,
    ):
        self.preprocessed_dir = Path(preprocessed_dir)
        self.transform = transform
        self.target_size = target_size
        self.normalize = normalize
        self.include_structured = include_structured
        self.include_text = include_text
        self.return_metadata = return_metadata

        # Paths
        self.hdf5_path = self.preprocessed_dir / "images.h5"
        self.structured_path = self.preprocessed_dir / "structured.parquet"
        self.text_path = self.preprocessed_dir / "text.parquet"

        # Validate HDF5 exists
        if not self.hdf5_path.exists():
            raise FileNotFoundError(f"HDF5 file not found: {self.hdf5_path}")

        # Load index from HDF5
        self._load_index()

        # Load structured data if requested
        self.structured_df = None
        if self.include_structured and self.structured_path.exists():
            self.structured_df = pd.read_parquet(self.structured_path)
            self.structured_df = self.structured_df.set_index("study_id")
            logger.info(f"Loaded structured data: {len(self.structured_df)} rows")

        # Load text data if requested
        self.text_df = None
        if self.include_text and self.text_path.exists():
            self.text_df = pd.read_parquet(self.text_path)
            self.text_df = self.text_df.set_index("study_id")
            logger.info(f"Loaded text data: {len(self.text_df)} rows")

        # Build default transform if none provided
        if self.transform is None:
            self.transform = self._build_default_transform()

        # Lazy HDF5 file handle (for multiprocessing)
        self._hdf5_file = None

    def _load_index(self) -> None:
        """Load the index mapping from HDF5 file."""
        with h5py.File(self.hdf5_path, "r") as f:
            if "index" not in f:
                raise ValueError("HDF5 file missing 'index' dataset")

            index_bytes = f["index"][:]
            self.index_df = pd.read_parquet(io.BytesIO(bytes(index_bytes)))

        self.study_ids = self.index_df["study_id"].tolist()
        self.idx_to_study = dict(zip(self.index_df["idx"], self.index_df["study_id"]))
        self.study_to_idx = dict(zip(self.index_df["study_id"], self.index_df["idx"]))

        logger.info(f"Loaded index with {len(self.study_ids)} images")

    def _build_default_transform(self) -> Callable:
        """Default transform: resize the full image to target_size (no augmentation)."""
        return build_image_transform(
            self.target_size, mode="resize", training=False, normalize=self.normalize,
        )

    def _get_hdf5_file(self) -> h5py.File:
        """Get HDF5 file handle (lazy loading for multiprocessing)."""
        if self._hdf5_file is None:
            self._hdf5_file = h5py.File(self.hdf5_path, "r")
        return self._hdf5_file

    def __len__(self) -> int:
        return len(self.study_ids)

    def __getitem__(self, idx: int) -> Union[torch.Tensor, dict]:
        """
        Get a sample by index.

        Args:
            idx: Sample index

        Returns:
            If include_structured/include_text/return_metadata=False:
                image tensor [3, H, W]
            Otherwise:
                dict with 'image' and optional 'structured', 'text', 'metadata'
        """
        f = self._get_hdf5_file()

        # Get study info from index
        row = self.index_df.iloc[idx]
        hdf5_idx = str(int(row["idx"]))
        study_id = int(row["study_id"])
        subject_id = int(row["subject_id"])

        # Load image from HDF5 - Shape: [1, H, W], float32, [0, 1]
        image = f["images"][hdf5_idx][:]

        # Ensure correct shape
        if image.ndim == 2:
            image = image[np.newaxis, ...]

        # Convert to torch tensor
        image = torch.from_numpy(image).float()

        # Apply transforms (expects [H, W] for ToPILImage)
        if self.transform is not None:
            image = self.transform(image.squeeze(0))

        # Simple return if no extra data requested
        if not (self.include_structured or self.include_text or self.return_metadata):
            return image

        # Build result dict
        result = {
            "image": image,
            "study_id": study_id,
            "subject_id": subject_id,
        }

        # Add structured features
        if self.include_structured and self.structured_df is not None:
            if study_id in self.structured_df.index:
                struct_row = self.structured_df.loc[study_id]
                result["structured"] = self._extract_structured_features(struct_row)
            else:
                result["structured"] = self._get_empty_structured_features()

        # Add text features
        if self.include_text and self.text_df is not None:
            if study_id in self.text_df.index:
                text_row = self.text_df.loc[study_id]
                result["text"] = self._extract_text_features(text_row)
            else:
                result["text"] = self._get_empty_text_features()

        # Add full metadata
        if self.return_metadata:
            if "metadata" in f and hdf5_idx in f["metadata"]:
                meta_json = f["metadata"][hdf5_idx][()]
                if isinstance(meta_json, bytes):
                    meta_json = meta_json.decode("utf-8")
                result["metadata"] = json.loads(meta_json)
            else:
                result["metadata"] = {}

        return result

    def _extract_structured_features(self, row: pd.Series) -> dict:
        """Extract structured features from parquet row."""
        # Demographics
        features = {
            "age": float(row.get("age", 0)) if pd.notna(row.get("age")) else 0.0,
            "gender_M": int(row.get("gender_M", 0)) if pd.notna(row.get("gender_M")) else 0,
        }

        # Triage vitals
        triage_cols = [
            "triage_temperature", "triage_heartrate", "triage_resprate",
            "triage_o2sat", "triage_sbp", "triage_dbp", "triage_acuity"
        ]
        for col in triage_cols:
            val = row.get(col)
            features[col] = float(val) if pd.notna(val) else 0.0

        # ED vitals (aggregated)
        vital_types = ["temperature", "heartrate", "resprate", "o2sat", "sbp", "dbp"]
        for vital in vital_types:
            for stat in ["mean", "min", "max"]:
                col = f"{vital}_{stat}"
                val = row.get(col)
                features[col] = float(val) if pd.notna(val) else 0.0

        # Labs
        lab_types = [
            "bicarbonate", "bnp", "bun", "calcium", "chloride", "creatinine",
            "glucose", "hematocrit", "hemoglobin", "lactate", "magnesium",
            "platelets", "potassium", "sodium", "troponin", "wbc"
        ]
        for lab in lab_types:
            col = f"lab_{lab}_mean"
            val = row.get(col)
            features[col] = float(val) if pd.notna(val) else 0.0

        # Availability flags
        for flag in ["has_triage", "has_labs", "has_ed_vitals"]:
            features[flag] = bool(row.get(flag, False))

        return features

    def _get_empty_structured_features(self) -> dict:
        """Return empty structured features dict."""
        return {
            "age": 0.0,
            "gender_M": 0,
            "has_triage": False,
            "has_labs": False,
            "has_ed_vitals": False,
        }

    def _extract_text_features(self, row: pd.Series) -> dict:
        """Extract text features from parquet row."""
        # Parse comma-separated tokens to tensor
        tokens_str = row.get("tokens", "")
        if tokens_str and isinstance(tokens_str, str):
            token_ids = [int(t) for t in tokens_str.split(",") if t.strip()]
            # Pad or truncate to 512
            if len(token_ids) < 512:
                token_ids = token_ids + [0] * (512 - len(token_ids))
            else:
                token_ids = token_ids[:512]
            tokens = torch.tensor(token_ids, dtype=torch.long)
        else:
            tokens = torch.zeros(512, dtype=torch.long)

        return {
            "tokens": tokens,
            "summary": str(row.get("summary", "")),
            "report": str(row.get("report", "")),
            "token_count": int(row.get("token_count", 0)),
            "has_report": bool(row.get("has_report", False)),
        }

    def _get_empty_text_features(self) -> dict:
        """Return empty text features dict."""
        return {
            "tokens": torch.zeros(512, dtype=torch.long),
            "summary": "",
            "report": "",
            "token_count": 0,
            "has_report": False,
        }

    def get_by_study_id(self, study_id: int) -> Union[torch.Tensor, dict]:
        """Get sample by study_id."""
        if study_id not in self.study_to_idx:
            raise KeyError(f"Study ID {study_id} not found in dataset")

        idx = self.index_df[self.index_df["study_id"] == study_id].index[0]
        return self[idx]

    def __del__(self):
        """Clean up HDF5 file handle."""
        if self._hdf5_file is not None:
            try:
                self._hdf5_file.close()
            except Exception:
                pass


class PreprocessedMAEDataset(Dataset):
    """
    Convenience dataset for MAE training that loads images only.

    Simplified wrapper around MIMICCXRDataset optimized for MAE pretraining:
    - Only loads images (no structured/text data)
    - Applies MAE-specific augmentations
    - Returns tensors directly (not dicts)

    Args:
        preprocessed_dir: Path to preprocessed cohort directory
        training: Whether to apply training augmentations
        target_size: Target image size (H, W)
        image_mode: "resize" (full view) or "center_crop" (legacy)
        augmentation: Optional overrides for get_mae_augmentations (e.g.
            crop_scale, horizontal_flip, rotation_degrees, gaussian_blur)
    """

    def __init__(
        self,
        preprocessed_dir: Union[str, Path],
        training: bool = True,
        target_size: tuple[int, int] = (224, 224),
        image_mode: str = "resize",
        augmentation: Optional[dict] = None,
    ):
        self.training = training
        self.target_size = target_size

        # Build appropriate transform
        transform = get_mae_augmentations(
            target_size, training, mode=image_mode, **(augmentation or {})
        )

        # Create underlying dataset
        self.dataset = MIMICCXRDataset(
            preprocessed_dir,
            transform=transform,
            target_size=target_size,
            normalize=True,
            include_structured=False,
            include_text=False,
            return_metadata=False,
        )

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.dataset[idx]


def get_mae_augmentations(
    target_size: tuple[int, int] = (224, 224),
    training: bool = True,
    mode: str = "resize",
    crop_scale: Optional[tuple[float, float]] = (0.5, 1.0),
    horizontal_flip: bool = True,
    rotation_degrees: int = 15,
    gaussian_blur: bool = True,
) -> Callable:
    """
    Get augmentation pipeline for MAE training.

    Based on medical_mae recommendations (defaults mirror MAEConfig):
    - Random resized crop of the full image (area scale 0.5-1.0)
    - Horizontal flip
    - Light rotation (up to 15 degrees)
    - Gaussian blur

    Input: Grayscale image [H, W] float32 in [0, 1] (full resolution ~3056x2544)
    Output: RGB tensor [3, target_size, target_size] ImageNet-normalized

    Args:
        target_size: Target image size (H, W)
        training: Whether to apply training augmentations
        mode: "resize" (full view) or "center_crop" (legacy); see build_image_transform
        crop_scale, horizontal_flip, rotation_degrees, gaussian_blur: Training augmentations

    Returns:
        Transform function
    """
    return build_image_transform(
        target_size,
        mode=mode,
        training=training,
        normalize=True,
        crop_scale=crop_scale,
        horizontal_flip=horizontal_flip,
        rotation_degrees=rotation_degrees,
        gaussian_blur=gaussian_blur,
    )


# Backwards compatibility aliases
MIMICCXRHybridDataset = MIMICCXRDataset  # Alias for old name
