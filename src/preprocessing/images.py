"""
Image preprocessing for CXR images.

Processes JPEG images from MIMIC-CXR-JPG and saves to HDF5 format
with chunking and compression for efficient storage and loading.

Output HDF5 structure:
    /images/{idx}  - Image tensor [C, H, W]
    /metadata/{idx} - JSON metadata string
    /index - DataFrame with study_id -> idx mapping
"""

import logging
from pathlib import Path
from typing import Optional
from concurrent.futures import ProcessPoolExecutor, as_completed
import json

import numpy as np
import h5py
import pandas as pd
from PIL import Image
from tqdm import tqdm

from ..config import Settings, get_settings
from ..datasets import MIMICCXRLoader

logger = logging.getLogger(__name__)


def normalize_image(
    img: np.ndarray,
    method: str = "minmax",
) -> np.ndarray:
    """
    Normalize image array.

    Args:
        img: Image array [H, W] or [H, W, C]
        method: "minmax" (0-1), "standardize" (mean=0, std=1), or "none"

    Returns:
        Normalized image as float32
    """
    img = img.astype(np.float32)

    if method == "minmax":
        img_min = img.min()
        img_max = img.max()
        if img_max > img_min:
            img = (img - img_min) / (img_max - img_min)
        else:
            img = np.zeros_like(img)
    elif method == "standardize":
        mean = img.mean()
        std = img.std()
        if std > 0:
            img = (img - mean) / std
        else:
            img = img - mean
    # "none" - just convert to float32

    return img


def load_and_process_image(
    image_path: Path,
    normalize_method: str = "minmax",
) -> Optional[np.ndarray]:
    """
    Load and preprocess a single CXR image.

    Args:
        image_path: Path to JPEG image
        normalize_method: Normalization method

    Returns:
        Processed image array [C, H, W] or None if failed
    """
    try:
        # Load image
        with Image.open(image_path) as img:
            # Convert to grayscale if needed
            if img.mode != "L":
                img = img.convert("L")

            # Convert to numpy array
            arr = np.array(img)

        # Normalize
        arr = normalize_image(arr, normalize_method)

        # Add channel dimension [H, W] -> [1, H, W]
        arr = arr[np.newaxis, ...]

        return arr

    except Exception as e:
        logger.warning(f"Failed to load image {image_path}: {e}")
        return None


def _process_batch_worker(args: tuple) -> list[dict]:
    """Worker function for parallel processing."""
    batch_data, cxr_base_path, normalize_method = args
    results = []

    for row in batch_data:
        subject_id = row["subject_id"]
        study_id = row["study_id"]

        # Construct image path
        subject_prefix = str(subject_id)[:2]
        study_dir = (
            Path(cxr_base_path)
            / "files"
            / f"p{subject_prefix}"
            / f"p{subject_id}"
            / f"s{study_id}"
        )

        # Find frontal view (PA preferred over AP)
        image_path = None
        for view in ["PA", "AP"]:
            candidates = list(study_dir.glob(f"*{view}*.jpg")) if study_dir.exists() else []
            if candidates:
                image_path = candidates[0]
                break

        # Fall back to any image
        if image_path is None and study_dir.exists():
            images = list(study_dir.glob("*.jpg"))
            if images:
                image_path = images[0]

        if image_path is None:
            results.append({
                "study_id": study_id,
                "subject_id": subject_id,
                "success": False,
                "error": "No image found",
                "image": None,
            })
            continue

        # Load and process
        arr = load_and_process_image(image_path, normalize_method)

        if arr is not None:
            results.append({
                "study_id": study_id,
                "subject_id": subject_id,
                "success": True,
                "error": None,
                "image": arr,
                "shape": arr.shape,
                "image_path": str(image_path),
            })
        else:
            results.append({
                "study_id": study_id,
                "subject_id": subject_id,
                "success": False,
                "error": "Failed to load",
                "image": None,
            })

    return results


class ImagePreprocessor:
    """Batch image preprocessing with HDF5 output."""

    def __init__(self, settings: Optional[Settings] = None):
        """Initialize preprocessor."""
        self.settings = settings or get_settings()
        self.config = self.settings.preprocessing

    def process_cohort(
        self,
        cohort: pd.DataFrame,
        output_path: Path,
        num_workers: Optional[int] = None,
        batch_size: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Process all images in a cohort and save to HDF5.

        Uses streaming writes to HDF5 to avoid memory accumulation.

        Args:
            cohort: Cohort DataFrame with subject_id, study_id
            output_path: Path for output HDF5 file
            num_workers: Number of parallel workers (uses config if None)
            batch_size: Batch size for processing (uses config if None)

        Returns:
            DataFrame with processing results (study_id, success, error, shape)
        """
        num_workers = num_workers if num_workers is not None else self.config.num_workers
        batch_size = batch_size if batch_size is not None else self.config.batch_size

        logger.info(f"Processing {len(cohort):,} images with {num_workers} workers...")

        # Prepare data for parallel processing
        cohort_data = cohort[["subject_id", "study_id"]].to_dict("records")

        # Split into batches
        batches = [
            cohort_data[i : i + batch_size]
            for i in range(0, len(cohort_data), batch_size)
        ]

        # Prepare worker args
        cxr_base = str(self.settings.paths.mimic_cxr_jpg)
        normalize = self.config.image_normalize
        worker_args = [(batch, cxr_base, normalize) for batch in batches]

        # Stream results to HDF5 to avoid memory accumulation
        output_path.parent.mkdir(parents=True, exist_ok=True)
        all_results = []
        index_data = []
        image_idx = 0
        successful_count = 0

        with h5py.File(output_path, "w") as f:
            # Create groups
            img_group = f.create_group("images")
            meta_group = f.create_group("metadata")

            # Use sequential processing if workers <= 0, otherwise parallel
            if num_workers <= 0:
                # Sequential processing (no multiprocessing)
                logger.info("Using sequential processing (no multiprocessing)")
                for args in tqdm(worker_args, desc="Processing batches"):
                    batch_results = _process_batch_worker(args)
                    for result in batch_results:
                        all_results.append({
                            "study_id": result["study_id"],
                            "subject_id": result["subject_id"],
                            "success": result["success"],
                            "error": result.get("error"),
                            "shape": str(result.get("shape", "")),
                        })
                        if result["success"]:
                            # Write immediately to HDF5
                            self._write_image_to_hdf5(
                                img_group, meta_group, result, image_idx
                            )
                            index_data.append({
                                "idx": image_idx,
                                "study_id": result["study_id"],
                                "subject_id": result["subject_id"],
                            })
                            image_idx += 1
                            successful_count += 1
            else:
                # Parallel processing
                with ProcessPoolExecutor(max_workers=num_workers) as executor:
                    futures = {
                        executor.submit(_process_batch_worker, args): i
                        for i, args in enumerate(worker_args)
                    }

                    with tqdm(total=len(futures), desc="Processing batches") as pbar:
                        for future in as_completed(futures):
                            batch_results = future.result()
                            for result in batch_results:
                                all_results.append({
                                    "study_id": result["study_id"],
                                    "subject_id": result["subject_id"],
                                    "success": result["success"],
                                    "error": result.get("error"),
                                    "shape": str(result.get("shape", "")),
                                })
                                if result["success"]:
                                    # Write immediately to HDF5
                                    self._write_image_to_hdf5(
                                        img_group, meta_group, result, image_idx
                                    )
                                    index_data.append({
                                        "idx": image_idx,
                                        "study_id": result["study_id"],
                                        "subject_id": result["subject_id"],
                                    })
                                    image_idx += 1
                                    successful_count += 1
                            pbar.update(1)

            # Save index as dataset at the end
            logger.info(f"Writing index for {successful_count:,} images...")
            index_df = pd.DataFrame(index_data)
            index_bytes = index_df.to_parquet()
            f.create_dataset("index", data=np.frombuffer(index_bytes, dtype=np.uint8))

        logger.info(f"Saved HDF5 to {output_path}")

        results_df = pd.DataFrame(all_results)
        success_rate = results_df["success"].mean() * 100
        logger.info(f"Image processing complete: {success_rate:.1f}% success rate")

        return results_df

    def _write_image_to_hdf5(
        self,
        img_group: h5py.Group,
        meta_group: h5py.Group,
        result: dict,
        idx: int,
    ) -> None:
        """Write a single image to HDF5 immediately (streaming write)."""
        image = result["image"]

        # Save image with compression
        img_group.create_dataset(
            str(idx),
            data=image,
            compression=self.config.image_compression,
            compression_opts=self.config.image_compression_level,
            chunks=True,
        )

        # Save metadata
        metadata = {
            "study_id": int(result["study_id"]),
            "subject_id": int(result["subject_id"]),
            "shape": list(image.shape),
            "image_path": result.get("image_path", ""),
        }
        meta_group.create_dataset(
            str(idx),
            data=json.dumps(metadata),
        )

    def _save_to_hdf5(
        self,
        images: list[dict],
        output_path: Path,
    ) -> None:
        """
        Save processed images to HDF5 file.

        Args:
            images: List of image results with 'image', 'study_id', etc.
            output_path: Output HDF5 file path
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Build index mapping
        index_data = []

        with h5py.File(output_path, "w") as f:
            # Create groups
            img_group = f.create_group("images")
            meta_group = f.create_group("metadata")

            for idx, result in enumerate(tqdm(images, desc="Writing HDF5")):
                study_id = result["study_id"]
                subject_id = result["subject_id"]
                image = result["image"]

                # Save image with compression
                img_group.create_dataset(
                    str(idx),
                    data=image,
                    compression=self.config.image_compression,
                    compression_opts=self.config.image_compression_level,
                    chunks=True,
                )

                # Save metadata
                metadata = {
                    "study_id": int(study_id),
                    "subject_id": int(subject_id),
                    "shape": list(image.shape),
                    "image_path": result.get("image_path", ""),
                }
                meta_group.create_dataset(
                    str(idx),
                    data=json.dumps(metadata),
                )

                index_data.append({
                    "idx": idx,
                    "study_id": study_id,
                    "subject_id": subject_id,
                })

            # Save index as dataset
            index_df = pd.DataFrame(index_data)
            index_bytes = index_df.to_parquet()
            f.create_dataset("index", data=np.frombuffer(index_bytes, dtype=np.uint8))

        logger.info(f"Saved HDF5 to {output_path}")

    @staticmethod
    def load_from_hdf5(
        hdf5_path: Path,
        study_ids: Optional[set[int]] = None,
    ) -> tuple[dict[int, np.ndarray], pd.DataFrame]:
        """
        Load images from HDF5 file.

        Args:
            hdf5_path: Path to HDF5 file
            study_ids: Specific study IDs to load (all if None)

        Returns:
            (images dict mapping study_id -> array, index DataFrame)
        """
        images = {}

        with h5py.File(hdf5_path, "r") as f:
            # Load index
            index_bytes = f["index"][:]
            import io
            index_df = pd.read_parquet(io.BytesIO(bytes(index_bytes)))

            # Filter if specific studies requested
            if study_ids is not None:
                index_df = index_df[index_df["study_id"].isin(study_ids)]

            # Load images
            for _, row in tqdm(index_df.iterrows(), total=len(index_df), desc="Loading images"):
                idx = row["idx"]
                study_id = row["study_id"]
                images[study_id] = f["images"][str(idx)][:]

        return images, index_df
