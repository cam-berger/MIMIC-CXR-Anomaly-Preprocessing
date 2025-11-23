"""
HDF5 writer for efficient image storage with memory-mapped loading support.

Stores images in chunked HDF5 format optimized for:
- Random access during training (memory-mapped)
- Compression to reduce storage
- Batch organization for Lambda deployment
"""
import h5py
import numpy as np
from pathlib import Path
from typing import Dict, Optional, List
import logging
import hashlib

logger = logging.getLogger(__name__)


class HDF5Writer:
    """
    Writer for storing images in HDF5 format with batch support.

    Features:
    - Chunked storage for efficient random access
    - Compression (gzip, lz4, etc.)
    - Batch organization (multiple files or groups)
    - Memory-mapped loading support
    - Integrity checksums
    """

    def __init__(
        self,
        output_dir: Path,
        compression: str = "gzip",
        compression_opts: Optional[int] = None,
        chunk_size: int = 100
    ):
        """
        Initialize HDF5 writer.

        Args:
            output_dir: Directory for output HDF5 files
            compression: Compression algorithm ("gzip", "lzf", None)
            compression_opts: Compression level (1-9 for gzip)
            chunk_size: Samples per chunk (for memory-mapped access)
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.compression = compression
        self.compression_opts = compression_opts if compression == "gzip" else None
        self.chunk_size = chunk_size

        self.current_file = None
        self.current_batch_id = None

        logger.info(f"Initialized HDF5Writer")
        logger.info(f"  Output dir: {self.output_dir}")
        logger.info(f"  Compression: {self.compression}")
        logger.info(f"  Chunk size: {self.chunk_size}")

    def create_batch_file(
        self,
        batch_id: int,
        split: str = "train",
        estimated_samples: Optional[int] = None
    ) -> h5py.File:
        """
        Create a new HDF5 file for a batch.

        Args:
            batch_id: Batch identifier (e.g., 0, 1, 2, ...)
            split: Dataset split ("train" or "val")
            estimated_samples: Estimated number of samples (for pre-allocation)

        Returns:
            Open HDF5 file handle
        """
        filename = self.output_dir / split / f"batch_{batch_id:04d}" / "images.h5"
        filename.parent.mkdir(parents=True, exist_ok=True)

        logger.info(f"Creating HDF5 batch file: {filename}")

        h5file = h5py.File(filename, 'w')

        # Store metadata as attributes
        h5file.attrs['batch_id'] = batch_id
        h5file.attrs['split'] = split
        h5file.attrs['compression'] = self.compression or "none"
        h5file.attrs['chunk_size'] = self.chunk_size

        if estimated_samples is not None:
            h5file.attrs['estimated_samples'] = estimated_samples

        self.current_file = h5file
        self.current_batch_id = batch_id

        return h5file

    def write_image(
        self,
        h5file: h5py.File,
        sample_id: str,
        image: np.ndarray,
        metadata: Optional[Dict] = None
    ):
        """
        Write a single image to HDF5 file.

        Args:
            h5file: Open HDF5 file handle
            sample_id: Unique sample identifier (e.g., "study_12345")
            image: Image array (C, H, W) or (H, W)
            metadata: Optional metadata dict to store as attributes
        """
        # Determine chunk shape based on image dimensions
        if image.ndim == 3:
            # (C, H, W) format
            chunk_shape = (1, image.shape[1], image.shape[2])
        elif image.ndim == 2:
            # (H, W) format
            chunk_shape = (image.shape[0], image.shape[1])
        else:
            raise ValueError(f"Image must be 2D or 3D, got shape {image.shape}")

        # Create dataset with chunking and compression
        dataset = h5file.create_dataset(
            f"images/{sample_id}",
            data=image,
            chunks=chunk_shape if image.nbytes > 1_000_000 else None,  # Only chunk large images
            compression=self.compression,
            compression_opts=self.compression_opts,
            dtype=image.dtype
        )

        # Store metadata as attributes
        if metadata:
            for key, value in metadata.items():
                # HDF5 attrs only support basic types
                if isinstance(value, (str, int, float, bool)):
                    dataset.attrs[key] = value
                elif isinstance(value, (list, tuple)):
                    # Convert lists to strings
                    dataset.attrs[key] = str(value)

        # Add shape and dtype info
        dataset.attrs['shape'] = image.shape
        dataset.attrs['dtype'] = str(image.dtype)
        dataset.attrs['size_bytes'] = image.nbytes

    def write_batch(
        self,
        batch_id: int,
        samples: List[Dict],
        split: str = "train"
    ) -> Path:
        """
        Write a batch of images to HDF5 file.

        Args:
            batch_id: Batch identifier
            samples: List of sample dicts with 'sample_id', 'image', 'metadata'
            split: Dataset split ("train" or "val")

        Returns:
            Path to created HDF5 file
        """
        logger.info(f"Writing batch {batch_id} with {len(samples)} samples to HDF5")

        # Create batch file
        h5file = self.create_batch_file(batch_id, split, len(samples))

        # Write each image
        for idx, sample in enumerate(samples):
            sample_id = sample['sample_id']
            image = sample['image']
            metadata = sample.get('metadata', {})

            try:
                self.write_image(h5file, sample_id, image, metadata)

                if (idx + 1) % 100 == 0:
                    logger.info(f"  Written {idx + 1}/{len(samples)} images")

            except Exception as e:
                logger.error(f"  Failed to write image for sample {sample_id}: {e}")
                continue

        # Store batch-level metadata
        h5file.attrs['total_samples'] = len(samples)
        h5file.attrs['successful_writes'] = len(h5file['images'].keys())

        # Close file
        file_path = Path(h5file.filename)
        h5file.close()

        logger.info(f"  HDF5 batch file created: {file_path}")
        logger.info(f"  File size: {file_path.stat().st_size / 1024 / 1024:.1f} MB")

        return file_path

    def compute_checksum(self, file_path: Path) -> str:
        """
        Compute SHA256 checksum for HDF5 file.

        Args:
            file_path: Path to HDF5 file

        Returns:
            SHA256 hex digest
        """
        sha256 = hashlib.sha256()

        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b''):
                sha256.update(chunk)

        return sha256.hexdigest()

    def get_batch_stats(self, file_path: Path) -> Dict:
        """
        Get statistics for a batch HDF5 file.

        Args:
            file_path: Path to HDF5 file

        Returns:
            Dictionary with batch statistics
        """
        with h5py.File(file_path, 'r') as h5file:
            num_samples = len(h5file['images'].keys())

            # Calculate total size
            total_bytes = 0
            for sample_id in h5file['images'].keys():
                dataset = h5file[f'images/{sample_id}']
                total_bytes += dataset.attrs.get('size_bytes', dataset.nbytes)

            # Get sample shapes
            shapes = []
            for sample_id in h5file['images'].keys():
                dataset = h5file[f'images/{sample_id}']
                shapes.append(tuple(dataset.attrs['shape']))

            stats = {
                'num_samples': num_samples,
                'total_size_mb': total_bytes / 1024 / 1024,
                'avg_size_mb': (total_bytes / num_samples) / 1024 / 1024 if num_samples > 0 else 0,
                'unique_shapes': list(set(shapes)),
                'batch_id': h5file.attrs.get('batch_id', -1),
                'split': h5file.attrs.get('split', 'unknown')
            }

            return stats

    def close(self):
        """Close current HDF5 file if open"""
        if self.current_file is not None:
            self.current_file.close()
            self.current_file = None
            self.current_batch_id = None


class HDF5Reader:
    """
    Reader for loading images from pre-compiled HDF5 files.

    Supports:
    - Memory-mapped loading (lazy loading)
    - Multi-batch datasets
    - Random access by sample ID
    """

    def __init__(self, hdf5_paths: List[Path]):
        """
        Initialize HDF5 reader.

        Args:
            hdf5_paths: List of paths to HDF5 files
        """
        self.hdf5_paths = [Path(p) for p in hdf5_paths]
        self.h5files = {}  # Lazy-loaded file handles
        self.sample_index = {}  # Map sample_id -> (file_path, dataset_path)

        logger.info(f"Initialized HDF5Reader with {len(self.hdf5_paths)} files")

        # Build sample index
        self._build_index()

    def _build_index(self):
        """Build index of sample IDs across all HDF5 files"""
        logger.info("Building sample index...")

        for file_path in self.hdf5_paths:
            with h5py.File(file_path, 'r') as h5file:
                for sample_id in h5file['images'].keys():
                    self.sample_index[sample_id] = (
                        file_path,
                        f"images/{sample_id}"
                    )

        logger.info(f"  Indexed {len(self.sample_index)} samples")

    def load_image(self, sample_id: str) -> Optional[np.ndarray]:
        """
        Load image for a specific sample ID.

        Args:
            sample_id: Sample identifier

        Returns:
            Image array or None if not found
        """
        if sample_id not in self.sample_index:
            logger.warning(f"Sample {sample_id} not found in index")
            return None

        file_path, dataset_path = self.sample_index[sample_id]

        # Lazy-load HDF5 file
        if file_path not in self.h5files:
            self.h5files[file_path] = h5py.File(file_path, 'r')

        h5file = self.h5files[file_path]
        image = h5file[dataset_path][:]  # Load to memory

        return image

    def close(self):
        """Close all open HDF5 files"""
        for h5file in self.h5files.values():
            h5file.close()
        self.h5files = {}
