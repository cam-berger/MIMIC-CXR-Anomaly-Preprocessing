"""
Full-resolution image loading and preprocessing for MIMIC-CXR
Maintains native resolution (~3000x2500) as specified in Step 2
"""
import numpy as np
from PIL import Image
from pathlib import Path
import torch
from typing import Optional, Tuple, Dict
import logging
import sys

# Add parent directory to path for base imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from base.processor import ImageProcessor

logger = logging.getLogger(__name__)


class FullResolutionImageLoader(ImageProcessor):
    """
    Load and preprocess MIMIC-CXR images at full resolution.
    NO downsampling to preserve fine-grained details.
    """

    def __init__(self, config: Dict):
        super().__init__(config)
        self.normalize_method = config['image']['normalize_method']
        self.use_augmentation = config['image']['augmentation']['enabled']

        logger.info(f"Initialized FullResolutionImageLoader")
        logger.info(f"  Normalization: {self.normalize_method}")
        logger.info(f"  Augmentation: {self.use_augmentation}")

    def validate_config(self) -> None:
        """Validate image processing configuration"""
        # Check required keys exist
        self.get_config_value('image', 'normalize_method', required=True)
        self.get_config_value('image', 'augmentation', 'enabled', required=True)

        # Validate normalization method
        valid_methods = ['minmax', 'standardize', 'none']
        norm_method = self.get_config_value('image', 'normalize_method')
        if norm_method not in valid_methods:
            raise ValueError(f"normalize_method must be one of {valid_methods}, got '{norm_method}'")

    def process(self, image_path: Path, **kwargs) -> Optional[Dict]:
        """Process a single image (implements BaseProcessor.process)"""
        try:
            image_tensor = self.load_image(image_path)
            return {
                'image': image_tensor,
                'path': str(image_path),
                'stats': self.get_image_stats(image_tensor)
            }
        except Exception as e:
            self._handle_error(e, f"processing {image_path}")
            return None

    def load_and_process(self, image_path: Path, **kwargs) -> Optional[torch.Tensor]:
        """Load and process an image (implements ImageProcessor.load_and_process)"""
        return self.load_image(image_path)

    def load_image(self, image_path: Path) -> torch.Tensor:
        """
        Load a single image at full resolution.

        Args:
            image_path: Path to .jpg file

        Returns:
            torch.Tensor: Image tensor [C, H, W] at native resolution
        """
        try:
            # Load image with PIL
            img = Image.open(image_path)

            # Convert to numpy array
            img_array = np.array(img)

            # Handle grayscale
            if len(img_array.shape) == 2:
                img_array = img_array[..., np.newaxis]  # Add channel dimension

            # Normalize
            img_normalized = self._normalize(img_array)

            # Convert to tensor [H, W, C] -> [C, H, W]
            img_tensor = torch.from_numpy(img_normalized).permute(2, 0, 1).float()

            return img_tensor

        except Exception as e:
            logger.error(f"Error loading image {image_path}: {e}")
            raise

    def load_study_images(self, study_dir: Path) -> Dict[str, torch.Tensor]:
        """
        Load all images for a study (multiple views).

        Args:
            study_dir: Directory containing study images

        Returns:
            Dict mapping view position to image tensor
        """
        if not study_dir.exists():
            logger.warning(f"Study directory does not exist: {study_dir}")
            return {}

        images = {}
        for img_file in study_dir.glob("*.jpg"):
            view_name = img_file.stem  # Use filename as key
            images[view_name] = self.load_image(img_file)

        logger.debug(f"Loaded {len(images)} images from {study_dir.name}")
        return images

    def _normalize(self, image: np.ndarray) -> np.ndarray:
        """
        Normalize image intensities.

        Args:
            image: Raw image array (0-255)

        Returns:
            Normalized image array
        """
        image = image.astype(np.float32)

        if self.normalize_method == 'minmax':
            # Scale to [0, 1]
            return image / 255.0

        elif self.normalize_method == 'standardize':
            # Zero mean, unit variance
            mean = image.mean()
            std = image.std()
            if std > 0:
                return (image - mean) / std
            else:
                return image - mean

        elif self.normalize_method == 'none':
            return image

        else:
            raise ValueError(f"Unknown normalization method: {self.normalize_method}")

    def augment(self, image: torch.Tensor) -> torch.Tensor:
        """
        Apply augmentations at full resolution.

        Args:
            image: Image tensor [C, H, W]

        Returns:
            Augmented image tensor
        """
        if not self.use_augmentation:
            return image

        aug_config = self.config['image']['augmentation']

        # Convert to numpy for augmentation
        img_np = image.permute(1, 2, 0).numpy()

        # Rotation (small angle)
        if aug_config['rotation_range'] > 0:
            angle = np.random.uniform(
                -aug_config['rotation_range'],
                aug_config['rotation_range']
            )
            img_np = self._rotate(img_np, angle)

        # Brightness adjustment
        if aug_config['brightness_range'] > 0:
            factor = 1.0 + np.random.uniform(
                -aug_config['brightness_range'],
                aug_config['brightness_range']
            )
            img_np = np.clip(img_np * factor, 0, 1)

        # Contrast adjustment
        if aug_config['contrast_range'] > 0:
            mean_val = img_np.mean()
            factor = 1.0 + np.random.uniform(
                -aug_config['contrast_range'],
                aug_config['contrast_range']
            )
            img_np = np.clip((img_np - mean_val) * factor + mean_val, 0, 1)

        # Convert back to tensor
        return torch.from_numpy(img_np).permute(2, 0, 1).float()

    def _rotate(self, image: np.ndarray, angle: float) -> np.ndarray:
        """Rotate image by given angle (degrees)"""
        from scipy.ndimage import rotate
        return rotate(image, angle, reshape=False, order=1, mode='constant', cval=0)

    def get_image_stats(self, image: torch.Tensor) -> Dict:
        """
        Calculate image statistics.

        Returns:
            Dict with shape, mean, std, min, max
        """
        return {
            'shape': tuple(image.shape),
            'mean': float(image.mean()),
            'std': float(image.std()),
            'min': float(image.min()),
            'max': float(image.max()),
            'size_mb': image.element_size() * image.nelement() / (1024 * 1024)
        }


class ImageCache:
    """
    Optional caching system for processed images to avoid reloading.
    """

    def __init__(self, cache_dir: Path, max_cache_size_gb: float = 10.0):
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_cache_size = max_cache_size_gb * 1024 * 1024 * 1024  # Convert to bytes
        logger.info(f"Image cache initialized at {cache_dir}")

    def save(self, study_id: int, image: torch.Tensor):
        """Save processed image to cache"""
        cache_file = self.cache_dir / f"{study_id}.pt"
        torch.save(image, cache_file)

    def load(self, study_id: int) -> Optional[torch.Tensor]:
        """Load processed image from cache"""
        cache_file = self.cache_dir / f"{study_id}.pt"
        if cache_file.exists():
            return torch.load(cache_file)
        return None

    def exists(self, study_id: int) -> bool:
        """Check if image is in cache"""
        cache_file = self.cache_dir / f"{study_id}.pt"
        return cache_file.exists()

    def clear(self):
        """Clear all cached images"""
        for cache_file in self.cache_dir.glob("*.pt"):
            cache_file.unlink()
        logger.info("Image cache cleared")
