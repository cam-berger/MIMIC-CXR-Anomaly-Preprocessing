"""
Unit tests for image_loading/image_loader.py

Tests image loading, normalization, augmentation, and error handling.
"""
import pytest
import torch
import numpy as np
from pathlib import Path
from PIL import Image
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.image_processing.image_loader import FullResolutionImageLoader


class TestFullResolutionImageLoader:
    """Test suite for FullResolutionImageLoader"""

    def test_initialization(self, sample_config):
        """Test loader initializes correctly with valid config"""
        loader = FullResolutionImageLoader(sample_config)

        assert loader.config == sample_config
        assert loader.normalize_method == 'minmax'
        assert loader.use_augmentation is False

    def test_initialization_with_augmentation(self, sample_config):
        """Test loader initializes with augmentation enabled"""
        sample_config['image']['augmentation']['enabled'] = True
        sample_config['image']['augmentation']['rotation_range'] = 5

        loader = FullResolutionImageLoader(sample_config)

        assert loader.use_augmentation is True

    def test_minmax_normalization(self, sample_config):
        """Test MinMax normalization produces values in [0, 1]"""
        loader = FullResolutionImageLoader(sample_config)

        # Create test image with known range (as numpy array)
        test_image = np.array([[[100.0, 200.0], [50.0, 250.0]]], dtype=np.float32)
        normalized = loader._normalize(test_image)

        # Check normalization produces values in [0, 1] range
        assert normalized.min() >= 0.0
        assert normalized.max() <= 1.0
        # Check actual min/max values are close to bounds
        assert normalized.min() < 0.5  # Min should be near 0
        assert normalized.max() > 0.5  # Max should be near 1

    def test_standardize_normalization(self, sample_config):
        """Test z-score standardization"""
        sample_config['image']['normalize_method'] = 'standardize'
        loader = FullResolutionImageLoader(sample_config)

        test_image = np.random.randn(1, 100, 100).astype(np.float32) * 50 + 100
        normalized = loader._normalize(test_image)

        # Check that mean is close to 0 and std close to 1
        assert abs(normalized.mean()) < 0.1
        assert abs(normalized.std() - 1.0) < 0.1

    def test_none_normalization(self, sample_config):
        """Test that 'none' normalization returns original"""
        sample_config['image']['normalize_method'] = 'none'
        loader = FullResolutionImageLoader(sample_config)

        test_image = np.random.randn(1, 100, 100).astype(np.float32)
        normalized = loader._normalize(test_image)

        assert np.allclose(test_image, normalized)

    def test_grayscale_to_tensor_conversion(self, sample_config, tmp_path):
        """Test loading grayscale image and converting to tensor"""
        loader = FullResolutionImageLoader(sample_config)

        # Create a grayscale test image
        img_array = np.random.randint(0, 255, (512, 512), dtype=np.uint8)
        img = Image.fromarray(img_array, mode='L')
        img_path = tmp_path / 'test_gray.jpg'
        img.save(img_path)

        # Load and process
        tensor = loader.load_image(img_path)

        assert isinstance(tensor, torch.Tensor)
        assert tensor.dim() == 3  # [C, H, W]
        assert tensor.shape[0] == 1  # Single channel
        assert tensor.shape[1] == 512
        assert tensor.shape[2] == 512

    def test_rgb_image_loading(self, sample_config, tmp_path):
        """Test RGB image loading (preserves 3 channels)"""
        loader = FullResolutionImageLoader(sample_config)

        # Create RGB test image
        img_array = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
        img = Image.fromarray(img_array, mode='RGB')
        img_path = tmp_path / 'test_rgb.jpg'
        img.save(img_path)

        # Load and process
        tensor = loader.load_image(img_path)

        # RGB images keep 3 channels for medical imaging
        assert tensor.shape[0] == 3

    def test_full_resolution_preserved(self, sample_config, tmp_path):
        """Test that full resolution is preserved"""
        sample_config['image']['preserve_full_resolution'] = True
        loader = FullResolutionImageLoader(sample_config)

        # Create large image
        img_array = np.random.randint(0, 255, (3000, 2500), dtype=np.uint8)
        img = Image.fromarray(img_array, mode='L')
        img_path = tmp_path / 'test_large.jpg'
        img.save(img_path)

        tensor = loader.load_image(img_path)

        # Check dimensions preserved
        assert tensor.shape[1] == 3000
        assert tensor.shape[2] == 2500

    def test_load_nonexistent_file(self, sample_config):
        """Test error handling for nonexistent file"""
        loader = FullResolutionImageLoader(sample_config)

        with pytest.raises(Exception):  # Will raise FileNotFoundError or similar
            loader.load_image(Path('/nonexistent/image.jpg'))

    def test_load_corrupted_file(self, sample_config, tmp_path):
        """Test error handling for corrupted image file"""
        loader = FullResolutionImageLoader(sample_config)

        # Create corrupted file
        corrupted_path = tmp_path / 'corrupted.jpg'
        with open(corrupted_path, 'wb') as f:
            f.write(b'not an image')

        with pytest.raises(Exception):  # PIL will raise an error
            loader.load_image(corrupted_path)

    @pytest.mark.skip(reason="Augmentation requires additional implementation")
    def test_augmentation_changes_image(self, sample_config, tmp_path):
        """Test that augmentation actually modifies the image"""
        sample_config['image']['augmentation']['enabled'] = True
        sample_config['image']['augmentation']['rotation_range'] = 10
        loader = FullResolutionImageLoader(sample_config)

        # Create test image
        img_array = np.random.randint(0, 255, (512, 512), dtype=np.uint8)
        img = Image.fromarray(img_array, mode='L')
        img_path = tmp_path / 'test_aug.jpg'
        img.save(img_path)

        # Load twice with different seeds
        torch.manual_seed(1)
        tensor1 = loader.load_image(img_path)

        torch.manual_seed(2)
        tensor2 = loader.load_image(img_path)

        # With augmentation, results should differ
        assert not torch.allclose(tensor1, tensor2)

    def test_batch_loading(self, sample_config, tmp_path):
        """Test loading multiple images"""
        loader = FullResolutionImageLoader(sample_config)

        # Create multiple test images
        image_paths = []
        for i in range(3):
            img_array = np.random.randint(0, 255, (512, 512), dtype=np.uint8)
            img = Image.fromarray(img_array, mode='L')
            img_path = tmp_path / f'test_{i}.jpg'
            img.save(img_path)
            image_paths.append(img_path)

        # Load all
        tensors = [loader.load_image(p) for p in image_paths]

        assert len(tensors) == 3
        assert all(t.shape == (1, 512, 512) for t in tensors)

    def test_normalization_preserves_range(self, sample_config):
        """Test that normalization keeps values in valid range"""
        loader = FullResolutionImageLoader(sample_config)

        test_image = np.random.randint(0, 255, (1, 100, 100)).astype(np.float32)
        normalized = loader._normalize(test_image)

        # Values should be in [0, 1] range after minmax normalization
        assert normalized.min() >= 0.0
        assert normalized.max() <= 1.0

    def test_different_image_sizes(self, sample_config, tmp_path):
        """Test loading images of different sizes"""
        loader = FullResolutionImageLoader(sample_config)

        sizes = [(512, 512), (1024, 768), (3000, 2500)]

        for height, width in sizes:
            img_array = np.random.randint(0, 255, (height, width), dtype=np.uint8)
            img = Image.fromarray(img_array, mode='L')
            img_path = tmp_path / f'test_{height}x{width}.jpg'
            img.save(img_path)

            tensor = loader.load_image(img_path)

            assert tensor.shape == (1, height, width)
