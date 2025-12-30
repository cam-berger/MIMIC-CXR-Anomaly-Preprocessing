"""
pytest configuration and fixtures for NaN/Inf stability testing.

Provides fixtures for model components, test data, and utilities.
"""

import pytest
import torch
import numpy as np
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.models.mae import MaskedAutoencoder
from src.models.multimodal import MultimodalClassifier, CrossAttentionFusion
from src.models.losses import CLIPLoss, SupConLoss, AsymmetricFocalLoss, MultiTaskLoss


@pytest.fixture
def device():
    """Get device for testing (prefer CUDA if available)."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def mae_model(device):
    """Create MAE model for testing."""
    model = MaskedAutoencoder(
        img_size=224,
        patch_size=16,
        in_channels=3,
        embed_dim=192,
        depth=4,
        num_heads=3,
        decoder_embed_dim=96,
        decoder_depth=2,
        decoder_num_heads=3,
        mlp_ratio=4.0,
        mask_ratio=0.75,
        norm_pix_loss=True,
    ).to(device)
    model.eval()
    return model


@pytest.fixture
def multimodal_classifier(device):
    """Create MultimodalClassifier for testing."""
    model = MultimodalClassifier(
        mae_checkpoint=None,  # No pretrained weights for testing
        num_labels=14,
        embed_dim=768,
        struct_input_dim=44,
        struct_hidden_dim=256,
        contrastive_dim=128,
        freeze_text=False,  # Don't freeze for testing
    ).to(device)
    model.eval()
    return model


@pytest.fixture
def cross_attention_fusion(device):
    """Create standalone CrossAttentionFusion for testing."""
    fusion = CrossAttentionFusion(
        embed_dim=768,
        num_heads=8,
        dropout=0.1,
    ).to(device)
    fusion.eval()
    return fusion


@pytest.fixture
def loss_functions(device):
    """Create loss functions for testing."""
    return {
        "clip": CLIPLoss(init_temperature=0.07).to(device),
        "supcon": SupConLoss(temperature=0.1).to(device),
        "focal": AsymmetricFocalLoss(gamma_neg=4, gamma_pos=0, clip=0.05).to(device),
        "multitask": MultiTaskLoss().to(device),
    }


@pytest.fixture
def normal_image(device):
    """Create normal test image."""
    # Simulate 224x224 RGB image normalized to [0, 1]
    # MAE expects 3-channel input (grayscale is converted to 3-channel in preprocessing)
    batch_size = 4
    img = torch.rand(batch_size, 3, 224, 224, device=device)
    return img


@pytest.fixture
def edge_case_images(device):
    """Create edge-case images for testing."""
    batch_size = 4
    # All images are 3-channel (MAE expects RGB input)
    return {
        "zeros": torch.zeros(batch_size, 3, 224, 224, device=device),
        "ones": torch.ones(batch_size, 3, 224, 224, device=device),
        "constant": torch.full((batch_size, 3, 224, 224), 0.5, device=device),
        "low_variance": torch.randn(batch_size, 3, 224, 224, device=device) * 0.001 + 0.5,
        "high_values": torch.full((batch_size, 3, 224, 224), 100.0, device=device),
        "with_nan": torch.rand(batch_size, 3, 224, 224, device=device),
        "with_inf": torch.rand(batch_size, 3, 224, 224, device=device),
    }


@pytest.fixture
def normal_embeddings(device):
    """Create normal test embeddings."""
    batch_size = 4
    embed_dim = 768
    img_emb = torch.randn(batch_size, embed_dim, device=device)
    text_emb = torch.randn(batch_size, embed_dim, device=device)
    return {"img": img_emb, "text": text_emb}


@pytest.fixture
def edge_case_embeddings(device):
    """Create edge-case embeddings for testing."""
    batch_size = 4
    embed_dim = 768

    # Zero vectors (zero L2 norm)
    zero_emb = torch.zeros(batch_size, embed_dim, device=device)

    # Very small norm
    tiny_emb = torch.randn(batch_size, embed_dim, device=device) * 1e-10

    # Very large norm
    large_emb = torch.randn(batch_size, embed_dim, device=device) * 100.0

    # Near FP16 limits
    fp16_max = torch.full((batch_size, embed_dim), 65000.0, device=device)

    # With NaN
    with_nan = torch.randn(batch_size, embed_dim, device=device)
    with_nan[0, :10] = float('nan')

    # With Inf
    with_inf = torch.randn(batch_size, embed_dim, device=device)
    with_inf[1, :10] = float('inf')
    with_inf[2, :10] = float('-inf')

    return {
        "zero": zero_emb,
        "tiny": tiny_emb,
        "large": large_emb,
        "fp16_max": fp16_max,
        "with_nan": with_nan,
        "with_inf": with_inf,
    }


@pytest.fixture
def normal_structured(device):
    """Create normal structured features."""
    batch_size = 4
    structured_dim = 128
    return torch.randn(batch_size, structured_dim, device=device)


@pytest.fixture
def edge_case_structured(device):
    """Create edge-case structured features."""
    batch_size = 4
    structured_dim = 128

    # All zeros
    zeros = torch.zeros(batch_size, structured_dim, device=device)

    # With NaN
    with_nan = torch.randn(batch_size, structured_dim, device=device)
    with_nan[:, :10] = float('nan')

    # With Inf
    with_inf = torch.randn(batch_size, structured_dim, device=device)
    with_inf[:, :10] = float('inf')

    return {
        "zeros": zeros,
        "with_nan": with_nan,
        "with_inf": with_inf,
    }


@pytest.fixture
def chexpert_labels(device):
    """Create CheXpert labels for testing."""
    batch_size = 4
    num_classes = 14
    # Random binary labels with some -1 (uncertain) and NaN (missing)
    labels = torch.randint(0, 2, (batch_size, num_classes), device=device, dtype=torch.float)
    # Add some uncertain labels (-1)
    labels[0, :3] = -1.0
    # Add some missing labels (NaN)
    labels[1, :3] = float('nan')
    return labels


@pytest.fixture
def gradscaler():
    """Create GradScaler for testing."""
    from torch.cuda.amp import GradScaler
    return GradScaler()


@pytest.fixture
def mock_preprocessed_data(tmp_path, device):
    """Create mock preprocessed data directory for integration testing."""
    import h5py
    import pandas as pd

    # Create HDF5 file with images
    images_path = tmp_path / "images.h5"
    with h5py.File(images_path, 'w') as f:
        for i in range(10):
            img = np.random.rand(1, 224, 224).astype(np.float32)
            f.create_dataset(f'images/{i}', data=img)
            f.create_dataset(f'metadata/{i}', data=np.string_(f'{{"study_id": "s{i}", "subject_id": "p{i}"}}'))

    # Create structured data
    structured_data = {
        'study_id': [f's{i}' for i in range(10)],
        'subject_id': [f'p{i}' for i in range(10)],
        'age': np.random.randint(20, 90, 10),
        'gender': np.random.choice([0, 1], 10),
    }
    # Add lab features
    for lab in ['wbc', 'hemoglobin', 'platelet']:
        for stat in ['mean', 'min', 'max']:
            structured_data[f'lab_{lab}_{stat}'] = np.random.randn(10)

    structured_df = pd.DataFrame(structured_data)
    structured_df.to_parquet(tmp_path / "structured.parquet")

    # Create text data
    text_data = {
        'study_id': [f's{i}' for i in range(10)],
        'subject_id': [f'p{i}' for i in range(10)],
        'report': [f'Test report {i}' for i in range(10)],
        'summary': [f'Test summary {i}' for i in range(10)],
        'tokens': [100 + i for i in range(10)],
    }
    text_df = pd.DataFrame(text_data)
    text_df.to_parquet(tmp_path / "text.parquet")

    return tmp_path


# Utility fixtures

@pytest.fixture
def assert_finite():
    """Helper function to assert all tensor values are finite."""
    def _assert_finite(tensor, name="tensor"):
        assert torch.isfinite(tensor).all(), f"{name} contains NaN or Inf"
    return _assert_finite


@pytest.fixture
def assert_no_nan():
    """Helper function to assert tensor contains no NaN."""
    def _assert_no_nan(tensor, name="tensor"):
        assert not torch.isnan(tensor).any(), f"{name} contains NaN"
    return _assert_no_nan


@pytest.fixture
def assert_no_inf():
    """Helper function to assert tensor contains no Inf."""
    def _assert_no_inf(tensor, name="tensor"):
        assert not torch.isinf(tensor).any(), f"{name} contains Inf"
    return _assert_no_inf


@pytest.fixture
def inject_nan():
    """Utility to inject NaN into tensor."""
    def _inject_nan(tensor, rate=0.1):
        """Inject NaN into random positions."""
        mask = torch.rand_like(tensor) < rate
        tensor = tensor.clone()
        tensor[mask] = float('nan')
        return tensor
    return _inject_nan


@pytest.fixture
def inject_inf():
    """Utility to inject Inf into tensor."""
    def _inject_inf(tensor, rate=0.1):
        """Inject Inf into random positions."""
        mask = torch.rand_like(tensor) < rate
        tensor = tensor.clone()
        tensor[mask] = float('inf')
        return tensor
    return _inject_inf
