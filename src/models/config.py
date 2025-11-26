"""
Configuration for MAE training and anomaly detection.

Based on MODEL_TRAINING_RESEARCH.md recommendations for medical imaging.
"""

from dataclasses import dataclass, field
from typing import Optional, List, Tuple
from pathlib import Path


@dataclass
class MAEConfig:
    """
    Configuration for Masked Autoencoder training.

    Key hyperparameters from medical_mae research:
    - mask_ratio: 0.75 for medical images (vs 0.90 for natural images)
    - epochs: 800-1600 for good representations
    - input_size: 224 (or 384 for higher resolution)
    """

    # Model architecture
    model_type: str = "vit_base"  # "vit_small", "vit_base", "vit_large"
    img_size: int = 224
    patch_size: int = 16
    in_channels: int = 3
    mask_ratio: float = 0.75  # 75% masked for medical images

    # Encoder
    embed_dim: int = 768
    depth: int = 12
    num_heads: int = 12

    # Decoder
    decoder_embed_dim: int = 512
    decoder_depth: int = 8
    decoder_num_heads: int = 16

    # Training
    epochs: int = 800
    batch_size: int = 64
    base_lr: float = 1.5e-4  # Base learning rate
    min_lr: float = 1e-6
    warmup_epochs: int = 40
    weight_decay: float = 0.05

    # Optimizer
    optimizer: str = "adamw"
    betas: Tuple[float, float] = (0.9, 0.95)

    # Loss
    norm_pix_loss: bool = True

    # Data augmentation (moderate for medical)
    crop_scale: Tuple[float, float] = (0.5, 1.0)
    horizontal_flip: bool = True
    rotation_degrees: int = 15
    gaussian_blur: bool = True

    # Hardware
    num_workers: int = 8
    pin_memory: bool = True
    mixed_precision: bool = True

    # Logging
    log_interval: int = 100
    save_interval: int = 50
    eval_interval: int = 10

    def get_model_kwargs(self) -> dict:
        """Get kwargs for MaskedAutoencoder constructor."""
        return {
            "img_size": self.img_size,
            "patch_size": self.patch_size,
            "in_channels": self.in_channels,
            "embed_dim": self.embed_dim,
            "depth": self.depth,
            "num_heads": self.num_heads,
            "decoder_embed_dim": self.decoder_embed_dim,
            "decoder_depth": self.decoder_depth,
            "decoder_num_heads": self.decoder_num_heads,
            "mask_ratio": self.mask_ratio,
            "norm_pix_loss": self.norm_pix_loss,
        }


@dataclass
class AnomalyDetectionConfig:
    """Configuration for anomaly detection phase."""

    # Threshold method
    threshold_method: str = "percentile"  # "percentile", "statistical", "max"
    threshold_percentile: float = 99.0
    threshold_k_std: float = 3.0

    # Reconstruction-based
    num_masks: int = 1  # Number of masks to average over
    mask_ratio: float = 0.75

    # Embedding-based
    knn_k: int = 10
    gmm_components: int = 10

    # Ensemble weights
    ensemble_weights: dict = field(default_factory=lambda: {
        "reconstruction": 0.4,
        "knn": 0.3,
        "gmm": 0.3,
    })


@dataclass
class TrainingConfig:
    """Full training configuration."""

    # Paths - preprocessed directory format (recommended)
    # Directory structure: {train_dir}/images.h5, structured.parquet, text.parquet
    train_dir: Optional[Path] = None  # e.g., output/preprocessed/normal_train
    val_dir: Optional[Path] = None    # e.g., output/preprocessed/normal_val

    # Legacy paths (for backwards compatibility)
    data_dir: Path = Path("output/preprocessed")
    train_hdf5: Optional[Path] = None
    val_hdf5: Optional[Path] = None
    cohort_csv: Optional[Path] = None

    # Output paths
    output_dir: Path = Path("output/models")
    checkpoint_dir: Path = Path("output/checkpoints")

    # Model config
    mae: MAEConfig = field(default_factory=MAEConfig)
    anomaly: AnomalyDetectionConfig = field(default_factory=AnomalyDetectionConfig)

    # Training phases
    pretrain: bool = True
    fit_anomaly: bool = True
    evaluate: bool = True

    # Device
    device: str = "cuda"
    distributed: bool = False
    world_size: int = 1
    local_rank: int = 0

    # Reproducibility
    seed: int = 42

    def __post_init__(self):
        """Create output directories."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)


# Preset configurations
def get_debug_config() -> TrainingConfig:
    """Small configuration for debugging."""
    config = TrainingConfig()
    config.mae.model_type = "vit_small"
    config.mae.embed_dim = 384
    config.mae.depth = 6
    config.mae.num_heads = 6
    config.mae.decoder_embed_dim = 256
    config.mae.decoder_depth = 2
    config.mae.decoder_num_heads = 8
    config.mae.epochs = 10
    config.mae.batch_size = 8
    return config


def get_fast_config() -> TrainingConfig:
    """Fast configuration for quick experiments."""
    config = TrainingConfig()
    config.mae.model_type = "vit_small"
    config.mae.embed_dim = 384
    config.mae.depth = 12
    config.mae.num_heads = 6
    config.mae.decoder_embed_dim = 256
    config.mae.decoder_depth = 4
    config.mae.decoder_num_heads = 8
    config.mae.epochs = 100
    config.mae.batch_size = 32
    return config


def get_base_config() -> TrainingConfig:
    """Base configuration (recommended for most cases)."""
    config = TrainingConfig()
    # Uses default MAEConfig (ViT-Base)
    config.mae.epochs = 800
    config.mae.batch_size = 64
    return config


def get_large_config() -> TrainingConfig:
    """Large configuration for best results."""
    config = TrainingConfig()
    config.mae.model_type = "vit_large"
    config.mae.embed_dim = 1024
    config.mae.depth = 24
    config.mae.num_heads = 16
    config.mae.decoder_embed_dim = 512
    config.mae.decoder_depth = 8
    config.mae.decoder_num_heads = 16
    config.mae.epochs = 1600
    config.mae.batch_size = 32
    return config
