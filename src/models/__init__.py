"""
Unsupervised learning models for chest X-ray anomaly detection.

This module implements a Masked Autoencoder (MAE) based approach for
self-supervised pretraining on normal chest X-rays, following the
recommendations from MODEL_TRAINING_RESEARCH.md.

Components:
- MIMICCXRDataset: HDF5 dataset loader for preprocessed images
- VisionTransformerEncoder: ViT encoder for image patches
- MaskedAutoencoder: MAE for self-supervised pretraining
- AnomalyDetector: Reconstruction-based anomaly detection
"""

from .dataset import MIMICCXRDataset, MIMICCXRHybridDataset
from .mae import MaskedAutoencoder
from .anomaly import ReconstructionAnomalyDetector, EmbeddingAnomalyDetector

__all__ = [
    "MIMICCXRDataset",
    "MIMICCXRHybridDataset",
    "MaskedAutoencoder",
    "ReconstructionAnomalyDetector",
    "EmbeddingAnomalyDetector",
]
