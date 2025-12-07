"""
Models for chest X-ray anomaly detection and classification.

This module implements:
- MAE: Masked Autoencoder for self-supervised pretraining
- Multimodal Classifier: Image + Text + Structured data fusion
- Anomaly Detection: Reconstruction, embedding, and ensemble methods

Components:
- MIMICCXRDataset: HDF5 dataset loader for preprocessed images
- MultimodalClassificationDataset: Dataset with CheXpert labels
- MaskedAutoencoder: MAE for self-supervised pretraining
- MultimodalClassifier: Classifier with cross-attention fusion
- AnomalyDetector: Multiple anomaly detection methods
"""

from .dataset import MIMICCXRDataset, PreprocessedMAEDataset, get_mae_augmentations
from .mae import MaskedAutoencoder
from .anomaly import (
    ReconstructionAnomalyDetector,
    EmbeddingAnomalyDetector,
    EnsembleAnomalyDetector,
    MultimodalEnsembleDetector,
)
from .classification_dataset import MultimodalClassificationDataset, collate_multimodal
from .multimodal import (
    MultimodalClassifier,
    TextEncoder,
    StructuredEncoder,
    CrossAttentionFusion,
)
from .losses import (
    CLIPLoss,
    SupConLoss,
    AsymmetricFocalLoss,
    MultiTaskLoss,
)

__all__ = [
    # Datasets
    "MIMICCXRDataset",
    "PreprocessedMAEDataset",
    "get_mae_augmentations",
    "MultimodalClassificationDataset",
    "collate_multimodal",
    # Models
    "MaskedAutoencoder",
    "MultimodalClassifier",
    "TextEncoder",
    "StructuredEncoder",
    "CrossAttentionFusion",
    # Losses
    "CLIPLoss",
    "SupConLoss",
    "AsymmetricFocalLoss",
    "MultiTaskLoss",
    # Anomaly Detection
    "ReconstructionAnomalyDetector",
    "EmbeddingAnomalyDetector",
    "EnsembleAnomalyDetector",
    "MultimodalEnsembleDetector",
]
