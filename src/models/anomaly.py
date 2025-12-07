"""
Anomaly detection methods for chest X-rays using pretrained MAE.

Implements multiple anomaly detection approaches:
1. Reconstruction-based: Uses MAE reconstruction error
2. Embedding-based: Uses distance in latent space (k-NN, GMM)
3. Ensemble: Combines multiple methods

Based on MODEL_TRAINING_RESEARCH.md recommendations.
"""

import logging
from typing import Optional, Dict, List, Tuple, Union
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.neighbors import NearestNeighbors
from sklearn.mixture import GaussianMixture
from sklearn.metrics import roc_auc_score, average_precision_score
from tqdm import tqdm

from .mae import MaskedAutoencoder

logger = logging.getLogger(__name__)


class ReconstructionAnomalyDetector:
    """
    Anomaly detection based on MAE reconstruction error.

    Approach: Normal images should have low reconstruction error after
    MAE pretraining on normal data. Anomalous images will have higher
    reconstruction error because the model hasn't learned to reconstruct them.

    Threshold Methods:
    - percentile: threshold = np.percentile(train_errors, p)
    - statistical: threshold = mean + k * std
    - max: threshold = max(train_errors)
    """

    def __init__(
        self,
        model: MaskedAutoencoder,
        device: str = "cuda",
        mask_ratio: float = 0.75,
        num_masks: int = 1,
    ):
        """
        Initialize detector.

        Args:
            model: Pretrained MAE model
            device: Device to run inference on
            mask_ratio: Mask ratio for reconstruction
            num_masks: Number of random masks to average over
        """
        self.model = model.to(device)
        self.model.eval()
        self.device = device
        self.mask_ratio = mask_ratio
        self.num_masks = num_masks

        # Threshold computed from normal data
        self.threshold = None
        self.train_errors = None

    def compute_reconstruction_error(
        self,
        images: torch.Tensor,
        per_pixel: bool = False,
    ) -> torch.Tensor:
        """
        Compute reconstruction error for images.

        Args:
            images: Input images [B, C, H, W]
            per_pixel: If True, return per-pixel error map

        Returns:
            If per_pixel: Error map [B, C, H, W]
            Else: Scalar error per image [B]
        """
        images = images.to(self.device)

        with torch.no_grad():
            errors = []
            for _ in range(self.num_masks):
                # Get reconstruction
                reconstructed = self.model.reconstruct(images, self.mask_ratio)

                # Compute per-pixel error
                error = (images - reconstructed) ** 2

                if not per_pixel:
                    # Mean over all dimensions except batch
                    error = error.mean(dim=(1, 2, 3))

                errors.append(error)

            # Average over multiple masks
            error = torch.stack(errors).mean(dim=0)

        return error

    def fit(
        self,
        dataloader: DataLoader,
        percentile: float = 99.0,
        method: str = "percentile",
        k_std: float = 3.0,
    ) -> None:
        """
        Fit threshold on normal training data.

        Args:
            dataloader: DataLoader with normal images
            percentile: Percentile for threshold (if method='percentile')
            method: 'percentile', 'statistical', or 'max'
            k_std: Number of std deviations for statistical method
        """
        logger.info("Computing reconstruction errors on training data...")

        all_errors = []
        for batch in tqdm(dataloader, desc="Fitting"):
            # Handle different batch formats
            if isinstance(batch, dict):
                images = batch["image"]
            elif isinstance(batch, (list, tuple)):
                images = batch[0]
            else:
                images = batch

            errors = self.compute_reconstruction_error(images)
            all_errors.append(errors.cpu())

        self.train_errors = torch.cat(all_errors).numpy()

        # Set threshold
        if method == "percentile":
            self.threshold = np.percentile(self.train_errors, percentile)
        elif method == "statistical":
            mean = np.mean(self.train_errors)
            std = np.std(self.train_errors)
            self.threshold = mean + k_std * std
        elif method == "max":
            self.threshold = np.max(self.train_errors)
        else:
            raise ValueError(f"Unknown method: {method}")

        logger.info(f"Threshold set to {self.threshold:.6f} ({method} method)")
        logger.info(f"Train errors: mean={np.mean(self.train_errors):.6f}, "
                   f"std={np.std(self.train_errors):.6f}, "
                   f"max={np.max(self.train_errors):.6f}")

    def score(self, images: torch.Tensor) -> np.ndarray:
        """
        Compute anomaly scores for images.

        Args:
            images: Input images [B, C, H, W]

        Returns:
            Anomaly scores [B] (higher = more anomalous)
        """
        errors = self.compute_reconstruction_error(images)
        return errors.cpu().numpy()

    def predict(self, images: torch.Tensor) -> np.ndarray:
        """
        Predict anomaly labels (0 = normal, 1 = anomalous).

        Args:
            images: Input images [B, C, H, W]

        Returns:
            Binary predictions [B]
        """
        if self.threshold is None:
            raise ValueError("Must call fit() before predict()")

        scores = self.score(images)
        return (scores > self.threshold).astype(int)

    def get_anomaly_heatmap(self, images: torch.Tensor) -> torch.Tensor:
        """
        Get per-pixel anomaly heatmap.

        Args:
            images: Input images [B, C, H, W]

        Returns:
            Heatmap [B, 1, H, W] (higher = more anomalous)
        """
        error_map = self.compute_reconstruction_error(images, per_pixel=True)
        # Average over channels
        heatmap = error_map.mean(dim=1, keepdim=True)
        return heatmap


class EmbeddingAnomalyDetector:
    """
    Anomaly detection based on embedding distance.

    Approach: Normal images should have embeddings close to the cluster
    of normal training embeddings. Anomalous images will be far from
    this cluster.

    Methods:
    - knn: Distance to k nearest neighbors
    - gmm: Negative log-likelihood under GMM fitted on normal data
    """

    def __init__(
        self,
        model: MaskedAutoencoder,
        device: str = "cuda",
        method: str = "knn",
        k: int = 10,
        n_components: int = 10,
    ):
        """
        Initialize detector.

        Args:
            model: Pretrained MAE model (uses encoder)
            device: Device to run inference on
            method: 'knn' or 'gmm'
            k: Number of neighbors for k-NN
            n_components: Number of components for GMM
        """
        self.model = model.to(device)
        self.model.eval()
        self.device = device
        self.method = method
        self.k = k
        self.n_components = n_components

        # Fitted models
        self.nn_model = None
        self.gmm_model = None
        self.train_embeddings = None
        self.threshold = None

    def _encode(self, images: torch.Tensor) -> np.ndarray:
        """Encode images to embeddings."""
        images = images.to(self.device)
        with torch.no_grad():
            embeddings = self.model.encode(images)
        return embeddings.cpu().numpy()

    def fit(
        self,
        dataloader: DataLoader,
        percentile: float = 99.0,
    ) -> None:
        """
        Fit detector on normal training data.

        Args:
            dataloader: DataLoader with normal images
            percentile: Percentile for threshold
        """
        logger.info("Computing embeddings for training data...")

        all_embeddings = []
        for batch in tqdm(dataloader, desc="Encoding"):
            if isinstance(batch, dict):
                images = batch["image"]
            elif isinstance(batch, (list, tuple)):
                images = batch[0]
            else:
                images = batch

            embeddings = self._encode(images)
            all_embeddings.append(embeddings)

        self.train_embeddings = np.concatenate(all_embeddings, axis=0)
        logger.info(f"Computed {len(self.train_embeddings)} embeddings")

        # Fit method-specific model
        if self.method == "knn":
            logger.info(f"Fitting k-NN with k={self.k}...")
            self.nn_model = NearestNeighbors(n_neighbors=self.k, metric="cosine")
            self.nn_model.fit(self.train_embeddings)

            # Compute scores on training data
            distances, _ = self.nn_model.kneighbors(self.train_embeddings)
            train_scores = distances.mean(axis=1)

        elif self.method == "gmm":
            logger.info(f"Fitting GMM with {self.n_components} components...")
            self.gmm_model = GaussianMixture(
                n_components=self.n_components,
                covariance_type="full",
                random_state=42,
            )
            self.gmm_model.fit(self.train_embeddings)

            # Compute scores on training data (negative log-likelihood)
            train_scores = -self.gmm_model.score_samples(self.train_embeddings)

        else:
            raise ValueError(f"Unknown method: {self.method}")

        # Set threshold
        self.threshold = np.percentile(train_scores, percentile)
        logger.info(f"Threshold set to {self.threshold:.6f}")

    def score(self, images: torch.Tensor) -> np.ndarray:
        """
        Compute anomaly scores for images.

        Args:
            images: Input images [B, C, H, W]

        Returns:
            Anomaly scores [B] (higher = more anomalous)
        """
        embeddings = self._encode(images)

        if self.method == "knn":
            distances, _ = self.nn_model.kneighbors(embeddings)
            scores = distances.mean(axis=1)
        elif self.method == "gmm":
            scores = -self.gmm_model.score_samples(embeddings)
        else:
            raise ValueError(f"Unknown method: {self.method}")

        return scores

    def predict(self, images: torch.Tensor) -> np.ndarray:
        """
        Predict anomaly labels (0 = normal, 1 = anomalous).

        Args:
            images: Input images [B, C, H, W]

        Returns:
            Binary predictions [B]
        """
        if self.threshold is None:
            raise ValueError("Must call fit() before predict()")

        scores = self.score(images)
        return (scores > self.threshold).astype(int)

    def get_embeddings(self, images: torch.Tensor) -> np.ndarray:
        """
        Get embeddings for visualization/analysis.

        Args:
            images: Input images [B, C, H, W]

        Returns:
            Embeddings [B, D]
        """
        return self._encode(images)


class EnsembleAnomalyDetector:
    """
    Ensemble of multiple anomaly detection methods.

    Combines reconstruction-based and embedding-based scores
    for more robust anomaly detection.
    """

    def __init__(
        self,
        model: MaskedAutoencoder,
        device: str = "cuda",
        weights: Optional[Dict[str, float]] = None,
    ):
        """
        Initialize ensemble detector.

        Args:
            model: Pretrained MAE model
            device: Device to run inference on
            weights: Optional weights for each method
                     Default: {'reconstruction': 0.4, 'knn': 0.3, 'gmm': 0.3}
        """
        self.device = device

        # Initialize individual detectors
        self.recon_detector = ReconstructionAnomalyDetector(model, device)
        self.knn_detector = EmbeddingAnomalyDetector(model, device, method="knn")
        self.gmm_detector = EmbeddingAnomalyDetector(model, device, method="gmm")

        # Set weights
        self.weights = weights or {
            "reconstruction": 0.4,
            "knn": 0.3,
            "gmm": 0.3,
        }

        # Normalization parameters
        self.score_means = {}
        self.score_stds = {}
        self.threshold = None

    def fit(
        self,
        dataloader: DataLoader,
        percentile: float = 99.0,
    ) -> None:
        """
        Fit all detectors on normal training data.

        Args:
            dataloader: DataLoader with normal images
            percentile: Percentile for threshold
        """
        # Fit individual detectors
        logger.info("Fitting reconstruction detector...")
        self.recon_detector.fit(dataloader, percentile=percentile)

        logger.info("Fitting k-NN detector...")
        self.knn_detector.fit(dataloader, percentile=percentile)

        logger.info("Fitting GMM detector...")
        self.gmm_detector.fit(dataloader, percentile=percentile)

        # Compute ensemble scores on training data for normalization
        logger.info("Computing ensemble normalization parameters...")
        all_scores = {"reconstruction": [], "knn": [], "gmm": []}

        for batch in tqdm(dataloader, desc="Normalizing"):
            if isinstance(batch, dict):
                images = batch["image"]
            elif isinstance(batch, (list, tuple)):
                images = batch[0]
            else:
                images = batch

            all_scores["reconstruction"].append(self.recon_detector.score(images))
            all_scores["knn"].append(self.knn_detector.score(images))
            all_scores["gmm"].append(self.gmm_detector.score(images))

        # Compute normalization parameters
        for method in all_scores:
            scores = np.concatenate(all_scores[method])
            self.score_means[method] = np.mean(scores)
            self.score_stds[method] = np.std(scores)

        # Compute ensemble threshold
        ensemble_scores = self._compute_ensemble_scores(all_scores)
        self.threshold = np.percentile(ensemble_scores, percentile)
        logger.info(f"Ensemble threshold: {self.threshold:.6f}")

    def _normalize_score(self, score: np.ndarray, method: str) -> np.ndarray:
        """Normalize score to zero mean and unit variance."""
        return (score - self.score_means[method]) / (self.score_stds[method] + 1e-8)

    def _compute_ensemble_scores(self, scores_dict: Dict[str, List[np.ndarray]]) -> np.ndarray:
        """Compute weighted ensemble scores from dictionary of scores."""
        # Concatenate all batches
        scores = {k: np.concatenate(v) for k, v in scores_dict.items()}

        # Normalize and weight
        ensemble = np.zeros(len(scores["reconstruction"]))
        for method, weight in self.weights.items():
            normalized = self._normalize_score(scores[method], method)
            ensemble += weight * normalized

        return ensemble

    def score(
        self,
        images: torch.Tensor,
        return_components: bool = False,
    ) -> Union[np.ndarray, Tuple[np.ndarray, Dict[str, np.ndarray]]]:
        """
        Compute ensemble anomaly scores.

        Args:
            images: Input images [B, C, H, W]
            return_components: Whether to return individual scores

        Returns:
            Ensemble scores [B]
            If return_components: (ensemble_scores, {method: scores})
        """
        scores = {
            "reconstruction": self.recon_detector.score(images),
            "knn": self.knn_detector.score(images),
            "gmm": self.gmm_detector.score(images),
        }

        # Normalize and combine
        ensemble = np.zeros(len(scores["reconstruction"]))
        for method, weight in self.weights.items():
            normalized = self._normalize_score(scores[method], method)
            ensemble += weight * normalized

        if return_components:
            return ensemble, scores
        return ensemble

    def predict(self, images: torch.Tensor) -> np.ndarray:
        """
        Predict anomaly labels (0 = normal, 1 = anomalous).

        Args:
            images: Input images [B, C, H, W]

        Returns:
            Binary predictions [B]
        """
        if self.threshold is None:
            raise ValueError("Must call fit() before predict()")

        scores = self.score(images)
        return (scores > self.threshold).astype(int)


def evaluate_anomaly_detection(
    scores: np.ndarray,
    labels: np.ndarray,
) -> Dict[str, float]:
    """
    Evaluate anomaly detection performance.

    Args:
        scores: Anomaly scores (higher = more anomalous)
        labels: Ground truth labels (0 = normal, 1 = anomalous)

    Returns:
        Dictionary with AUROC and AUPRC metrics
    """
    auroc = roc_auc_score(labels, scores)
    auprc = average_precision_score(labels, scores)

    return {
        "AUROC": auroc,
        "AUPRC": auprc,
    }


class MultimodalEnsembleDetector:
    """
    Multimodal ensemble anomaly detector.

    Combines multiple detection signals from the multimodal classifier:
    1. Classification confidence: Max probability of any pathology
    2. MAE reconstruction error: How well can we reconstruct the image
    3. Embedding distance: Distance to normal embeddings in fused space
    4. Prediction entropy: Uncertainty in predictions

    This provides a more comprehensive anomaly score than image-only methods
    by leveraging clinical context from text and structured data.
    """

    def __init__(
        self,
        classifier,  # MultimodalClassifier
        mae_model: MaskedAutoencoder,
        device: str = "cuda",
        weights: Optional[Dict[str, float]] = None,
        k: int = 10,
    ):
        """
        Initialize multimodal ensemble detector.

        Args:
            classifier: Trained MultimodalClassifier
            mae_model: Pretrained MAE model (for reconstruction)
            device: Device to run inference on
            weights: Optional weights for each method
                     Default: {'confidence': 0.3, 'reconstruction': 0.25,
                              'embedding': 0.25, 'entropy': 0.2}
            k: Number of neighbors for k-NN embedding distance
        """
        self.classifier = classifier.to(device)
        self.classifier.eval()
        self.mae = mae_model.to(device)
        self.mae.eval()
        self.device = device
        self.k = k

        # Default weights
        self.weights = weights or {
            "confidence": 0.3,
            "reconstruction": 0.25,
            "embedding": 0.25,
            "entropy": 0.2,
        }

        # k-NN model for embedding distance
        self.nn_model = None
        self.train_embeddings = None

        # Normalization parameters
        self.score_means = {}
        self.score_stds = {}
        self.threshold = None

    def fit(
        self,
        dataloader,  # DataLoader with multimodal data
        percentile: float = 99.0,
    ) -> None:
        """
        Fit detector on normal training data.

        Args:
            dataloader: DataLoader with normal samples (multimodal format)
            percentile: Percentile for threshold
        """
        logger.info("Fitting multimodal ensemble detector...")

        all_scores = {
            "confidence": [],
            "reconstruction": [],
            "embedding": [],
            "entropy": [],
        }
        all_embeddings = []

        for batch in tqdm(dataloader, desc="Fitting multimodal detector"):
            # Extract batch data
            images = batch["image"].to(self.device)
            text_tokens = batch["text_tokens"].to(self.device)
            attention_mask = batch.get("attention_mask")
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.device)
            structured = batch["structured"].to(self.device)

            with torch.no_grad():
                # Get classifier outputs
                logits, _, _, fused_emb, _, _ = self.classifier(
                    images, text_tokens, structured, attention_mask,
                    return_embeddings=True,
                )
                probs = torch.sigmoid(logits)

                # 1. Classification confidence (max pathology prob)
                # For normal images, all pathologies should have low probability
                conf_scores = probs.max(dim=1)[0].cpu().numpy()
                all_scores["confidence"].append(conf_scores)

                # 2. MAE reconstruction error
                reconstructed = self.mae.reconstruct(images, mask_ratio=0.75)
                recon_error = ((images - reconstructed) ** 2).mean(dim=(1, 2, 3))
                all_scores["reconstruction"].append(recon_error.cpu().numpy())

                # 3. Embedding (for k-NN fitting later)
                all_embeddings.append(fused_emb.cpu().numpy())

                # 4. Prediction entropy
                # Higher entropy = more uncertain = potentially anomalous
                entropy = -(probs * torch.log(probs + 1e-8) +
                           (1 - probs) * torch.log(1 - probs + 1e-8)).sum(dim=1)
                all_scores["entropy"].append(entropy.cpu().numpy())

        # Fit k-NN on embeddings
        self.train_embeddings = np.concatenate(all_embeddings, axis=0)
        logger.info(f"Fitting k-NN on {len(self.train_embeddings)} embeddings...")
        self.nn_model = NearestNeighbors(n_neighbors=self.k, metric="cosine")
        self.nn_model.fit(self.train_embeddings)

        # Compute embedding distances for training data
        distances, _ = self.nn_model.kneighbors(self.train_embeddings)
        all_scores["embedding"] = [distances.mean(axis=1)]

        # Concatenate all scores
        for method in all_scores:
            all_scores[method] = np.concatenate(all_scores[method])

        # Compute normalization parameters
        for method in all_scores:
            self.score_means[method] = np.mean(all_scores[method])
            self.score_stds[method] = np.std(all_scores[method])
            logger.info(f"  {method}: mean={self.score_means[method]:.4f}, "
                       f"std={self.score_stds[method]:.4f}")

        # Compute ensemble threshold
        ensemble_scores = self._combine_scores(all_scores)
        self.threshold = np.percentile(ensemble_scores, percentile)
        logger.info(f"Ensemble threshold: {self.threshold:.6f}")

    def _normalize_score(self, score: np.ndarray, method: str) -> np.ndarray:
        """Normalize score to zero mean and unit variance."""
        return (score - self.score_means[method]) / (self.score_stds[method] + 1e-8)

    def _combine_scores(self, scores_dict: Dict[str, np.ndarray]) -> np.ndarray:
        """Combine normalized scores with weights."""
        ensemble = np.zeros(len(scores_dict["confidence"]))
        for method, weight in self.weights.items():
            if method in scores_dict:
                normalized = self._normalize_score(scores_dict[method], method)
                ensemble += weight * normalized
        return ensemble

    def score(
        self,
        images: torch.Tensor,
        text_tokens: torch.Tensor,
        structured: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        return_components: bool = False,
    ) -> Union[np.ndarray, Tuple[np.ndarray, Dict[str, np.ndarray]]]:
        """
        Compute multimodal ensemble anomaly scores.

        Args:
            images: Input images [B, 3, H, W]
            text_tokens: Text token IDs [B, seq_len]
            structured: Structured features [B, num_features]
            attention_mask: Text attention mask [B, seq_len]
            return_components: Whether to return individual scores

        Returns:
            Ensemble scores [B]
            If return_components: (ensemble_scores, {method: scores})
        """
        images = images.to(self.device)
        text_tokens = text_tokens.to(self.device)
        structured = structured.to(self.device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)

        with torch.no_grad():
            # Get classifier outputs
            logits, _, _, fused_emb, _, _ = self.classifier(
                images, text_tokens, structured, attention_mask,
                return_embeddings=True,
            )
            probs = torch.sigmoid(logits)

            # Individual scores
            scores = {}

            # 1. Classification confidence
            scores["confidence"] = probs.max(dim=1)[0].cpu().numpy()

            # 2. Reconstruction error
            reconstructed = self.mae.reconstruct(images, mask_ratio=0.75)
            scores["reconstruction"] = ((images - reconstructed) ** 2).mean(
                dim=(1, 2, 3)
            ).cpu().numpy()

            # 3. Embedding distance
            embeddings = fused_emb.cpu().numpy()
            distances, _ = self.nn_model.kneighbors(embeddings)
            scores["embedding"] = distances.mean(axis=1)

            # 4. Entropy
            scores["entropy"] = -(
                probs * torch.log(probs + 1e-8) +
                (1 - probs) * torch.log(1 - probs + 1e-8)
            ).sum(dim=1).cpu().numpy()

        # Combine scores
        ensemble = self._combine_scores(scores)

        if return_components:
            return ensemble, scores
        return ensemble

    def score_batch(
        self,
        batch: Dict[str, torch.Tensor],
        return_components: bool = False,
    ) -> Union[np.ndarray, Tuple[np.ndarray, Dict[str, np.ndarray]]]:
        """
        Convenience method to score a batch dict from DataLoader.

        Args:
            batch: Batch dict with 'image', 'text_tokens', 'structured', etc.
            return_components: Whether to return individual scores

        Returns:
            Same as score()
        """
        return self.score(
            images=batch["image"],
            text_tokens=batch["text_tokens"],
            structured=batch["structured"],
            attention_mask=batch.get("attention_mask"),
            return_components=return_components,
        )

    def predict(
        self,
        images: torch.Tensor,
        text_tokens: torch.Tensor,
        structured: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> np.ndarray:
        """
        Predict anomaly labels (0 = normal, 1 = anomalous).

        Args:
            images: Input images [B, 3, H, W]
            text_tokens: Text token IDs [B, seq_len]
            structured: Structured features [B, num_features]
            attention_mask: Text attention mask [B, seq_len]

        Returns:
            Binary predictions [B]
        """
        if self.threshold is None:
            raise ValueError("Must call fit() before predict()")

        scores = self.score(images, text_tokens, structured, attention_mask)
        return (scores > self.threshold).astype(int)

    def get_pathology_probabilities(
        self,
        images: torch.Tensor,
        text_tokens: torch.Tensor,
        structured: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Get per-pathology probabilities for interpretability.

        Args:
            Same as score()

        Returns:
            probabilities: [B, num_labels] array
            label_names: List of pathology names
        """
        from .classification_dataset import MultimodalClassificationDataset

        images = images.to(self.device)
        text_tokens = text_tokens.to(self.device)
        structured = structured.to(self.device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)

        with torch.no_grad():
            logits, _, _ = self.classifier(
                images, text_tokens, structured, attention_mask
            )
            probs = torch.sigmoid(logits).cpu().numpy()

        return probs, MultimodalClassificationDataset.PATHOLOGY_LABELS


def compute_optimal_threshold(
    scores: np.ndarray,
    labels: np.ndarray,
    metric: str = "f1",
) -> Tuple[float, Dict[str, float]]:
    """
    Compute optimal threshold using validation data.

    Args:
        scores: Anomaly scores
        labels: Ground truth labels
        metric: Metric to optimize ('f1', 'precision', 'recall')

    Returns:
        (optimal_threshold, metrics_at_threshold)
    """
    from sklearn.metrics import precision_recall_curve, f1_score

    precisions, recalls, thresholds = precision_recall_curve(labels, scores)

    if metric == "f1":
        f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)
        best_idx = np.argmax(f1_scores)
    elif metric == "precision":
        # Find threshold that achieves target recall (e.g., 0.9)
        target_recall = 0.9
        valid_idx = recalls >= target_recall
        if valid_idx.sum() > 0:
            best_idx = np.where(valid_idx)[0][np.argmax(precisions[valid_idx])]
        else:
            best_idx = 0
    elif metric == "recall":
        # Find threshold that achieves target precision (e.g., 0.9)
        target_precision = 0.9
        valid_idx = precisions >= target_precision
        if valid_idx.sum() > 0:
            best_idx = np.where(valid_idx)[0][np.argmax(recalls[valid_idx])]
        else:
            best_idx = -1
    else:
        raise ValueError(f"Unknown metric: {metric}")

    optimal_threshold = thresholds[best_idx] if best_idx < len(thresholds) else thresholds[-1]

    # Compute metrics at optimal threshold
    predictions = (scores >= optimal_threshold).astype(int)
    from sklearn.metrics import precision_score, recall_score, accuracy_score

    metrics = {
        "threshold": optimal_threshold,
        "precision": precision_score(labels, predictions),
        "recall": recall_score(labels, predictions),
        "f1": f1_score(labels, predictions),
        "accuracy": accuracy_score(labels, predictions),
    }

    return optimal_threshold, metrics
