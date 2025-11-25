#!/usr/bin/env python3
"""
Detect anomalies in chest X-rays using a pretrained MAE model.

This script runs inference on new images using the trained anomaly
detection pipeline. It supports:
- Single image inference
- Batch processing from HDF5 or directory
- Visualization of anomaly heatmaps

Usage:
    # Single image
    python detect_anomalies.py --image path/to/xray.jpg --model output/models/mae_best.pt

    # Batch from HDF5
    python detect_anomalies.py --hdf5 output/preprocessed/test/images.h5 --model output/models/mae_best.pt

    # With visualization
    python detect_anomalies.py --image path/to/xray.jpg --model output/models/mae_best.pt --visualize

Example output:
    Image: xray.jpg
    Anomaly Score: 0.0234
    Prediction: NORMAL (threshold: 0.0456)
    Component Scores:
      - Reconstruction: 0.0189
      - k-NN Distance: 0.0312
      - GMM Score: 0.0201
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Optional, List, Dict

import numpy as np
import torch
from torch.utils.data import DataLoader
from PIL import Image
import torchvision.transforms as T
from tqdm import tqdm

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.models.mae import MaskedAutoencoder
from src.models.dataset import MIMICCXRDataset, get_mae_augmentations
from src.models.anomaly import (
    ReconstructionAnomalyDetector,
    EmbeddingAnomalyDetector,
    EnsembleAnomalyDetector,
)
from src.models.config import MAEConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)


def load_model(
    model_path: Path,
    device: str = "cuda",
) -> MaskedAutoencoder:
    """Load pretrained MAE model."""
    logger.info(f"Loading model from {model_path}")

    checkpoint = torch.load(model_path, map_location=device)

    # Get config from checkpoint
    if "config" in checkpoint:
        config = checkpoint["config"]
        if isinstance(config, dict):
            mae_config = MAEConfig(**config) if "mae" not in config else MAEConfig(**config["mae"])
        else:
            mae_config = MAEConfig()
    else:
        mae_config = MAEConfig()

    # Create model
    model = MaskedAutoencoder(**mae_config.get_model_kwargs())

    # Load weights
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model = model.to(device)
    model.eval()

    return model


def load_anomaly_detector(
    detector_path: Path,
    model: MaskedAutoencoder,
    device: str = "cuda",
) -> EnsembleAnomalyDetector:
    """Load fitted anomaly detector parameters."""
    logger.info(f"Loading anomaly detector from {detector_path}")

    checkpoint = torch.load(detector_path, map_location="cpu")

    detector = EnsembleAnomalyDetector(model, device=device)

    # Restore parameters
    detector.threshold = checkpoint["threshold"]
    detector.score_means = checkpoint["score_means"]
    detector.score_stds = checkpoint["score_stds"]
    detector.weights = checkpoint.get("weights", detector.weights)

    detector.recon_detector.threshold = checkpoint["recon_threshold"]
    detector.recon_detector.train_errors = checkpoint["recon_train_errors"]

    detector.knn_detector.threshold = checkpoint["knn_threshold"]
    detector.gmm_detector.threshold = checkpoint["gmm_threshold"]

    # Note: For k-NN and GMM, we'd need to refit on training data
    # or save the fitted models. This is a simplified version.

    return detector


def preprocess_image(
    image_path: Path,
    img_size: int = 224,
) -> torch.Tensor:
    """Load and preprocess a single image."""
    transform = T.Compose([
        T.Resize((img_size, img_size)),
        T.Grayscale(num_output_channels=3),
        T.ToTensor(),
        T.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])

    img = Image.open(image_path)
    if img.mode != "L":
        img = img.convert("L")

    tensor = transform(img)
    return tensor.unsqueeze(0)  # Add batch dimension


def detect_single_image(
    model: MaskedAutoencoder,
    image_path: Path,
    threshold: float = 0.05,
    device: str = "cuda",
    visualize: bool = False,
) -> Dict:
    """
    Detect anomaly in a single image.

    Args:
        model: Pretrained MAE model
        image_path: Path to image file
        threshold: Anomaly threshold
        device: Device to use
        visualize: Whether to generate visualization

    Returns:
        Dictionary with anomaly results
    """
    # Preprocess image
    image = preprocess_image(image_path).to(device)

    # Create simple reconstruction detector
    detector = ReconstructionAnomalyDetector(model, device=device)
    detector.threshold = threshold

    # Compute score
    with torch.no_grad():
        score = detector.score(image)[0]
        prediction = "ANOMALOUS" if score > threshold else "NORMAL"

        result = {
            "image_path": str(image_path),
            "anomaly_score": float(score),
            "prediction": prediction,
            "threshold": threshold,
        }

        # Generate heatmap if requested
        if visualize:
            heatmap = detector.get_anomaly_heatmap(image)
            result["heatmap"] = heatmap.cpu().numpy()

            # Also get reconstruction
            reconstructed = model.reconstruct(image)
            result["reconstruction"] = reconstructed.cpu().numpy()
            result["original"] = image.cpu().numpy()

    return result


def detect_batch(
    model: MaskedAutoencoder,
    dataloader: DataLoader,
    threshold: float = 0.05,
    device: str = "cuda",
) -> List[Dict]:
    """
    Detect anomalies in a batch of images.

    Args:
        model: Pretrained MAE model
        dataloader: DataLoader with images
        threshold: Anomaly threshold
        device: Device to use

    Returns:
        List of results for each image
    """
    detector = ReconstructionAnomalyDetector(model, device=device)
    detector.threshold = threshold

    results = []

    for batch in tqdm(dataloader, desc="Detecting anomalies"):
        if isinstance(batch, dict):
            images = batch["image"].to(device)
            study_ids = batch.get("study_id", [None] * len(images))
        elif isinstance(batch, (list, tuple)):
            images = batch[0].to(device)
            if len(batch) > 1 and isinstance(batch[1], dict):
                study_ids = batch[1].get("study_id", [None] * len(images))
            else:
                study_ids = [None] * len(images)
        else:
            images = batch.to(device)
            study_ids = [None] * len(images)

        # Compute scores
        with torch.no_grad():
            scores = detector.score(images)

        for i, (score, study_id) in enumerate(zip(scores, study_ids)):
            results.append({
                "study_id": study_id,
                "anomaly_score": float(score),
                "prediction": "ANOMALOUS" if score > threshold else "NORMAL",
            })

    return results


def visualize_result(
    result: Dict,
    output_path: Optional[Path] = None,
) -> None:
    """Visualize anomaly detection result."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Original image
    original = result["original"][0].transpose(1, 2, 0)
    # Denormalize
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    original = std * original + mean
    original = np.clip(original, 0, 1)

    axes[0].imshow(original)
    axes[0].set_title("Original")
    axes[0].axis("off")

    # Reconstruction
    recon = result["reconstruction"][0].transpose(1, 2, 0)
    recon = std * recon + mean
    recon = np.clip(recon, 0, 1)

    axes[1].imshow(recon)
    axes[1].set_title("Reconstruction")
    axes[1].axis("off")

    # Heatmap
    heatmap = result["heatmap"][0, 0]
    axes[2].imshow(heatmap, cmap="hot")
    axes[2].set_title(f"Anomaly Heatmap (Score: {result['anomaly_score']:.4f})")
    axes[2].axis("off")

    plt.suptitle(
        f"Prediction: {result['prediction']} "
        f"(Threshold: {result['threshold']:.4f})",
        fontsize=14,
    )
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        logger.info(f"Saved visualization to {output_path}")
    else:
        plt.show()

    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Detect anomalies in chest X-rays",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Input
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--image", type=Path,
        help="Path to single image file"
    )
    input_group.add_argument(
        "--hdf5", type=Path,
        help="Path to HDF5 file with images"
    )
    input_group.add_argument(
        "--directory", type=Path,
        help="Directory containing image files"
    )

    # Model
    parser.add_argument(
        "--model", type=Path, required=True,
        help="Path to trained MAE model checkpoint"
    )
    parser.add_argument(
        "--detector", type=Path, default=None,
        help="Path to fitted anomaly detector (optional)"
    )

    # Options
    parser.add_argument(
        "--threshold", type=float, default=0.05,
        help="Anomaly threshold (default: 0.05)"
    )
    parser.add_argument(
        "--batch-size", type=int, default=32,
        help="Batch size for batch processing"
    )
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use"
    )

    # Output
    parser.add_argument(
        "--output", type=Path, default=None,
        help="Output file for results (JSON)"
    )
    parser.add_argument(
        "--visualize", action="store_true",
        help="Generate visualizations"
    )
    parser.add_argument(
        "--viz-output", type=Path, default=None,
        help="Output path for visualizations"
    )

    args = parser.parse_args()

    # Load model
    model = load_model(args.model, args.device)

    # Process input
    if args.image:
        # Single image
        logger.info(f"Processing single image: {args.image}")
        result = detect_single_image(
            model, args.image,
            threshold=args.threshold,
            device=args.device,
            visualize=args.visualize,
        )

        # Print results
        print(f"\nImage: {result['image_path']}")
        print(f"Anomaly Score: {result['anomaly_score']:.4f}")
        print(f"Prediction: {result['prediction']} (threshold: {result['threshold']:.4f})")

        # Visualize
        if args.visualize and "heatmap" in result:
            viz_path = args.viz_output or Path(args.image).with_suffix(".anomaly.png")
            visualize_result(result, viz_path)

        results = [result]

    elif args.hdf5:
        # Batch from HDF5
        logger.info(f"Processing HDF5 file: {args.hdf5}")

        transform = get_mae_augmentations(target_size=(224, 224), training=False)
        dataset = MIMICCXRDataset(
            args.hdf5,
            transform=transform,
            return_metadata=True,
        )
        dataloader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=4,
        )

        results = detect_batch(model, dataloader, args.threshold, args.device)

        # Print summary
        n_anomalous = sum(1 for r in results if r["prediction"] == "ANOMALOUS")
        print(f"\nProcessed {len(results)} images")
        print(f"Anomalous: {n_anomalous} ({100*n_anomalous/len(results):.1f}%)")
        print(f"Normal: {len(results) - n_anomalous} ({100*(len(results)-n_anomalous)/len(results):.1f}%)")

    elif args.directory:
        # Batch from directory
        logger.info(f"Processing directory: {args.directory}")

        image_extensions = {".jpg", ".jpeg", ".png", ".dcm"}
        image_files = [
            f for f in args.directory.iterdir()
            if f.suffix.lower() in image_extensions
        ]

        results = []
        for image_path in tqdm(image_files, desc="Processing images"):
            result = detect_single_image(
                model, image_path,
                threshold=args.threshold,
                device=args.device,
                visualize=False,
            )
            results.append(result)

        # Print summary
        n_anomalous = sum(1 for r in results if r["prediction"] == "ANOMALOUS")
        print(f"\nProcessed {len(results)} images")
        print(f"Anomalous: {n_anomalous} ({100*n_anomalous/len(results):.1f}%)")

    # Save results
    if args.output:
        # Remove non-serializable fields
        serializable_results = []
        for r in results:
            sr = {k: v for k, v in r.items() if k not in ["heatmap", "reconstruction", "original"]}
            serializable_results.append(sr)

        with open(args.output, "w") as f:
            json.dump(serializable_results, f, indent=2)
        logger.info(f"Saved results to {args.output}")


if __name__ == "__main__":
    main()
