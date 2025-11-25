"""
Preprocessing pipeline orchestrator.

Coordinates all preprocessing steps:
1. Load cohort
2. Process images -> HDF5
3. Process structured data -> Parquet
4. Process text -> Parquet
5. Generate manifest with statistics

Output directory structure:
    output/preprocessed/{cohort_name}/
    ├── images.h5
    ├── structured.parquet
    ├── text.parquet
    └── manifest.json
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd

from ..config import Settings, get_settings
from .images import ImagePreprocessor
from .structured import StructuredPreprocessor
from .text import TextPreprocessor

logger = logging.getLogger(__name__)


class PreprocessingPipeline:
    """Orchestrates the complete preprocessing pipeline."""

    def __init__(self, settings: Optional[Settings] = None):
        """Initialize pipeline."""
        self.settings = settings or get_settings()

        self.image_processor = ImagePreprocessor(settings)
        self.structured_processor = StructuredPreprocessor(settings)
        self.text_processor = TextPreprocessor(settings)

    def process_cohort(
        self,
        cohort_path: Path,
        output_name: str,
        process_images: bool = True,
        process_structured: bool = True,
        process_text: bool = True,
        enable_summarization: bool = False,
        include_context: bool = True,
        num_workers: int = 4,
    ) -> dict:
        """
        Run full preprocessing pipeline on a cohort.

        Args:
            cohort_path: Path to cohort parquet file
            output_name: Name for output directory (e.g., "normal_train")
            process_images: Whether to process images
            process_structured: Whether to process structured data
            process_text: Whether to process text
            enable_summarization: Whether to use Claude for text summarization
            include_context: Whether to include clinical context in summarization
            num_workers: Number of parallel workers for image processing

        Returns:
            Dictionary with processing statistics
        """
        start_time = datetime.now()
        logger.info(f"Starting preprocessing pipeline: {output_name}")
        logger.info(f"Cohort: {cohort_path}")

        # Load cohort
        cohort = pd.read_parquet(cohort_path)
        logger.info(f"Loaded cohort: {len(cohort):,} samples")

        # Setup output directory
        output_dir = self.settings.paths.preprocessed_dir / output_name
        output_dir.mkdir(parents=True, exist_ok=True)

        stats = {
            "cohort_name": output_name,
            "cohort_path": str(cohort_path),
            "total_samples": len(cohort),
            "start_time": start_time.isoformat(),
            "modalities": {},
        }

        # Process images
        if process_images:
            logger.info("=" * 60)
            logger.info("PROCESSING IMAGES")
            logger.info("=" * 60)

            image_output = output_dir / "images.h5"
            image_results = self.image_processor.process_cohort(
                cohort=cohort,
                output_path=image_output,
                num_workers=num_workers,
            )

            stats["modalities"]["images"] = {
                "output_path": str(image_output),
                "total": len(image_results),
                "success": int(image_results["success"].sum()),
                "failed": int((~image_results["success"]).sum()),
                "success_rate": float(image_results["success"].mean()),
            }

            # Save image results
            image_results.to_parquet(output_dir / "image_results.parquet", index=False)

        # Process structured data
        if process_structured:
            logger.info("=" * 60)
            logger.info("PROCESSING STRUCTURED DATA")
            logger.info("=" * 60)

            structured_output = output_dir / "structured.parquet"
            structured_results = self.structured_processor.process_cohort(
                cohort=cohort,
                output_path=structured_output,
            )

            # Calculate statistics
            has_triage = structured_results.get("has_triage", pd.Series([False])).sum()
            has_labs = structured_results.get("has_labs", pd.Series([False])).sum()
            has_vitals = structured_results.get("has_ed_vitals", pd.Series([False])).sum()

            stats["modalities"]["structured"] = {
                "output_path": str(structured_output),
                "total": len(structured_results),
                "with_triage": int(has_triage),
                "with_labs": int(has_labs),
                "with_ed_vitals": int(has_vitals),
            }

        # Process text
        if process_text:
            logger.info("=" * 60)
            logger.info("PROCESSING TEXT")
            logger.info("=" * 60)

            text_output = output_dir / "text.parquet"
            text_results = self.text_processor.process_cohort(
                cohort=cohort,
                output_path=text_output,
                enable_summarization=enable_summarization,
                include_context=include_context,
            )

            has_report = text_results.get("has_report", pd.Series([False])).sum()
            avg_tokens = text_results.get("token_count", pd.Series([0])).mean()

            stats["modalities"]["text"] = {
                "output_path": str(text_output),
                "total": len(text_results),
                "with_report": int(has_report),
                "avg_token_count": float(avg_tokens),
                "summarization_enabled": enable_summarization,
                "include_context": include_context,
            }

        # Finalize
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()

        stats["end_time"] = end_time.isoformat()
        stats["duration_seconds"] = duration
        stats["samples_per_second"] = len(cohort) / duration if duration > 0 else 0

        # Save manifest
        manifest_path = output_dir / "manifest.json"
        with open(manifest_path, "w") as f:
            json.dump(stats, f, indent=2)

        logger.info("=" * 60)
        logger.info("PREPROCESSING COMPLETE")
        logger.info("=" * 60)
        logger.info(f"Duration: {duration:.1f} seconds ({duration/60:.1f} minutes)")
        logger.info(f"Speed: {stats['samples_per_second']:.2f} samples/second")
        logger.info(f"Output: {output_dir}")

        return stats

    def process_all_cohorts(
        self,
        cohorts_dir: Optional[Path] = None,
        **kwargs,
    ) -> dict:
        """
        Process all cohort files in directory.

        Args:
            cohorts_dir: Directory containing cohort parquet files
            **kwargs: Arguments passed to process_cohort

        Returns:
            Dictionary with all processing statistics
        """
        cohorts_dir = cohorts_dir or self.settings.paths.cohort_dir

        all_stats = {}

        # Find all cohort files
        cohort_files = sorted(cohorts_dir.glob("*.parquet"))
        logger.info(f"Found {len(cohort_files)} cohort files in {cohorts_dir}")

        for cohort_path in cohort_files:
            # Extract name from filename (e.g., "normal_train.parquet" -> "normal_train")
            output_name = cohort_path.stem

            # Skip full cohorts (only process train/val splits)
            if output_name.endswith("_full"):
                logger.info(f"Skipping full cohort: {output_name}")
                continue

            stats = self.process_cohort(
                cohort_path=cohort_path,
                output_name=output_name,
                **kwargs,
            )

            all_stats[output_name] = stats

        return all_stats

    @staticmethod
    def load_preprocessed(
        preprocessed_dir: Path,
    ) -> dict:
        """
        Load preprocessed data from directory.

        Args:
            preprocessed_dir: Directory containing preprocessed files

        Returns:
            Dictionary with loaded data:
            - "images": HDF5 file handle (must be closed by caller)
            - "structured": DataFrame
            - "text": DataFrame
            - "manifest": dict
        """
        import h5py

        result = {}

        # Load manifest
        manifest_path = preprocessed_dir / "manifest.json"
        if manifest_path.exists():
            with open(manifest_path) as f:
                result["manifest"] = json.load(f)

        # Load images (return handle, don't load into memory)
        images_path = preprocessed_dir / "images.h5"
        if images_path.exists():
            result["images"] = h5py.File(images_path, "r")

        # Load structured
        structured_path = preprocessed_dir / "structured.parquet"
        if structured_path.exists():
            result["structured"] = pd.read_parquet(structured_path)

        # Load text
        text_path = preprocessed_dir / "text.parquet"
        if text_path.exists():
            result["text"] = pd.read_parquet(text_path)

        return result
