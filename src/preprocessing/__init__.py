"""
Preprocessing pipeline for multimodal MIMIC-CXR data.

Outputs:
- Images: HDF5 file with chunked, compressed tensors
- Structured data: Parquet file with labs, vitals, demographics
- Text: Parquet file with reports, summaries, tokens

All preprocessing is designed for efficient batch processing with
multiprocessing support.
"""

from .images import ImagePreprocessor
from .structured import StructuredPreprocessor
from .text import TextPreprocessor
from .pipeline import PreprocessingPipeline

__all__ = [
    "ImagePreprocessor",
    "StructuredPreprocessor",
    "TextPreprocessor",
    "PreprocessingPipeline",
]
