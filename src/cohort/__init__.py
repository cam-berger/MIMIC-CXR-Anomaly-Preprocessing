"""
Cohort generation for MIMIC-CXR anomaly detection.

Two cohorts are generated:
1. Normal cohort (~33k): CXR studies with "No Finding" for unsupervised pretraining
2. Anomalous cohort (~200k): CXR studies with pathological findings for classification
"""

from .builder import CohortBuilder

__all__ = ["CohortBuilder"]
