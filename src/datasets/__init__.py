"""
Dataset loaders for MIMIC data sources.

Linking Keys Across Datasets:
=============================
- subject_id: Patient identifier (consistent across ALL MIMIC datasets)
- hadm_id: Hospital admission ID (links MIMIC-IV tables)
- stay_id: ED stay ID (links MIMIC-IV-ED tables)
- study_id: Radiology study ID (links MIMIC-CXR tables)
- dicom_id: Individual image ID within a study

Relationships:
=============
- One subject can have multiple hospital admissions (hadm_id)
- One subject can have multiple ED stays (stay_id)
- One subject can have multiple radiology studies (study_id)
- One study can have multiple images/views (dicom_id)
- An ED stay MAY result in a hospital admission (stay_id -> hadm_id)
- A radiology study has a datetime that can be matched to ED/admission windows
"""

from .mimic_iv import MIMICIVLoader
from .mimic_iv_ed import MIMICIVEDLoader
from .mimic_cxr import MIMICCXRLoader
from .cxr_pro import CXRPROLoader
from .linker import DatasetLinker

__all__ = [
    "MIMICIVLoader",
    "MIMICIVEDLoader",
    "MIMICCXRLoader",
    "CXRPROLoader",
    "DatasetLinker",
]
