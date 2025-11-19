"""
Configuration settings for normal case identification.
"""
from dataclasses import dataclass, field
from typing import List, Set


@dataclass
class FilterConfig:
    """Configuration for filtering normal cases."""

    # Radiology Report Criteria
    require_no_finding: bool = True
    no_finding_value: float = 1.0
    exclude_pathology_labels: List[str] = field(default_factory=lambda: [
        "Atelectasis",
        "Cardiomegaly",
        "Consolidation",
        "Edema",
        "Enlarged Cardiomediastinum",
        "Fracture",
        "Lung Lesion",
        "Lung Opacity",
        "Pleural Effusion",
        "Pleural Other",
        "Pneumonia",
        "Pneumothorax",
        "Support Devices"
    ])

    # Clinical Context Criteria
    acceptable_dispositions: Set[str] = field(default_factory=lambda: {
        "HOME",
        "DISCHARGED",
        "LEFT WITHOUT BEING SEEN",
        "LEFT AGAINST MEDICAL ADVICE"
    })

    excluded_dispositions: Set[str] = field(default_factory=lambda: {
        "ADMITTED",
        "ELOPED",
        "EXPIRED"
    })

    # Critical diagnosis exclusions (ICD-9 and ICD-10 code prefixes)
    critical_diagnosis_patterns: List[str] = field(default_factory=lambda: [
        # Sepsis
        "995.9", "785.52", "A41", "A40", "R65",
        # Pneumonia
        "480", "481", "482", "483", "484", "485", "486", "J12", "J13", "J14", "J15", "J16", "J17", "J18",
        # Acute MI
        "410", "I21", "I22",
        # CHF
        "428", "I50",
        # Respiratory failure
        "518.81", "518.82", "J96",
        # Pulmonary embolism
        "415.1", "I26",
        # Pneumothorax
        "512", "J93",
        # Pleural effusion
        "511", "J90", "J91",
        # ARDS
        "518.82", "J80",
    ])

    # Time window for matching ED stays with CXR studies (hours)
    time_window_hours: int = 24

    # Minimum age for inclusion
    min_age: int = 18

    # Train/validation split
    validation_fraction: float = 0.15
    random_seed: int = 42

    # Verbose logging
    verbose: bool = True


@dataclass
class ProcessingConfig:
    """Configuration for data processing."""

    # Chunk size for reading large CSV files
    chunk_size: int = 50000

    # Number of samples to validate manually
    validation_sample_size: int = 100

    # Enable parallel processing
    use_parallel: bool = False
    n_jobs: int = -1

    # Memory optimization
    use_low_memory: bool = True
    dtype_optimization: bool = True
