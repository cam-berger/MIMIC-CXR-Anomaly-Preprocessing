"""
Configuration settings using environment variables.

Set these environment variables or create a .env file:
    MIMIC_CXR_JPG_PATH=/path/to/mimic-cxr-jpg/2.1.0
    MIMIC_IV_PATH=/path/to/mimiciv/3.1
    MIMIC_IV_ED_PATH=/path/to/mimic-iv-ed/2.2
    MIMIC_IV_NOTE_PATH=/path/to/mimic-iv-note/2.2
    CXR_PRO_PATH=/path/to/cxr-pro/1.0.0
    OUTPUT_PATH=/path/to/output
    ANTHROPIC_API_KEY=sk-...
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
from functools import lru_cache

# Try to load .env file if python-dotenv is available
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


@dataclass
class DataPaths:
    """Paths to MIMIC datasets - loaded from environment variables."""

    # Base paths from environment
    mimic_cxr_jpg: Path = field(default_factory=lambda: Path(os.environ.get(
        "MIMIC_CXR_JPG_PATH", "/data/mimic-cxr-jpg/2.1.0"
    )))
    mimic_iv: Path = field(default_factory=lambda: Path(os.environ.get(
        "MIMIC_IV_PATH", "/data/mimiciv/3.1"
    )))
    mimic_iv_ed: Path = field(default_factory=lambda: Path(os.environ.get(
        "MIMIC_IV_ED_PATH", "/data/mimic-iv-ed/2.2"
    )))
    mimic_iv_note: Path = field(default_factory=lambda: Path(os.environ.get(
        "MIMIC_IV_NOTE_PATH", "/data/mimic-iv-note/2.2"
    )))
    cxr_pro: Path = field(default_factory=lambda: Path(os.environ.get(
        "CXR_PRO_PATH", "/data/cxr-pro/1.0.0"
    )))
    output: Path = field(default_factory=lambda: Path(os.environ.get(
        "OUTPUT_PATH", "./output"
    )))

    # Derived paths - MIMIC-CXR-JPG
    @property
    def cxr_chexpert(self) -> Path:
        return self.mimic_cxr_jpg / "mimic-cxr-2.0.0-chexpert.csv.gz"

    @property
    def cxr_metadata(self) -> Path:
        return self.mimic_cxr_jpg / "mimic-cxr-2.0.0-metadata.csv.gz"

    @property
    def cxr_split(self) -> Path:
        return self.mimic_cxr_jpg / "mimic-cxr-2.0.0-split.csv.gz"

    @property
    def cxr_images(self) -> Path:
        return self.mimic_cxr_jpg / "files"

    # Derived paths - MIMIC-IV (hosp module)
    @property
    def iv_hosp(self) -> Path:
        return self.mimic_iv / "hosp"

    @property
    def iv_patients(self) -> Path:
        return self.iv_hosp / "patients.csv.gz"

    @property
    def iv_admissions(self) -> Path:
        return self.iv_hosp / "admissions.csv.gz"

    @property
    def iv_diagnoses_icd(self) -> Path:
        return self.iv_hosp / "diagnoses_icd.csv.gz"

    @property
    def iv_procedures_icd(self) -> Path:
        return self.iv_hosp / "procedures_icd.csv.gz"

    @property
    def iv_labevents(self) -> Path:
        return self.iv_hosp / "labevents.csv.gz"

    @property
    def iv_d_labitems(self) -> Path:
        return self.iv_hosp / "d_labitems.csv.gz"

    @property
    def iv_d_icd_diagnoses(self) -> Path:
        return self.iv_hosp / "d_icd_diagnoses.csv.gz"

    @property
    def iv_d_icd_procedures(self) -> Path:
        return self.iv_hosp / "d_icd_procedures.csv.gz"

    @property
    def iv_transfers(self) -> Path:
        return self.iv_hosp / "transfers.csv.gz"

    # Derived paths - MIMIC-IV-ED
    @property
    def ed_stays(self) -> Path:
        return self.mimic_iv_ed / "ed" / "edstays.csv.gz"

    @property
    def ed_diagnosis(self) -> Path:
        return self.mimic_iv_ed / "ed" / "diagnosis.csv.gz"

    @property
    def ed_triage(self) -> Path:
        return self.mimic_iv_ed / "ed" / "triage.csv.gz"

    @property
    def ed_vitalsign(self) -> Path:
        return self.mimic_iv_ed / "ed" / "vitalsign.csv.gz"

    # Derived paths - MIMIC-IV-Note
    @property
    def note_discharge(self) -> Path:
        return self.mimic_iv_note / "note" / "discharge.csv.gz"

    @property
    def note_radiology(self) -> Path:
        return self.mimic_iv_note / "note" / "radiology.csv.gz"

    # Derived paths - CXR-PRO (radiology reports with prior refs removed)
    @property
    def cxr_pro_train(self) -> Path:
        return self.cxr_pro / "mimic_train_impressions.csv"

    @property
    def cxr_pro_test(self) -> Path:
        return self.cxr_pro / "mimic_test_impressions.csv"

    # Output paths
    @property
    def cohort_dir(self) -> Path:
        return self.output / "cohorts"

    @property
    def preprocessed_dir(self) -> Path:
        return self.output / "preprocessed"

    def validate(self) -> list[str]:
        """Check which paths exist and return list of missing ones."""
        missing = []
        critical_paths = [
            ("MIMIC-CXR-JPG", self.mimic_cxr_jpg),
            ("MIMIC-IV", self.mimic_iv),
            ("MIMIC-IV-ED", self.mimic_iv_ed),
        ]
        for name, path in critical_paths:
            if not path.exists():
                missing.append(f"{name}: {path}")
        return missing


@dataclass
class CohortConfig:
    """Configuration for cohort generation."""

    # Radiology filtering (CheXpert labels)
    chexpert_no_finding_threshold: float = 1.0  # Require explicit "No Finding"

    # Temporal matching
    ed_cxr_time_window_hours: int = 24  # CXR must be within this window of ED visit

    # Clinical filtering
    min_age: int = 18
    acceptable_dispositions: frozenset = field(default_factory=lambda: frozenset({
        "HOME", "DISCHARGED", "LEFT WITHOUT BEING SEEN",
        "LEFT AGAINST MEDICAL ADVICE", "OTHER"
    }))

    # Diagnoses to exclude from "normal" cohort (ICD-9/10 prefixes)
    excluded_diagnosis_patterns: tuple = (
        # Sepsis
        "995.9", "A41", "A40", "R65.2",
        # Pneumonia
        "480", "481", "482", "483", "484", "485", "486",
        "J12", "J13", "J14", "J15", "J16", "J17", "J18",
        # Acute MI
        "410", "I21", "I22",
        # Respiratory failure
        "518.81", "518.82", "518.84", "J96",
        # ARDS
        "518.82", "J80",
        # Pulmonary embolism
        "415.1", "I26",
    )

    # Train/validation split
    validation_fraction: float = 0.15
    random_seed: int = 42


@dataclass
class PreprocessingConfig:
    """Configuration for preprocessing pipeline."""

    # Image processing
    image_normalize: str = "minmax"  # "minmax", "standardize", or "none"
    image_chunk_size: int = 100  # Images per HDF5 chunk
    image_compression: str = "gzip"  # HDF5 compression
    image_compression_level: int = 4

    # Structured data (labs, vitals)
    labs_time_window_before_hours: int = 48
    labs_time_window_after_hours: int = 24
    priority_labs: tuple = (
        "wbc", "hemoglobin", "hematocrit", "platelets",
        "glucose", "creatinine", "bun", "sodium", "potassium",
        "chloride", "bicarbonate", "calcium", "magnesium",
        "lactate", "troponin", "bnp", "procalcitonin",
    )
    priority_vitals: tuple = (
        "temperature", "heartrate", "resprate",
        "o2sat", "sbp", "dbp", "pain",
    )

    # Text processing
    tokenizer_model: str = "emilyalsentzer/Bio_ClinicalBERT"
    max_text_length: int = 512
    use_claude_summarization: bool = True
    claude_model: str = "claude-sonnet-4-5-20250929"

    # Processing
    num_workers: int = 4
    batch_size: int = 100


@dataclass
class Settings:
    """Main settings container."""

    paths: DataPaths = field(default_factory=DataPaths)
    cohort: CohortConfig = field(default_factory=CohortConfig)
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    anthropic_api_key: Optional[str] = field(default_factory=lambda: os.environ.get("ANTHROPIC_API_KEY"))

    def __post_init__(self):
        # Ensure output directories exist
        self.paths.output.mkdir(parents=True, exist_ok=True)
        self.paths.cohort_dir.mkdir(parents=True, exist_ok=True)
        self.paths.preprocessed_dir.mkdir(parents=True, exist_ok=True)


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
