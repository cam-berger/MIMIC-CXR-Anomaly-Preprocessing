"""
Data paths configuration for MIMIC datasets.
"""
from pathlib import Path
from typing import Dict


class DataPaths:
    """Central configuration for all MIMIC dataset paths."""

    def __init__(self):
        # Base directories
        self.MIMIC_CXR_BASE = Path("/media/dev/MIMIC_DATA/mimic-cxr-jpg")
        self.MIMIC_DATA_BASE = Path("/home/dev/Documents/Portfolio/MIMIC_Data/physionet.org/files")

        # MIMIC-CXR-JPG paths
        self.CXR_CHEXPERT = self.MIMIC_CXR_BASE / "mimic-cxr-2.0.0-chexpert.csv.gz"
        self.CXR_NEGBIO = self.MIMIC_CXR_BASE / "mimic-cxr-2.0.0-negbio.csv.gz"
        self.CXR_METADATA = self.MIMIC_CXR_BASE / "mimic-cxr-2.0.0-metadata.csv.gz"
        self.CXR_SPLIT = self.MIMIC_CXR_BASE / "mimic-cxr-2.0.0-split.csv.gz"
        self.CXR_IMAGES = self.MIMIC_CXR_BASE / "files"

        # MIMIC-IV-ED paths
        self.ED_BASE = self.MIMIC_DATA_BASE / "mimic-iv-ed" / "2.2" / "ed"
        self.ED_STAYS = self.ED_BASE / "edstays.csv"
        self.ED_DIAGNOSIS = self.ED_BASE / "diagnosis.csv"
        self.ED_TRIAGE = self.ED_BASE / "triage.csv"
        self.ED_VITALSIGN = self.ED_BASE / "vitalsign.csv"

        # MIMIC-IV paths
        self.IV_BASE = self.MIMIC_DATA_BASE / "mimiciv" / "3.1" / "hosp"
        self.IV_PATIENTS = self.IV_BASE / "patients.csv"
        self.IV_ADMISSIONS = self.IV_BASE / "admissions.csv"
        self.IV_DIAGNOSES = self.IV_BASE / "diagnoses_icd.csv"
        self.IV_LABEVENTS = self.IV_BASE / "labevents.csv"
        self.IV_TRANSFERS = self.IV_BASE / "transfers.csv"

    def validate_paths(self) -> Dict[str, bool]:
        """Validate that all required paths exist."""
        paths_to_check = {
            "CXR_CHEXPERT": self.CXR_CHEXPERT,
            "CXR_METADATA": self.CXR_METADATA,
            "ED_STAYS": self.ED_STAYS,
            "ED_DIAGNOSIS": self.ED_DIAGNOSIS,
            "IV_PATIENTS": self.IV_PATIENTS,
            "IV_ADMISSIONS": self.IV_ADMISSIONS,
        }

        results = {}
        for name, path in paths_to_check.items():
            results[name] = path.exists()

        return results

    def get_output_dir(self, subdir: str = "") -> Path:
        """Get output directory for results."""
        base_output = Path("/home/dev/Documents/Portfolio/MIMIC/MIMIC-CXR-Anomaly-Preprocessing/output")
        if subdir:
            output_dir = base_output / subdir
        else:
            output_dir = base_output
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir
