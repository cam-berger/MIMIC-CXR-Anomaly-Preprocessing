"""
Path management for Step 2 preprocessing pipeline
"""
from pathlib import Path
import yaml


class Step2Paths:
    """Manages all file paths for Step 2 preprocessing"""

    def __init__(self, config_path: Path | str):
        """Initialize paths from config file

        Args:
            config_path: Path to YAML configuration file
        """
        self.config_path = Path(config_path)

        # Load configuration
        with open(self.config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        # Get paths from config
        data_config = self.config['data']

        # Input paths (cohort CSVs from Step 1)
        self.step1_train = Path(data_config['step1_cohort_train'])
        self.step1_val = Path(data_config['step1_cohort_val'])

        # MIMIC data paths
        self.mimic_cxr_base = Path(data_config['mimic_cxr_base'])
        self.mimic_iv_base = Path(data_config['mimic_iv_base'])
        self.mimic_ed_base = Path(data_config['mimic_ed_base'])

        # Output paths
        self.output_base = Path(data_config['output_base'])

        # Create output directories
        self._create_directories()

    def _create_directories(self):
        """Create all necessary output directories"""
        self.output_base.mkdir(parents=True, exist_ok=True)

        # Create subdirectories for train and val splits
        for split in ['train', 'val']:
            split_dir = self.output_base / split
            split_dir.mkdir(parents=True, exist_ok=True)

            # Create modality-specific subdirectories
            (split_dir / 'images').mkdir(exist_ok=True)
            (split_dir / 'structured_features').mkdir(exist_ok=True)
            (split_dir / 'text_features').mkdir(exist_ok=True)
            (split_dir / 'metadata').mkdir(exist_ok=True)

    def get_image_path(self, subject_id: int, study_id: int) -> Path:
        """Get path to MIMIC-CXR study directory

        Args:
            subject_id: Patient subject ID
            study_id: CXR study ID

        Returns:
            Path to study directory containing DICOM images
        """
        # MIMIC-CXR directory structure: files/p{first_2_digits}/p{subject_id}/s{study_id}/
        subject_str = str(subject_id)
        subject_prefix = f"p{subject_str[:2]}"
        subject_dir = f"p{subject_str}"
        study_dir = f"s{study_id}"

        return self.mimic_cxr_base / "files" / subject_prefix / subject_dir / study_dir

    def get_lab_events_path(self) -> Path:
        """Get path to MIMIC-IV labevents file"""
        return self.mimic_iv_base / "hosp" / "labevents.csv"

    def get_vital_signs_path(self) -> Path:
        """Get path to MIMIC-IV-ED vitalsign file"""
        return self.mimic_ed_base / "ed" / "vitalsign.csv"

    def get_triage_path(self) -> Path:
        """Get path to MIMIC-IV-ED triage file"""
        return self.mimic_ed_base / "ed" / "triage.csv"

    def __repr__(self):
        return f"Step2Paths(config={self.config_path}, output={self.output_base})"
