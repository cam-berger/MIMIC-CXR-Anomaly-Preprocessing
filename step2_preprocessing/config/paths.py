"""
Path configuration for Step 2 preprocessing
"""
from pathlib import Path
import yaml


class Step2Paths:
    """Centralized path management for Step 2"""

    def __init__(self, config_path=None):
        if config_path is None:
            config_path = Path(__file__).parent / "config.yaml"

        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        # Base directories
        self.project_root = Path(__file__).parent.parent

        # Step 1 outputs
        self.step1_train = Path(self.config['data']['step1_cohort_train'])
        self.step1_val = Path(self.config['data']['step1_cohort_val'])

        # MIMIC data
        self.mimic_cxr_base = Path(self.config['data']['mimic_cxr_base'])
        self.mimic_iv_base = Path(self.config['data']['mimic_iv_base'])
        self.mimic_ed_base = Path(self.config['data']['mimic_ed_base'])

        # Output directories
        self.output_base = self.project_root / self.config['data']['output_base']
        self.processed_images = self.output_base / "processed_images"
        self.structured_features = self.output_base / "structured_features"
        self.text_summaries = self.output_base / "text_summaries"
        self.tokenized_text = self.output_base / "tokenized_text"
        self.multimodal_data = self.output_base / "multimodal_dataset"

        # Create output directories
        self._create_directories()

    def _create_directories(self):
        """Create all output directories"""
        directories = [
            self.output_base,
            self.processed_images / "train",
            self.processed_images / "val",
            self.structured_features,
            self.text_summaries,
            self.tokenized_text,
            self.multimodal_data,
        ]
        for dir_path in directories:
            dir_path.mkdir(parents=True, exist_ok=True)

    def get_image_path(self, subject_id, study_id):
        """Get path to MIMIC-CXR image directory"""
        # MIMIC-CXR directory structure: files/p{XX}/p{subject_id}/s{study_id}/
        subject_str = str(subject_id)
        prefix = f"p{subject_str[:2]}"
        patient_dir = f"p{subject_id}"
        study_dir = f"s{study_id}"

        return self.mimic_cxr_base / "files" / prefix / patient_dir / study_dir

    def get_lab_events_path(self):
        """Get path to MIMIC-IV lab events"""
        return self.mimic_iv_base / "hosp" / "labevents.csv"

    def get_vital_signs_path(self):
        """Get path to MIMIC-ED vital signs"""
        return self.mimic_ed_base / "ed" / "vitalsign.csv"

    def get_triage_path(self):
        """Get path to MIMIC-ED triage"""
        return self.mimic_ed_base / "ed" / "triage.csv"
