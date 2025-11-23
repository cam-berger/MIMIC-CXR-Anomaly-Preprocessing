"""
Configuration loader utilities.
"""
from pathlib import Path
from typing import Dict


class PathsConfig:
    """Simple paths configuration for processors."""

    def __init__(self, config: Dict):
        self.config = config

        # MIMIC data paths
        self.mimic_cxr_base = Path(config['data']['mimic_cxr_base'])
        self.mimic_iv_base = Path(config['data']['mimic_iv_base'])
        self.mimic_ed_base = Path(config['data']['mimic_ed_base'])

        # CXR paths
        self.cxr_images = self.mimic_cxr_base / "files"

        # ED paths
        self.ed_vitalsign = self.mimic_ed_base / "ed" / "vitalsign.csv"
        self.ed_triage = self.mimic_ed_base / "ed" / "triage.csv"

        # IV paths
        self.iv_labevents = self.mimic_iv_base / "hosp" / "labevents.csv"

    def get_cxr_images_path(self) -> Path:
        """Get path to CXR images directory"""
        return self.cxr_images

    def get_vital_signs_path(self) -> Path:
        """Get path to vital signs CSV"""
        return self.ed_vitalsign

    def get_triage_path(self) -> Path:
        """Get path to triage CSV"""
        return self.ed_triage

    def get_lab_events_path(self) -> Path:
        """Get path to lab events CSV"""
        return self.iv_labevents


def load_paths_config(config: Dict) -> PathsConfig:
    """
    Load paths configuration from config dict.

    Args:
        config: Configuration dictionary

    Returns:
        PathsConfig instance
    """
    return PathsConfig(config)
