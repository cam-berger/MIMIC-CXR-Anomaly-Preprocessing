"""
Pytest configuration and fixtures for step2_preprocessing tests.

This module provides shared fixtures for all tests including:
- Sample configurations
- Mock processors
- Test data
- Temporary directories
"""
import pytest
import yaml
import tempfile
import torch
import pandas as pd
from pathlib import Path
from typing import Dict, Any
from unittest.mock import Mock


@pytest.fixture
def sample_config() -> Dict[str, Any]:
    """Minimal valid configuration for testing"""
    return {
        'image': {
            'preserve_full_resolution': True,
            'normalize_method': 'minmax',
            'augmentation': {
                'enabled': False,
                'rotation_range': 0
            }
        },
        'structured': {
            'missing_token': 'NOT_DONE',
            'encoding_method': 'aggregated',
            'temporal_features': {
                'enabled': True
            },
            'priority_labs': ['hemoglobin', 'wbc', 'creatinine'],
            'priority_vitals': ['heartrate', 'sbp', 'dbp']
        },
        'text': {
            'ner': {
                'model': 'en_core_sci_md',
                'extract_entities': True
            },
            'summarization': {
                'use_claude': False,  # Disabled for testing
                'model': 'claude-sonnet-4-5-20250929',
                'temperature': 0.0,
                'max_summary_length': 500
            },
            'note_rewriting': {
                'enabled': False,
                'use_claude': False,
                'model': 'claude-sonnet-4-5-20250929',
                'temperature': 0.0,
                'max_rewrite_length': 2000
            },
            'retrieval': {
                'use_entity_based': True,
                'use_semantic_fallback': True,
                'max_sentences': 20,
                'similarity_threshold': 0.3
            },
            'tokenizer': {
                'model': 'emilyalsentzer/Bio_ClinicalBERT',
                'max_length': 512,
                'padding': 'max_length',
                'truncation': True
            }
        },
        'processing': {
            'batch_size': 1,
            'num_workers': 0  # Single threaded for testing
        }
    }


@pytest.fixture
def sample_config_file(tmp_path, sample_config) -> Path:
    """Create a temporary config YAML file"""
    config_file = tmp_path / 'config.yaml'
    with open(config_file, 'w') as f:
        yaml.dump(sample_config, f)
    return config_file


@pytest.fixture
def temp_output_dir(tmp_path) -> Path:
    """Create temporary output directory structure"""
    output_dir = tmp_path / 'output'
    output_dir.mkdir()

    # Create subdirectories
    (output_dir / 'train' / 'images').mkdir(parents=True)
    (output_dir / 'train' / 'structured_features').mkdir(parents=True)
    (output_dir / 'train' / 'text_features').mkdir(parents=True)
    (output_dir / 'train' / 'metadata').mkdir(parents=True)

    (output_dir / 'val' / 'images').mkdir(parents=True)
    (output_dir / 'val' / 'structured_features').mkdir(parents=True)
    (output_dir / 'val' / 'text_features').mkdir(parents=True)
    (output_dir / 'val' / 'metadata').mkdir(parents=True)

    return output_dir


@pytest.fixture
def sample_image() -> torch.Tensor:
    """Create a sample medical image tensor"""
    # Grayscale image 512x512
    return torch.rand(1, 512, 512)


@pytest.fixture
def sample_clinical_note() -> str:
    """Sample clinical note with abbreviations"""
    return """Patient c/o c/p x 2hrs. Hx HTN on lisinopril, DM2 on metformin.
Vitals: HR 88, BP 135/85, SpO2 98%. No SOB, no diaphoresis. CXR ordered."""


@pytest.fixture
def sample_vitals_df() -> pd.DataFrame:
    """Sample vital signs DataFrame"""
    return pd.DataFrame({
        'subject_id': [10000] * 3,
        'stay_id': [30000] * 3,
        'charttime': pd.to_datetime([
            '2020-01-01 10:00',
            '2020-01-01 12:00',
            '2020-01-01 14:00'
        ]),
        'temperature': [98.6, 98.8, 99.0],
        'heartrate': [75, 78, 80],
        'resprate': [16, 18, 18],
        'o2sat': [98, 97, 98],
        'sbp': [120, 125, 122],
        'dbp': [80, 82, 81]
    })


@pytest.fixture
def sample_labs_df() -> pd.DataFrame:
    """Sample lab results DataFrame"""
    return pd.DataFrame({
        'subject_id': [10000] * 3,
        'charttime': pd.to_datetime([
            '2020-01-01 09:00',
            '2020-01-01 09:00',
            '2020-01-01 09:00'
        ]),
        'itemid': [51222, 51301, 50912],  # Hemoglobin, WBC, Creatinine
        'valuenum': [13.5, 8.2, 1.0],
        'valueuom': ['g/dL', 'K/uL', 'mg/dL']
    })


@pytest.fixture
def sample_cohort_df(tmp_path) -> pd.DataFrame:
    """Sample cohort DataFrame for testing"""
    return pd.DataFrame({
        'subject_id': [10000, 10001, 10002],
        'study_id': [50000, 50001, 50002],
        'dicom_id': ['abc123', 'def456', 'ghi789'],
        'study_datetime': pd.to_datetime([
            '2020-01-01 10:30',
            '2020-01-02 14:00',
            '2020-01-03 09:15'
        ]),
        'stay_id': [30000, 30001, 30002]
    })


@pytest.fixture
def mock_image_loader(sample_config):
    """Mock image loader for testing"""
    mock = Mock()
    mock.load_and_process.return_value = torch.rand(1, 512, 512)
    mock.config = sample_config['image']
    return mock


@pytest.fixture
def mock_temporal_processor(sample_config):
    """Mock temporal processor for testing"""
    mock = Mock()
    mock.extract_features.return_value = {
        'vital_heartrate': {
            'is_missing': False,
            'last_value': 80.0,
            'trend_slope': 1.5,
            'mean_value': 78.0
        }
    }
    mock.config = sample_config['structured']
    return mock


@pytest.fixture
def mock_text_processor(sample_config):
    """Mock text processor for testing"""
    mock = Mock()
    mock.process_note.return_value = {
        'summary': 'Patient presents with chest pain.',
        'tokens': {
            'input_ids': torch.randint(0, 1000, (512,)),
            'attention_mask': torch.ones(512),
            'num_tokens': 25,
            'is_truncated': False
        },
        'num_entities': 5,
        'entities': ['chest pain', 'hypertension', 'diabetes'],
        'context_sentences': 3
    }
    mock.rewrite_note.side_effect = lambda x: x  # Return original
    mock.config = sample_config['text']
    return mock


@pytest.fixture(autouse=True)
def reset_torch_random():
    """Reset torch random seed before each test for reproducibility"""
    torch.manual_seed(42)
    yield
