"""
Unit tests for integration/multimodal_dataset.py

Tests dataset loading, sample retrieval, and multi-modality integration.
"""
import pytest
import sys
import pandas as pd
import torch
from pathlib import Path
from unittest.mock import Mock, patch

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.integration.multimodal_dataset import MultimodalMIMICDataset


class TestMultimodalDatasetInitialization:
    """Test dataset initialization"""

    def test_initialization_with_all_modalities(
        self, sample_config, sample_cohort_df, tmp_path,
        mock_image_loader, mock_temporal_processor, mock_text_processor
    ):
        """Test initialization with all modalities enabled"""
        # Save cohort to file
        cohort_path = tmp_path / 'cohort.csv'
        sample_cohort_df.to_csv(cohort_path, index=False)

        # Mock paths
        class MockPaths:
            mimic_cxr_base = tmp_path / 'cxr'
            mimic_iv_base = tmp_path / 'iv'
            mimic_ed_base = tmp_path / 'ed'

        with patch('src.integration.multimodal_dataset.FullResolutionImageLoader', return_value=mock_image_loader), \
             patch('src.integration.multimodal_dataset.TemporalFeatureExtractor', return_value=mock_temporal_processor), \
             patch('src.integration.multimodal_dataset.ClinicalNoteProcessor', return_value=mock_text_processor):

            dataset = MultimodalMIMICDataset(
                cohort_csv_path=cohort_path,
                config=sample_config,
                paths=MockPaths(),
                anthropic_api_key="fake_key",
                split='train',
                load_images=True,
                load_structured=True,
                load_text=True
            )

            assert dataset.load_images is True
            assert dataset.load_structured is True
            assert dataset.load_text is True
            assert len(dataset.cohort) == 3

    def test_initialization_skip_images(
        self, sample_config, sample_cohort_df, tmp_path,
        mock_temporal_processor, mock_text_processor
    ):
        """Test initialization with images skipped"""
        cohort_path = tmp_path / 'cohort.csv'
        sample_cohort_df.to_csv(cohort_path, index=False)

        class MockPaths:
            mimic_cxr_base = tmp_path / 'cxr'
            mimic_iv_base = tmp_path / 'iv'
            mimic_ed_base = tmp_path / 'ed'

        with patch('src.integration.multimodal_dataset.TemporalFeatureExtractor', return_value=mock_temporal_processor), \
             patch('src.integration.multimodal_dataset.ClinicalNoteProcessor', return_value=mock_text_processor):

            dataset = MultimodalMIMICDataset(
                cohort_csv_path=cohort_path,
                config=sample_config,
                paths=MockPaths(),
                anthropic_api_key="fake_key",
                split='train',
                load_images=False,
                load_structured=True,
                load_text=True
            )

            assert dataset.load_images is False
            assert dataset.image_loader is None


class TestMultimodalDatasetSampleRetrieval:
    """Test sample retrieval and __getitem__"""

    def test_getitem_returns_all_modalities(
        self, sample_config, sample_cohort_df, tmp_path,
        mock_image_loader, mock_temporal_processor, mock_text_processor
    ):
        """Test that __getitem__ returns all modalities"""
        cohort_path = tmp_path / 'cohort.csv'
        sample_cohort_df.to_csv(cohort_path, index=False)

        class MockPaths:
            mimic_cxr_base = tmp_path / 'cxr'
            mimic_iv_base = tmp_path / 'iv'
            mimic_ed_base = tmp_path / 'ed'

        # Create mock DICOM file
        (tmp_path / 'cxr' / 'files' / 'p10' / 'p10000' / 's50000').mkdir(parents=True)
        dicom_file = tmp_path / 'cxr' / 'files' / 'p10' / 'p10000' / 's50000' / 'abc123.jpg'
        dicom_file.touch()

        with patch('src.integration.multimodal_dataset.FullResolutionImageLoader', return_value=mock_image_loader), \
             patch('src.integration.multimodal_dataset.TemporalFeatureExtractor', return_value=mock_temporal_processor), \
             patch('src.integration.multimodal_dataset.ClinicalNoteProcessor', return_value=mock_text_processor):

            dataset = MultimodalMIMICDataset(
                cohort_csv_path=cohort_path,
                config=sample_config,
                paths=MockPaths(),
                anthropic_api_key="fake_key",
                split='train',
                load_images=True,
                load_structured=True,
                load_text=True
            )

            sample = dataset[0]

            # Check all modalities present
            assert 'image' in sample
            assert 'structured' in sample
            assert 'text' in sample
            assert 'metadata' in sample

    def test_getitem_with_skipped_modality(
        self, sample_config, sample_cohort_df, tmp_path,
        mock_temporal_processor, mock_text_processor
    ):
        """Test __getitem__ with one modality skipped"""
        cohort_path = tmp_path / 'cohort.csv'
        sample_cohort_df.to_csv(cohort_path, index=False)

        class MockPaths:
            mimic_cxr_base = tmp_path / 'cxr'
            mimic_iv_base = tmp_path / 'iv'
            mimic_ed_base = tmp_path / 'ed'

        with patch('src.integration.multimodal_dataset.TemporalFeatureExtractor', return_value=mock_temporal_processor), \
             patch('src.integration.multimodal_dataset.ClinicalNoteProcessor', return_value=mock_text_processor):

            dataset = MultimodalMIMICDataset(
                cohort_csv_path=cohort_path,
                config=sample_config,
                paths=MockPaths(),
                anthropic_api_key="fake_key",
                split='train',
                load_images=False,
                load_structured=True,
                load_text=True
            )

            sample = dataset[0]

            # Image should be None
            assert sample['image'] is None
            assert sample['structured'] is not None
            assert sample['text'] is not None

    def test_len_returns_cohort_size(
        self, sample_config, sample_cohort_df, tmp_path
    ):
        """Test that __len__ returns correct cohort size"""
        cohort_path = tmp_path / 'cohort.csv'
        sample_cohort_df.to_csv(cohort_path, index=False)

        class MockPaths:
            mimic_cxr_base = tmp_path / 'cxr'
            mimic_iv_base = tmp_path / 'iv'
            mimic_ed_base = tmp_path / 'ed'

        with patch('src.integration.multimodal_dataset.FullResolutionImageLoader'), \
             patch('src.integration.multimodal_dataset.TemporalFeatureExtractor'), \
             patch('src.integration.multimodal_dataset.ClinicalNoteProcessor'):

            dataset = MultimodalMIMICDataset(
                cohort_csv_path=cohort_path,
                config=sample_config,
                paths=MockPaths(),
                anthropic_api_key="fake_key",
                split='train',
                load_images=False,
                load_structured=False,
                load_text=False
            )

            assert len(dataset) == 3


class TestMultimodalDatasetErrorHandling:
    """Test error handling during sample loading"""

    def test_image_loading_error_logged(
        self, sample_config, sample_cohort_df, tmp_path
    ):
        """Test that image loading errors are properly logged"""
        cohort_path = tmp_path / 'cohort.csv'
        sample_cohort_df.to_csv(cohort_path, index=False)

        class MockPaths:
            mimic_cxr_base = tmp_path / 'cxr'
            mimic_iv_base = tmp_path / 'iv'
            mimic_ed_base = tmp_path / 'ed'

        # Mock image loader that raises error
        mock_loader = Mock()
        mock_loader.load_and_process.side_effect = FileNotFoundError("Image not found")

        with patch('src.integration.multimodal_dataset.FullResolutionImageLoader', return_value=mock_loader), \
             patch('src.integration.multimodal_dataset.TemporalFeatureExtractor'), \
             patch('src.integration.multimodal_dataset.ClinicalNoteProcessor'):

            dataset = MultimodalMIMICDataset(
                cohort_csv_path=cohort_path,
                config=sample_config,
                paths=MockPaths(),
                anthropic_api_key="fake_key",
                split='train',
                load_images=True,
                load_structured=False,
                load_text=False
            )

            sample = dataset[0]

            # Image should be None due to error
            assert sample['image'] is None
            # Error should be logged
            assert len(sample['metadata']['error_log']) > 0

    def test_structured_loading_error_logged(
        self, sample_config, sample_cohort_df, tmp_path
    ):
        """Test that structured data loading errors are logged"""
        cohort_path = tmp_path / 'cohort.csv'
        sample_cohort_df.to_csv(cohort_path, index=False)

        class MockPaths:
            mimic_cxr_base = tmp_path / 'cxr'
            mimic_iv_base = tmp_path / 'iv'
            mimic_ed_base = tmp_path / 'ed'

        # Mock processor that raises error
        mock_processor = Mock()
        mock_processor.extract_features.side_effect = Exception("Data error")

        with patch('src.integration.multimodal_dataset.TemporalFeatureExtractor', return_value=mock_processor):

            dataset = MultimodalMIMICDataset(
                cohort_csv_path=cohort_path,
                config=sample_config,
                paths=MockPaths(),
                anthropic_api_key="fake_key",
                split='train',
                load_images=False,
                load_structured=True,
                load_text=False
            )

            sample = dataset[0]

            # Structured should be None due to error
            assert sample['structured'] is None
            # Error should be logged
            assert len(sample['metadata']['error_log']) > 0


class TestMultimodalDatasetMetadata:
    """Test metadata extraction and storage"""

    def test_metadata_contains_required_fields(
        self, sample_config, sample_cohort_df, tmp_path
    ):
        """Test that metadata contains all required fields"""
        cohort_path = tmp_path / 'cohort.csv'
        sample_cohort_df.to_csv(cohort_path, index=False)

        class MockPaths:
            mimic_cxr_base = tmp_path / 'cxr'
            mimic_iv_base = tmp_path / 'iv'
            mimic_ed_base = tmp_path / 'ed'

        with patch('src.integration.multimodal_dataset.FullResolutionImageLoader'), \
             patch('src.integration.multimodal_dataset.TemporalFeatureExtractor'), \
             patch('src.integration.multimodal_dataset.ClinicalNoteProcessor'):

            dataset = MultimodalMIMICDataset(
                cohort_csv_path=cohort_path,
                config=sample_config,
                paths=MockPaths(),
                anthropic_api_key="fake_key",
                split='train',
                load_images=False,
                load_structured=False,
                load_text=False
            )

            sample = dataset[0]
            metadata = sample['metadata']

            # Required fields
            assert 'subject_id' in metadata
            assert 'study_id' in metadata
            assert 'dicom_id' in metadata
            assert 'study_datetime' in metadata
            assert 'split' in metadata
            assert 'error_log' in metadata

    def test_split_label_correct(
        self, sample_config, sample_cohort_df, tmp_path
    ):
        """Test that split label is correctly set"""
        cohort_path = tmp_path / 'cohort.csv'
        sample_cohort_df.to_csv(cohort_path, index=False)

        class MockPaths:
            mimic_cxr_base = tmp_path / 'cxr'
            mimic_iv_base = tmp_path / 'iv'
            mimic_ed_base = tmp_path / 'ed'

        for split_name in ['train', 'val', 'test']:
            with patch('src.integration.multimodal_dataset.FullResolutionImageLoader'), \
                 patch('src.integration.multimodal_dataset.TemporalFeatureExtractor'), \
                 patch('src.integration.multimodal_dataset.ClinicalNoteProcessor'):

                dataset = MultimodalMIMICDataset(
                    cohort_csv_path=cohort_path,
                    config=sample_config,
                    paths=MockPaths(),
                    anthropic_api_key="fake_key",
                    split=split_name,
                    load_images=False,
                    load_structured=False,
                    load_text=False
                )

                sample = dataset[0]
                assert sample['metadata']['split'] == split_name
