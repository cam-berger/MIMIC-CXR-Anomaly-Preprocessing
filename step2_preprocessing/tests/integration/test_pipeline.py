"""
Integration tests for the complete step2 preprocessing pipeline.

Tests end-to-end processing of samples through all modalities.
"""
import pytest
import sys
import torch
import json
import pandas as pd
from pathlib import Path
from unittest.mock import patch

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


class TestEndToEndPipeline:
    """Integration tests for complete pipeline"""

    @pytest.fixture
    def integration_cohort(self, tmp_path):
        """Create small cohort for integration testing"""
        cohort = pd.DataFrame({
            'subject_id': [10000, 10001],
            'study_id': [50000, 50001],
            'dicom_id': ['abc123', 'def456'],
            'study_datetime': pd.to_datetime(['2020-01-01 10:00', '2020-01-02 14:00']),
            'stay_id': [30000, 30001]
        })

        cohort_path = tmp_path / 'test_cohort.csv'
        cohort.to_csv(cohort_path, index=False)
        return cohort_path

    def test_pipeline_with_all_skip_flags(
        self, sample_config, integration_cohort, tmp_path
    ):
        """Test pipeline runs with all modalities skipped"""
        from src.integration.multimodal_dataset import MultimodalMIMICDataset

        class MockPaths:
            mimic_cxr_base = tmp_path / 'cxr'
            mimic_iv_base = tmp_path / 'iv'
            mimic_ed_base = tmp_path / 'ed'

        dataset = MultimodalMIMICDataset(
            cohort_csv_path=integration_cohort,
            config=sample_config,
            paths=MockPaths(),
            anthropic_api_key=None,
            split='train',
            load_images=False,
            load_structured=False,
            load_text=False
        )

        # Should still work with all modalities skipped
        assert len(dataset) == 2

        sample = dataset[0]
        assert sample['image'] is None
        assert sample['structured'] is None
        assert sample['text'] is None
        assert sample['metadata'] is not None

    def test_pipeline_processes_multiple_samples(
        self, sample_config, integration_cohort, tmp_path,
        mock_image_loader, mock_temporal_processor, mock_text_processor
    ):
        """Test pipeline can process multiple samples"""
        from src.integration.multimodal_dataset import MultimodalMIMICDataset

        class MockPaths:
            mimic_cxr_base = tmp_path / 'cxr'
            mimic_iv_base = tmp_path / 'iv'
            mimic_ed_base = tmp_path / 'ed'

        with patch('src.integration.multimodal_dataset.FullResolutionImageLoader', return_value=mock_image_loader), \
             patch('src.integration.multimodal_dataset.TemporalFeatureExtractor', return_value=mock_temporal_processor), \
             patch('src.integration.multimodal_dataset.ClinicalNoteProcessor', return_value=mock_text_processor):

            dataset = MultimodalMIMICDataset(
                cohort_csv_path=integration_cohort,
                config=sample_config,
                paths=MockPaths(),
                anthropic_api_key="fake_key",
                split='train',
                load_images=True,
                load_structured=True,
                load_text=True
            )

            # Process all samples
            samples = [dataset[i] for i in range(len(dataset))]

            assert len(samples) == 2
            for sample in samples:
                assert 'image' in sample
                assert 'structured' in sample
                assert 'text' in sample
                assert 'metadata' in sample

    def test_sample_saving_and_loading(
        self, sample_config, tmp_path,
        mock_image_loader, mock_temporal_processor, mock_text_processor
    ):
        """Test that samples can be saved and loaded correctly"""
        # Create sample data
        sample_key = "s10000_study50000"

        # Create output directories
        output_dir = tmp_path / 'output' / 'train'
        (output_dir / 'images').mkdir(parents=True)
        (output_dir / 'structured_features').mkdir(parents=True)
        (output_dir / 'text_features').mkdir(parents=True)
        (output_dir / 'metadata').mkdir(parents=True)

        # Create mock sample
        image = torch.rand(1, 512, 512)
        structured = {
            'vital_heartrate': {
                'is_missing': False,
                'last_value': 75.0,
                'mean_value': 76.0
            }
        }
        text = {
            'summary': 'Patient presents with chest pain.',
            'tokens': {
                'input_ids': torch.randint(0, 1000, (512,)),
                'attention_mask': torch.ones(512),
                'num_tokens': 25
            },
            'num_entities': 5
        }
        metadata = {
            'subject_id': 10000,
            'study_id': 50000,
            'split': 'train'
        }

        # Save image
        torch.save(image, output_dir / 'images' / f'{sample_key}.pt')

        # Save structured
        with open(output_dir / 'structured_features' / f'{sample_key}.json', 'w') as f:
            json.dump(structured, f)

        # Save text
        torch.save(text, output_dir / 'text_features' / f'{sample_key}.pt')

        # Save metadata
        with open(output_dir / 'metadata' / f'{sample_key}.json', 'w') as f:
            json.dump(metadata, f)

        # Load back
        loaded_image = torch.load(output_dir / 'images' / f'{sample_key}.pt')
        assert torch.allclose(image, loaded_image)

        with open(output_dir / 'structured_features' / f'{sample_key}.json') as f:
            loaded_structured = json.load(f)
        assert loaded_structured == structured

        loaded_text = torch.load(output_dir / 'text_features' / f'{sample_key}.pt')
        assert loaded_text['summary'] == text['summary']

        with open(output_dir / 'metadata' / f'{sample_key}.json') as f:
            loaded_metadata = json.load(f)
        assert loaded_metadata == metadata

    def test_error_recovery_continues_processing(
        self, sample_config, integration_cohort, tmp_path
    ):
        """Test that pipeline continues after errors in individual samples"""
        from src.integration.multimodal_dataset import MultimodalMIMICDataset

        class MockPaths:
            mimic_cxr_base = tmp_path / 'cxr'
            mimic_iv_base = tmp_path / 'iv'
            mimic_ed_base = tmp_path / 'ed'

        # Mock image loader that fails on first sample
        from unittest.mock import Mock
        mock_loader = Mock()
        call_count = [0]

        def side_effect_loader(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                raise Exception("Simulated error")
            return torch.rand(1, 512, 512)

        mock_loader.load_and_process.side_effect = side_effect_loader

        with patch('src.integration.multimodal_dataset.FullResolutionImageLoader', return_value=mock_loader):

            dataset = MultimodalMIMICDataset(
                cohort_csv_path=integration_cohort,
                config=sample_config,
                paths=MockPaths(),
                anthropic_api_key=None,
                split='train',
                load_images=True,
                load_structured=False,
                load_text=False
            )

            # First sample should have error
            sample0 = dataset[0]
            assert sample0['image'] is None
            assert len(sample0['metadata']['error_log']) > 0

            # Second sample should succeed
            sample1 = dataset[1]
            assert sample1['image'] is not None
            assert len(sample1['metadata']['error_log']) == 0

    def test_config_variations(self, sample_config, integration_cohort, tmp_path):
        """Test pipeline with different config variations"""
        from src.integration.multimodal_dataset import MultimodalMIMICDataset

        class MockPaths:
            mimic_cxr_base = tmp_path / 'cxr'
            mimic_iv_base = tmp_path / 'iv'
            mimic_ed_base = tmp_path / 'ed'

        # Test different normalization methods
        for norm_method in ['minmax', 'standardize', 'none']:
            config = sample_config.copy()
            config['image']['normalize_method'] = norm_method

            with patch('src.integration.multimodal_dataset.FullResolutionImageLoader'), \
                 patch('src.integration.multimodal_dataset.TemporalFeatureExtractor'), \
                 patch('src.integration.multimodal_dataset.ClinicalNoteProcessor'):

                dataset = MultimodalMIMICDataset(
                    cohort_csv_path=integration_cohort,
                    config=config,
                    paths=MockPaths(),
                    anthropic_api_key=None,
                    split='train',
                    load_images=False,
                    load_structured=False,
                    load_text=False
                )

                # Should initialize without errors
                assert len(dataset) == 2

        # Test different encoding methods
        for encoding in ['aggregated', 'sequential']:
            config = sample_config.copy()
            config['structured']['encoding_method'] = encoding

            with patch('src.integration.multimodal_dataset.FullResolutionImageLoader'), \
                 patch('src.integration.multimodal_dataset.TemporalFeatureExtractor'), \
                 patch('src.integration.multimodal_dataset.ClinicalNoteProcessor'):

                dataset = MultimodalMIMICDataset(
                    cohort_csv_path=integration_cohort,
                    config=config,
                    paths=MockPaths(),
                    anthropic_api_key=None,
                    split='train',
                    load_images=False,
                    load_structured=False,
                    load_text=False
                )

                # Should initialize without errors
                assert len(dataset) == 2
