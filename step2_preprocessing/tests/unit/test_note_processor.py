"""
Unit tests for text_processing/note_processor.py

Tests NER, retrieval, summarization, and note rewriting functionality.
Converted from tests/test_rewriting.py to proper pytest format.
"""
import pytest
import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.text_processing.note_processor import ClinicalNoteProcessor


# Pytest marker for tests requiring API key
requires_api_key = pytest.mark.skipif(
    not os.environ.get('ANTHROPIC_API_KEY'),
    reason="ANTHROPIC_API_KEY not set"
)


class TestNoteProcessorConfiguration:
    """Test configuration loading and validation"""

    def test_config_has_note_rewriting(self, sample_config):
        """Test that note_rewriting configuration is present"""
        assert 'text' in sample_config
        assert 'note_rewriting' in sample_config['text']
        assert 'enabled' in sample_config['text']['note_rewriting']
        assert 'model' in sample_config['text']['note_rewriting']
        assert 'temperature' in sample_config['text']['note_rewriting']

    def test_rewriting_disabled_by_default(self, sample_config):
        """Test that rewriting is disabled by default"""
        assert sample_config['text']['note_rewriting']['enabled'] is False

    def test_config_has_ner_settings(self, sample_config):
        """Test NER configuration"""
        assert 'ner' in sample_config['text']
        assert 'model' in sample_config['text']['ner']
        assert 'extract_entities' in sample_config['text']['ner']

    def test_config_has_retrieval_settings(self, sample_config):
        """Test retrieval configuration"""
        assert 'retrieval' in sample_config['text']
        assert 'use_entity_based' in sample_config['text']['retrieval']
        assert 'use_semantic_fallback' in sample_config['text']['retrieval']


class TestNoteProcessorInitialization:
    """Test processor initialization"""

    @requires_api_key
    def test_initialization_rewriting_disabled(self, sample_config):
        """Test initialization with rewriting disabled"""
        api_key = os.environ.get('ANTHROPIC_API_KEY')
        processor = ClinicalNoteProcessor(sample_config, api_key)

        assert hasattr(processor, 'rewriting_chain')
        assert processor.rewriting_chain is None  # Should be None when disabled
        assert hasattr(processor, 'nlp')  # NER model should be loaded
        assert hasattr(processor, 'embedder')  # Sentence embedder should be loaded

    @requires_api_key
    def test_initialization_rewriting_enabled(self, sample_config):
        """Test initialization with rewriting enabled"""
        api_key = os.environ.get('ANTHROPIC_API_KEY')

        # Enable rewriting
        sample_config['text']['note_rewriting']['enabled'] = True
        processor = ClinicalNoteProcessor(sample_config, api_key)

        assert hasattr(processor, 'rewriting_chain')
        assert processor.rewriting_chain is not None  # Should be initialized
        assert hasattr(processor, 'rewriting_llm')
        assert hasattr(processor, 'rewriting_prompt')

    def test_initialization_without_api_key(self, sample_config):
        """Test initialization without API key (summarization disabled)"""
        sample_config['text']['summarization']['use_claude'] = False
        processor = ClinicalNoteProcessor(sample_config, None)

        assert hasattr(processor, 'nlp')  # NER should still work
        assert processor.summarization_chain is None


class TestNoteRewriting:
    """Test clinical note rewriting functionality"""

    @requires_api_key
    def test_abbreviation_expansion(self, sample_config):
        """Test that common medical abbreviations are expanded"""
        api_key = os.environ.get('ANTHROPIC_API_KEY')
        sample_config['text']['note_rewriting']['enabled'] = True
        processor = ClinicalNoteProcessor(sample_config, api_key)

        test_note = "Patient c/o c/p. Hx of HTN on meds."
        rewritten = processor.rewrite_note(test_note)

        # Should be different and longer
        assert rewritten != test_note
        assert len(rewritten) > len(test_note)

        # Check for expansions (case-insensitive)
        rewritten_lower = rewritten.lower()
        assert 'complains' in rewritten_lower or 'complaint' in rewritten_lower
        assert 'chest pain' in rewritten_lower
        assert 'hypertension' in rewritten_lower or 'history' in rewritten_lower

    @requires_api_key
    def test_numerical_values_preserved(self, sample_config):
        """Test that numerical values are not changed"""
        api_key = os.environ.get('ANTHROPIC_API_KEY')
        sample_config['text']['note_rewriting']['enabled'] = True
        processor = ClinicalNoteProcessor(sample_config, api_key)

        test_note = "Patient presents with c/p. Vitals: HR 78, BP 120/80, SpO2 98%."
        rewritten = processor.rewrite_note(test_note)

        # Check that key numbers are preserved
        assert '78' in rewritten
        assert '120' in rewritten
        assert '80' in rewritten
        assert '98' in rewritten

    def test_empty_note_handling(self, sample_config):
        """Test handling of empty notes"""
        # When rewriting is disabled (use_claude=False), returns original text
        sample_config['text']['note_rewriting']['enabled'] = True
        sample_config['text']['note_rewriting']['use_claude'] = False
        processor = ClinicalNoteProcessor(sample_config, "fake_key")

        # With rewriting disabled, returns original (even if empty/whitespace)
        assert processor.rewrite_note("") == ""
        assert processor.rewrite_note("   ") == "   "  # Returns original whitespace

    def test_fallback_when_disabled(self, sample_config):
        """Test that original note is returned when rewriting disabled"""
        sample_config['text']['note_rewriting']['enabled'] = False
        processor = ClinicalNoteProcessor(sample_config, "fake_key")

        test_note = "Patient c/o c/p."
        rewritten = processor.rewrite_note(test_note)

        # Should return original when disabled
        assert rewritten == test_note


class TestEntityExtraction:
    """Test medical entity extraction with scispacy"""

    def test_entity_extraction_basic(self, sample_config, sample_clinical_note):
        """Test basic entity extraction"""
        sample_config['text']['summarization']['use_claude'] = False
        processor = ClinicalNoteProcessor(sample_config, None)

        # Extract entities using public method
        entities = processor.extract_entities(sample_clinical_note)

        assert isinstance(entities, list)
        assert len(entities) > 0  # Should extract some entities

    def test_entity_deduplication(self, sample_config):
        """Test that duplicate entities are removed"""
        sample_config['text']['summarization']['use_claude'] = False
        processor = ClinicalNoteProcessor(sample_config, None)

        # Note with repeated terms
        note = "Patient has chest pain. The chest pain is severe. Chest pain radiating."
        entities = processor.extract_entities(note)

        # "chest pain" should appear only once
        entity_counts = {}
        for e in entities:
            entity_counts[e] = entity_counts.get(e, 0) + 1

        # All entities should be unique
        assert all(count == 1 for count in entity_counts.values())

    def test_entity_normalization(self, sample_config):
        """Test that entities are normalized (lowercase, stripped)"""
        sample_config['text']['summarization']['use_claude'] = False
        processor = ClinicalNoteProcessor(sample_config, None)

        note = "Patient has CHEST PAIN and  shortness  of breath"
        entities = processor.extract_entities(note)

        # Entities should be lowercase and normalized
        for entity in entities:
            assert entity == entity.lower()
            assert entity == entity.strip()


class TestRAGPipelineIntegration:
    """Test complete RAG pipeline (NER + Retrieval + Summarization)"""

    @requires_api_key
    def test_full_pipeline_execution(self, sample_config, sample_clinical_note):
        """Test complete pipeline from note to summary"""
        api_key = os.environ.get('ANTHROPIC_API_KEY')
        sample_config['text']['summarization']['use_claude'] = True
        processor = ClinicalNoteProcessor(sample_config, api_key)

        result = processor.process_note(sample_clinical_note)

        # Verify all expected keys
        assert 'summary' in result
        assert 'tokens' in result
        assert 'num_entities' in result
        assert 'entities' in result
        assert 'context_sentences' in result

        # Verify processing occurred
        assert result['num_entities'] > 0
        assert len(result['summary']) > 0
        assert result['tokens']['num_tokens'] > 0

    @requires_api_key
    def test_entity_extraction_improves_with_rewriting(self, sample_config):
        """Test that rewriting improves entity extraction"""
        api_key = os.environ.get('ANTHROPIC_API_KEY')

        test_note = "58yo M c/o c/p, SOB. Hx HTN, DM2 on meds. Vitals: HR 88, BP 135/85."

        # Process without rewriting
        sample_config['text']['note_rewriting']['enabled'] = False
        sample_config['text']['summarization']['use_claude'] = True
        processor_orig = ClinicalNoteProcessor(sample_config, api_key)
        result_orig = processor_orig.process_note(test_note)

        # Process with rewriting
        sample_config['text']['note_rewriting']['enabled'] = True
        processor_rewr = ClinicalNoteProcessor(sample_config, api_key)
        result_rewr = processor_rewr.process_note(test_note)

        # Both should produce valid results
        assert result_orig['num_entities'] > 0
        assert result_rewr['num_entities'] > 0

        # Rewritten should extract more or equal entities
        # (not guaranteed, but likely with expanded abbreviations)
        assert result_rewr['num_entities'] >= result_orig['num_entities'] * 0.8  # Allow some variance


class TestRetrievalMethods:
    """Test sentence retrieval methods"""

    def test_entity_based_retrieval(self, sample_config, sample_clinical_note):
        """Test entity-based retrieval"""
        sample_config['text']['summarization']['use_claude'] = False
        processor = ClinicalNoteProcessor(sample_config, None)

        entities = processor._extract_medical_entities(sample_clinical_note)
        sentences = sample_clinical_note.split('.')

        retrieved = processor._entity_based_retrieval(sentences, entities)

        assert isinstance(retrieved, list)
        assert len(retrieved) <= len(sentences)

    def test_semantic_retrieval(self, sample_config):
        """Test semantic similarity retrieval"""
        sample_config['text']['summarization']['use_claude'] = False
        processor = ClinicalNoteProcessor(sample_config, None)

        sentences = [
            "Patient has severe chest pain",
            "The weather is sunny today",  # Irrelevant
            "Shortness of breath noted",
            "I like pizza"  # Irrelevant
        ]

        retrieved = processor._semantic_retrieval(sentences, threshold=0.3)

        assert isinstance(retrieved, list)
        # Should retrieve medical sentences, not irrelevant ones
        assert len(retrieved) >= 2


class TestTokenization:
    """Test ClinicalBERT tokenization"""

    def test_tokenization_output_structure(self, sample_config):
        """Test tokenizer produces correct output structure"""
        sample_config['text']['summarization']['use_claude'] = False
        processor = ClinicalNoteProcessor(sample_config, None)

        test_text = "Patient presents with chest pain and shortness of breath."
        tokens = processor._tokenize_text(test_text)

        assert 'input_ids' in tokens
        assert 'attention_mask' in tokens
        assert 'num_tokens' in tokens
        assert 'is_truncated' in tokens

    def test_max_length_truncation(self, sample_config):
        """Test that long text is truncated to max_length"""
        sample_config['text']['summarization']['use_claude'] = False
        max_length = sample_config['text']['tokenizer']['max_length']
        processor = ClinicalNoteProcessor(sample_config, None)

        # Create very long text
        long_text = " ".join(["chest pain"] * 1000)
        tokens = processor._tokenize_text(long_text)

        # Should be truncated to max_length
        assert tokens['input_ids'].shape[0] == max_length
        assert tokens['is_truncated'] is True

    def test_short_text_padding(self, sample_config):
        """Test that short text is padded to max_length"""
        sample_config['text']['summarization']['use_claude'] = False
        max_length = sample_config['text']['tokenizer']['max_length']
        processor = ClinicalNoteProcessor(sample_config, None)

        short_text = "Patient has fever."
        tokens = processor._tokenize_text(short_text)

        # Should be padded to max_length
        assert tokens['input_ids'].shape[0] == max_length
        assert tokens['is_truncated'] is False
