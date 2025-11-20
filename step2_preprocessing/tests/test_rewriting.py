"""
Test suite for clinical note rewriting functionality.

Tests the note rewriting feature that expands abbreviations and standardizes
clinical notes before processing through the RAG pipeline.
"""

import os
import sys
import yaml
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.text_processing.note_processor import ClinicalNoteProcessor


def load_config():
    """Load configuration from config.yaml"""
    config_path = Path(__file__).parent.parent / 'config' / 'config.yaml'
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def test_config_loading():
    """Test that note_rewriting configuration loads correctly"""
    config = load_config()

    assert 'text' in config
    assert 'note_rewriting' in config['text']
    assert 'enabled' in config['text']['note_rewriting']
    assert 'model' in config['text']['note_rewriting']
    assert 'temperature' in config['text']['note_rewriting']

    print("Configuration test passed")


def test_rewriting_disabled_by_default():
    """Test that rewriting is disabled by default in config"""
    config = load_config()

    assert config['text']['note_rewriting']['enabled'] == False
    print("Rewriting disabled by default: OK")


def test_rewriting_initialization():
    """Test that ClinicalNoteProcessor initializes correctly with rewriting disabled"""
    config = load_config()
    api_key = os.environ.get('ANTHROPIC_API_KEY')

    if api_key is None:
        print("SKIP: No API key available for initialization test")
        return

    # Test with rewriting disabled
    processor = ClinicalNoteProcessor(config, api_key)
    assert hasattr(processor, 'rewriting_chain')
    assert processor.rewriting_chain is None  # Should be None when disabled

    print("Processor initialization (rewriting disabled): OK")


def test_rewriting_with_enabled_config():
    """Test that rewriting chain initializes when enabled"""
    config = load_config()
    api_key = os.environ.get('ANTHROPIC_API_KEY')

    if api_key is None:
        print("SKIP: No API key available for rewriting test")
        return

    # Enable rewriting for this test
    config['text']['note_rewriting']['enabled'] = True

    processor = ClinicalNoteProcessor(config, api_key)
    assert hasattr(processor, 'rewriting_chain')
    assert processor.rewriting_chain is not None  # Should be initialized when enabled
    assert hasattr(processor, 'rewriting_llm')
    assert hasattr(processor, 'rewriting_prompt')

    print("Processor initialization (rewriting enabled): OK")


def test_abbreviation_expansion():
    """Test that common medical abbreviations are expanded"""
    config = load_config()
    api_key = os.environ.get('ANTHROPIC_API_KEY')

    if api_key is None:
        print("SKIP: No API key available for abbreviation test")
        return

    # Enable rewriting
    config['text']['note_rewriting']['enabled'] = True
    processor = ClinicalNoteProcessor(config, api_key)

    # Test note with common abbreviations
    test_note = "Patient c/o c/p. Hx of HTN on meds."
    rewritten = processor.rewrite_note(test_note)

    # Check that abbreviations were expanded
    assert rewritten != test_note  # Should be different
    assert len(rewritten) > len(test_note)  # Should be longer

    # Check for common expansions (case-insensitive)
    rewritten_lower = rewritten.lower()
    assert 'complains' in rewritten_lower or 'complaint' in rewritten_lower  # c/o
    assert 'chest pain' in rewritten_lower  # c/p
    assert 'hypertension' in rewritten_lower or 'history' in rewritten_lower  # HTN or Hx

    print(f"Abbreviation expansion test passed")
    print(f"  Original: {test_note}")
    print(f"  Rewritten: {rewritten}")


def test_numerical_values_preserved():
    """Test that numerical values are not changed"""
    config = load_config()
    api_key = os.environ.get('ANTHROPIC_API_KEY')

    if api_key is None:
        print("SKIP: No API key available for numerical preservation test")
        return

    # Enable rewriting
    config['text']['note_rewriting']['enabled'] = True
    processor = ClinicalNoteProcessor(config, api_key)

    # Test note with specific numerical values
    test_note = "Patient presents with c/p. Vitals: HR 78, BP 120/80, SpO2 98%."
    rewritten = processor.rewrite_note(test_note)

    # Check that key numbers are preserved
    assert '78' in rewritten
    assert '120' in rewritten
    assert '80' in rewritten
    assert '98' in rewritten

    print("Numerical value preservation test passed")
    print(f"  Original: {test_note}")
    print(f"  Rewritten: {rewritten}")


def test_empty_note_handling():
    """Test handling of empty notes"""
    config = load_config()
    api_key = os.environ.get('ANTHROPIC_API_KEY')

    if api_key is None:
        print("SKIP: No API key available")
        return

    config['text']['note_rewriting']['enabled'] = True
    processor = ClinicalNoteProcessor(config, api_key)

    # Test empty string
    assert processor.rewrite_note("") == ""

    # Test whitespace only
    assert processor.rewrite_note("   ") == ""

    print("Empty note handling test passed")


def test_fallback_on_disabled():
    """Test that original note is returned when rewriting is disabled"""
    config = load_config()
    api_key = os.environ.get('ANTHROPIC_API_KEY')

    if api_key is None:
        print("SKIP: No API key available")
        return

    # Rewriting disabled
    config['text']['note_rewriting']['enabled'] = False
    processor = ClinicalNoteProcessor(config, api_key)

    test_note = "Patient c/o c/p."
    rewritten = processor.rewrite_note(test_note)

    # Should return original when disabled
    assert rewritten == test_note

    print("Fallback test (disabled) passed")


def test_integration_with_rag_pipeline():
    """Test that rewritten notes flow correctly through RAG pipeline"""
    config = load_config()
    api_key = os.environ.get('ANTHROPIC_API_KEY')

    if api_key is None:
        print("SKIP: No API key available for RAG integration test")
        return

    # Test with rewriting enabled
    config['text']['note_rewriting']['enabled'] = True
    processor = ClinicalNoteProcessor(config, api_key)

    test_note = "Patient c/o c/p and SOB. Hx of HTN."

    # Process through complete pipeline
    result = processor.process_note(test_note)

    # Verify all expected keys are present
    assert 'summary' in result
    assert 'tokens' in result
    assert 'num_entities' in result
    assert 'entities' in result
    assert 'context_sentences' in result

    # Verify some processing occurred
    assert result['num_entities'] > 0
    assert len(result['summary']) > 0

    print("RAG pipeline integration test passed")
    print(f"  Entities extracted: {result['num_entities']}")
    print(f"  Summary length: {len(result['summary'])} chars")


def test_comparison_original_vs_rewritten():
    """Compare entity extraction from original vs rewritten notes"""
    config = load_config()
    api_key = os.environ.get('ANTHROPIC_API_KEY')

    if api_key is None:
        print("SKIP: No API key available for comparison test")
        return

    test_note = "58yo M c/o c/p, SOB. Hx HTN, DM2 on meds. Vitals: HR 88, BP 135/85."

    # Process with rewriting disabled
    config['text']['note_rewriting']['enabled'] = False
    processor_orig = ClinicalNoteProcessor(config, api_key)
    result_orig = processor_orig.process_note(test_note)

    # Process with rewriting enabled
    config['text']['note_rewriting']['enabled'] = True
    processor_rewr = ClinicalNoteProcessor(config, api_key)
    result_rewr = processor_rewr.process_note(test_note)

    print("\nComparison: Original vs Rewritten through RAG")
    print(f"  Original entities: {result_orig['num_entities']}")
    print(f"  Rewritten entities: {result_rewr['num_entities']}")
    print(f"  Original summary length: {len(result_orig['summary'])} chars")
    print(f"  Rewritten summary length: {len(result_rewr['summary'])} chars")

    # Both should produce valid results
    assert result_orig['num_entities'] > 0
    assert result_rewr['num_entities'] > 0

    print("Comparison test passed")


def run_all_tests():
    """Run all tests"""
    print("=" * 80)
    print("RUNNING NOTE REWRITING TESTS")
    print("=" * 80)
    print()

    tests = [
        test_config_loading,
        test_rewriting_disabled_by_default,
        test_rewriting_initialization,
        test_rewriting_with_enabled_config,
        test_empty_note_handling,
        test_fallback_on_disabled,
        test_abbreviation_expansion,
        test_numerical_values_preserved,
        test_integration_with_rag_pipeline,
        test_comparison_original_vs_rewritten,
    ]

    passed = 0
    skipped = 0
    failed = 0

    for test_func in tests:
        test_name = test_func.__name__
        print(f"\nRunning: {test_name}")
        print("-" * 80)

        try:
            test_func()
            passed += 1
        except AssertionError as e:
            print(f"FAILED: {e}")
            failed += 1
        except Exception as e:
            if "SKIP" in str(e):
                skipped += 1
            else:
                print(f"ERROR: {e}")
                failed += 1

    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    print(f"Passed: {passed}")
    print(f"Skipped: {skipped}")
    print(f"Failed: {failed}")
    print(f"Total: {len(tests)}")

    if failed == 0:
        print("\nAll tests passed!")
    else:
        print(f"\n{failed} test(s) failed")

    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
