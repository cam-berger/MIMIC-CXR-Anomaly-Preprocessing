#!/usr/bin/env python3
"""
Test script for Claude + LangChain integration.
Tests the text processing pipeline with RAG.
"""
import os
import sys
from pathlib import Path
import yaml

# Check API key
api_key = os.environ.get('ANTHROPIC_API_KEY')
if not api_key:
    print("❌ ERROR: ANTHROPIC_API_KEY environment variable not set!")
    print("\nPlease set it with:")
    print('  export ANTHROPIC_API_KEY="sk-ant-your-key-here"')
    sys.exit(1)

print("="*80)
print("CLAUDE + LANGCHAIN TEST")
print("="*80)
print(f"✓ API Key found: {api_key[:20]}...{api_key[-10:]}")

# Load configuration
config_path = Path(__file__).parent / 'config' / 'config.yaml'
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

print(f"✓ Config loaded: {config_path}")
print(f"  Model: {config['text']['summarization']['model']}")
print(f"  Max tokens: {config['text']['summarization']['max_summary_length']}")

# Initialize text processor
print("\nInitializing ClinicalNoteProcessor...")
try:
    from src.text_processing.note_processor import ClinicalNoteProcessor

    processor = ClinicalNoteProcessor(config, api_key)
    print("✓ Text processor initialized successfully!")

except Exception as e:
    print(f"❌ ERROR: Failed to initialize processor: {e}")
    sys.exit(1)

# Test with sample clinical note
print("\n" + "="*80)
print("TESTING WITH SAMPLE CLINICAL NOTE")
print("="*80)

sample_note = """
Chief Complaint: Chest pain and shortness of breath

History of Present Illness:
58-year-old male with history of hypertension and diabetes presents to the ED with
acute onset chest pain starting 2 hours ago. Pain is substernal, pressure-like,
radiating to left arm. Associated with shortness of breath and diaphoresis.

Past Medical History:
- Hypertension (HTN) on lisinopril
- Type 2 diabetes mellitus on metformin
- Hyperlipidemia on atorvastatin
- Former smoker (quit 5 years ago, 20 pack-year history)

Physical Examination:
Vitals: BP 145/92, HR 98, RR 22, SpO2 94% on room air, Temp 37.2°C
General: Anxious, diaphoretic
Cardiovascular: Regular rate and rhythm, no murmurs
Respiratory: Bilateral basilar crackles, decreased breath sounds at bases
Abdomen: Soft, non-tender

Assessment and Plan:
1. Chest pain - rule out acute coronary syndrome
   - Troponin, EKG, chest x-ray ordered
   - Aspirin, nitroglycerin given
   - Cardiology consult pending
2. Shortness of breath - possible pulmonary edema vs pneumonia
   - Chest x-ray to evaluate
   - Supplemental oxygen
3. Diabetes - continue home medications
"""

print("\nSample note preview (first 200 chars):")
print(sample_note[:200] + "...\n")

# Process the note
print("Processing note with RAG pipeline...")
print("  [1/4] Extracting medical entities (scispacy NER)...")
try:
    entities = processor.extract_entities(sample_note)
    print(f"      ✓ Found {len(entities)} entities: {entities[:5]}...")
except Exception as e:
    print(f"      ❌ ERROR: {e}")
    sys.exit(1)

print("  [2/4] Retrieving relevant sentences...")
try:
    relevant_sentences = processor.retrieve_relevant_sentences(sample_note, entities)
    print(f"      ✓ Retrieved {len(relevant_sentences)} relevant sentences")
except Exception as e:
    print(f"      ❌ ERROR: {e}")
    sys.exit(1)

print("  [3/4] Generating summary with Claude (RAG)...")
try:
    summary = processor.summarize_with_claude(relevant_sentences)
    print(f"      ✓ Summary generated ({len(summary)} characters)")
except Exception as e:
    print(f"      ❌ ERROR: {e}")
    print(f"      This usually means API key issue or rate limiting")
    sys.exit(1)

print("  [4/4] Tokenizing with ClinicalBERT...")
try:
    tokens = processor.tokenize(summary)
    print(f"      ✓ Tokenized: {tokens['num_tokens']} tokens")
except Exception as e:
    print(f"      ❌ ERROR: {e}")
    sys.exit(1)

# Display results
print("\n" + "="*80)
print("RESULTS")
print("="*80)

print(f"\nExtracted Entities ({len(entities)}):")
for i, ent in enumerate(entities[:10], 1):
    print(f"  {i}. {ent}")
if len(entities) > 10:
    print(f"  ... and {len(entities) - 10} more")

print(f"\nRetrieved Sentences ({len(relevant_sentences)}):")
for i, sent in enumerate(relevant_sentences[:3], 1):
    print(f"  {i}. {sent[:100]}...")

print(f"\n📝 Claude-Generated Summary:")
print("-" * 80)
print(summary)
print("-" * 80)

print(f"\n✓ Tokenization:")
print(f"  Input tokens: {tokens['num_tokens']}")
print(f"  Truncated: {tokens['is_truncated']}")

# Test complete pipeline
print("\n" + "="*80)
print("TESTING COMPLETE PIPELINE")
print("="*80)

try:
    result = processor.process_note(sample_note)
    print("✓ Complete pipeline executed successfully!")
    print(f"\n  Summary: {result['summary'][:150]}...")
    print(f"  Entities: {result['num_entities']}")
    print(f"  Tokens: {result['tokens']['num_tokens']}")
    print(f"  Context sentences: {result['context_sentences']}")
except Exception as e:
    print(f"❌ ERROR in complete pipeline: {e}")
    sys.exit(1)

print("\n" + "="*80)
print("✅ ALL TESTS PASSED!")
print("="*80)
print("\nYour Claude + LangChain integration is working correctly!")
print("\nNext steps:")
print("  1. Test on real sample: python test_sample.py --api-key YOUR_KEY")
print("  2. Run small batch: python main.py --max-samples 2 --anthropic-api-key YOUR_KEY")
print("  3. Run full dataset: python main.py --anthropic-api-key YOUR_KEY")
