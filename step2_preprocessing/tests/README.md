# Step 2 Tests

This directory contains test scripts for the multimodal preprocessing pipeline.

## Test Scripts

- `test_setup.py` - Verifies environment setup (scispacy, ClinicalBERT, etc.)
- `test_claude.py` - Tests Claude + LangChain integration for text summarization
- `test_sample.py` - Tests complete pipeline on a single sample

## Running Tests

```bash
# Test environment setup
python tests/test_setup.py

# Test Claude integration (requires ANTHROPIC_API_KEY)
ANTHROPIC_API_KEY="your-key" python tests/test_claude.py

# Test sample processing
python tests/test_sample.py --max-samples 1
```

## Adding New Tests

When adding new test scripts:
1. Name them `test_*.py`
2. Add clear documentation at the top
3. Update this README
