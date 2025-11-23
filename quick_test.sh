#!/bin/bash
# Quick essential pre-deployment checks

set -e

echo "========================================"
echo "QUICK PRE-DEPLOYMENT CHECKS"
echo "========================================"
echo ""

# Check Python
echo "1. Python version..."
python --version
echo "✓"
echo ""

# Check dependencies
echo "2. Checking dependencies..."
python -c "import torch; print(f'  ✓ PyTorch {torch.__version__}')"
python -c "import h5py; print(f'  ✓ h5py {h5py.__version__}')"
python -c "import pyarrow; print(f'  ✓ pyarrow {pyarrow.__version__}')"
python -c "import pandas as pd; print(f'  ✓ pandas {pd.__version__}')"
python -c "import spacy; print(f'  ✓ spacy {spacy.__version__}')"
python -c "import spacy; nlp = spacy.load('en_core_sci_md'); print('  ✓ scispacy model loaded')"
python -c "import transformers; print(f'  ✓ transformers {transformers.__version__}')"
python -c "from langchain_anthropic import ChatAnthropic; print('  ✓ langchain-anthropic installed')"
echo ""

# Check configuration
echo "3. Checking configuration..."
if grep -q 'claude-sonnet-4-5-20250929' step2_preprocessing/config/config.yaml; then
    echo "  ✓ Claude Sonnet 4.5 configured"
fi
if grep -q 'preserve_full_resolution: true' step2_preprocessing/config/config.yaml; then
    echo "  ✓ Full resolution enabled"
fi
if grep -q 'cxr_pro: true' step2_preprocessing/config/config.yaml; then
    echo "  ✓ CXR-PRO enabled"
fi
echo ""

# Check cohort file
echo "4. Checking validation cohort..."
if [ ! -f "step2_preprocessing/cohorts/validation_subset_200.csv" ]; then
    if [ -f "output/cohorts/normal_cohort_validation.csv" ]; then
        echo "  Creating validation subset..."
        mkdir -p step2_preprocessing/cohorts
        head -201 output/cohorts/normal_cohort_validation.csv > step2_preprocessing/cohorts/validation_subset_200.csv
        echo "  ✓ Created validation subset (200 samples)"
    else
        echo "  ✗ Validation cohort not found. Run Step 1 first: python main.py"
        exit 1
    fi
else
    echo "  ✓ Validation cohort exists"
fi
echo ""

# Run small test
echo "5. Running small preprocessing test (5 samples)..."
cd step2_preprocessing
python main.py --max-samples 5 --output-dir ./output_test_5 --val-only 2>&1 | tail -20
cd ..

if [ -d "step2_preprocessing/output_test_5/val/images" ]; then
    NUM_IMAGES=$(ls step2_preprocessing/output_test_5/val/images/*.pt 2>/dev/null | wc -l)
    NUM_TEXT=$(ls step2_preprocessing/output_test_5/val/text_features/*.pt 2>/dev/null | wc -l)
    NUM_STRUCT=$(ls step2_preprocessing/output_test_5/val/structured_features/*.json 2>/dev/null | wc -l)

    echo ""
    echo "  Results:"
    echo "    Images: $NUM_IMAGES"
    echo "    Text: $NUM_TEXT"
    echo "    Structured: $NUM_STRUCT"

    if [ "$NUM_IMAGES" -ge 1 ] && [ "$NUM_TEXT" -ge 1 ] && [ "$NUM_STRUCT" -ge 1 ]; then
        echo "  ✓ Preprocessing test PASSED"
    else
        echo "  ✗ Preprocessing test FAILED - missing outputs"
        exit 1
    fi
fi

echo ""
echo "========================================"
echo "✓ ALL QUICK CHECKS PASSED"
echo "========================================"
echo ""
echo "System is ready for pre-compilation testing!"
echo ""
echo "Next steps:"
echo "  1. Test pre-compilation with 10 samples:"
echo "     python run_precompilation.py --cohort step2_preprocessing/cohorts/validation_subset_200.csv --output-dir ./precompiled_test --split val --batch-size 0 --max-samples 10"
echo ""
echo "  2. For comprehensive testing, see PRE_DEPLOYMENT_TESTING.md"
