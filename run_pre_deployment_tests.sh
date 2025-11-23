#!/bin/bash
# Quick pre-deployment testing script
# Runs essential tests before Lambda deployment

set -e  # Exit on error

echo "========================================================================"
echo "PRE-DEPLOYMENT TESTING SUITE"
echo "========================================================================"
echo ""

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Test counter
TESTS_PASSED=0
TESTS_FAILED=0

run_test() {
    local test_name=$1
    local test_cmd=$2

    echo -e "${YELLOW}Running: ${test_name}${NC}"
    if eval "$test_cmd"; then
        echo -e "${GREEN}✓ PASSED${NC}"
        ((TESTS_PASSED++))
    else
        echo -e "${RED}✗ FAILED${NC}"
        ((TESTS_FAILED++))
    fi
    echo ""
}

# Check if we're in the right directory
if [ ! -f "README.md" ] || [ ! -d "step2_preprocessing" ]; then
    echo -e "${RED}Error: Must run from project root directory${NC}"
    exit 1
fi

echo "========================================================================"
echo "PHASE 1: DEPENDENCY CHECK"
echo "========================================================================"
echo ""

run_test "Check Python version (>=3.8)" "python --version | grep -E 'Python 3\.([8-9]|1[0-9])'"
run_test "Check PyTorch installed" "python -c 'import torch; print(f\"PyTorch {torch.__version__}\")'"
run_test "Check h5py installed" "python -c 'import h5py; print(f\"h5py {h5py.__version__}\")'"
run_test "Check pyarrow installed" "python -c 'import pyarrow; print(f\"pyarrow {pyarrow.__version__}\")'"
run_test "Check pandas installed" "python -c 'import pandas; print(f\"pandas {pandas.__version__}\")'"
run_test "Check spacy installed" "python -c 'import spacy; print(f\"spacy {spacy.__version__}\")'"
run_test "Check scispacy model" "python -c 'import spacy; spacy.load(\"en_core_sci_md\")'"
run_test "Check transformers installed" "python -c 'import transformers; print(f\"transformers {transformers.__version__}\")'"
run_test "Check langchain installed" "python -c 'from langchain_anthropic import ChatAnthropic'"

echo "========================================================================"
echo "PHASE 2: UNIT TESTS"
echo "========================================================================"
echo ""

cd step2_preprocessing

if [ -d "tests/unit" ]; then
    run_test "Image processor unit tests" "pytest tests/unit/test_image_loader.py -v --tb=short 2>&1 | tail -20"
    run_test "Temporal processor unit tests" "pytest tests/unit/test_temporal_processor.py -v --tb=short 2>&1 | tail -20"
    run_test "Note processor unit tests" "pytest tests/unit/test_note_processor.py -v --tb=short 2>&1 | tail -20"
    run_test "Multimodal dataset unit tests" "pytest tests/unit/test_multimodal_dataset.py -v --tb=short 2>&1 | tail -20"
else
    echo -e "${YELLOW}⚠ Unit tests directory not found, skipping...${NC}"
fi

cd ..

echo "========================================================================"
echo "PHASE 3: INTEGRATION TESTS"
echo "========================================================================"
echo ""

# Check if validation cohort exists
if [ ! -f "step2_preprocessing/cohorts/validation_subset_200.csv" ]; then
    echo -e "${YELLOW}⚠ Validation cohort not found. Creating subset...${NC}"

    # Check if main validation cohort exists
    if [ -f "output/cohorts/normal_cohort_validation.csv" ]; then
        mkdir -p step2_preprocessing/cohorts
        head -201 output/cohorts/normal_cohort_validation.csv > step2_preprocessing/cohorts/validation_subset_200.csv
        echo -e "${GREEN}✓ Created validation subset (200 samples)${NC}"
    else
        echo -e "${RED}✗ Main validation cohort not found. Run Step 1 first!${NC}"
        echo "  Run: python main.py"
        exit 1
    fi
fi

# Test small-scale preprocessing (5 samples)
echo -e "${YELLOW}Running small-scale preprocessing test (5 samples)...${NC}"
run_test "Small-scale preprocessing (5 samples)" "cd step2_preprocessing && python main.py --max-samples 5 --output-dir ./output_test_5 --val-only 2>&1 | tail -30 && cd .."

# Check outputs were created
if [ -d "step2_preprocessing/output_test_5/val" ]; then
    run_test "Check image outputs exist" "[ -d step2_preprocessing/output_test_5/val/images ] && [ \$(ls step2_preprocessing/output_test_5/val/images/*.pt 2>/dev/null | wc -l) -ge 1 ]"
    run_test "Check text outputs exist" "[ -d step2_preprocessing/output_test_5/val/text_features ] && [ \$(ls step2_preprocessing/output_test_5/val/text_features/*.pt 2>/dev/null | wc -l) -ge 1 ]"
    run_test "Check structured outputs exist" "[ -d step2_preprocessing/output_test_5/val/structured_features ] && [ \$(ls step2_preprocessing/output_test_5/val/structured_features/*.json 2>/dev/null | wc -l) -ge 1 ]"
fi

# Test CXR-PRO integration
if [ -f "step2_preprocessing/test_cxr_pro_integration.py" ]; then
    run_test "CXR-PRO integration test" "python step2_preprocessing/test_cxr_pro_integration.py 2>&1 | tail -30"
fi

echo "========================================================================"
echo "PHASE 4: PRE-COMPILATION TESTS"
echo "========================================================================"
echo ""

# Test small-scale pre-compilation (3 samples)
echo -e "${YELLOW}Running small-scale pre-compilation test (3 samples)...${NC}"
if [ -f "test_precompilation.py" ]; then
    run_test "Small-scale pre-compilation (3 samples)" "python test_precompilation.py 2>&1 | tail -30"
else
    echo -e "${YELLOW}⚠ test_precompilation.py not found, skipping...${NC}"
fi

# Test pipeline stages
if [ -f "test_pipeline_stages.py" ]; then
    echo -e "${YELLOW}Running pipeline stages test...${NC}"
    run_test "Pipeline stages test" "python test_pipeline_stages.py 2>&1 | tail -50"
else
    echo -e "${YELLOW}⚠ test_pipeline_stages.py not found, skipping...${NC}"
fi

echo "========================================================================"
echo "PHASE 5: DATA QUALITY VALIDATION"
echo "========================================================================"
echo ""

# Validate image data
if [ -d "step2_preprocessing/output_test_5/val/images" ]; then
    echo -e "${YELLOW}Validating image data quality...${NC}"
    run_test "Image shape validation" "python -c '
import torch
import glob
imgs = glob.glob(\"step2_preprocessing/output_test_5/val/images/*.pt\")
if imgs:
    img = torch.load(imgs[0])
    print(f\"Shape: {img.shape}, dtype: {img.dtype}, range: [{img.min():.3f}, {img.max():.3f}]\")
    assert img.ndim == 3, \"Image should be 3D (C,H,W)\"
    assert img.min() >= 0 and img.max() <= 1.1, \"Image range should be [0,1] for minmax\"
    print(\"✓ Image validation passed\")
' 2>&1"
fi

# Validate text data
if [ -d "step2_preprocessing/output_test_5/val/text_features" ]; then
    echo -e "${YELLOW}Validating text data quality...${NC}"
    run_test "Text feature validation" "python -c '
import torch
import glob
texts = glob.glob(\"step2_preprocessing/output_test_5/val/text_features/*.pt\")
if texts:
    text = torch.load(texts[0])
    print(f\"Summary length: {len(text[\"summary\"])}, Entities: {text[\"num_entities\"]}, Tokens: {text[\"tokens\"][\"num_tokens\"]}\")
    assert \"summary\" in text, \"Missing summary\"
    assert \"tokens\" in text, \"Missing tokens\"
    assert text[\"tokens\"][\"num_tokens\"] > 0, \"Should have at least some tokens\"
    print(\"✓ Text validation passed\")
' 2>&1"
fi

# Validate structured data
if [ -d "step2_preprocessing/output_test_5/val/structured_features" ]; then
    echo -e "${YELLOW}Validating structured data quality...${NC}"
    run_test "Structured feature validation" "python -c '
import json
import glob
structs = glob.glob(\"step2_preprocessing/output_test_5/val/structured_features/*.json\")
if structs:
    with open(structs[0]) as f:
        data = json.load(f)
    print(f\"Labs: {len(data[\"labs\"])}, Vitals: {len(data[\"vitals\"])}\")
    assert \"labs\" in data, \"Missing labs\"
    assert \"vitals\" in data, \"Missing vitals\"
    assert len(data[\"vitals\"]) > 0, \"Should have some vitals\"
    print(\"✓ Structured data validation passed\")
' 2>&1"
fi

echo "========================================================================"
echo "PHASE 6: CONFIGURATION VALIDATION"
echo "========================================================================"
echo ""

run_test "Check Claude Sonnet 4.5 configured" "grep -q 'claude-sonnet-4-5-20250929' step2_preprocessing/config/config.yaml"
run_test "Check full resolution enabled" "grep -q 'preserve_full_resolution: true' step2_preprocessing/config/config.yaml"
run_test "Check CXR-PRO enabled" "grep -q 'cxr_pro: true' step2_preprocessing/config/config.yaml"
run_test "Check NOT_DONE token configured" "grep -q 'missing_token: \"NOT_DONE\"' step2_preprocessing/config/config.yaml"

echo "========================================================================"
echo "TEST SUMMARY"
echo "========================================================================"
echo ""

TOTAL_TESTS=$((TESTS_PASSED + TESTS_FAILED))
echo -e "Total tests run: ${TOTAL_TESTS}"
echo -e "${GREEN}Passed: ${TESTS_PASSED}${NC}"
echo -e "${RED}Failed: ${TESTS_FAILED}${NC}"
echo ""

if [ $TESTS_FAILED -eq 0 ]; then
    echo -e "${GREEN}========================================================================"
    echo "✓ ALL TESTS PASSED - READY FOR LAMBDA DEPLOYMENT!"
    echo "========================================================================${NC}"
    echo ""
    echo "Next steps:"
    echo "1. Review PRE_DEPLOYMENT_TESTING.md for detailed testing guide"
    echo "2. Run medium-scale test (100 samples) to validate performance"
    echo "3. Build full pre-compiled dataset for Lambda deployment"
    echo ""
    exit 0
else
    echo -e "${RED}========================================================================"
    echo "✗ SOME TESTS FAILED - REVIEW ERRORS BEFORE DEPLOYMENT"
    echo "========================================================================${NC}"
    echo ""
    echo "Review the failed tests above and fix issues before deploying to Lambda."
    echo "See PRE_DEPLOYMENT_TESTING.md for troubleshooting guidance."
    echo ""
    exit 1
fi
