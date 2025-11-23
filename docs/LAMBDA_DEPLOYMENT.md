# Lambda GPU Deployment Guide
## MIMIC-CXR Preprocessing Validation (200 Samples)

**Objective**: Validate preprocessing pipeline on 200 samples to ensure MAE-readiness before full-scale Step 3 implementation.

**Instance**: 1x NVIDIA GH200 Grace Hopper
**Estimated Time**: 4-5 hours
**Estimated Cost**: $32-41 ($8/hr × 4-5 hours)

---

## Pre-Deployment Validation Checklist

**Complete these checks BEFORE Lambda deployment** to ensure smooth preprocessing and avoid costly failures on the GPU instance.

### Local Environment Verification

#### 1. Cohort Generation ✅
```bash
# Verify new stratified cohort exists
ls -lh step2_preprocessing/cohorts/validation_subset_200.csv
# Expected: ~30KB, 201 lines (200 samples + header)

# Verify demographic balance
python3 generate_stratified_cohort.py \
  --input output/output_test/cohorts/normal_cohort_validation.csv \
  --output /tmp/test_cohort.csv \
  --n-samples 200 \
  --stratify-by gender anchor_age

# Check output shows balanced gender/age distribution
```

**Expected Output**:
- Gender: ~59% F / 41% M
- Age groups: Proportional across 18-30, 31-45, 46-60, 61-75, 76+
- Mean age: ~48 years

#### 2. Data Extraction ✅
```bash
# Verify all 5 components extracted
ls -ld validation_data_subset/*

# Component checklist:
# ✓ CXR images: ~850 JPG files (200 studies)
# ✓ MIMIC-IV: 4 structured data files (patients, admissions, labevents, d_labitems)
# ✓ MIMIC-ED: 7 ED tables
# ✓ CXR-PRO: mimic_train_impressions.csv (66MB, 371k reports)
# ✓ DICOM metadata: mimic-cxr-2.0.0-metadata.csv.gz (16MB)

# Verify CXR image count
find validation_data_subset/cxr/files -name "*.jpg" | wc -l
# Expected: ~850 files

# Verify CXR-PRO reports present
wc -l validation_data_subset/cxr-pro/mimic_train_impressions.csv
# Expected: 371952 lines

# Verify DICOM metadata present
ls -lh validation_data_subset/mimic-cxr-2.0.0-metadata.csv.gz
# Expected: ~16MB compressed
```

#### 3. Archive Creation ✅
```bash
# Verify compressed archives exist and are reasonable size
ls -lh *.tar.gz

# Expected sizes:
# validation_data_subset.tar.gz: ~1.9GB (compressed from 3.7GB)
# step2_preprocessing.tar.gz: ~80MB (code + configs)
```

#### 4. Local Path Validation (Optional but Recommended) ✅
```bash
# Test path validation script locally
./validate_deployment_paths.sh step2_preprocessing/config/config_validation.yaml

# This will FAIL locally (paths point to Lambda filesystem)
# But verifies script works and shows what paths will be checked
```

### Pre-Transfer Checklist

Before transferring to Lambda GPU, ensure:

- [ ] **Cohort generated**: validation_subset_200.csv exists with 200 stratified samples
- [ ] **Data extracted**: validation_data_subset/ contains all 5 components (3.7GB uncompressed)
- [ ] **CXR-PRO verified**: mimic_train_impressions.csv present (prevents 93.5% failure)
- [ ] **DICOM metadata verified**: mimic-cxr-2.0.0-metadata.csv.gz present (adds 10 acquisition features)
- [ ] **Archives created**: Both .tar.gz files exist (~2GB total compressed)
- [ ] **Archive sizes reasonable**: validation_data_subset.tar.gz ~1.9GB, step2_preprocessing.tar.gz ~80MB
- [ ] **Extraction script updated**: Latest version with 5 steps (includes DICOM)
- [ ] **Validation script present**: validate_deployment_paths.sh executable
- [ ] **Lambda credentials ready**: API key for Anthropic Claude (if using summarization)
- [ ] **Cost budgeted**: Expect $32-40 for 4-5 hours on 1x GH200

### What These Checks Prevent

**93.5% Failure Scenario** (Previous Lambda Run):
- **Root cause**: CXR-PRO reports path not configured
- **Symptom**: Text sequences showed 2 tokens (empty), structured data 93.5% empty
- **Prevention**: Steps 2 and 4 above verify CXR-PRO reports present and paths validated

**DICOM Features Missing**:
- **Impact**: Model can't distinguish AP portable vs PA standard (20-30% FP reduction lost)
- **Prevention**: Step 2 verifies DICOM metadata file present

**Incomplete Extraction**:
- **Impact**: Preprocessing fails on missing files, wastes Lambda time ($8/hr)
- **Prevention**: Step 2 component checklist verifies all 5 data sources

**Unbalanced Demographics**:
- **Impact**: Results not representative of full population
- **Prevention**: Step 1 verifies stratified sampling by gender/age

### Quick Validation Summary

```bash
# One-liner to check everything
echo "Cohort: $(wc -l < step2_preprocessing/cohorts/validation_subset_200.csv) lines" && \
echo "CXR images: $(find validation_data_subset/cxr/files -name "*.jpg" 2>/dev/null | wc -l) files" && \
echo "CXR-PRO: $([ -f validation_data_subset/cxr-pro/mimic_train_impressions.csv ] && echo "✓" || echo "✗")" && \
echo "DICOM: $([ -f validation_data_subset/mimic-cxr-2.0.0-metadata.csv.gz ] && echo "✓" || echo "✗")" && \
echo "Archives: $(ls *.tar.gz 2>/dev/null | wc -l) files" && \
echo "Total size: $(du -sh validation_data_subset.tar.gz step2_preprocessing.tar.gz 2>/dev/null | awk '{sum+=$1} END {print sum}')"

# Expected output:
# Cohort: 201 lines
# CXR images: ~850 files
# CXR-PRO: ✓
# DICOM: ✓
# Archives: 2 files
# Total size: ~2GB
```

If all checks pass, proceed to Lambda deployment below.

---

## Quick Start

### 1. Local Preparation (30 min)

```bash
cd /home/dev/Documents/Portfolio/MIMIC/MIMIC-CXR-Anomaly-Preprocessing

# Extract 200-sample subset from full MIMIC dataset
chmod +x extract_validation_subset.sh
./extract_validation_subset.sh

# Compress for transfer
tar -czf validation_data_subset.tar.gz validation_data_subset/
tar -czf step2_preprocessing.tar.gz step2_preprocessing/

# Check sizes
ls -lh *.tar.gz
# Expected: validation_data_subset.tar.gz (~3-10GB), step2_preprocessing.tar.gz (~50-100MB)
```

### 2. Launch Lambda GPU Instance

- Go to https://cloud.lambdalabs.com/instances
- Select: **1x NVIDIA GH200 Grace Hopper**
- Region: us-west or us-east (choose lowest latency)
- OS: Ubuntu 22.04 LTS with CUDA 12.x
- Launch instance and note the IP address

### 3. Transfer Data to Lambda GPU (15 min)

```bash
# Replace <LAMBDA_IP> with your instance IP
export LAMBDA_IP=xxx.xxx.xxx.xxx

# Transfer compressed archives
rsync -avz --progress validation_data_subset.tar.gz ubuntu@$LAMBDA_IP:~/
rsync -avz --progress step2_preprocessing.tar.gz ubuntu@$LAMBDA_IP:~/
```

### 4. Setup Environment on Lambda GPU (15 min)

```bash
# SSH into Lambda GPU
ssh ubuntu@$LAMBDA_IP

# Create workspace and extract
mkdir -p ~/mimic-cxr-validation
cd ~/mimic-cxr-validation
mv ~/validation_data_subset.tar.gz ~/step2_preprocessing.tar.gz .
tar -xzf validation_data_subset.tar.gz
tar -xzf step2_preprocessing.tar.gz

# Setup Python environment
cd step2_preprocessing
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
pip install scispacy
pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.1/en_core_sci_md-0.5.1.tar.gz

# Verify GPU
python3 -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0)}')"
# Expected: CUDA: True, GPU: NVIDIA GH200
```

### 5. Configure Data Paths (5 min)

```bash
cd ~/mimic-cxr-validation/step2_preprocessing

# Update config file paths
sed -i 's|/media/dev/MIMIC_DATA/mimic-cxr-jpg|/home/ubuntu/mimic-cxr-validation/validation_data_subset/cxr|g' config/config_validation.yaml
sed -i 's|/home/dev/Documents/Portfolio/MIMIC_Data/physionet.org/files/mimiciv/3.1|/home/ubuntu/mimic-cxr-validation/validation_data_subset/mimic-iv|g' config/config_validation.yaml
sed -i 's|/home/dev/Documents/Portfolio/MIMIC_Data/physionet.org/files/mimic-iv-ed/2.2|/home/ubuntu/mimic-cxr-validation/validation_data_subset/mimic-ed|g' config/config_validation.yaml
sed -i 's|cxr_pro_reports:.*|cxr_pro_reports: "/home/ubuntu/mimic-cxr-validation/validation_data_subset/cxr-pro/mimic_train_impressions.csv"|g' config/config_validation.yaml
sed -i 's|dicom_metadata_path:.*|dicom_metadata_path: "/home/ubuntu/mimic-cxr-validation/validation_data_subset/mimic-cxr-2.0.0-metadata.csv.gz"|g' config/config_validation.yaml

# IMPORTANT: Validate all paths before proceeding
cd ~/mimic-cxr-validation
./validate_deployment_paths.sh step2_preprocessing/config/config_validation.yaml

# If validation fails, DO NOT proceed - fix paths first!
# This prevents 93.5% failure rate from missing CXR-PRO configuration

# Set Anthropic API key
export ANTHROPIC_API_KEY='your-anthropic-api-key-here'
```

### 6. Run Preprocessing Pipeline (2-3 hours)

```bash
cd ~/mimic-cxr-validation/step2_preprocessing
source venv/bin/activate

# Run preprocessing with validation config
python3 main.py \
  --config config/config_validation.yaml \
  --anthropic-api-key $ANTHROPIC_API_KEY \
  --train-only \
  --skip-on-error \
  2>&1 | tee preprocessing_validation.log

# Monitor progress in separate terminal
tail -f preprocessing_validation.log

# Monitor GPU usage
watch -n 2 nvidia-smi
```

**Expected runtime**:
- Image processing: 30-60 min (200 full-resolution CXRs)
- Structured features: 10-20 min (labs/vitals extraction)
- Text processing: 60-90 min (NER + Claude summarization)
- **Total: 2-3 hours**

### 7. Validate MAE Readiness (15 min)

```bash
cd ~/mimic-cxr-validation/step2_preprocessing
source venv/bin/activate

# Run validation script
python3 validate_mae_readiness.py \
  --output-dir output/validation_200 \
  --report-path output/validation_200/mae_readiness_report.json \
  2>&1 | tee validation_report.log

# View results
cat validation_report.log
```

**Success criteria**: ≥95% of samples fully valid (all modalities)

Expected output:
```
================================================================================
MAE READINESS ASSESSMENT
================================================================================
✓ READY FOR MAE TRAINING (95.0%+ success rate)
  - All modalities properly formatted
  - Image tensors: [C,H,W] normalized [0,1]
  - Text tokens: ClinicalBERT format ≤512 tokens
  - Structured features: Temporal aggregations present
================================================================================
```

### 8. Retrieve Results (10 min)

```bash
# On LOCAL machine
cd /home/dev/Documents/Portfolio/MIMIC/MIMIC-CXR-Anomaly-Preprocessing
mkdir -p validation_results

# Download validation report and logs
scp ubuntu@$LAMBDA_IP:~/mimic-cxr-validation/step2_preprocessing/output/validation_200/mae_readiness_report.json \
  ./validation_results/

scp ubuntu@$LAMBDA_IP:~/mimic-cxr-validation/step2_preprocessing/preprocessing_validation.log \
  ./validation_results/

scp ubuntu@$LAMBDA_IP:~/mimic-cxr-validation/step2_preprocessing/validation_report.log \
  ./validation_results/

scp ubuntu@$LAMBDA_IP:~/mimic-cxr-validation/step2_preprocessing/output/validation_200/processing_stats.json \
  ./validation_results/

# View results locally
cat validation_results/mae_readiness_report.json | jq '.'
```

### 9. Cleanup

```bash
# IMPORTANT: Terminate Lambda GPU instance to stop billing!
# Via Lambda Cloud Dashboard:
#   - Navigate to Instances
#   - Select your instance
#   - Click "Terminate"
```

### 10. Post-Deployment Comparison & Analysis

After retrieving results, compare the new run against the previous baseline to quantify improvements.

#### A. Quick Success Metrics

```bash
cd /home/dev/Documents/Portfolio/MIMIC/MIMIC-CXR-Anomaly-Preprocessing/validation_results

# View MAE readiness report
cat mae_readiness_report.json | jq '.summary'

# Key metrics to check:
# 1. Overall success rate (target: ≥95%)
# 2. Text sequence length (target: 50-200 tokens, not 2)
# 3. Structured feature completeness (target: <5% empty, not 93.5%)
# 4. DICOM features present (target: 10 fields per sample)
```

#### B. Comparison with Previous Run (November 22, 2025)

**Previous Lambda Run Results** (Before CXR-PRO & DICOM fixes):
```json
{
  "total_samples": 200,
  "successful_samples": 13,
  "failed_samples": 187,
  "success_rate": 6.5%,

  "text_issues": {
    "mean_sequence_length": 2.0,
    "median_sequence_length": 2.0,
    "issue": "Empty text sequences (CXR-PRO path missing)"
  },

  "structured_issues": {
    "empty_count": 187,
    "empty_percentage": 93.5%,
    "issue": "All vitals showing 'NOT_DONE' (ED data or CXR-PRO missing)"
  },

  "dicom_features": {
    "present": false,
    "fields_per_sample": 0
  }
}
```

**Expected New Run Results** (After CXR-PRO & DICOM fixes):
```json
{
  "total_samples": 200,
  "successful_samples": 190-200,
  "success_rate": "95-100%",

  "text_improvements": {
    "mean_sequence_length": "50-200 tokens",
    "median_sequence_length": "~100 tokens",
    "fix_applied": "CXR-PRO reports path configured"
  },

  "structured_improvements": {
    "empty_count": "<10 samples",
    "empty_percentage": "<5%",
    "fix_applied": "All 5 data paths validated before preprocessing"
  },

  "dicom_features": {
    "present": true,
    "fields_per_sample": 10,
    "features": ["view_pa", "view_ap", "view_lateral", "orientation_erect",
                 "orientation_recumbent", "is_portable", "image_rows_normalized",
                 "image_cols_normalized", "num_views", "orientation_unknown"]
  }
}
```

#### C. Detailed Comparison Script

Create a comparison script to analyze both runs:

```python
# compare_lambda_runs.py
import json
import pandas as pd
from pathlib import Path

def compare_runs(old_path, new_path):
    """Compare two Lambda preprocessing runs"""

    # Load reports
    with open(old_path) as f:
        old_data = json.load(f)
    with open(new_path) as f:
        new_data = json.load(f)

    print("="*70)
    print("Lambda Preprocessing Run Comparison")
    print("="*70)

    # Success rate comparison
    old_success = old_data.get('success_rate', 0)
    new_success = new_data.get('success_rate', 0)
    improvement = new_success - old_success

    print(f"\n📊 SUCCESS RATE:")
    print(f"  Previous run: {old_success:.1f}%")
    print(f"  New run:      {new_success:.1f}%")
    print(f"  Improvement:  +{improvement:.1f} percentage points")

    # Text processing comparison
    old_text_len = old_data.get('text_seq_length_stats', {}).get('mean', 0)
    new_text_len = new_data.get('text_seq_length_stats', {}).get('mean', 0)

    print(f"\n📝 TEXT PROCESSING:")
    print(f"  Previous mean seq length: {old_text_len:.1f} tokens")
    print(f"  New mean seq length:      {new_text_len:.1f} tokens")
    if new_text_len > 10:
        print(f"  ✅ Text processing FIXED (was {old_text_len:.0f}, now {new_text_len:.0f})")

    # Structured data comparison
    old_empty = old_data.get('structured_empty_count', 0)
    new_empty = new_data.get('structured_empty_count', 0)
    old_empty_pct = 100 * old_empty / 200
    new_empty_pct = 100 * new_empty / 200

    print(f"\n🏥 STRUCTURED DATA:")
    print(f"  Previous empty: {old_empty}/200 ({old_empty_pct:.1f}%)")
    print(f"  New empty:      {new_empty}/200 ({new_empty_pct:.1f}%)")
    if new_empty_pct < 10:
        print(f"  ✅ Structured data FIXED (reduced from {old_empty_pct:.1f}% to {new_empty_pct:.1f}%)")

    # DICOM features (new in this run)
    new_dicom_count = new_data.get('dicom_features_count', 0)
    if new_dicom_count > 0:
        print(f"\n🩻 DICOM METADATA (NEW):")
        print(f"  Samples with DICOM features: {new_dicom_count}/200")
        print(f"  Features per sample: 10 (view position, orientation, portable, dimensions)")
        print(f"  ✅ NEW FEATURE: Image acquisition context available")

    # Overall verdict
    print(f"\n{'='*70}")
    if new_success >= 95 and new_text_len > 10 and new_empty_pct < 10:
        print("✅ VALIDATION PASSED - All improvements verified")
        print("   Ready to proceed with full dataset preprocessing and MAE training")
    else:
        print("⚠️  VALIDATION NEEDS REVIEW - Some metrics below target")
        print("   Review failed samples before proceeding")
    print(f"{'='*70}\n")

# Run comparison
compare_runs(
    'validation_results/old_run/mae_readiness_report.json',
    'validation_results/mae_readiness_report.json'
)
```

#### D. Key Improvements to Verify

| Metric | Previous Run | Target (New Run) | What Fixed It |
|--------|--------------|------------------|---------------|
| **Success Rate** | 6.5% (13/200) | ≥95% (190+/200) | CXR-PRO path + validation script |
| **Text Seq Length** | 2.0 tokens (empty) | 50-200 tokens | CXR-PRO reports loaded |
| **Structured Empty** | 93.5% (187/200) | <5% (<10/200) | All 5 paths validated |
| **DICOM Features** | 0 fields | 10 fields/sample | New metadata integration |
| **Processing Time** | N/A | 0.4-0.5s/sample | Baseline for full run |

#### E. Sample-Level Validation

Check a few specific samples to verify all modalities present:

```bash
# Check a random sample's structured features
cat output/validation_200/train/structured_features/s10874533_study54444686.json | jq . | head -30

# Expected to see:
# - First 10 fields: DICOM metadata (view_pa, view_ap, etc.)
# - Next fields: Vitals (temperature, heart_rate, resp_rate, etc.)
# - Final fields: Labs (if available)

# Verify DICOM features present
cat output/validation_200/train/structured_features/s10874533_study54444686.json | jq 'keys | length'
# Expected: ~40-60 fields (10 DICOM + 30-50 clinical features)

# Check portable detection works
grep -A 1 "is_portable" output/validation_200/train/structured_features/*.json | head -20
# Should see is_portable: 1.0 for some samples (AP portable studies)
```

#### F. Cost-Benefit Analysis

```bash
# Calculate improvement ROI
OLD_SUCCESS_RATE=6.5
NEW_SUCCESS_RATE=95.0  # Target
LAMBDA_COST_PER_HOUR=8
PREPROCESSING_HOURS=4

echo "Improvement Analysis:"
echo "  Successful samples (old): $((200 * $OLD_SUCCESS_RATE / 100))"
echo "  Successful samples (new): $((200 * $NEW_SUCCESS_RATE / 100))"
echo "  Additional usable samples: $((200 * ($NEW_SUCCESS_RATE - $OLD_SUCCESS_RATE) / 100))"
echo ""
echo "Cost per usable sample:"
echo "  Old run: \$$(echo "scale=2; $LAMBDA_COST_PER_HOUR * $PREPROCESSING_HOURS / (200 * $OLD_SUCCESS_RATE / 100)" | bc)/sample"
echo "  New run: \$$(echo "scale=2; $LAMBDA_COST_PER_HOUR * $PREPROCESSING_HOURS / (200 * $NEW_SUCCESS_RATE / 100)" | bc)/sample"
```

**Expected Output**:
```
Improvement Analysis:
  Successful samples (old): 13
  Successful samples (new): 190
  Additional usable samples: 177

Cost per usable sample:
  Old run: $2.46/sample (wasted 93.5% of GPU time!)
  New run: $0.17/sample (efficient use of resources)
```

#### G. Decision Criteria

**Proceed to full dataset if**:
- ✅ Success rate ≥95%
- ✅ Text sequences 50-200 tokens (not 2)
- ✅ Structured features <5% empty (not 93.5%)
- ✅ DICOM features present in all samples (10 fields each)
- ✅ Processing time reasonable (<1s per sample)

**Investigate further if**:
- ❌ Success rate <90%
- ❌ Text sequences still ~2 tokens
- ❌ Structured features >10% empty
- ❌ DICOM features missing
- ❌ Processing time >5s per sample

---

## Decision Point

### If Validation PASSES (≥95% success)

**Next steps**:
1. ✅ Preprocessing pipeline validated and MAE-ready
2. 📋 **Plan Step 3: Multimodal MAE Implementation**
   - Design MAE architecture (image/text/structured encoders)
   - Implement tokenization modules
   - Create training pipeline
   - Estimate compute requirements

### If Validation FAILS (<95% success)

**Debug process**:
1. Analyze errors in `mae_readiness_report.json`
2. Check failed sample details in processing logs
3. Fix preprocessing code issues
4. Re-run validation on failed samples
5. Iterate until ≥95% success

---

## File Structure After Processing

```
output/validation_200/
└── train/
    ├── images/                 # 200 .pt files (PyTorch tensors)
    │   └── s{subject_id}_study{study_id}.pt
    ├── text_features/          # 200 .pt files (ClinicalBERT tokens + summaries)
    │   └── s{subject_id}_study{study_id}.pt
    ├── structured_features/    # 200 .json files (labs/vitals temporal aggregations)
    │   └── s{subject_id}_study{study_id}.json
    ├── metadata/               # 200 .json files (sample metadata)
    │   └── s{subject_id}_study{study_id}.json
    └── processing_stats.json   # Overall statistics
```

---

## Troubleshooting

### CUDA Out of Memory
```bash
# Check GPU memory
nvidia-smi

# Solution: Already using batch_size=1 and memory mapping
# If still failing, disable GPU for specific modality:
# In config_validation.yaml, set: processing.use_gpu: false
```

### Data Path Errors
```bash
# Verify data paths exist
ls -lh ~/mimic-cxr-validation/validation_data_subset/cxr/files/
ls -lh ~/mimic-cxr-validation/validation_data_subset/mimic-iv/hosp/
ls -lh ~/mimic-cxr-validation/validation_data_subset/mimic-ed/ed/

# If missing, re-run extraction script locally and re-transfer
```

### Claude API Rate Limits
```bash
# Check API key is set
echo $ANTHROPIC_API_KEY

# If hitting rate limits, the pipeline will retry automatically (max_retries=2)
# Monitor in logs for "Claude summarization failed" errors
```

### Slow Processing
```bash
# Check bottleneck
nvidia-smi  # GPU utilization
htop        # CPU/memory usage
iotop       # Disk I/O

# Common causes:
# - Low GPU utilization: I/O bound (expected for image loading)
# - High network I/O: Claude API calls (expected for text processing)
# - Disk I/O: Image reading from compressed archives (use uncompressed for speed)
```

---

## Cost Breakdown

| Component | Cost |
|-----------|------|
| Lambda GPU (1xGH200, 4-5 hrs) | $32-40 |
| Claude API (summarization disabled) | $0 |
| Data transfer (egress) | $0 (small dataset) |
| **Total** | **$32-40** |

**Cost optimization**:
- Use spot instances if available (30-50% discount)
- Monitor progress; terminate if stuck
- Text summarization disabled in validation config (saves $10-20)

---

## Validation Checklist

Before running:
- [ ] Lambda GPU instance launched (1xGH200)
- [ ] Data extracted and transferred (~5-15GB)
- [ ] Dependencies installed (PyTorch, scispacy, LangChain)
- [ ] Config paths updated for Lambda filesystem (all 4: CXR, MIMIC-IV, MIMIC-ED, **CXR-PRO**)
- [ ] **Path validation passed** (`./validate_deployment_paths.sh config/config_validation.yaml`)
- [ ] Anthropic API key set
- [ ] GPU verified (`nvidia-smi`)

After running:
- [ ] Preprocessing completed without crashes
- [ ] 200 samples processed (check `processing_stats.json`)
- [ ] Validation report generated (`mae_readiness_report.json`)
- [ ] Success rate ≥95%
- [ ] Results downloaded to local machine
- [ ] **Lambda GPU instance terminated**

---

## Troubleshooting

### Issue 1: Text Processing Produces Empty Sequences (2 tokens)

**Symptoms**:
- `text_seq_length_stats` shows mean=2.0, median=2.0 (empty sequences)
- All samples have minimal text tokens instead of expected 50-200 tokens
- Text appears valid in cohort CSV but not loaded by processor

**Root Cause**: CXR-PRO reports path not configured or incorrect in `config_validation.yaml`

**Diagnosis**:
```bash
# Check if CXR-PRO path exists in config
grep "cxr_pro_reports" config/config_validation.yaml

# Should output something like:
# cxr_pro_reports: "/home/ubuntu/mimic-cxr-validation/validation_data_subset/cxr-pro/mimic_train_impressions.csv"

# If empty, the field is missing!
```

**Fix**:
```bash
# Add CXR-PRO path to config if missing
echo '  cxr_pro_reports: "/home/ubuntu/mimic-cxr-validation/validation_data_subset/cxr-pro/mimic_train_impressions.csv"' >> config/config_validation.yaml

# Or use sed to update existing path
sed -i 's|cxr_pro_reports:.*|cxr_pro_reports: "/home/ubuntu/mimic-cxr-validation/validation_data_subset/cxr-pro/mimic_train_impressions.csv"|g' config/config_validation.yaml
```

**Prevention**: Run `validate_deployment_paths.sh` before preprocessing (see below)

---

### Issue 2: Structured Features Show "NOT_DONE" (Empty)

**Symptoms**:
- 93.5% of samples have `structured_status: "Empty"` in validation report
- All vital signs show `is_missing: true` and value `NOT_DONE`
- ED vitals exist in raw data but not extracted

**Root Cause**: MIMIC-IV-ED path not configured correctly

**Diagnosis**:
```bash
# Check if ED vitals file exists
ls -lh /home/ubuntu/mimic-cxr-validation/validation_data_subset/mimic-ed/ed/vitalsign.csv

# If missing, check config path
grep "mimic_ed_base" config/config_validation.yaml
```

**Fix**:
```bash
# Update MIMIC-ED path
sed -i 's|/home/dev/Documents/Portfolio/MIMIC_Data/physionet.org/files/mimic-iv-ed/2.2|/home/ubuntu/mimic-cxr-validation/validation_data_subset/mimic-ed|g' config/config_validation.yaml
```

---

### Validation Script (Recommended)

**Before running preprocessing**, validate all data paths:

```bash
# Run path validation script
./validate_deployment_paths.sh step2_preprocessing/config/config_validation.yaml

# Expected output:
# ✅ All paths validated successfully!
# 1. MIMIC-CXR Images: ✅ PASS (434 JPG files)
# 2. MIMIC-IV Structured Data: ✅ PASS
# 3. MIMIC-IV-ED Data: ✅ PASS
# 4. CXR-PRO Radiology Reports: ✅ PASS (371,951 reports, 66MB)
# 5. DICOM Metadata: ✅ PASS (377k images, view position & orientation)
```

**If validation fails**, fix paths before preprocessing to avoid 93.5% failure rate!

---

### DICOM Metadata Integration (NEW)

**Purpose**: Provides image acquisition context to prevent misclassifications due to imaging technique.

**Features extracted from DICOM metadata**:
- **View Position**: PA, AP, LATERAL (one-hot encoded)
- **Patient Orientation**: Erect vs. Recumbent (binary)
- **Portable indicator**: Detects "CHEST (PORTABLE AP)" procedures
- **Image dimensions**: Normalized pixel dimensions (proxy for field of view)
- **Number of views**: Indicates study comprehensiveness

**Why this matters**:
- AP portable (patient supine) shows enlarged cardiac silhouette → prevents false cardiomegaly detections
- LATERAL views show different anatomy → model knows view type
- Recumbent position affects fluid distribution → prevents false edema detections

**Configuration**: Already included in `config_validation.yaml`:
```yaml
data:
  dicom_metadata_path: "/path/to/mimic-cxr-2.0.0-metadata.csv.gz"
```

**Coverage**:
- 227,835 studies (96.8% have view position, 90.0% have orientation)
- 58.2% AP, 37.7% PA, 47.2% LATERAL
- 81.3% Erect, 10.1% Recumbent
- 49.7% Portable procedures

---

### Issue 3: Missing Dependencies

**Symptoms**: `ModuleNotFoundError` during preprocessing

**Common missing packages**:
```bash
pip install sentence-transformers  # For text embeddings
pip install scispacy  # For medical NER
pip install en-core-sci-md  # ScispaCy model (350MB)
```

---

### Issue 4: Lambda GPU Instance Costs

**Cost monitoring**:
- 1x NVIDIA GH200 Grace Hopper: **$3.69/hour**
- 200-sample validation: ~3 hours = **~$11**
- **Always terminate instance after downloading results!**

```bash
# Check instance status from local machine
ssh ubuntu@<LAMBDA_IP> "uptime"

# Terminate from Lambda dashboard after confirming downloads
```

---

## Additional Resources

- **Lambda GPU Docs**: https://docs.lambdalabs.com/
- **PyTorch CUDA Guide**: https://pytorch.org/get-started/locally/
- **Anthropic API Docs**: https://docs.anthropic.com/

---

**Created**: 2025-11-20
**Purpose**: Validate MIMIC-CXR preprocessing pipeline (Step 2) before MAE training (Step 3)
**Status**: Ready for deployment
