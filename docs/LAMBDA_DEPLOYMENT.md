# Lambda GPU Deployment Guide
## MIMIC-CXR Preprocessing Validation (200 Samples)

**Objective**: Validate preprocessing pipeline on 200 samples to ensure MAE-readiness before full-scale Step 3 implementation.

**Instance**: 1x NVIDIA GH200 Grace Hopper
**Estimated Time**: 4-5 hours
**Estimated Cost**: $32-41 ($8/hr × 4-5 hours)

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
