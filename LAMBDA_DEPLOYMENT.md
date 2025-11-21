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
- [ ] Config paths updated for Lambda filesystem
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

## Additional Resources

- **Lambda GPU Docs**: https://docs.lambdalabs.com/
- **PyTorch CUDA Guide**: https://pytorch.org/get-started/locally/
- **Anthropic API Docs**: https://docs.anthropic.com/

---

**Created**: 2025-11-20
**Purpose**: Validate MIMIC-CXR preprocessing pipeline (Step 2) before MAE training (Step 3)
**Status**: Ready for deployment
