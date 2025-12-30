# Configuration Guide

Complete reference for configuring the MIMIC-CXR Anomaly Detection Pipeline.

---

## Table of Contents

1. [Environment Variables](#environment-variables)
2. [Training Configurations](#training-configurations)
3. [Preprocessing Options](#preprocessing-options)
4. [Hyperparameter Reference](#hyperparameter-reference)

---

## Environment Variables

### Required Data Paths

Set these in a `.env` file or export as environment variables:

```bash
# MIMIC-CXR: Chest X-ray images (JPG format)
MIMIC_CXR_JPG_PATH=/path/to/mimic-cxr-jpg/2.1.0

# MIMIC-IV: Hospital records (patients, admissions, labs)
MIMIC_IV_PATH=/path/to/mimiciv/3.1

# MIMIC-IV-ED: Emergency department data (vitals, triage)
MIMIC_IV_ED_PATH=/path/to/mimic-iv-ed/2.2

# CXR-PRO: Radiology report impressions
CXR_PRO_PATH=/path/to/cxr-pro/1.0.0

# Output directory for processed data and models
OUTPUT_PATH=./output
```

### Optional Configuration

```bash
# Claude API for text summarization (optional)
ANTHROPIC_API_KEY=sk-ant-...

# Disable Claude API calls (use clinical context only)
# Set to empty string or omit entirely
ANTHROPIC_API_KEY=
```

### Example .env File

```bash
# .env - Copy to project root and customize paths

# Required: MIMIC Dataset Paths
MIMIC_CXR_JPG_PATH=/media/dev/MIMIC_DATA/mimic-cxr-jpg/2.1.0
MIMIC_IV_PATH=/media/dev/MIMIC_DATA/mimiciv/3.1
MIMIC_IV_ED_PATH=/media/dev/MIMIC_DATA/mimic-iv-ed/2.2
CXR_PRO_PATH=/media/dev/MIMIC_DATA/cxr-pro/1.0.0
OUTPUT_PATH=./output

# Optional: Claude API for summarization
ANTHROPIC_API_KEY=sk-ant-api03-...
```

---

## Training Configurations

### Available Presets

Three training configurations are available in `src/models/config.py`:

| Config | Purpose | Epochs | Batch | LR | Image Size |
|--------|---------|--------|-------|-----|------------|
| `debug` | Quick testing | 2 | 2 | 1e-4 | 224 |
| `fast` | Development | 10 | 8 | 5e-5 | 384 |
| `base` | Production | 30 | 16 | 3e-5 | 512 |

### Using Configurations

```bash
# Debug: Quick validation (2 epochs)
python train_classifier.py --config debug

# Fast: Development testing (10 epochs)
python train_classifier.py --config fast

# Base: Production training (30 epochs)
python train_classifier.py --config base
```

### Configuration Details

#### Debug Config
```python
{
    "epochs": 2,
    "batch_size": 2,
    "learning_rate": 1e-4,
    "img_size": 224,
    "freeze_mae": True,      # Always frozen for speed
    "warmup_epochs": 0,
    "weight_decay": 0.01,
}
```

#### Fast Config
```python
{
    "epochs": 10,
    "batch_size": 8,
    "learning_rate": 5e-5,
    "img_size": 384,
    "freeze_mae": True,
    "warmup_epochs": 1,
    "weight_decay": 0.01,
}
```

#### Base Config
```python
{
    "epochs": 30,
    "batch_size": 16,
    "learning_rate": 3e-5,
    "img_size": 512,
    "freeze_mae": True,       # Frozen first 5 epochs
    "unfreeze_epoch": 5,      # Then gradually unfreezes
    "warmup_epochs": 2,
    "weight_decay": 0.05,
}
```

### Custom Configuration

Override any parameter via command line:

```bash
python train_classifier.py \
    --config base \
    --epochs 50 \
    --batch-size 8 \
    --learning-rate 1e-5 \
    --img-size 1024
```

---

## Preprocessing Options

### Cohort Building

```bash
# Build normal cohort (for MAE pretraining)
python build_cohort.py --normal-only

# Build anomalous cohort (for classification)
python build_cohort.py --anomalous-only

# Custom output directory
python build_cohort.py --output-dir ./custom_cohorts
```

### Data Preprocessing

```bash
# Standard preprocessing
python preprocess.py \
    --cohort output/cohorts/normal_train.parquet \
    --workers 8

# Leak-free mode (REQUIRED for classification)
python preprocess.py \
    --cohort output/cohorts/anomalous_train.parquet \
    --leak-free \
    --enable-summarization \
    --workers 8

# Skip specific modalities
python preprocess.py \
    --skip-images \     # Skip image processing
    --skip-structured \ # Skip labs/vitals
    --skip-text         # Skip text processing
```

### Preprocessing Flags

| Flag | Description |
|------|-------------|
| `--leak-free` | Exclude radiology reports (prevents CheXpert label leakage) |
| `--enable-summarization` | Use Claude API for text summarization |
| `--workers N` | Number of parallel workers (default: 8) |
| `--skip-images` | Skip image preprocessing |
| `--skip-structured` | Skip structured data (labs/vitals) |
| `--skip-text` | Skip text preprocessing |

---

## Hyperparameter Reference

### MAE Pretraining (`train_mae.py`)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--img-size` | 1024 | Input image resolution |
| `--patch-size` | 16 | ViT patch size |
| `--mask-ratio` | 0.75 | Percentage of patches to mask |
| `--epochs` | 800 | Total training epochs |
| `--batch-size` | 4 | Samples per batch |
| `--learning-rate` | 1.5e-4 | Base learning rate |
| `--weight-decay` | 0.05 | AdamW weight decay |
| `--warmup-epochs` | 40 | Linear warmup epochs |

### Classifier Training (`train_classifier.py`)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--config` | `base` | Preset configuration (debug/fast/base) |
| `--epochs` | 30 | Total training epochs |
| `--batch-size` | 16 | Samples per batch |
| `--learning-rate` | 3e-5 | Base learning rate |
| `--img-size` | 512 | Input image resolution |
| `--freeze-mae` | True | Freeze MAE encoder initially |
| `--unfreeze-epoch` | 5 | Epoch to start unfreezing |
| `--llrd-factor` | 0.9 | Layer-wise LR decay factor |

### Loss Function Weights

| Loss | Default Weight | Purpose |
|------|----------------|---------|
| Asymmetric Focal | 1.0 | Multi-label classification |
| CLIP | 0.3 | Image-text contrastive alignment |
| SupCon | 0.3 | Supervised contrastive learning |

### Memory Optimization

| Image Size | batch_size | VRAM Required |
|------------|------------|---------------|
| 224 | 32 | ~8 GB |
| 512 | 16 | ~24 GB |
| 1024 | 4 | ~68 GB |
| 1024 | 8 | ~97 GB (GH200 only) |

---

## GPU Memory Guidelines

### Recommended Settings by GPU

| GPU | VRAM | Max Batch (1024px) | Max Batch (512px) |
|-----|------|-------------------|-------------------|
| RTX 3090 | 24 GB | 1-2 | 8 |
| RTX 4090 | 24 GB | 2 | 8-12 |
| A100 | 40 GB | 2-3 | 12-16 |
| A100 | 80 GB | 4-6 | 24-32 |
| GH200 | 97 GB | 4-8 | 32 |

### Out of Memory Solutions

1. **Reduce batch size**: Most effective
2. **Reduce image size**: `--img-size 512` instead of 1024
3. **Enable gradient checkpointing**: Trades compute for memory
4. **Use mixed precision**: `--amp` (enabled by default)

---

## Production Settings

### Full Training Run (Recommended)

```bash
# Classifier training on GH200 or equivalent
python train_classifier.py \
    --config base \
    --train-dir output/preprocessed/anomalous_train \
    --val-dir output/preprocessed/anomalous_val \
    --chexpert-csv /path/to/mimic-cxr-2.0.0-chexpert.csv.gz \
    --mae-checkpoint output/models/mae_final.pt \
    --epochs 50 \
    --batch-size 16 \
    --img-size 1024 \
    --num-workers 16
```

### Expected Results

With the settings above (50 epochs, full dataset):
- **Macro AUROC**: 0.701
- **Macro AUPRC**: 0.899
- **Training Time**: ~36 hours on GH200
- **Cost**: ~$54 on Lambda Cloud

---

## See Also

- [ARCHITECTURE.md](ARCHITECTURE.md) - Technical architecture details
- [LAMBDA_DEPLOYMENT.md](LAMBDA_DEPLOYMENT.md) - GPU deployment guide
- [DATA_SCHEMA.md](DATA_SCHEMA.md) - Preprocessed data format
- [Main README](../README.md) - Quick start guide
