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

Classifier presets live in `src/models/config.py` (`get_classifier_config`); unset fields use `ClassifierConfig` defaults:

| Config | Purpose | Epochs | Batch | LR | Warmup | MAE frozen for | Image Size |
|--------|---------|--------|-------|-----|--------|----------------|------------|
| `debug` | Quick testing | 2 | 4 | 5e-5 | 1 | 100 epochs (always) | 224 |
| `fast` | Development | 10 | 16 | 5e-5 | 2 | 2 epochs | 224 |
| `base` | Production | 50 | 16 | 5e-5 | 5 | 5 epochs | 224 |

The image size defaults to 224 in every preset; pass `--img-size` (e.g. 1024, as in the December 2024 production run) explicitly.

### Using Configurations

```bash
# Debug: Quick validation (2 epochs)
python train_classifier.py --config debug --train-dir ... --chexpert-csv ...

# Base: Production training at 1024px with a pretrained MAE
python train_classifier.py --config base --img-size 1024 \
    --train-dir output/preprocessed/anomalous_train \
    --val-dir output/preprocessed/anomalous_val \
    --chexpert-csv /path/to/mimic-cxr-2.0.0-chexpert.csv.gz \
    --mae-checkpoint output/models/mae_final.pt

# Resume: configuration and data paths come from the checkpoint
python train_classifier.py --resume output/checkpoints/classifier_latest.pt
```

On resume, flags that would change the run (epochs, batch size, LR, image
size/mode, loss weights, freeze schedule, text model) are rejected; hardware
flags (`--device`, `--num-workers`) can change.

### Key ClassifierConfig Fields

| Field | Default | Description |
|-------|---------|-------------|
| `img_size` | 224 | Input resolution |
| `image_mode` | `resize` | `resize` (whole radiograph) or `center_crop` (legacy crop from native resolution; ~13% of the image at 1024) |
| `text_model_name` | `emilyalsentzer/Bio_ClinicalBERT` | Must match the preprocessing tokenizer (`PreprocessingConfig.tokenizer_model`); checked at startup |
| `freeze_mae_epochs` | 5 | Epochs before the MAE encoder unfreezes |
| `unfreeze_warmup_epochs` | 1 | LR warmup for the encoder after it unfreezes |
| `lr_decay` | 0.9 | Per-block layer-wise LR decay (block *i* of 12 gets `0.9^(12-i)`) |
| `grad_clip` | 1.0 | Gradient clipping (over the optimizer's parameters) |
| `max_consecutive_skips` | 50 | Fail after this many skipped (non-finite) batches in a row |
| `cls_weight` / `clip_weight` / `supcon_weight` | 1.0 / 0.3 / 0.3 | Loss weights (0 disables a term) |

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

| Flag | Default | Description |
|------|---------|-------------|
| `--config` | `base` | Preset (debug: ViT-S, bs 8; fast: ViT-S, bs 32; base: ViT-B, bs 64, 800 epochs) |
| `--img-size` | 224 | Input resolution (production used 1024) |
| `--image-mode` | `resize` | `resize` or `center_crop` (legacy) |
| `--patch-size` | 16 | ViT patch size |
| `--mask-ratio` | 0.75 | Fraction of patches masked |
| `--epochs` | preset | Total training epochs |
| `--batch-size` | preset | Samples per batch |
| `--lr` | 1.5e-4 | AdamW learning rate (not scaled by batch size) |
| `--num-workers` | 8 | Data loader workers (0 = main process) |
| `--resume` | - | Checkpoint to resume (config and paths come from it) |

Weight decay (0.05), warmup (40 epochs), gradient clipping (`grad_clip=1.0`) and the augmentations (`crop_scale`, `horizontal_flip`, `rotation_degrees`, `gaussian_blur`) are `MAEConfig` fields.

### Classifier Training (`train_classifier.py`)

| Flag | Default | Description |
|------|---------|-------------|
| `--config` | `base` | Preset (debug/fast/base) |
| `--epochs` | preset | Total training epochs |
| `--batch-size` | preset | Samples per batch |
| `--lr` | 5e-5 | Base learning rate (head; encoder blocks get layer-wise decay) |
| `--img-size` | 224 | Input resolution |
| `--image-mode` | `resize` | `resize` or `center_crop` (legacy) |
| `--text-model` | Bio_ClinicalBERT | Text encoder; must match the preprocessing tokenizer |
| `--freeze-mae-epochs` | preset | Epochs before the MAE encoder unfreezes (100 = always frozen) |
| `--cls-weight` / `--clip-weight` / `--supcon-weight` | 1.0 / 0.3 / 0.3 | Loss weights |
| `--mae-checkpoint` | - | Pretrained MAE weights (missing encoder weights raise an error) |
| `--resume` | - | Checkpoint to resume (config and paths come from it) |

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
