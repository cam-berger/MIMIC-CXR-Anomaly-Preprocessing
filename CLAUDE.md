# CLAUDE.md - AI Assistant Guide

This document provides guidance for AI assistants working with the MIMIC-CXR Anomaly Detection Pipeline codebase.

## Project Overview

This is a **medical imaging pipeline** for chest X-ray anomaly detection using multimodal deep learning. The pipeline processes MIMIC datasets (images, labs, vitals, reports) and trains models to detect pathologies in chest X-rays.

**Production Results (December 2024)**: Achieved **Macro AUROC 0.701, AUPRC 0.899** on 12 CheXpert pathology classes after 50 epochs of training on Lambda GH200 GPU (~$54 cost). These results predate the September 2026 fixes (text tokenizer mismatch, degenerate cross-attention, unnormalized labs, 1024px center crop, frozen MAE); treat them as a baseline to beat, not as the model's capability. See `docs/NEXT_ITERATION_PLAN.md`.

**Key Goal**: Train multimodal classifiers combining imaging, clinical text, and structured data for robust pathology detection.

## Quick Reference

### Entry Points

| Script | Purpose | Example |
|--------|---------|---------|
| `build_cohort.py` | Build patient cohorts from MIMIC data | `python build_cohort.py` |
| `preprocess.py` | Preprocess cohorts into ML-ready format | `python preprocess.py --workers 8` |
| `train_mae.py` | Train Masked Autoencoder model | `python train_mae.py --config base` |
| `train_classifier.py` | Train multimodal classifier | `python train_classifier.py --config base` |
| `detect_anomalies.py` | Run anomaly detection on new data | `python detect_anomalies.py` |

### Environment Setup

```bash
# Required environment variables (or .env file)
MIMIC_CXR_JPG_PATH=/path/to/mimic-cxr-jpg/2.1.0
MIMIC_IV_PATH=/path/to/mimiciv/3.1
MIMIC_IV_ED_PATH=/path/to/mimic-iv-ed/2.2
CXR_PRO_PATH=/path/to/cxr-pro/1.0.0
OUTPUT_PATH=./output
ANTHROPIC_API_KEY=sk-ant-...  # Optional, for text summarization
```

### Key Commands

```bash
# Install dependencies
pip install -r requirements.txt
python -m spacy download en_core_sci_md

# MAE Pretraining Pipeline (unsupervised on normal X-rays)
python build_cohort.py --normal-only      # Step 1: Build normal cohort
python preprocess.py --workers 8          # Step 2: Preprocess data
python train_mae.py --config base         # Step 3: Train MAE

# Classification Pipeline (supervised on anomalous X-rays)
python build_cohort.py --anomalous-only   # Step 1: Build anomalous cohort
python preprocess.py --leak-free \        # Step 2: Preprocess (leak-free!)
    --enable-summarization --workers 8
python train_classifier.py --config base  # Step 3: Train classifier

# Quick test
python train_mae.py --config debug --epochs 2 --batch-size 2 --skip-anomaly
```

**IMPORTANT**: For classification training, always use `--leak-free` to prevent CheXpert label leakage (labels are extracted from radiology reports).

## Codebase Structure

```
MIMIC-CXR-Anomaly-Preprocessing/
├── src/                          # Main source code
│   ├── config/
│   │   └── settings.py           # Configuration from environment (DataPaths, CohortConfig, etc.)
│   ├── datasets/                 # Data loaders for MIMIC datasets
│   │   ├── mimic_iv.py           # Hospital data (patients, labs, diagnoses)
│   │   ├── mimic_iv_ed.py        # ED data (stays, vitals, triage)
│   │   ├── mimic_cxr.py          # X-ray data (images, CheXpert labels)
│   │   ├── cxr_pro.py            # Radiology report text
│   │   └── linker.py             # Cross-dataset record linking
│   ├── cohort/
│   │   └── builder.py            # Cohort building logic with filters
│   ├── preprocessing/
│   │   ├── images.py             # Image processing -> HDF5
│   │   ├── structured.py         # Labs/vitals -> Parquet
│   │   ├── text.py               # Text processing -> Parquet
│   │   └── pipeline.py           # Pipeline orchestration
│   ├── models/
│   │   ├── mae.py                # Masked Autoencoder implementation
│   │   ├── dataset.py            # PyTorch datasets (MIMICCXRDataset, PreprocessedMAEDataset)
│   │   ├── multimodal.py         # Multimodal classifier (TextEncoder, CrossAttentionFusion)
│   │   ├── classification_dataset.py  # Dataset with CheXpert labels
│   │   ├── losses.py             # Loss functions (CLIP, SupCon, Focal)
│   │   ├── anomaly.py            # Anomaly detection (reconstruction, embedding, ensemble)
│   │   └── config.py             # Training configurations (MAE + classifier presets)
│   ├── training/
│   │   └── loop.py               # Shared training helpers (robust step, LR schedule, checkpoints)
│   └── utils/
│       └── io.py                 # Logging utilities
├── docs/                         # Documentation
│   ├── ARCHITECTURE.md           # Technical architecture deep-dive
│   ├── DATA_SCHEMA.md            # Preprocessed data format specification
│   └── CONFIGURATION_GUIDE.md    # Configuration options and tradeoffs
├── build_cohort.py               # CLI: Build cohorts
├── preprocess.py                 # CLI: Preprocess data
├── train_mae.py                  # CLI: Train MAE model
├── train_classifier.py           # CLI: Train multimodal classifier
├── detect_anomalies.py           # CLI: Run anomaly detection
├── requirements.txt              # Python dependencies
└── .env.example                  # Environment variable template
```

## Key Concepts

### Dataset Linking Keys

Understanding these IDs is critical for working with MIMIC data:

| Key | Description | Scope |
|-----|-------------|-------|
| `subject_id` | Patient identifier | All MIMIC datasets |
| `hadm_id` | Hospital admission ID | MIMIC-IV |
| `stay_id` | ED stay ID | MIMIC-IV-ED |
| `study_id` | Radiology study ID | MIMIC-CXR |
| `dicom_id` | Individual image ID | MIMIC-CXR |

### Pipeline Stages

**Stage 1: Cohort Building** (`build_cohort.py`)
- Filters CXR studies by CheXpert labels
- `--normal-only`: "No Finding" = 1.0 for MAE pretraining (~20k studies after filtering)
- `--anomalous-only`: Any pathology = 1.0 for classification (~32.5k studies after filtering)
- Links to ED visits within 24-hour window
- Outputs: `output/cohorts/{normal,anomalous}_{train,val}.parquet`

**Stage 2: Preprocessing** (`preprocess.py`)
- Processes images -> HDF5 (full resolution, min-max normalized)
- Processes structured data -> Parquet (labs, vitals, demographics)
- Processes text -> Parquet (reports OR clinical context)
- **IMPORTANT**: Use `--leak-free` for classification to prevent CheXpert label leakage
- Outputs: `output/preprocessed/{cohort_name}/images.h5`, `structured.parquet`, `text.parquet`

**Stage 3: MAE Training** (`train_mae.py`)
- Self-supervised pretraining on normal X-rays
- 75% masking ratio (medical imaging optimal)
- ViT encoder with asymmetric decoder
- Outputs: `output/models/mae_final.pt`

**Stage 4: Classification Training** (`train_classifier.py`)
- Supervised training on anomalous X-rays with CheXpert labels
- Multimodal: Image (MAE encoder) + Text (Bio_ClinicalBERT, same model as the preprocessing tokenizer) + Structured
- Loss: Asymmetric Focal + CLIP + Supervised Contrastive
- Outputs: `output/models/classifier_best.pt`, `classifier_final.pt`; checkpoints in `output/checkpoints/`
- Resume: `python train_classifier.py --resume output/checkpoints/classifier_latest.pt` (config and data paths come from the checkpoint)

### Data Flow

```
Raw MIMIC Data → Cohort Building → Preprocessing → Training
     │                 │                │             │
     ├── MIMIC-CXR     ├── Normal       ├── images.h5 ├── MAE Model
     ├── MIMIC-IV      ├── Anomalous    ├── structured│
     ├── MIMIC-IV-ED   ├── Split        ├── text      ├── Classifier
     └── CXR-PRO       └── cohorts/     └── (leak-free)
```

## Development Workflow

### Making Changes

1. **Configuration Changes**: Edit `src/config/settings.py` for pipeline settings or `src/models/config.py` for training configs

2. **Adding New Features**: Follow the existing processor patterns:
   - Image: `src/preprocessing/images.py`
   - Structured: `src/preprocessing/structured.py`
   - Text: `src/preprocessing/text.py`

3. **Dataset Loaders**: Add new data sources in `src/datasets/`

### Testing Changes

```bash
# Quick validation with small sample
python build_cohort.py --normal-only -v
python preprocess.py --workers 4 --cohort output/cohorts/normal_val.parquet

# Debug MAE training
python train_mae.py --config debug --train-dir output/preprocessed/normal_train --epochs 2 --skip-anomaly
```

### Common Patterns

**Loading Configuration**:
```python
from src.config import get_settings
settings = get_settings()
print(settings.paths.cxr_images)  # Path to CXR images
```

**Using Dataset Loaders**:
```python
from src.datasets import MIMICCXRLoader, DatasetLinker
cxr = MIMICCXRLoader()
normal_studies = cxr.get_normal_studies()  # CheXpert "No Finding"
```

**Loading Preprocessed Data**:
```python
from src.preprocessing import PreprocessingPipeline
data = PreprocessingPipeline.load_preprocessed(Path("output/preprocessed/normal_train"))
# data["images"] - HDF5 file handle
# data["structured"] - pandas DataFrame
# data["text"] - pandas DataFrame
```

## Important Design Decisions

### 1. Full Resolution Images
Images are stored at native resolution (~3000x2500 pixels) in HDF5. This preserves fine-grained details needed for anomaly detection. Memory-intensive but critical for medical accuracy.

### 2. Missing Values Are Informative
Missing labs/vitals are not imputed as fake values. Clinical-context text uses a `"NOT_DONE"` token; the classifier receives NaN, and `StructuredEncoder` standardizes observed values (log-transforming heavy-tailed labs, with statistics fitted on the training set and saved in the checkpoint) and appends a missingness indicator per feature. Medical missingness is informative (test not ordered = clinical judgment).

### 3. Full-View Resize (`image_mode="resize"`)
Training and inference resize the whole radiograph to `img_size` (`build_image_transform` in `src/models/dataset.py`, shared by all datasets and `detect_anomalies.py`). The earlier `center_crop` mode, which crops `img_size` pixels from the native ~3056x2544 image, kept ~13% of the image at 1024px (0.65% at 224px) and cut off costophrenic angles, apices and lateral lung fields. It remains available via `--image-mode center_crop` only to reproduce old models; checkpoints saved before `image_mode` existed load as `center_crop`.

### 4. Claude Summarization (Optional)
Text summarization uses Claude API when enabled. Includes clinical context (demographics, vitals, labs) for richer summaries. Can be disabled to reduce costs.

### 5. Leak-Free Mode for Classification
CheXpert labels are NLP-extracted from radiology reports. To prevent label leakage when training classifiers:
- Use `--leak-free` flag during preprocessing
- Text features use only information available when the X-ray is taken: demographics, chief complaint, triage vitals/acuity, labs
- Radiology report text is excluded, and so are ED/hospital discharge diagnoses, procedures and disposition (`format_clinical_context(include_outcomes=False)`): ICD codes such as J90 (pleural effusion) encode the labels
- Labs are aggregated up to the study time only (`labs_time_window_after_hours = 0`); later labs can be ordered because of the X-ray's findings
- Any new feature must pass the same test: would it be known at image acquisition, independently of the image's findings?
- See `docs/ARCHITECTURE.md` section "CheXpert Label Leakage Prevention"

### 6. Training Stability (root cause fixed September 2026)
**Problem**: Training hit cascade failures: after one non-finite gradient, every later step corrupted the weights until the circuit breaker ended the epoch, and the same thing happened every following epoch.

**Actual root cause**: `text_clip_proj` (and, after unfreezing, the MAE `cls_token`/`pos_embed`) were trainable but not in the optimizer. `optimizer.zero_grad()` never cleared their gradients and `scaler.unscale_()` never checked them, so `clip_grad_norm_(model.parameters())` spread a single stale Inf/NaN from them into every later update. The December 2024 GradScaler reset, weight-revert fuse, Adam-state wipe and CrossAttention NaN masking treated symptoms; they have been removed.

**Current mechanism** (`src/training/loop.py`, used by both training scripts):
- `backward_and_step`: clips over exactly the optimizer's parameters and never applies a non-finite update (GradScaler skips it under AMP; skipped explicitly in FP32). Gradients are cleared every step.
- `check_trainable_params_in_optimizer`: fails fast if any trainable parameter is missing from the optimizer (checked every epoch, after freezing/unfreezing).
- Batches with a non-finite loss are skipped before backward. NaNs are not masked inside the model, so they reach this check.
- `ConsecutiveSkipGuard`: raises after `max_consecutive_skips` skipped batches in a row, instead of silently continuing.
- The MAE encoder unfreezes with its own LR warmup (`unfreeze_warmup_epochs`) and per-block layer-wise LR decay.
- Structured features are normalized (raw NT-proBNP up to 70,000 overflowed fp16).
- Kept: `safe_normalize()` for zero-norm vectors (Fix #3) and the MAE `1e-5` epsilon (Fix #4).

**Tests**: `tests/test_training_fixes.py` (regression tests for each fix, including end-to-end train/resume/inference runs), `tests/test_multimodal_stability.py`, `tests/test_nan_handling.py`.

**Baseline Metrics**: `tests/baseline_metrics.md` documents the original cascade (its root-cause section predates the diagnosis above).

## Key Files to Know

| File | Purpose | When to Edit |
|------|---------|--------------|
| `src/config/settings.py` | All configuration dataclasses | Adding config options |
| `src/cohort/builder.py` | Cohort filtering logic | Changing filter criteria |
| `src/preprocessing/pipeline.py` | Pipeline orchestration | Adding processing steps |
| `src/preprocessing/text.py` | Text processing, leak-free mode | Text feature changes |
| `src/models/mae.py` | MAE architecture | Model architecture changes |
| `src/models/multimodal.py` | Classifier architecture | Classification model changes |
| `src/models/losses.py` | Loss functions (CLIP, SupCon, Focal) | Loss modifications |
| `src/models/dataset.py` | PyTorch datasets | Data loading changes |
| `src/models/config.py` | Training presets (debug/fast/base) for MAE and classifier | Training hyperparameters |
| `src/training/loop.py` | Optimizer step, LR schedule, checkpoint I/O | Training-loop behavior |

## Output Data Schema

### images.h5 (HDF5)
```
/images/{idx}     - Image tensor [1, H, W], float32, [0,1] normalized
/metadata/{idx}   - JSON: {study_id, subject_id, shape, image_path}
/index            - Parquet: study_id -> idx mapping
```

### structured.parquet
Key columns: `subject_id`, `study_id`, `age`, `gender`, `triage_*`, `*_mean/min/max/std`, `lab_*_mean/min/max/count`, `has_*` flags

### text.parquet
Key columns: `subject_id`, `study_id`, `report`, `clinical_context`, `summary`, `tokens`, `token_count`, `has_report`

**Note**: In `--leak-free` mode, `report` and `report_clean` are empty; `summary` contains clinical context summary only.

## Performance Considerations

### Memory
- Full-resolution images: ~29 MB each
- Training batch size limited to 1-4 for high-res
- Labs loaded in chunks to avoid OOM

### Speed Bottlenecks
1. **Claude API**: Rate-limited, ~1-5s per call
2. **Lab Events**: Large file (~10 GB), requires chunked loading
3. **Image I/O**: Use SSD storage for 2-3x speedup

### GPU Requirements (1024x1024 training)
| Model | batch_size=2 | batch_size=4 |
|-------|--------------|--------------|
| ViT-Small | ~8 GB | ~15 GB |
| ViT-Base | ~35 GB | ~68 GB |

## Common Issues

### "Missing required data paths"
Set environment variables or create `.env` file with MIMIC dataset paths.

### Out of Memory
- Reduce `--batch-size` or `--workers`
- Use `--img-size 512` for smaller images
- Labs are chunked automatically

### Slow Processing
- Increase `--workers` (up to CPU cores)
- Use SSD storage
- Disable Claude summarization if not needed

### Claude API Errors
- Set `ANTHROPIC_API_KEY` environment variable
- Use `--text-only` flag to skip other modalities
- Disable with `--no-summarization` to skip entirely

### Classification Training Issues
- **Data Leakage**: If classifier achieves suspiciously high accuracy, check if `--leak-free` was used during preprocessing. CheXpert labels are extracted from radiology reports - feeding report text leaks labels.
- **Missing Labels**: Some studies have uncertain (-1.0) or missing (NaN) labels. These are masked out during training automatically.
- **NaN Loss / Exploding Gradients**: see section 6. Skipped batches are logged per epoch (`skipped_loss`, `skipped_grad`). A few skipped steps early in AMP training are the GradScaler calibrating its loss scale. A long run of skips raises an error by design; investigate the data or the learning rate rather than raising the limit.
- **"token ids do not match the text encoder"**: `text.parquet` was tokenized with a different tokenizer than `text_model_name`. Re-run text preprocessing or pass `--text-model` with the tokenizer used in preprocessing.
- **"trainable parameter(s) are not in the optimizer"**: a new module was added outside `MultimodalClassifier.get_layer_groups()`; add it there.

## Git Conventions

- Feature branches from main
- Descriptive commit messages
- Don't commit large data files (`.h5`, `.parquet`, `.csv`)
- Keep `.env` out of version control

## Documentation References

- `docs/ARCHITECTURE.md` - Technical architecture and production training results
- `docs/DATA_SCHEMA.md` - Complete output schema specification
- `docs/CONFIGURATION_GUIDE.md` - All configuration options and tradeoffs
- `docs/LAMBDA_DEPLOYMENT.md` - GPU deployment guide with cost breakdown
- `docs/NEXT_ITERATION_PLAN.md` - What the September 2026 review means for the results, and the plan for the next iteration
- `docs/CHANGELOG.md` - Change history
- `README.md` - User-facing documentation, tutorials, and future improvements
- `tests/baseline_metrics.md` - Cascade failure analysis and fix validation
