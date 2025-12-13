# CLAUDE.md - AI Assistant Guide

This document provides guidance for AI assistants working with the MIMIC-CXR Anomaly Detection Pipeline codebase.

## Project Overview

This is a **medical data preprocessing pipeline** that prepares chest X-ray images and clinical data from MIMIC datasets for training anomaly detection AI models. The pipeline identifies "normal" chest X-rays from healthy patients and preprocesses multimodal data (images, labs, vitals, reports) into ML-ready formats.

**Key Goal**: Train models to detect abnormalities in chest X-rays using unsupervised learning on normal samples.

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
│   │   └── config.py             # Training configurations (debug, fast, base)
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
- `--normal-only`: "No Finding" = 1.0 for MAE pretraining (~33k studies)
- `--anomalous-only`: Any pathology = 1.0 for classification (~32k studies)
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
- Multimodal: Image (MAE encoder) + Text (ClinicalBERT) + Structured
- Loss: Asymmetric Focal + CLIP + Supervised Contrastive
- Outputs: `output/models/classifier.pt`

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

### 2. NOT_DONE Token for Missing Values
Missing labs/vitals use `"NOT_DONE"` token instead of imputation. Medical missingness is informative (test not ordered = clinical judgment). Models should learn this.

### 3. Center Crop for MAE Training
Training uses center crop from full-resolution images. Chest X-rays are radiologist-centered, so center crops consistently capture lung fields.

### 4. Claude Summarization (Optional)
Text summarization uses Claude API when enabled. Includes clinical context (demographics, vitals, labs) for richer summaries. Can be disabled to reduce costs.

### 5. Leak-Free Mode for Classification
CheXpert labels are NLP-extracted from radiology reports. To prevent label leakage when training classifiers:
- Use `--leak-free` flag during preprocessing
- Text features use only clinical context (vitals, labs, chief complaint)
- Radiology report text is excluded
- See `docs/ARCHITECTURE.md` section "CheXpert Label Leakage Prevention"

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
| `src/models/config.py` | Training presets (debug/fast/base) | Training hyperparameters |

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
- **NaN Loss During Training**: The training loop has bulletproof NaN handling:
  - Batches with NaN/Inf loss are automatically skipped (no backward pass)
  - Weight integrity fuse: reverts to last-good checkpoint if parameters become corrupted
  - ~1-2% of batches may produce NaN due to edge-case data; this is normal and handled safely
  - Known problematic study_ids are blacklisted in `src/models/classification_dataset.py`
  - All loss functions compute in FP32 for numerical stability (even with mixed precision training)

## Git Conventions

- Feature branches from main
- Descriptive commit messages
- Don't commit large data files (`.h5`, `.parquet`, `.csv`)
- Keep `.env` out of version control

## Documentation References

- `docs/ARCHITECTURE.md` - Deep technical architecture details
- `docs/DATA_SCHEMA.md` - Complete output schema specification
- `docs/CONFIGURATION_GUIDE.md` - All configuration options and tradeoffs
- `README.md` - User-facing documentation and tutorials
