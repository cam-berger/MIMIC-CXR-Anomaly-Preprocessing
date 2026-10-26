# CLAUDE.md - AI Assistant Guide

This document provides guidance for AI assistants working with the MIMIC-CXR Anomaly Detection Pipeline codebase.

## v2 (branch `v2`; v2.0 done, v2.1 next)

v2 tests modern pretrained chest X-ray models (RAD-DINO, MedSigLIP, DINOv2) on the same 12-class task: zero-shot, frozen embeddings with linear probes, then one GPU fine-tune. Image-only, with a fixed evaluation protocol. v2.0 is done: the fine-tuned RAD-DINO at 518 px is the baseline (0.787 [0.769, 0.803] on `test_clean`); the results log is §9 of the plan. M4 (`v2/context.py`, `v2/fuse.py`) adds text written before the X-ray (indication, ED chief complaint, prior report) under the rules pre-registered in §6 of the plan; the current report's findings and impression are never an input. M4b (`v2/attend.py`, `v2/stack.py`, prior-study labels in `v2/fuse.py`) tests text-steered attention over cached patch tokens (gate G4b), the prior report as its CheXpert labels, and a text combiner fit on val; judge text gains under the v2 label policy (§6 M4b amendment). M5 (`v2/lineup.py`, `prepare_data --population all --split-scheme official`) lines our models up with published MIMIC-CXR results on the official split, in its own output directory (`BENCH_OUT`); M5b adds `finetune --head attention` and the val combiner there (`run.sh bench_ft` on a GPU, `bench_stack` on the Mac). M6 (`v2/textenc.py`, `v2/textft.py`, `fuse --text-encoder`) tries BERT-family encoders of that text on the benchmark, general (`bert`) against medical (`biomedbert`, `clinicalbert`), frozen and fine-tuned, with gate G6 (`run.sh bench_text` on the Mac). M7 (`v2/external.py`) is external validation: the benchmark's runs against a radiologist's labels on the official test split (`run.sh m7a`), and the CheXpert test set in its own output directory, `EXT_OUT` (`run.sh m7b`, gate G7). CheXpert falls under the same data-handling rules as MIMIC. M8 (`v2/replicate.py`) repeats the benchmark fine-tunes with a second seed and the attention head for 6 epochs (`run.sh m8_gpu` on a GPU, `m8` on the Mac); each headline verdict must hold for both seeds. **Read `docs/V2_PLAN.md` first**: it defines the task, splits, metrics, milestones (M0-M8), decision gates and the results log. **`v2/README.md`** explains how to run it (setup, `v2/scripts/run.sh` stages, the GPU session). v2 code lives in the standalone `v2/` package with tests in `tests/v2/`; it does not modify the v1 pipeline described below. The per-class names in `docs/RESULTS.md` are shifted for the last four classes (see `docs/V2_PLAN.md` §2).

v2 rules: metrics are computed only in `v2/evaluate.py`, from the one prediction format written by `v2.common.write_predictions`; label policy and splits live in `v2/common.py` and `v2/prepare_data.py`; model choices use `val`, never a test set. Before changing anything, run `pytest tests/v2 -q` and `python -m v2.smoke` (synthetic end-to-end check with a planted signal, ~2 min).

**MIMIC data handling (PhysioNet data use agreement and its Sept 2025 LLM guidance):** never read, print or list raw MIMIC content into the conversation. That covers CSV or parquet rows, images, report text, the v2 manifest, `context.parquet`, embeddings (including the `*_tokens*.npy` caches and their `.json` sidecars, and M6's `text_*.npz`) and predictions. Run the v2 commands and relay only their aggregate output (`audit.md`, `reports/*.md`, `summary.md`). Keep record-level diagnostics in local files. See `v2/README.md` §9. transformers is pinned in `v2/requirements.txt` because DINOv2 embeddings at non-native sizes differ between versions.

## v3 (branch `v3`, from `v2`)

v3 tests TypeSafe's Jev, a hosted classification model, as the combiner of v2's image estimates (`rad-dino_518_ft_attn_s0`) and the pre-imaging text, on M5's benchmark. It runs a context ladder of arms (`jev_est` → `jev_est_indication` → `jev_est_text`, plus `jev_text` as a control) and gate J1 (`jev_est_text` against v2's val combiner of the same inputs). The image itself comes in v3.1. v3.0 is done: val chose the yes/no question format, and **J1 FAILS** (Jev 0.803 against the stack's 0.827 on the official test, -0.024 [-0.033, -0.016]); adding text to Jev adds nothing detectable. v3.1 is done: showing Jev the X-ray adds nothing detectable either (J2 -0.001 [-0.005, 0.005]), and Jev's own reading of the X-ray scores 0.716 against RAD-DINO's 0.814. **Read `docs/V3_PLAN.md` first** (pre-registered 2026-10-26); `v3/README.md` explains how to run it. The code lives in `v3/` and reuses v2 unchanged, with tests in `tests/v3/`. v3 works in its own output dir `V3_OUT`, which links into `BENCH_OUT` and never writes there. Metrics come only from `v2/evaluate.py`. Before changing anything, run `pytest tests/v3 -q` and `python -m v3.smoke`. Jev requests carry MIMIC text: the user verified TypeSafe's zero data retention. The v2 rules on record-level data apply to `V3_OUT/jev_cache/`, `predictions/` and `logs/`.

## Project Overview

This is a **medical imaging pipeline** for chest X-ray anomaly detection using multimodal deep learning. The pipeline processes MIMIC datasets (images, labs, vitals, reports) and trains models to detect pathologies in chest X-rays.

**Production Results (December 2025)**: Achieved **Macro AUROC 0.701, AUPRC 0.899** on 12 CheXpert pathology classes after 50 epochs of training on Lambda GH200 GPU (~$54 cost).

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

### 6. NaN/Inf Stability Fixes (December 2025)
**Problem**: Training experienced catastrophic cascade failures starting at Epoch 2, where GradScaler corruption caused 100% batch failure rate for remaining 27 epochs.

**Root Cause**: Creating new `GradScaler()` object lost optimizer state synchronization, causing permanent corruption.

**Fixes Implemented** (all validated with comprehensive test suite):

1. **Fix #1 - GradScaler Reset** (CRITICAL - `train_classifier.py:430-437`)
   - Changed from `scaler = GradScaler()` (creates new object)
   - To `scaler.load_state_dict(initial_state)` (resets existing object)
   - **Impact**: Eliminates cascade failures, enables full 30-epoch training

2. **Fix #2 - CrossAttention NaN Guards** (`src/models/multimodal.py:257-302`)
   - Added input sanitization: `torch.nan_to_num()` + `clamp(-10, 10)`
   - Added output sanitization after attention operations
   - **Impact**: Prevents NaN propagation through attention layers

3. **Fix #3 - Safe Normalization** (`src/models/multimodal.py:29-69`, `losses.py`)
   - Created `safe_normalize()` utility to handle zero-norm vectors
   - Replaced 6 instances of `F.normalize()` which returned NaN for zero vectors
   - **Impact**: Stabilizes CLIP and SupCon contrastive losses

4. **Fix #4 - MAE Epsilon** (`src/models/mae.py:525`)
   - Increased epsilon from `1e-6` to `1e-5` for FP16 safety
   - **Impact**: Handles low-variance patches without numerical issues

**Test Coverage**: 21 unit tests + 8 integration tests (see `tests/` directory)

**Production Results (December 2025)**:
- **Macro AUROC: 0.701** (50 epochs, full dataset)
- **Macro AUPRC: 0.899**
- Training completion: 50/50 epochs (zero cascade failures)
- NaN batch rate: <0.5% (properly handled by skip logic)
- Weight corruptions: 0 (GradScaler reset fix validated)
- Circuit breaker triggers: 0

**Baseline Metrics**: See `tests/baseline_metrics.md` for detailed cascade failure analysis.

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
- **NaN Loss During Training** ✅ **FIXED (December 2025)**:
  - **Previous Issue**: Catastrophic cascade failures caused 100% batch corruption after Epoch 2 due to GradScaler reset bug
  - **Current Status**: All 4 NaN/Inf fixes implemented and validated (see section 6 above)
  - Batches with NaN/Inf loss are automatically skipped (no backward pass)
  - Weight integrity fuse: reverts to last-good checkpoint if parameters become corrupted
  - GradScaler properly resets state (preserves optimizer synchronization)
  - CrossAttention has NaN guards to prevent propagation
  - Safe normalization prevents division-by-zero in contrastive losses
  - ~1-2% of batches may produce NaN due to edge-case data; this is normal and handled safely
  - Expected behavior: Zero cascade failures, <0.5% NaN rate, 30/30 epochs complete
  - See `tests/baseline_metrics.md` for before/after analysis

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
- `docs/NAN_STABILITY_FIXES.md` - NaN/Inf stability fixes documentation
- `README.md` - User-facing documentation, tutorials, and future improvements
- `tests/baseline_metrics.md` - Cascade failure analysis and fix validation
