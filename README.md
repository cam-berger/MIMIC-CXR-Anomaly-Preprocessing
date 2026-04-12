# MIMIC-CXR Anomaly Detection Pipeline

A data preprocessing pipeline that prepares chest X-ray images and clinical data for training AI models to detect medical abnormalities.

## Table of Contents
- [What This Project Does](#what-this-project-does)
- [Results](#results)
- [Background: The Problem We're Solving](#background-the-problem-were-solving)
- [The Data: MIMIC Datasets](#the-data-mimic-datasets)
- [How It Works](#how-it-works)
- [Getting Started](#getting-started)
- [Project Structure](#project-structure)
- [Detailed Walkthrough](#detailed-walkthrough)
- [CLI Reference](#cli-reference)
- [Future Improvements](#future-improvements)
- [MAE Reconstruction Analysis](#mae-reconstruction-analysis)
- [References](#references)

---

## What This Project Does

This pipeline takes raw medical data from multiple sources and transforms it into a clean, organized format ready for machine learning. Specifically, it:

1. **Identifies "normal" chest X-rays** (~20,000 studies) - X-rays from patients who were healthy and sent home
2. **Identifies "abnormal" chest X-rays** (~32,500 studies) - X-rays showing various pathologies with CheXpert labels
3. **Combines multiple data types** for each X-ray:
   - The image itself
   - Lab results (blood tests, etc.)
   - Vital signs (heart rate, blood pressure, etc.)
   - The radiologist's written report (excluded in leak-free mode)

The output is ready-to-use datasets for training anomaly detection models.

### Dataset Summary

| Cohort | Purpose | Final Size | Train/Val Split |
|--------|---------|------------|-----------------|
| **Normal** | MAE pretraining (unsupervised) | ~20,000 | 17,000 / 3,000 |
| **Anomalous** | Classification (supervised) | ~32,500 | 27,576 / 4,922 |

---

## Results

Production model trained on Lambda Cloud GH200 GPU (December 2025):

| Metric | Value |
|--------|-------|
| **Macro AUROC** | **0.701** |
| **Macro AUPRC** | **0.899** |
| Training Time | ~36 hours |
| Training Cost | ~$54 |

### Top Performing Classes

| Class | AUROC | AUPRC |
|-------|-------|-------|
| Edema | 0.878 | 0.934 |
| Consolidation | 0.840 | 0.825 |
| No_Finding | 0.821 | 0.928 |
| Pneumonia | 0.812 | 0.672 |
| Cardiomegaly | 0.808 | 0.965 |

See [docs/RESULTS.md](docs/RESULTS.md) for complete per-class metrics, confusion matrices, and threshold analysis.

---

## Background: The Problem We're Solving

### Why Anomaly Detection?

Traditional medical AI typically trains on labeled examples: "This X-ray shows pneumonia, this one shows a fracture." But labeling medical images is expensive and time-consuming—it requires expert radiologists.

**Anomaly detection** takes a different approach:
1. Train on thousands of "normal" images (unsupervised learning)
2. The model learns what "healthy" looks like
3. At test time, anything that deviates significantly from "normal" is flagged as potentially abnormal

This is valuable because:
- We have many more unlabeled images than labeled ones
- The model can potentially catch abnormalities it was never explicitly trained on
- It's similar to how radiologists actually work—they learn normal anatomy first

### Why Multiple Data Types (Multimodal)?

A chest X-ray alone doesn't tell the full story. Consider:
- A 25-year-old athlete with chest pain → probably not a heart attack
- A 75-year-old diabetic with the same X-ray → much more concerning

By combining:
- **Images** (what the radiologist sees)
- **Clinical data** (age, vital signs, lab results)
- **Text** (the radiologist's interpretation)

...we can build more accurate, context-aware models.

---

## The Data: MIMIC Datasets

This project uses four datasets from [PhysioNet](https://physionet.org/), all part of the MIMIC (Medical Information Mart for Intensive Care) family:

### MIMIC-CXR-JPG (Chest X-ray Images)
- **What**: 377,000+ chest X-ray images in JPEG format
- **Key data**: Images, CheXpert labels (automated disease detection), study metadata
- **Why we need it**: The actual images we're training on

### MIMIC-IV (Hospital Records)
- **What**: Comprehensive hospital data for 300,000+ patients
- **Key data**: Demographics, lab results, diagnoses (ICD codes), procedures
- **Why we need it**: Clinical context—what was happening with the patient?

### MIMIC-IV-ED (Emergency Department)
- **What**: Emergency department visits for 450,000+ encounters
- **Key data**: Triage vitals, ED diagnoses, disposition (sent home vs admitted)
- **Why we need it**: Many X-rays happen in the ED; this tells us the clinical context

### CXR-PRO (Radiology Reports)
- **What**: Cleaned radiology report text for MIMIC-CXR images
- **Key data**: The "Impression" section of radiology reports (with prior references removed)
- **Why we need it**: Expert interpretation in natural language

### Data Provenance

| Dataset | Version | PhysioNet ID | Citation |
|---------|---------|--------------|----------|
| MIMIC-CXR-JPG | 2.1.0 | [mimic-cxr-jpg](https://physionet.org/content/mimic-cxr-jpg/2.1.0/) | Johnson et al., 2019 |
| MIMIC-IV | 3.1 | [mimiciv](https://physionet.org/content/mimiciv/3.1/) | Johnson et al., 2023 |
| MIMIC-IV-ED | 2.2 | [mimic-iv-ed](https://physionet.org/content/mimic-iv-ed/2.2/) | Johnson et al., 2023 |
| CXR-PRO | 1.0.0 | [cxr-pro](https://physionet.org/content/cxr-pro/1.0.0/) | Ramesh et al., 2022 |

**Access Requirements**: All datasets require PhysioNet credentialed access and completion of the CITI training course.

### How They Link Together

All datasets share a **`subject_id`** (patient identifier). Here's how records connect:

```
Patient (subject_id: 12345678)
    │
    ├── ED Visit (stay_id: 11111)
    │   ├── Triage: HR 88, BP 120/80, SpO2 98%
    │   ├── ED Diagnosis: Chest pain (R07.9)
    │   └── Disposition: Discharged home
    │
    ├── Hospital Admission (hadm_id: 22222) [if admitted]
    │   ├── Labs: WBC 8.2, Hemoglobin 14.1
    │   ├── Diagnoses: ...
    │   └── Procedures: ...
    │
    └── Chest X-ray Study (study_id: 33333)
        ├── Image: /files/p12/p12345678/s33333/abc123.jpg
        ├── CheXpert Labels: No Finding = 1.0
        └── Report: "No acute cardiopulmonary abnormality..."
```

The tricky part is **temporal matching**—connecting an X-ray to the right ED visit. We match based on timestamps: the X-ray must have been taken within ±24 hours of the ED visit.

---

## How It Works

The pipeline has two main steps:

### Step 1: Cohort Building (`build_cohort.py`)

A "cohort" is a defined group of patients/studies for analysis. We build two:

**Normal Cohort** (for MAE pretraining)
```
Start: 377,110 X-ray studies
  │
  ├─ Filter: CheXpert "No Finding" = 1.0 → ~95,000
  │    (Automated label says no disease detected)
  │
  ├─ Filter: Match to ED visit within ±24 hours → ~52,000
  │    (Need clinical context from ED visit)
  │
  ├─ Filter: Patient was sent home → ~22,000
  │    (Admitted patients might have been sick)
  │
  ├─ Filter: No critical diagnoses (sepsis, MI, etc.) → ~20,000
  │    (Extra safety check)
  │
  └─ Filter: Age ≥ 18 → ~20,000 final
      └─ Split: 17,000 train / 3,000 val
```

**Anomalous Cohort** (for classification)
```
Start: 377,110 X-ray studies
  │
  ├─ Filter: Any CheXpert pathology = 1.0 → ~200,000
  │    (At least one finding detected)
  │
  ├─ Filter: Match to ED visit within ±24 hours → ~55,000
  │    (Need clinical context from ED visit)
  │
  ├─ Filter: Age ≥ 18 → ~52,000
  │
  └─ Filter: Valid for all modalities → ~32,500 final
      └─ Split: 27,576 train / 4,922 val
```

Each cohort includes all linked data: demographics, vitals, labs, diagnoses, reports.

### Step 2: Preprocessing (`preprocess.py`)

Raw data → ML-ready format:

**Images → HDF5**
```python
# Raw: JPEG files scattered across directories
/files/p10/p10000032/s50414267/02aa804e-bde0afdd-112c0b34-7bc16630-4e384014.jpg

# Processed: Single HDF5 file with normalized tensors
images.h5
  └── images/0 → numpy array [1, 2544, 3056], dtype=float32, range [0,1]
  └── images/1 → ...
  └── index → DataFrame mapping study_id to array index
```

**Structured Data → Parquet**
```python
# Raw: Separate CSV files with millions of rows
labevents.csv (40 million rows)
vitalsign.csv (millions of rows)

# Processed: One row per study with aggregated features
structured.parquet
  study_id | age | gender | triage_hr | triage_bp | lab_wbc_mean | lab_glucose_mean | ...
  33333    | 65  | M      | 88        | 120       | 8.2          | 112.0            | ...
```

**Text → Parquet**
```python
# Raw: Report text
"No acute cardiopulmonary abnormality. The lungs are clear..."

# Processed: Cleaned text + optional AI summary + tokens
text.parquet
  study_id | report | clinical_context | summary | tokens | token_count
  33333    | "No.." | "65 yo male..."  | "Normal"| [101,..| 45
```

The text processing can optionally use Claude AI to generate summaries that incorporate clinical context (age, vitals, labs, diagnoses).

### Data Leakage Policy

**Critical for research validity**: CheXpert labels are NLP-extracted from radiology reports. Using report text to predict these labels is trivial (the model just "reads" the diagnosis). We enforce strict temporal boundaries:

**Anchor Time** = CXR acquisition timestamp

| Data Source | Included | Condition |
|-------------|----------|-----------|
| Demographics (age, gender) | ✅ | Always available |
| ED triage vitals | ✅ | Recorded at ED arrival (before imaging) |
| Labs | ✅ | Only if `charttime ≤ anchor_time` |
| Chief complaint | ✅ | Recorded at triage (before imaging) |
| ED diagnoses (ICD codes) | ⚠️ | Use with caution—may reflect post-imaging workup |
| Radiology report text | ❌ | **Excluded in leak-free mode** (post-imaging) |
| Hospital procedures | ❌ | Occur after ED evaluation |

**Usage:**
```bash
# For classification training: ALWAYS use --leak-free
python preprocess.py --leak-free --cohort anomalous_train.parquet

# For MAE pretraining: leak-free not required (no labels predicted)
python preprocess.py --cohort normal_train.parquet
```

In `--leak-free` mode, text features contain only clinical context (demographics, vitals, labs, chief complaint)—no radiology findings.

---

## Getting Started

### Prerequisites

1. **Python 3.10+**
2. **PhysioNet Account** with credentialed access to MIMIC datasets
   - Complete the required training at https://physionet.org/
   - Sign data use agreements for each dataset

### Installation

```bash
# 1. Clone repository
git clone <repo-url>
cd MIMIC-CXR-Anomaly-Preprocessing

# 2. Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Download NLP model
python -m spacy download en_core_sci_md
```

### Configuration

Create a `.env` file (or set environment variables):

```bash
# Required: Paths to downloaded MIMIC datasets
MIMIC_CXR_JPG_PATH=/path/to/mimic-cxr-jpg/2.1.0
MIMIC_IV_PATH=/path/to/mimiciv/3.1
MIMIC_IV_ED_PATH=/path/to/mimic-iv-ed/2.2
CXR_PRO_PATH=/path/to/cxr-pro/1.0.0

# Output directory
OUTPUT_PATH=./output

# Optional: For AI-powered text summarization
ANTHROPIC_API_KEY=sk-ant-...
```

### Running the Pipeline

```bash
# Step 1: Build cohorts (5-15 minutes)
python build_cohort.py

# Step 2: Preprocess all cohorts (hours, depending on size)
python preprocess.py --workers 8

# Or preprocess with AI summarization
python preprocess.py --enable-summarization --workers 8
```

---

## Project Structure

```
MIMIC-CXR-Anomaly-Preprocessing/
│
├── src/                          # Main source code
│   │
│   ├── config/
│   │   └── settings.py           # Configuration from environment variables
│   │
│   ├── datasets/                 # Data loaders for each MIMIC dataset
│   │   ├── __init__.py           # Documents how datasets link together
│   │   ├── mimic_iv.py           # Hospital data: patients, labs, diagnoses
│   │   ├── mimic_iv_ed.py        # ED data: stays, vitals, triage
│   │   ├── mimic_cxr.py          # X-ray data: images, CheXpert labels
│   │   ├── cxr_pro.py            # Report text
│   │   └── linker.py             # Links records across datasets
│   │
│   ├── cohort/
│   │   └── builder.py            # Builds normal and anomalous cohorts
│   │
│   ├── preprocessing/
│   │   ├── images.py             # Image processing → HDF5
│   │   ├── structured.py         # Labs/vitals processing → Parquet
│   │   ├── text.py               # Text processing → Parquet
│   │   └── pipeline.py           # Coordinates all preprocessing
│   │
│   └── utils/
│       └── io.py                 # Logging setup
│
├── build_cohort.py               # CLI entry point for cohort building
├── preprocess.py                 # CLI entry point for preprocessing
├── requirements.txt              # Python dependencies
├── .env.example                  # Example configuration file
└── README.md                     # This file
```

---

## Detailed Walkthrough

### Understanding the Dataset Loaders

Each loader in `src/datasets/` handles one MIMIC dataset:

```python
from src.datasets import MIMICCXRLoader

# Initialize (reads from paths in environment variables)
cxr = MIMICCXRLoader()

# Get all X-rays labeled as "No Finding"
normal_studies = cxr.get_normal_studies()
# Returns DataFrame: subject_id, study_id, No Finding, Atelectasis, ...

# Get path to an image
image_path = cxr.get_image_path(
    subject_id=10000032,
    study_id=50414267,
    dicom_id="02aa804e-bde0afdd-112c0b34-7bc16630-4e384014"
)
# Returns: Path("/data/mimic-cxr-jpg/files/p10/p10000032/s50414267/02aa...jpg")
```

### Understanding the Linker

The `DatasetLinker` connects records across datasets:

```python
from src.datasets import DatasetLinker, MIMICCXRLoader, MIMICIVEDLoader

linker = DatasetLinker()
cxr = MIMICCXRLoader()
ed = MIMICIVEDLoader()

# Link X-rays to ED visits
linked = linker.link_cxr_to_ed(
    cxr_studies=cxr.get_normal_studies(),
    cxr_metadata=cxr.metadata,
    ed_stays=ed.edstays,
    time_window_hours=24  # X-ray within 24 hours of ED visit
)
# Returns DataFrame with columns from both: subject_id, study_id, stay_id, ...
```

### Understanding Cohort Building

The `CohortBuilder` applies filtering logic:

```python
from src.cohort import CohortBuilder

builder = CohortBuilder()

# Build normal cohort with all filters
normal = builder.build_normal_cohort(
    require_ed_match=True,      # Must have matching ED visit
    filter_dispositions=True,   # Must have been sent home
    filter_diagnoses=True,      # No critical diagnoses
    include_reports=True        # Include radiology report text
)

# Split into train/validation
train, val = builder.split_cohort(normal, test_size=0.15)
```

### Understanding Preprocessing

The `PreprocessingPipeline` processes all modalities:

```python
from src.preprocessing import PreprocessingPipeline
from pathlib import Path

pipeline = PreprocessingPipeline()

# Process a single cohort
stats = pipeline.process_cohort(
    cohort_path=Path("output/cohorts/normal_train.parquet"),
    output_name="normal_train",
    process_images=True,
    process_structured=True,
    process_text=True,
    enable_summarization=False,  # Set True to use Claude AI
    num_workers=4
)

# Output structure:
# output/preprocessed/normal_train/
#   ├── images.h5           # All images in one HDF5 file
#   ├── structured.parquet  # Demographics, vitals, labs
#   ├── text.parquet        # Reports and summaries
#   └── manifest.json       # Processing statistics
```

### Loading Preprocessed Data

```python
from src.preprocessing import PreprocessingPipeline
from pathlib import Path
import h5py

# Load all preprocessed data
data = PreprocessingPipeline.load_preprocessed(
    Path("output/preprocessed/normal_train")
)

# Access structured data (pandas DataFrame)
structured = data["structured"]
print(structured.columns)
# ['study_id', 'subject_id', 'age', 'gender', 'triage_heartrate', ...]

# Access text data (pandas DataFrame)
text = data["text"]
print(text.columns)
# ['study_id', 'report', 'clinical_context', 'summary', 'tokens', ...]

# Access images (HDF5 file - must close when done)
images_h5 = data["images"]
first_image = images_h5["images"]["0"][:]  # numpy array
images_h5.close()
```

---

## CLI Reference

### build_cohort.py

```bash
python build_cohort.py [OPTIONS]

Options:
  --normal-only          Build only the normal cohort
  --anomalous-only       Build only the anomalous cohort
  --no-ed-match          Don't require ED stay match (more samples, less context)
  --no-reports           Don't include radiology reports
  --validation-fraction  Fraction for validation split (default: 0.15)
  --output-dir PATH      Override output directory
  --log-file PATH        Write logs to file
  -v, --verbose          Show debug output

Examples:
  # Build both cohorts with defaults
  python build_cohort.py

  # Build only normal cohort, verbose output
  python build_cohort.py --normal-only -v

  # Build with larger validation set
  python build_cohort.py --validation-fraction 0.2
```

### preprocess.py

```bash
python preprocess.py [OPTIONS]

Options:
  --cohort PATH          Process only this cohort file
  --images-only          Process only images
  --structured-only      Process only structured data
  --text-only            Process only text
  --enable-summarization Use Claude AI for text summaries
  --no-context           Don't include clinical context in summaries
  --workers N            Parallel workers for images (default: 4)
  --cohorts-dir PATH     Directory containing cohort files
  --output-dir PATH      Override output directory
  --log-file PATH        Write logs to file
  -v, --verbose          Show debug output

Examples:
  # Process all cohorts with 8 workers
  python preprocess.py --workers 8

  # Process only images for one cohort
  python preprocess.py --cohort output/cohorts/normal_train.parquet --images-only

  # Process text with AI summarization
  python preprocess.py --text-only --enable-summarization
```

---

## Troubleshooting

### Common Issues

**"Missing required data paths"**
- Check that all environment variables are set correctly
- Verify the paths exist and contain the expected files

**"No records after ED matching"**
- The temporal matching might be too strict
- Try `--no-ed-match` to skip ED matching (fewer features but more samples)

**Out of memory during lab processing**
- Labs are streamed in chunks, but may still use significant memory
- Reduce batch size or process fewer samples

**Slow image processing**
- Increase `--workers` (up to number of CPU cores)
- Images are I/O bound; an SSD helps significantly

---

## Next Steps

After preprocessing, the data is ready for:

1. **Unsupervised pretraining** on normal cohort
   - Train a Masked Autoencoder (MAE) or similar
   - Model learns "normal" chest X-ray features

2. **Supervised fine-tuning** on anomalous cohort
   - Add classification head for specific pathologies
   - Fine-tune on labeled abnormal examples

3. **Anomaly detection**
   - Use reconstruction error or embedding distance
   - Flag images that deviate from learned "normal"

---

## MAE Training (Step 3)

The project includes a complete MAE training pipeline for learning "normal" chest X-ray representations.

### train_mae.py

```bash
python train_mae.py [OPTIONS]

Options:
  --config {debug,fast,base,large}  Preset configuration (default: base)
  --train-dir PATH      Path to preprocessed training data
  --val-dir PATH        Path to preprocessed validation data
  --output-dir PATH     Directory for model outputs
  --checkpoint-dir PATH Directory for checkpoints
  --img-size SIZE       Input image size (default: 224, recommended: 1024 for CXR)
  --epochs N            Number of training epochs
  --batch-size N        Batch size per GPU
  --num-workers N       DataLoader workers (default: 8)
  --skip-anomaly        Skip anomaly detection phase (MAE pretraining only)
  --resume PATH         Resume from checkpoint
  -v, --verbose         Verbose output

Examples:
  # Debug run (2 epochs, small model)
  python train_mae.py --config debug --train-dir output/preprocessed/normal_train \
    --epochs 2 --batch-size 2 --skip-anomaly

  # Full training on high-resolution images
  python train_mae.py --config base --train-dir output/preprocessed/normal_train \
    --val-dir output/preprocessed/normal_val --img-size 1024 --epochs 800 \
    --batch-size 4 --skip-anomaly
```

### Image Processing Strategy

The MAE training uses **center crop** from full-resolution X-rays to consistently capture lung fields:

```
Full resolution image (~3056×2544)
         ↓
    Center Crop (1024×1024)
         ↓
   Augmentations (flip, rotate, blur)
         ↓
   3-channel conversion + ImageNet normalize
         ↓
   Output: [3, 1024, 1024] tensor
```

**Why center crop?** Chest X-rays are centered by radiologists. A center crop from full resolution:
- Captures the entire lung field consistently
- Avoids edge artifacts from random cropping
- Preserves anatomical context for anomaly detection
- Works with variable input resolutions (224 to 1024+)

### Configuration Presets

| Preset | Model | Embed Dim | Epochs | Batch Size | Use Case |
|--------|-------|-----------|--------|------------|----------|
| `debug` | ViT-Small | 384 | 10 | 8 | Quick testing |
| `fast` | ViT-Small | 384 | 100 | 32 | Experiments |
| `base` | ViT-Base | 768 | 800 | 64 | Production training |
| `large` | ViT-Large | 1024 | 1600 | 32 | Best results |

### Memory Requirements (1024×1024 input)

| Model | batch_size=2 | batch_size=4 | batch_size=6 |
|-------|--------------|--------------|--------------|
| ViT-Small | ~8 GB | ~15 GB | ~22 GB |
| ViT-Base | ~35 GB | ~68 GB | OOM (>97GB) |
| ViT-Large | ~70 GB | OOM | OOM |

*Tested on NVIDIA GH200 (97GB VRAM)*

### Output Files

```
output/models/
├── mae_final.pt           # Final trained model
├── config.json            # Training configuration
└── training_history.json  # Loss curves and metrics

output/checkpoints/
├── checkpoint_epoch_50.pt  # Periodic checkpoints
├── checkpoint_epoch_100.pt
└── ...
```

---

## Classifier Training (Step 4)

After MAE pretraining, the classifier is trained on the **anomalous cohort** (32,498 studies with CheXpert pathology labels) to detect 12 chest X-ray findings.

### train_classifier.py

```bash
python train_classifier.py [OPTIONS]

Options:
  --config {debug,fast,base}  Preset configuration (default: base)
  --train-dir PATH      Path to preprocessed training data (anomalous cohort)
  --val-dir PATH        Path to preprocessed validation data
  --chexpert-csv PATH   CheXpert labels CSV (mimic-cxr-2.0.0-chexpert.csv.gz)
  --mae-checkpoint PATH Pretrained MAE model
  --resume PATH         Resume from classifier checkpoint
  --img-size SIZE       Input image size (default: 1024)
  --epochs N            Number of training epochs (default: 30)
  --batch-size N        Batch size per GPU (default: 16)
  --num-workers N       DataLoader workers (default: 0 for stability)
  --device {cuda,cpu}   Training device

Examples:
  # Debug run with frozen encoder
  python train_classifier.py --config debug --epochs 2 --batch-size 2

  # Full training (frozen encoder, stable)
  python train_classifier.py --config debug \
    --train-dir output/preprocessed/anomalous_train \
    --val-dir output/preprocessed/anomalous_val \
    --chexpert-csv /path/to/mimic-cxr-2.0.0-chexpert.csv.gz \
    --mae-checkpoint output/models/mae_final.pt \
    --img-size 1024 --epochs 30 --batch-size 16
```

### Architecture

The multimodal classifier combines three modalities:

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  Chest X-ray    │    │   Clinical      │    │   Structured    │
│    Image        │    │   Text          │    │     Data        │
└────────┬────────┘    └────────┬────────┘    └────────┬────────┘
         │                      │                      │
         ▼                      ▼                      ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   MAE Encoder   │    │  ClinicalBERT   │    │   MLP Encoder   │
│  (frozen/fine)  │    │   Encoder       │    │                 │
└────────┬────────┘    └────────┬────────┘    └────────┬────────┘
         │                      │                      │
         └──────────────────────┼──────────────────────┘
                                │
                                ▼
                     ┌─────────────────┐
                     │  Cross-Attention │
                     │     Fusion       │
                     └────────┬────────┘
                              │
                              ▼
                     ┌─────────────────┐
                     │  Classification  │
                     │     Head         │
                     └────────┬────────┘
                              │
                              ▼
                       12 CheXpert Labels
```

**Loss Functions:**
- **Asymmetric Focal Loss**: Handles class imbalance
- **CLIP Loss**: Image-text alignment
- **Supervised Contrastive Loss**: Feature learning

### Configuration Presets

| Preset | freeze_mae_epochs | Learning Rate | Use Case |
|--------|------------------|---------------|----------|
| `debug` | 100 (always frozen) | 1e-4 | Stable training, limited VRAM |
| `fast` | 2 | 5e-5 | Quick experiments |
| `base` | 5 | 1e-4 | Production training |

**Important:** Use `--config debug` (freeze_mae_epochs=100) to keep MAE encoder frozen. Unfreezing requires ~90GB VRAM for backpropagation through the full ViT.

### Memory Requirements (1024×1024, BS=16)

| Configuration | MAE Frozen | MAE Unfrozen |
|---------------|------------|--------------|
| ViT-Base | ~28 GB | ~90 GB (OOM on <97GB) |

### Training Details

See [docs/RESULTS.md](docs/RESULTS.md) for:
- Complete per-class performance metrics
- Confusion matrix analysis
- Threshold optimization
- ROC and PR curves

See [docs/LAMBDA_DEPLOYMENT.md](docs/LAMBDA_DEPLOYMENT.md) for deployment guide

### Output Files

```
output/models/
├── classifier_best.pt    # Best validation AUROC checkpoint
├── classifier_final.pt   # Final epoch checkpoint
└── classifier_training.log  # Training logs with metrics
```

---

## References

### Datasets
- [MIMIC-CXR-JPG](https://physionet.org/content/mimic-cxr-jpg/) - Chest X-ray images
- [MIMIC-IV](https://physionet.org/content/mimiciv/) - Hospital records
- [MIMIC-IV-ED](https://physionet.org/content/mimic-iv-ed/) - Emergency department data
- [CXR-PRO](https://physionet.org/content/cxr-pro/) - Radiology reports

### Key Papers
- [MIMIC-CXR Database](https://www.nature.com/articles/s41597-019-0322-0)
- [CheXpert Labeler](https://arxiv.org/abs/1901.07031)
- [Masked Autoencoders (MAE)](https://arxiv.org/abs/2111.06377)
- [Asymmetric Loss for Multi-Label Classification](https://arxiv.org/abs/2009.14119)

### How to Cite

If you use this pipeline or the MIMIC datasets, please cite:

```bibtex
@article{johnson2019mimic,
  title={MIMIC-CXR, a de-identified publicly available database of chest radiographs with free-text reports},
  author={Johnson, Alistair EW and Pollard, Tom J and Greenbaum, Nathaniel R and Lungren, Matthew P and Deng, Chih-ying and Peng, Yifan and Lu, Zhiyong and Mark, Roger G and Berkowitz, Seth J and Horng, Steven},
  journal={Scientific Data},
  volume={6},
  number={1},
  pages={317},
  year={2019},
  publisher={Nature Publishing Group}
}

@article{johnson2023mimiciv,
  title={MIMIC-IV, a freely accessible electronic health record dataset},
  author={Johnson, Alistair EW and Bulgarelli, Lucas and Shen, Lu and Gayles, Alvin and Shammout, Ayad and Horng, Steven and Pollard, Tom J and Hao, Sicheng and Moody, Benjamin and Gow, Brian and others},
  journal={Scientific Data},
  volume={10},
  number={1},
  pages={1},
  year={2023},
  publisher={Nature Publishing Group}
}

@article{irvin2019chexpert,
  title={CheXpert: A large chest radiograph dataset with uncertainty labels and expert comparison},
  author={Irvin, Jeremy and Rajpurkar, Pranav and Ko, Michael and Yu, Yifan and Ciurea-Ilcus, Silviana and Chute, Chris and Marklund, Henrik and Haghgoo, Behzad and Ball, Robyn and Shpanskaya, Katie and others},
  journal={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={33},
  pages={590--597},
  year={2019}
}
```

---

## Future Improvements

### MAE Fine-Tuning Strategies

The current pipeline uses a **frozen MAE encoder** during classification to avoid overfitting and reduce memory requirements. Several strategies could improve feature extraction:

| Strategy | Description | Expected Improvement |
|----------|-------------|---------------------|
| **Gradual Unfreezing** | Progressively unfreeze encoder layers (last→first) over training | 5-10% AUROC |
| **Layer-wise Learning Rate Decay (LLRD)** | Lower learning rates for earlier layers (e.g., 0.1× per layer) | Better fine-tuning stability |
| **Domain-Specific Pretraining** | Continue MAE pretraining on larger CXR datasets (CheXpert, NIH ChestX-ray14) | Better medical feature representations |
| **Patch Size Optimization** | Experiment with 8×8 (fine detail) or 32×32 (global context) patches | Task-dependent improvements |

**Implementation**: Modify `src/models/multimodal.py` to support unfreezing schedules and LLRD.

### NLP Embeddings and Summarization

Current text encoding uses **ClinicalBERT** with clinical context only (leak-free mode). Advanced approaches:

| Approach | Model | Use Case |
|----------|-------|----------|
| **PubMedBERT** | `microsoft/BiomedNLP-PubMedBERT-base` | Scientific literature understanding |
| **BioClinicalBERT** | `emilyalsentzer/Bio_ClinicalBERT` | MIMIC clinical notes (current) |
| **RadBERT** | `StanfordAIMI/RadBERT` | Radiology-specific reports |
| **GatorTron** | `UFNLP/gatortron-base` | Largest clinical model (8.9B params) |

**Additional NLP Improvements:**
- **Entity Extraction**: Extract clinical entities (findings, anatomy, severity) using scispaCy or MedCAT
- **Temporal Modeling**: Model report sequences for patients with multiple studies using LSTM/Transformer
- **Summarization Fine-Tuning**: Train BART/T5 on MIMIC reports for domain-specific summarization
- **Negation Detection**: Use NegEx or transformer-based approaches to handle "no evidence of..."

### RAG (Retrieval-Augmented Generation)

Integrate retrieval systems to enhance predictions with similar historical cases:

| Component | Implementation | Benefit |
|-----------|----------------|---------|
| **Similar Case Retrieval** | FAISS index on MAE embeddings | Find visually similar X-rays |
| **Knowledge Base Integration** | Link to RadLex, SNOMED-CT ontologies | Structured medical knowledge |
| **Explainable Predictions** | Generate natural language explanations using retrieved context | Clinical interpretability |
| **Few-Shot Learning** | Use retrieved examples as in-context examples for rare pathologies | Better rare class detection |

**Implementation Path:**
1. Build FAISS index from MAE encoder embeddings
2. Retrieve top-k similar cases during inference
3. Aggregate predictions/explanations from retrieved cases
4. Fine-tune with contrastive learning on retrieved pairs

### Image Processing Alternatives

The current implementation uses **CenterCrop** to extract a fixed-size region from full-resolution X-rays:

```
Original: ~3056×2544 → CenterCrop(1024) → 1024×1024
Coverage: ~13% of original pixels (33% width × 40% height)
```

| Approach | Pros | Cons |
|----------|------|------|
| **CenterCrop** (current) | Preserves native resolution, consistent framing | Loses peripheral lung fields |
| **Resize** | Captures entire anatomy, no information loss | 3× downscale reduces fine detail |
| **Multi-Scale** | Use both crop and resize features | Increased compute, best of both |

**Multi-Scale Vision Implementation:**
```python
# Extract features at multiple scales
global_features = mae_encoder(resize(image, 224))    # Full image context
local_features = mae_encoder(center_crop(image, 224))  # Fine detail
combined = concat([global_features, local_features])
```

### Additional Improvements

| Category | Approach | Description |
|----------|----------|-------------|
| **Contrastive Learning** | CLIP-style pretraining | Joint image-text embedding space using radiology reports |
| **Uncertainty Quantification** | MC Dropout / Deep Ensembles | Calibrated confidence scores for clinical deployment |
| **Active Learning** | Uncertainty sampling | Prioritize labeling uncertain samples |
| **Test-Time Augmentation** | Multiple crops + voting | Improve inference robustness |
| **Ensemble Methods** | Multi-seed training | Combine diverse models for better generalization |

### Research Directions

- **Longitudinal Analysis**: Track patient X-rays over time to detect disease progression
- **Multi-Task Learning**: Joint training on detection, segmentation, and report generation
- **Federated Learning**: Train across institutions without sharing patient data
- **Causal Inference**: Identify causal relationships between clinical variables and outcomes

---

## MAE Reconstruction Analysis

The MAE model learns to reconstruct "normal" chest X-rays during pretraining. When presented with anomalous images, the model struggles to accurately reconstruct abnormal regions, producing higher reconstruction error. This error signal serves as the basis for anomaly detection.

### Reconstruction Error by Pathology

We analyzed reconstruction error across all 13 pathology classes in the validation set:

| Rank | Pathology | MSE | Detectability |
|------|-----------|-----|---------------|
| 1 | Pleural Other | 0.000907 | Highest |
| 2 | Pneumothorax | 0.000873 | |
| 3 | Pneumonia | 0.000867 | |
| 4 | Fracture | 0.000864 | |
| 5 | Lung Lesion | 0.000816 | |
| 6 | Edema | 0.000814 | |
| 7 | Atelectasis | 0.000622 | |
| 8 | Pleural Effusion | 0.000618 | |
| 9 | Lung Opacity | 0.000544 | |
| 10 | Cardiomegaly | 0.000511 | |
| 11 | Enlarged Cardiomediastinum | 0.000509 | |
| 12 | Support Devices | 0.000418 | |
| 13 | Consolidation | 0.000356 | Lowest |

### Key Findings

**High reconstruction error** (Pleural Other, Pneumothorax, Pneumonia, Fracture):
- These conditions introduce visual patterns that deviate significantly from normal anatomy
- Sharp discontinuities (fractures) or absence of lung markings (pneumothorax) are particularly hard to reconstruct
- Best candidates for MAE-based anomaly detection

**Moderate reconstruction error** (Lung Lesion, Edema, Atelectasis, Pleural Effusion):
- Intermediate difficulty involving opacity changes or fluid accumulation
- Distributed error patterns across lung fields

**Lower reconstruction error** (Cardiomegaly, Support Devices, Consolidation):
- May have more predictable patterns or overlap with variations learned from normal images
- Cardiomegaly shows error concentrated around cardiac border

### Visual Comparisons

Each comparison shows the original image, MAE reconstruction, squared error heatmap, and anomaly overlay:

| Pathology | Example |
|-----------|---------|
| Atelectasis | ![Atelectasis](docs/assets/reconstruction_comparisons/Atelectasis.png) |
| Cardiomegaly | ![Cardiomegaly](docs/assets/reconstruction_comparisons/Cardiomegaly.png) |
| Pneumothorax | ![Pneumothorax](docs/assets/reconstruction_comparisons/Pneumothorax.png) |
| Support Devices | ![Support Devices](docs/assets/reconstruction_comparisons/Support_Devices.png) |

See [docs/MAE_RECONSTRUCTION_ANALYSIS.md](docs/MAE_RECONSTRUCTION_ANALYSIS.md) for complete visual comparisons of all 13 pathology classes with detailed clinical descriptions.

---

## License

Research use only. Requires PhysioNet credentialed access to MIMIC datasets.
