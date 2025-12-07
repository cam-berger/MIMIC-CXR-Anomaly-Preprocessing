# MIMIC-CXR Anomaly Detection Pipeline

A data preprocessing pipeline that prepares chest X-ray images and clinical data for training AI models to detect medical abnormalities.

## Table of Contents
- [What This Project Does](#what-this-project-does)
- [Background: The Problem We're Solving](#background-the-problem-were-solving)
- [The Data: MIMIC Datasets](#the-data-mimic-datasets)
- [How It Works](#how-it-works)
- [Getting Started](#getting-started)
- [Project Structure](#project-structure)
- [Detailed Walkthrough](#detailed-walkthrough)
- [CLI Reference](#cli-reference)

---

## What This Project Does

This pipeline takes raw medical data from multiple sources and transforms it into a clean, organized format ready for machine learning. Specifically, it:

1. **Identifies "normal" chest X-rays** (~33,000 studies) - X-rays from patients who were healthy and sent home
2. **Identifies "abnormal" chest X-rays** (~200,000 studies) - X-rays showing various medical conditions
3. **Combines multiple data types** for each X-ray:
   - The image itself
   - Lab results (blood tests, etc.)
   - Vital signs (heart rate, blood pressure, etc.)
   - The radiologist's written report
   - Diagnoses and procedures

The output is ready-to-use datasets for training anomaly detection models.

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

**Normal Cohort (~33k studies)**
```
Start: 377,000 X-ray studies
  │
  ├─ Filter: CheXpert "No Finding" = 1.0 → ~95,000
  │    (Automated label says no disease detected)
  │
  ├─ Filter: Match to ED visit within 24 hours → ~52,000
  │    (We need clinical context)
  │
  ├─ Filter: Patient was sent home → ~22,000
  │    (Admitted patients might have been sick)
  │
  ├─ Filter: No critical diagnoses (sepsis, MI, etc.) → ~20,000
  │    (Extra safety check)
  │
  └─ Filter: Age ≥ 18 → ~20,000 (final normal cohort)
```

**Anomalous Cohort (~200k studies)**
```
Start: 377,000 X-ray studies
  │
  ├─ Filter: Any pathology label = 1.0 → ~200,000
  │    (CheXpert detected something abnormal)
  │
  └─ Filter: Age ≥ 18 → ~200,000 (final anomalous cohort)
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

## References

### Datasets
- [MIMIC-CXR-JPG](https://physionet.org/content/mimic-cxr-jpg/) - Chest X-ray images
- [MIMIC-IV](https://physionet.org/content/mimiciv/) - Hospital records
- [MIMIC-IV-ED](https://physionet.org/content/mimic-iv-ed/) - Emergency department data
- [CXR-PRO](https://physionet.org/content/cxr-pro/) - Radiology reports

### Key Papers
- [MIMIC-CXR Database](https://www.nature.com/articles/s41597-019-0322-0)
- [CheXpert Labeler](https://arxiv.org/abs/1901.07031)

---

## Future Improvements

### Image Processing Alternatives

The current implementation uses **CenterCrop** to extract a fixed-size region from full-resolution X-rays:

```
Original: ~3056×2544 → CenterCrop(1024) → 1024×1024
Coverage: ~13% of original pixels (33% width × 40% height)
```

**Potential improvement**: Use **Resize** instead of CenterCrop to preserve the entire image:

```python
# Current (CenterCrop):
T.CenterCrop(target_size)  # Takes center portion only

# Alternative (Resize):
T.Resize(target_size)  # Scales entire image
```

| Approach | Pros | Cons |
|----------|------|------|
| **CenterCrop** (current) | Preserves native resolution, consistent framing | Loses peripheral lung fields (~86% of image) |
| **Resize** (alternative) | Captures entire anatomy, no information loss | 3× downscale reduces fine detail |

**When to consider Resize:**
- Detecting peripheral pathology (pneumothorax, pleural effusions)
- Analyzing diaphragm or shoulder regions
- When global context matters more than fine detail

**When CenterCrop is preferred:**
- Central pathology (cardiomegaly, mediastinal masses)
- When training resolution is limited (224×224)
- When consistent anatomical framing is important

This is configurable in `src/models/dataset.py` in the `get_mae_augmentations()` function.

---

## License

Research use only. Requires PhysioNet credentialed access to MIMIC datasets.
