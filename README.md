# MIMIC-CXR Anomaly Detection Pipeline

A unified preprocessing pipeline for multimodal chest X-ray anomaly detection using MIMIC datasets.

## Overview

This pipeline processes data from multiple MIMIC datasets to create cohorts for:
1. **Unsupervised pretraining** on ~33k normal CXR studies
2. **Supervised classification** on ~200k anomalous CXR studies

## Quick Start

### Prerequisites

- Python 3.10+
- PhysioNet credentialed access to:
  - MIMIC-CXR-JPG v2.1.0
  - MIMIC-IV v3.1
  - MIMIC-IV-ED v2.2
  - CXR-PRO v1.0.0

### Installation

```bash
# Clone repository
cd MIMIC-CXR-Anomaly-Preprocessing

# Install dependencies
pip install -r requirements.txt

# Download scispacy model
python -m spacy download en_core_sci_md
```

### Configuration

Set environment variables (or create `.env` file):

```bash
export MIMIC_CXR_JPG_PATH=/path/to/mimic-cxr-jpg/2.1.0
export MIMIC_IV_PATH=/path/to/mimiciv/3.1
export MIMIC_IV_ED_PATH=/path/to/mimic-iv-ed/2.2
export CXR_PRO_PATH=/path/to/cxr-pro/1.0.0
export OUTPUT_PATH=./output
export ANTHROPIC_API_KEY=sk-...  # Optional: for text summarization
```

### Usage

```bash
# Step 1: Build cohorts
python build_cohort.py

# Step 2: Preprocess data
python preprocess.py
```

## Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────┐
│  MIMIC Data Sources                                         │
├──────────────┬──────────────┬──────────────┬───────────────┤
│ MIMIC-CXR-JPG│ MIMIC-IV     │ MIMIC-IV-ED  │ CXR-PRO       │
│ 377k studies │ 299k patients│ 449k ED stays│ 371k reports  │
│ Images+Labels│ Labs/Dx/Proc │ Vitals/Triage│ Report text   │
└──────┬───────┴──────┬───────┴──────┬───────┴───────┬───────┘
       │              │              │               │
       └──────────────┴──────────────┴───────────────┘
                           │
                    ┌──────▼──────┐
                    │   Linker    │  Links via subject_id,
                    │             │  study_id, hadm_id, stay_id
                    └──────┬──────┘
                           │
              ┌────────────┴────────────┐
              ▼                         ▼
    ┌─────────────────┐       ┌─────────────────┐
    │  Normal Cohort  │       │ Anomalous Cohort│
    │   ~33k studies  │       │  ~200k studies  │
    │  (No Finding)   │       │  (Pathologies)  │
    └────────┬────────┘       └────────┬────────┘
             │                         │
             └────────────┬────────────┘
                          ▼
              ┌───────────────────────┐
              │   Preprocessing       │
              ├───────────┬───────────┤
              │  Images   │ Structured│  Text
              │  → HDF5   │ → Parquet │  → Parquet
              └───────────┴───────────┘
```

## Project Structure

```
├── src/
│   ├── config/
│   │   └── settings.py      # Environment-based configuration
│   ├── datasets/
│   │   ├── mimic_iv.py      # MIMIC-IV loader (patients, labs, diagnoses)
│   │   ├── mimic_iv_ed.py   # MIMIC-IV-ED loader (ED stays, vitals)
│   │   ├── mimic_cxr.py     # MIMIC-CXR-JPG loader (images, labels)
│   │   ├── cxr_pro.py       # CXR-PRO loader (radiology reports)
│   │   └── linker.py        # Cross-dataset linking
│   ├── cohort/
│   │   └── builder.py       # Cohort generation (normal + anomalous)
│   ├── preprocessing/
│   │   ├── images.py        # Batch image processing → HDF5
│   │   ├── structured.py    # Labs, vitals → Parquet
│   │   ├── text.py          # Reports → Parquet
│   │   └── pipeline.py      # Orchestrates preprocessing
│   └── utils/
│       └── io.py            # Logging utilities
├── build_cohort.py          # CLI: cohort generation
├── preprocess.py            # CLI: preprocessing
└── requirements.txt         # Dependencies (pinned versions)
```

## Dataset Linking

All MIMIC datasets share a common `subject_id` for patients. Additional keys:

| Dataset | Key Fields | Linking |
|---------|-----------|---------|
| MIMIC-CXR-JPG | `subject_id`, `study_id`, `dicom_id` | Images + labels |
| CXR-PRO | `subject_id`, `study_id` | Radiology reports |
| MIMIC-IV-ED | `subject_id`, `stay_id`, `hadm_id` | ED visits |
| MIMIC-IV | `subject_id`, `hadm_id` | Hospital admissions |

**Temporal Matching**: CXR studies are matched to ED stays within ±24 hours of `study_datetime`.

## Output Format

### Cohorts (Parquet)
```
output/cohorts/
├── normal_full.parquet      # All normal studies
├── normal_train.parquet     # 85% train split
├── normal_val.parquet       # 15% validation split
├── anomalous_full.parquet   # All abnormal studies
├── anomalous_train.parquet
└── anomalous_val.parquet
```

### Preprocessed Data
```
output/preprocessed/{cohort_name}/
├── images.h5                # HDF5: chunked, compressed images
├── structured.parquet       # Labs, vitals, demographics
├── text.parquet             # Reports, summaries, tokens
└── manifest.json            # Processing statistics
```

## Features Extracted

### Cohort Features
- **Demographics**: age, gender
- **ED Visit**: intime, outtime, disposition, triage vitals
- **Diagnoses**: ED and hospital ICD codes
- **Procedures**: ICD procedure codes
- **Reports**: Radiology report text

### Preprocessed Features

| Modality | Format | Content |
|----------|--------|---------|
| **Images** | HDF5 | Normalized tensors [C, H, W] |
| **Structured** | Parquet | Labs (17 types), vitals (7 types), demographics |
| **Text** | Parquet | Report, summary, tokens, token count |

## CLI Options

### build_cohort.py
```bash
python build_cohort.py [OPTIONS]

Options:
  --normal-only          Build only normal cohort
  --anomalous-only       Build only anomalous cohort
  --no-ed-match          Don't require ED stay match
  --no-reports           Don't include radiology reports
  --validation-fraction  Validation split fraction (default: 0.15)
  --output-dir           Output directory
  -v, --verbose          Verbose output
```

### preprocess.py
```bash
python preprocess.py [OPTIONS]

Options:
  --cohort PATH          Process specific cohort file
  --images-only          Process only images
  --structured-only      Process only structured data
  --text-only            Process only text
  --enable-summarization Use Claude API for summarization
  --workers N            Parallel workers (default: 4)
  --cohorts-dir PATH     Directory with cohort files
  --output-dir PATH      Output directory
  -v, --verbose          Verbose output
```

## Performance

- **Cohort generation**: ~5-15 minutes
- **Preprocessing**: ~0.5-1 sec/sample with 4 workers (images + structured)
- **Storage**: ~30 MB/image (HDF5), ~5 KB/sample (structured + text)

## Requirements

See `requirements.txt` for pinned versions. Key dependencies:
- pandas, numpy, pyarrow, h5py
- torch, torchvision
- transformers, spacy, scispacy
- anthropic (optional, for summarization)

## License

Research use only. Requires PhysioNet credentialing for MIMIC data access.

## References

- MIMIC-CXR-JPG: https://physionet.org/content/mimic-cxr-jpg/
- MIMIC-IV: https://physionet.org/content/mimiciv/
- MIMIC-IV-ED: https://physionet.org/content/mimic-iv-ed/
- CXR-PRO: https://physionet.org/content/cxr-pro/
