# MIMIC-CXR Unsupervised Anomaly Detection Pipeline

End-to-end pipeline for identifying and preprocessing "normal" chest X-ray cases from MIMIC datasets, enabling unsupervised multimodal anomaly detection.

## Project Overview

This repository implements a two-step pipeline for MIMIC-CXR anomaly detection:

**Step 1: Normal Cohort Identification**
- Filters chest X-rays with "No Finding" labels and benign clinical courses
- Integrates MIMIC-CXR-JPG, MIMIC-IV-ED, and MIMIC-IV data
- Produces high-quality normal cohorts for unsupervised learning

**Step 2: Multimodal Data Preprocessing**
- Full-resolution image loading (~3000×2500 pixels)
- Temporal feature engineering from labs and vitals
- Clinical note summarization with NER and Claude LLM
- Prepares model-ready multimodal features

## Quick Start

### Prerequisites

- Python 3.8+
- Access to MIMIC datasets (requires PhysioNet credentialing):
  - MIMIC-CXR-JPG v2.1.0
  - MIMIC-IV v3.1
  - MIMIC-IV-ED v2.2

### Installation

```bash
# Clone/navigate to repository
cd MIMIC-CXR-Anomaly-Preprocessing

# Install Step 1 dependencies
pip install -r requirements.txt

# Install Step 2 dependencies
cd step2_preprocessing
pip install -r requirements.txt

# Install scispacy model
pip install scispacy
pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.1/en_core_sci_md-0.5.1.tar.gz

# Optional: Set Claude API key for text summarization
export ANTHROPIC_API_KEY='your-api-key-here'
```

### Basic Usage

```bash
# Step 1: Identify normal cohort
python main.py

# Step 2: Preprocess data (from step2_preprocessing directory)
cd step2_preprocessing
python main.py
```

---

## Step 1: Normal Cohort Identification

### Overview

Identifies "normal" cases by applying strict radiology and clinical filters across MIMIC datasets.

### Normal Case Definition

A case is considered "normal" if it meets ALL criteria:

**Radiology Criteria:**
- CheXpert "No Finding" label = 1.0
- All pathology labels (Pneumonia, Effusion, etc.) NOT positive
- No acute/significant abnormalities in report

**Clinical Context Criteria:**
- ED disposition: Discharged home (not admitted)
- No critical diagnoses (sepsis, MI, respiratory failure, etc.)
- No ICU admission if hospitalized
- No in-hospital death if hospitalized
- CXR performed during/near ED visit (24-hour window)
- Patient age ≥ 18 years

### Usage

```bash
# Basic usage
python main.py

# Custom options
python main.py --output-dir my_output --validation-samples 200

# Fast mode (skip hospital filtering)
python main.py --no-hospital-filter

# Debug mode
python main.py --log-level DEBUG
```

### Command-Line Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--output-dir` | Output directory for results | `output` |
| `--no-hospital-filter` | Skip hospital outcome filtering (faster) | False |
| `--validation-samples` | Number of samples for manual review | 100 |
| `--log-level` | Logging level (DEBUG/INFO/WARNING/ERROR) | INFO |
| `--skip-validation` | Skip data validation | False |
| `--optimize-memory` | Optimize memory usage | False |

### Output Files

```
output/
├── cohorts/
│   ├── normal_cohort_full.csv           # Complete normal cohort
│   ├── normal_cohort_train.csv          # Training set (85%)
│   └── normal_cohort_validation.csv     # Validation set (15%)
├── manual_review/
│   ├── sample_for_review.csv            # Random sample for verification
│   └── edge_cases_*.csv                 # Edge cases
├── reports/
│   ├── validation_report.txt            # Data quality report
│   └── summary_statistics.json          # Cohort statistics
└── logs/
    └── cohort_building_*.log            # Execution log
```

### Expected Results

- Normal cases identified: ~20,000-40,000 studies (10-20% of MIMIC-CXR)
- Unique subjects: ~15,000-25,000 patients
- Processing time: 5-15 minutes
- Memory usage: 4-8 GB peak

### Configuration

Edit `src/config/config.py` and `src/config/paths.py` to customize:

- Filter criteria (dispositions, diagnoses, time windows)
- Data paths
- Validation parameters
- Train/val split ratio

---

## Step 2: Multimodal Data Preprocessing

### Overview

Processes the normal cohort from Step 1 to extract three modalities with advanced feature engineering.

### Key Features

**Image Processing:**
- Full-resolution preservation (~3000×2500 pixels, no downsampling)
- MinMax [0,1] or z-score normalization
- Optional augmentation at full resolution
- Memory-efficient (~30MB per image)

**Structured Data:**
- NOT_DONE token for missing values (no imputation)
- Temporal features with trend analysis
- Time-aware aggregations
- Priority labs: Hemoglobin, WBC, creatinine, electrolytes
- Priority vitals: HR, BP, SpO2, temperature, RR

**Text Processing:**
- Optional clinical note rewriting (expands abbreviations, standardizes format)
- Medical NER with scispacy
- Entity-based + semantic retrieval
- Claude-3.5-Sonnet summarization via LangChain
- ClinicalBERT tokenization

### Configuration

Edit `step2_preprocessing/config/config.yaml`:

```yaml
# Image settings
image:
  preserve_full_resolution: true
  normalize_method: "minmax"      # or "standardize"
  augmentation:
    enabled: true
    rotation_range: 5

# Structured data
structured:
  missing_token: "NOT_DONE"
  encoding_method: "aggregated"   # or "sequential"

# Text processing
text:
  summarization:
    use_claude: true
    model: "claude-3-5-sonnet-latest"
    max_summary_length: 500

  # Optional: Note rewriting (disabled by default)
  note_rewriting:
    enabled: false  # Set to true to expand abbreviations
    model: "claude-sonnet-4-5-20250929"
    temperature: 0.0

# Data paths
data:
  step1_cohort_train: "../output/cohorts/normal_cohort_train.csv"
  step1_cohort_val: "../output/cohorts/normal_cohort_validation.csv"
  mimic_cxr_base: "/media/dev/MIMIC_DATA/mimic-cxr-jpg"
  mimic_iv_base: "/path/to/mimiciv/3.1"
  mimic_ed_base: "/path/to/mimic-iv-ed/2.2"
```

### Usage

```bash
cd step2_preprocessing

# Test on small subset
python main.py --max-samples 10

# Process full dataset
python main.py

# Skip specific modalities
python main.py --skip-text          # No Claude API key
python main.py --skip-images        # Structured + text only
python main.py --skip-structured    # Images + text only

# Process specific split
python main.py --train-only
python main.py --val-only

# Custom output
python main.py --output-dir /path/to/output
```

### Output Structure

```
step2_preprocessing/output/
├── preprocessing.log
├── preprocessing_summary.json
│
├── train/
│   ├── images/
│   │   └── s10000032_study50414267.pt     # [C, H, W] FloatTensor
│   ├── structured_features/
│   │   └── s10000032_study50414267.json   # Temporal features
│   ├── text_features/
│   │   └── s10000032_study50414267.pt     # Summary + tokens
│   ├── metadata/
│   │   └── s10000032_study50414267.json   # Sample metadata
│   └── processing_stats.json
│
└── val/
    └── [same structure]
```

### Output Formats

**Images** (.pt):
```python
import torch
image = torch.load('image.pt')
# Shape: [1, 3056, 2544] (C, H, W)
# Type: torch.FloatTensor
# Range: [0.0, 1.0] with minmax normalization
```

**Structured Features** (.json):
```json
{
  "vital_heartrate": {
    "is_missing": false,
    "last_value": 78.0,
    "trend_slope": -2.0,
    "mean_value": 80.0,
    "measurement_count": 3
  },
  "lab_hemoglobin": {
    "is_missing": true,
    "last_value": "NOT_DONE"
  }
}
```

**Text Features** (.pt):
```python
text_data = torch.load('text.pt')
# Keys: 'summary', 'tokens', 'num_entities', 'entities'
# text_data['tokens']['input_ids']: ClinicalBERT tokens
# text_data['summary']: Claude summary
```

### Clinical Note Rewriting (Optional)

An optional preprocessing step that standardizes clinical notes before entity extraction and summarization:

**Features:**
- Expands abbreviations (e.g., "c/o" → "complains of", "c/p" → "chest pain", "HTN" → "hypertension")
- Normalizes format with complete sentences and proper grammar
- Uses professional clinical tone with appropriate medical terminology
- Preserves all numerical values and factual details
- Maintains information completeness (no fabrication or omission)

**Configuration:**
```yaml
text:
  note_rewriting:
    enabled: false  # Set to true to enable
    model: "claude-sonnet-4-5-20250929"
    temperature: 0.0  # Deterministic
```

**Impact:**
- May improve entity extraction quality (more complete medical terms)
- Increases processing time (adds one Claude API call per note)
- Disabled by default to minimize API costs
- Recommended for production use with high-quality requirements

**Testing:**
```bash
# Run full test suite
cd step2_preprocessing
python tests/test_rewriting.py

# Run demo comparison (shows before/after with real notes)
python demo_rewriting_pipeline.py

# View implementation in notebook
jupyter notebook notebooks/RAG_Implementation.ipynb
# See cells 9-13 for rewriting implementation and comparison
```

**Test Results (Validated):**
- Configuration loading: ✓ Passed
- Initialization (enabled/disabled): ✓ Passed
- Empty note handling: ✓ Passed
- Fallback when disabled: ✓ Passed
- Abbreviation expansion: ✓ Passed ("c/o" → "complains of", "HTN" → "hypertension")
- Numerical preservation: ✓ Passed (all vital signs preserved: 78, 120/80, 98%)
- RAG pipeline integration: ✓ Passed (6 entities extracted, 378-char summary generated)
- Entity extraction improvement: ✓ Verified (7 → 13 entities with rewriting, 86% increase)

### Performance

**Processing time (per sample):**
- Image loading: 0.1-0.5s
- Structured data: 0.2-1.0s
- Text processing: 1-5s (with Claude API)

**Full dataset (~23k samples):**
- Without Claude: 2-4 hours
- With Claude: 8-12 hours (rate limits)

**Memory requirements:**
- RAM: 16+ GB recommended
- GPU: Optional (8+ GB VRAM for faster processing)
- Disk: ~50 GB for processed data

---

## Project Structure

```
MIMIC-CXR-Anomaly-Preprocessing/
├── main.py                          # Step 1 main script
├── requirements.txt                 # Step 1 dependencies
├── README.md                        # This file
├── .gitignore
│
├── src/                             # Step 1 source code
│   ├── config/
│   ├── data_loaders/
│   ├── filters/
│   ├── mergers/
│   ├── validators/
│   └── utils/
│
├── output/                          # Step 1 outputs (gitignored)
│   ├── cohorts/
│   ├── manual_review/
│   ├── reports/
│   └── logs/
│
└── step2_preprocessing/             # Step 2 pipeline
    ├── main.py
    ├── requirements.txt
    ├── setup.sh
    ├── config/
    │   └── config.yaml
    │
    ├── src/
    │   ├── image_processing/
    │   ├── structured_data/
    │   ├── text_processing/
    │   ├── integration/
    │   └── utils/
    │
    ├── tests/                       # Test scripts
    │   ├── README.md
    │   ├── test_setup.py
    │   ├── test_claude.py
    │   └── test_sample.py
    │
    ├── notebooks/                   # Jupyter notebooks
    │   ├── README.md
    │   ├── RAG_Implementation.ipynb
    │   ├── analyze_test_results.ipynb
    │   └── explore_cohort_outputs.ipynb
    │
    └── output/                      # Step 2 outputs (gitignored)
        ├── train/
        └── val/
```

---

## Complete Workflow

### 1. Identify Normal Cohort (Step 1)

```bash
# Run Step 1 to create normal cohort
python main.py

# Review validation report
cat output/reports/validation_report.txt

# Manual review (optional)
cat output/manual_review/sample_for_review.csv
```

**Output**: `output/cohorts/normal_cohort_train.csv` and `normal_cohort_validation.csv`

### 2. Preprocess Data (Step 2)

```bash
cd step2_preprocessing

# Test on small subset first
python main.py --max-samples 5 --skip-text --output-dir test_output

# Analyze test results
jupyter notebook notebooks/analyze_test_results.ipynb

# Process full dataset with Claude
export ANTHROPIC_API_KEY='your-key'
python main.py
```

**Output**: Preprocessed multimodal features in `output/train/` and `output/val/`

### 3. Load Preprocessed Data

```python
import torch
import json
from pathlib import Path

# Load one sample
sample_id = "s10000032_study50414267"
output_dir = Path("step2_preprocessing/output/train")

# Image
image = torch.load(output_dir / "images" / f"{sample_id}.pt")

# Structured data
with open(output_dir / "structured_features" / f"{sample_id}.json") as f:
    structured = json.load(f)

# Text
text = torch.load(output_dir / "text_features" / f"{sample_id}.pt")

print(f"Image shape: {image.shape}")
print(f"Structured features: {len(structured)}")
print(f"Summary: {text['summary'][:100]}...")
```

---

## Troubleshooting

### Step 1 Issues

**"Path not found":**
- Verify paths in `src/config/paths.py`
- Ensure proper permissions to MIMIC data

**"No normal cases found":**
- Check filter criteria in `src/config/config.py` (may be too strict)
- Review logs for filtering statistics

**Memory errors:**
- Use `--optimize-memory` flag
- Use `--no-hospital-filter` to skip hospital data
- Increase system RAM (>8GB recommended)

### Step 2 Issues

**"scispacy model not found":**
```bash
pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.1/en_core_sci_md-0.5.1.tar.gz
```

**"LangChain not available":**
```bash
pip install langchain langchain-anthropic anthropic
export ANTHROPIC_API_KEY='your-key'
```

**Claude API 404 errors:**
- Model name changed - use `claude-3-5-sonnet-latest` in config.yaml
- Verify API key has correct permissions

**"Lab events file too large":**
- Code loads labs in 100k row chunks to handle ~120M row file
- Ensure sufficient RAM (16+ GB)

**CUDA out of memory:**
- Reduce batch size in DataLoader
- Process on CPU: `CUDA_VISIBLE_DEVICES="" python main.py`

---

## Data Requirements

**MIMIC-CXR-JPG v2.1.0:**
- `mimic-cxr-2.0.0-chexpert.csv.gz` (CheXpert labels)
- `mimic-cxr-2.0.0-metadata.csv.gz` (Image metadata)
- `files/` (JPEG images organized by patient/study)

**MIMIC-IV-ED v2.2:**
- `ed/edstays.csv` (ED stay information)
- `ed/diagnosis.csv` (ED diagnoses)
- `ed/triage.csv` (Triage vitals and chief complaints)
- `ed/vitalsign.csv` (ED vital signs)

**MIMIC-IV v3.1:**
- `hosp/patients.csv` (Demographics)
- `hosp/admissions.csv` (Hospital admissions)
- `hosp/transfers.csv` (Transfer events)
- `hosp/labevents.csv.gz` (Lab results, ~120M rows)

---

## Next Steps

After completing both steps, you can:

1. **Train Models**: Use preprocessed data for multimodal anomaly detection
2. **Exploratory Analysis**: Analyze feature distributions, missing patterns
3. **Quality Validation**: Review image quality and text summaries
4. **Model Development**: Build and train unsupervised anomaly detection models

Example PyTorch Dataset:
```python
from step2_preprocessing.src.multimodal_dataset import MultimodalMIMICDataset

dataset = MultimodalMIMICDataset(
    cohort_path="step2_preprocessing/output/train",
    config=config
)

sample = dataset[0]
# Returns: {'image': tensor, 'structured': dict, 'text': dict, 'metadata': dict}
```

---

## License

This code is provided for research and educational purposes. Please ensure you have proper authorization and credentialing to access MIMIC datasets through PhysioNet.

## References

**MIMIC Datasets:**
- Johnson, A., Pollard, T., Mark, R. et al. MIMIC-CXR-JPG, a large publicly available database of labeled chest radiographs. *Sci Data* 6, 317 (2019).
- Johnson, A., Bulgarelli, L., Pollard, T. et al. MIMIC-IV-ED. PhysioNet (2023).
- Johnson, A., Bulgarelli, L., Shen, L. et al. MIMIC-IV. PhysioNet (2024).

**NLP/ML Tools:**
- scispacy: https://allenai.github.io/scispacy/
- ClinicalBERT: https://huggingface.co/emilyalsentzer/Bio_ClinicalBERT
- Claude: https://www.anthropic.com/claude
- LangChain: https://python.langchain.com/

## Citation

If you use this pipeline in your research, please cite:

```bibtex
@misc{mimic-cxr-anomaly-pipeline,
  title={MIMIC-CXR Unsupervised Anomaly Detection Pipeline},
  year={2024},
  note={Two-step pipeline for normal cohort identification and multimodal preprocessing.
        Implements full-resolution image loading, temporal feature engineering,
        and clinical note summarization with Claude LLM}
}
```

## Contact

For issues or questions, please open an issue in the project repository.
