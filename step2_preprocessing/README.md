# Step 2: Multimodal Data Preprocessing and Feature Engineering

Comprehensive preprocessing pipeline for MIMIC-CXR data that maintains full image resolution and implements advanced temporal feature engineering.

## Overview

This step processes the "normal" cohort identified in Step 1 and extracts three modalities:

1. **Images**: Full-resolution chest X-rays (~3000×2500 pixels) with normalization
2. **Structured Data**: Temporal features from labs and vitals with missing value tokens
3. **Text**: Clinical note summaries using NER, retrieval, and Claude LLM

## Key Features

### Image Processing
- **Full Resolution Preservation**: No downsampling to maintain fine-grained details
- **Flexible Normalization**: MinMax [0,1] or standardization (z-score)
- **Optional Augmentation**: Rotation, brightness, contrast at full resolution
- **Memory Efficient**: ~30MB per image with optional caching

### Structured Data Processing
- **NOT_DONE Token**: Explicit missing value representation instead of imputation
- **Temporal Features**: Time-aware aggregations with trend analysis
- **Dual Encoding**:
  - Aggregated (summary statistics + trends)
  - Sequential (RNN-ready time series)
- **Priority Labs**: Hemoglobin, WBC, creatinine, etc.
- **Priority Vitals**: HR, BP, SpO2, temperature, respiratory rate

### Text Processing
- **Medical NER**: scispacy entity extraction (diseases, symptoms, treatments)
- **Entity-Based Retrieval**: CLEAR method for precision
- **Semantic Fallback**: Sentence-transformers for recall
- **Claude Summarization**: LangChain + Anthropic Claude-3-Sonnet
- **ClinicalBERT Tokenization**: Bio_ClinicalBERT for model-ready text

## Installation

### 1. Create Conda Environment

```bash
conda create -n ml_env python=3.9
conda activate ml_env
```

### 2. Install Dependencies

```bash
cd step2_preprocessing
pip install -r requirements.txt
```

### 3. Install scispacy Model

```bash
pip install scispacy
pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.1/en_core_sci_md-0.5.1.tar.gz
```

### 4. Download ClinicalBERT (automatic on first run)

```bash
python -c "from transformers import AutoTokenizer; AutoTokenizer.from_pretrained('emilyalsentzer/Bio_ClinicalBERT')"
```

### 5. Set Up API Keys (Optional - for Claude summarization)

```bash
export ANTHROPIC_API_KEY='your-api-key-here'
```

Or pass via command line: `--anthropic-api-key YOUR_KEY`

### 6. Verify Setup

```bash
python test_setup.py
```

## Configuration

Configuration is managed in `config/config.yaml`. Key settings:

```yaml
image:
  preserve_full_resolution: true  # Maintain ~3000×2500 pixels
  normalize_method: "minmax"      # "minmax" or "standardize"

structured:
  missing_token: "NOT_DONE"       # Token for missing values
  encoding_method: "aggregated"   # "aggregated" or "sequential"

text:
  summarization:
    use_claude: true
    model: "claude-3-sonnet-20240229"
    max_summary_length: 512
```

### Path Configuration

Update paths in `config/config.yaml` to point to your data:

```yaml
data:
  step1_cohort_train: "/path/to/step1/output/cohort_train.csv"
  step1_cohort_val: "/path/to/step1/output/cohort_val.csv"
  mimic_cxr_base: "/media/dev/MIMIC_DATA/mimic-cxr-jpg"
  mimic_iv_base: "/home/dev/Documents/Portfolio/MIMIC_Data/physionet.org/files/mimiciv/3.1"
  mimic_ed_base: "/home/dev/Documents/Portfolio/MIMIC_Data/physionet.org/files/mimic-iv-ed/2.2"
```

## Usage

### Test Single Sample

```bash
# Test processing on sample 0
python test_sample.py --index 0 --split train

# Test with Claude API key
python test_sample.py --index 0 --api-key YOUR_KEY
```

### Process Small Subset (Testing)

```bash
# Process 10 samples from each split
python main.py --max-samples 10 --log-level INFO
```

### Process Full Dataset

```bash
# Process complete training and validation sets
python main.py

# Process only training set
python main.py --train-only

# Skip text processing (if no API key)
python main.py --skip-text

# Custom output directory
python main.py --output-dir /path/to/output
```

### Advanced Options

```bash
# Skip specific modalities
python main.py --skip-images        # Process only structured + text
python main.py --skip-structured    # Process only images + text
python main.py --skip-text          # Process only images + structured

# Process specific split
python main.py --train-only         # Only training set
python main.py --val-only           # Only validation set

# Debug mode
python main.py --log-level DEBUG --max-samples 5
```

## Output Structure

```
output/
├── preprocessing.log                    # Complete processing log
├── preprocessing_summary.json           # Overall statistics
│
├── train/
│   ├── images/
│   │   └── s10000032_study50414267.pt  # PyTorch tensor [C, H, W]
│   ├── structured_features/
│   │   └── s10000032_study50414267.json # Temporal features
│   ├── text_features/
│   │   └── s10000032_study50414267.pt   # Tokens + summary
│   ├── metadata/
│   │   └── s10000032_study50414267.json # Sample metadata
│   └── processing_stats.json            # Training set statistics
│
└── val/
    └── [same structure as train/]
```

### Output File Formats

#### Image Files (.pt)
```python
import torch
image = torch.load('s10000032_study50414267.pt')
# Shape: [1, 3056, 2544] (C, H, W)
# Type: torch.FloatTensor
# Range: [0.0, 1.0] (if minmax normalization)
```

#### Structured Features (.json)
```json
{
  "vital_heartrate": {
    "is_missing": false,
    "last_value": 78.0,
    "first_value": 82.0,
    "trend_slope": -2.0,
    "mean_value": 80.0,
    "measurement_count": 3,
    "time_span_hours": 2.5
  },
  "lab_hemoglobin": {
    "is_missing": true,
    "last_value": "NOT_DONE"
  }
}
```

#### Text Features (.pt)
```python
import torch
text_data = torch.load('s10000032_study50414267.pt')
# Keys: 'summary', 'tokens', 'num_entities', 'entities'
# text_data['tokens']['input_ids']: ClinicalBERT token IDs
# text_data['summary']: Claude-generated summary
```

## Architecture

```
MultimodalMIMICDataset
├── FullResolutionImageLoader
│   ├── Load .jpg from MIMIC-CXR
│   ├── Normalize (minmax/standardize)
│   └── Optional augmentation
│
├── TemporalFeatureExtractor
│   ├── Load MIMIC-IV labs
│   ├── Load MIMIC-ED vitals
│   ├── Temporal aggregation
│   └── Missing value tokens
│
└── ClinicalNoteProcessor
    ├── Medical NER (scispacy)
    ├── Entity-based retrieval
    ├── Semantic similarity
    ├── Claude summarization
    └── ClinicalBERT tokenization
```

## Memory Requirements

- **RAM**: 16+ GB recommended
  - Full-resolution image: ~30 MB
  - Batch size 4: ~120 MB + overhead

- **GPU**: Optional (8+ GB VRAM recommended)
  - Used for ClinicalBERT tokenization
  - Not required for preprocessing

- **Disk Space**:
  - Processed data: ~50 GB for 23k samples
  - Original MIMIC data: ~500+ GB

## Performance

Typical processing times (single sample):
- Image loading: 0.1-0.5s
- Structured data: 0.2-1.0s (depends on lab count)
- Text processing: 1-5s (with Claude API)

Expected total time for full dataset:
- Without Claude: ~2-4 hours
- With Claude: ~8-12 hours (API rate limits)

## Troubleshooting

### "scispacy model not found"
```bash
pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.1/en_core_sci_md-0.5.1.tar.gz
```

### "LangChain/Anthropic not available"
```bash
pip install langchain langchain-anthropic anthropic
export ANTHROPIC_API_KEY='your-key'
```

### "Image directory not found"
Check that `mimic_cxr_base` in config.yaml points to the correct MIMIC-CXR-JPG directory.

### "Lab events file too large"
The code loads labs in chunks (100k rows at a time) to handle the ~120M row labevents.csv file.

### CUDA out of memory
- Reduce batch size in DataLoader
- Process images on CPU: `CUDA_VISIBLE_DEVICES="" python main.py`

### Claude API rate limits
- The code processes sequentially to avoid rate limits
- For large datasets, consider running in batches or overnight

## Next Steps

After preprocessing, outputs can be used for:

1. **Model Training**: Load processed data with PyTorch DataLoader
2. **Exploratory Analysis**: Analyze feature distributions, missing patterns
3. **Quality Checks**: Validate image quality, text summaries
4. **Anomaly Detection**: Train multimodal models for CXR abnormality detection

## Citation

If you use this preprocessing pipeline, please cite:

```bibtex
@misc{mimic-cxr-preprocessing,
  title={Multimodal Preprocessing Pipeline for MIMIC-CXR},
  year={2024},
  note={Implements full-resolution image loading, temporal feature engineering,
        and clinical note summarization}
}
```

## References

- MIMIC-CXR: https://physionet.org/content/mimic-cxr-jpg/
- MIMIC-IV: https://physionet.org/content/mimiciv/
- MIMIC-IV-ED: https://physionet.org/content/mimic-iv-ed/
- scispacy: https://allenai.github.io/scispacy/
- ClinicalBERT: https://huggingface.co/emilyalsentzer/Bio_ClinicalBERT
- Claude: https://www.anthropic.com/claude
