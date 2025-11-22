# Data Schema Reference

Complete data schema documentation for the MIMIC-CXR preprocessing pipeline outputs.

## Table of Contents

1. [Overview](#overview)
2. [Step 1 Output: Cohort CSV Schema](#step-1-output-cohort-csv-schema)
3. [Step 2 Output: Multimodal Features](#step-2-output-multimodal-features)
4. [File Relationships and Loading](#file-relationships-and-loading)
5. [Example Records](#example-records)

---

## Overview

The pipeline produces two types of outputs:

**Step 1**: CSV files containing metadata for normal cases
- `normal_cohort_train.csv` (~17,000 rows × 28 columns)
- `normal_cohort_validation.csv` (~3,000 rows × 28 columns)

**Step 2**: PyTorch tensors and JSON files with preprocessed features
- Images: `*.pt` files (torch.Tensor)
- Structured data: `*.json` files (JSON)
- Text features: `*.pt` files (torch.Tensor)
- Metadata: `*.json` files (JSON)

---

## Step 1 Output: Cohort CSV Schema

### File Location

```
output/cohorts/
├── normal_cohort_full.csv          # Complete cohort (all normal cases)
├── normal_cohort_train.csv         # Training split (85%)
└── normal_cohort_validation.csv    # Validation split (15%)
```

### Column Specification (28 columns)

| Column | Type | Description | Example | Nullable |
|--------|------|-------------|---------|----------|
| `subject_id` | int64 | Patient identifier (MIMIC subject_id) | 10000032 | No |
| `study_id` | int64 | Radiology study identifier | 50414267 | No |
| `dicom_id` | str | DICOM series identifier | cf82ba73-01fe9f97-... | No |
| `study_datetime` | datetime | Study date and time (UTC) | 2180-04-15 03:26:00 | No |
| `ViewPosition` | str | X-ray view position | PA, AP, LATERAL | Yes |
| `image_count` | int64 | Number of images in study | 1, 2 | No |
| `no_finding` | float64 | CheXpert "No Finding" label | 1.0 | Yes |
| `atelectasis` | float64 | CheXpert pathology label | 0.0, 1.0, -1.0, NaN | Yes |
| `cardiomegaly` | float64 | CheXpert pathology label | 0.0, 1.0, -1.0, NaN | Yes |
| `consolidation` | float64 | CheXpert pathology label | 0.0, 1.0, -1.0, NaN | Yes |
| `edema` | float64 | CheXpert pathology label | 0.0, 1.0, -1.0, NaN | Yes |
| `enlarged_cardiomediastinum` | float64 | CheXpert pathology label | 0.0, 1.0, -1.0, NaN | Yes |
| `fracture` | float64 | CheXpert pathology label | 0.0, 1.0, -1.0, NaN | Yes |
| `lung_lesion` | float64 | CheXpert pathology label | 0.0, 1.0, -1.0, NaN | Yes |
| `lung_opacity` | float64 | CheXpert pathology label | 0.0, 1.0, -1.0, NaN | Yes |
| `pleural_effusion` | float64 | CheXpert pathology label | 0.0, 1.0, -1.0, NaN | Yes |
| `pleural_other` | float64 | CheXpert pathology label | 0.0, 1.0, -1.0, NaN | Yes |
| `pneumonia` | float64 | CheXpert pathology label | 0.0, 1.0, -1.0, NaN | Yes |
| `pneumothorax` | float64 | CheXpert pathology label | 0.0, 1.0, -1.0, NaN | Yes |
| `support_devices` | float64 | CheXpert label (lines, tubes) | 0.0, 1.0, -1.0, NaN | Yes |
| `stay_id` | int64 | ED stay identifier | 39824912 | Yes |
| `hadm_id` | int64 | Hospital admission identifier | 22345678 | Yes |
| `ed_intime` | datetime | ED admission time | 2180-04-15 02:15:00 | Yes |
| `ed_outtime` | datetime | ED discharge time | 2180-04-15 08:30:00 | Yes |
| `disposition` | str | ED disposition | HOME, ADMITTED | Yes |
| `anchor_age` | int64 | Patient age at study | 65 | Yes |
| `gender` | str | Patient gender | M, F | Yes |
| `race` | str | Patient race/ethnicity | WHITE, BLACK, ASIAN, ... | Yes |

### CheXpert Label Values

- `1.0`: Positive (finding present)
- `0.0`: Negative (finding absent)
- `-1.0`: Uncertain
- `NaN`: Not mentioned in report

**Normal cohort criteria**:
- `no_finding == 1.0`
- All pathology labels are NOT `1.0` (may be 0.0, -1.0, or NaN)

### ED Disposition Values

Acceptable dispositions (included in cohort):
- `HOME`: Discharged to home
- `DISCHARGED`: General discharge
- `LEFT WITHOUT BEING SEEN`: Left before evaluation
- `LEFT AGAINST MEDICAL ADVICE`: AMA discharge

Excluded dispositions:
- `ADMITTED`: Admitted to hospital
- `EXPIRED`: Died in ED
- `ELOPED`: Left without permission

### Data Types

```python
import pandas as pd

dtype_spec = {
    'subject_id': 'int64',
    'study_id': 'int64',
    'dicom_id': 'str',
    'study_datetime': 'datetime64[ns]',
    'ViewPosition': 'str',
    'image_count': 'int64',
    'no_finding': 'float64',
    # ... pathology labels (float64)
    'stay_id': 'Int64',  # Nullable integer
    'hadm_id': 'Int64',  # Nullable integer
    'ed_intime': 'datetime64[ns]',
    'ed_outtime': 'datetime64[ns]',
    'disposition': 'str',
    'anchor_age': 'Int64',
    'gender': 'str',
    'race': 'str'
}

# Load cohort
cohort = pd.read_csv(
    'output/cohorts/normal_cohort_train.csv',
    dtype=dtype_spec,
    parse_dates=['study_datetime', 'ed_intime', 'ed_outtime']
)
```

---

## Step 2 Output: Multimodal Features

### Directory Structure

```
step2_preprocessing/output/
├── preprocessing.log               # Processing log
├── preprocessing_summary.json      # Summary statistics
│
├── train/
│   ├── images/
│   │   ├── s10000032_study50414267.pt
│   │   ├── s10002764_study51234567.pt
│   │   └── ...
│   ├── structured_features/
│   │   ├── s10000032_study50414267.json
│   │   ├── s10002764_study51234567.json
│   │   └── ...
│   ├── text_features/
│   │   ├── s10000032_study50414267.pt
│   │   ├── s10002764_study51234567.pt
│   │   └── ...
│   ├── metadata/
│   │   ├── s10000032_study50414267.json
│   │   ├── s10002764_study51234567.json
│   │   └── ...
│   └── processing_stats.json
│
└── val/
    └── [same structure as train/]
```

### File Naming Convention

Format: `s{subject_id}_study{study_id}.{ext}`

Example: `s10000032_study50414267.pt`
- Subject ID: 10000032
- Study ID: 50414267

---

## Image Tensor Schema

### File Format

- **Extension**: `.pt` (PyTorch tensor)
- **Size**: ~29 MB per file
- **Tensor Shape**: `[C, H, W]` (Channel, Height, Width)
- **Typical Dimensions**: `[1, 3056, 2544]` for PA view

### Tensor Specification

```python
import torch

image = torch.load('s10000032_study50414267.pt')

# Properties
type(image)              # torch.FloatTensor
image.shape              # torch.Size([1, 3056, 2544])
image.dtype              # torch.float32
image.min()              # 0.0 (with minmax normalization)
image.max()              # 1.0 (with minmax normalization)
image.mean()             # ~0.5 ± 0.15
image.std()              # ~0.2 ± 0.1

# Memory
image.element_size()     # 4 bytes (float32)
image.nelement()         # 3056 * 2544 = 7,774,464
# Total: ~29.6 MB
```

### Normalization Methods

**MinMax Normalization** (default):
```python
# Range: [0.0, 1.0]
normalized = pixel_values / 255.0
```

**Standardization** (z-score):
```python
# Mean ≈ 0, Std ≈ 1
mean = image.mean()
std = image.std()
normalized = (image - mean) / std
```

### Channel Convention

- **C=1**: Grayscale (standard for chest X-rays)
- **C=3**: Converted to RGB (same grayscale repeated)

PyTorch convention: `[C, H, W]` (channels first)

### Loading Example

```python
import torch
from pathlib import Path

# Load image
image_path = Path('output/train/images/s10000032_study50414267.pt')
image = torch.load(image_path)

print(f"Shape: {image.shape}")
print(f"Range: [{image.min():.3f}, {image.max():.3f}]")
print(f"Mean: {image.mean():.3f}, Std: {image.std():.3f}")
print(f"Size: {image.element_size() * image.nelement() / 1024**2:.1f} MB")

# Visualize
import matplotlib.pyplot as plt

plt.imshow(image[0], cmap='gray')
plt.title('Chest X-ray (Full Resolution)')
plt.axis('off')
plt.show()
```

---

## Structured Features Schema

### File Format

- **Extension**: `.json` (JSON)
- **Size**: ~2 KB per file
- **Structure**: Dictionary of feature dictionaries

### Schema Specification

```python
{
  "vital_{vital_name}": {
    "is_missing": bool,
    "measurement_count": int,
    "last_value": float | "NOT_DONE",
    "first_value": float | "NOT_DONE",
    "trend_slope": float,
    "mean_value": float,
    "std_value": float,
    "min_value": float,
    "max_value": float,
    "time_span_hours": float,
    "avg_time_between_measurements": float
  },
  "lab_{lab_name}": {
    # Same structure as vitals
  }
}
```

### Field Descriptions

| Field | Type | Description | Unit |
|-------|------|-------------|------|
| `is_missing` | bool | True if measurement not performed | - |
| `measurement_count` | int | Number of measurements recorded | count |
| `last_value` | float or str | Most recent value (or "NOT_DONE") | varies |
| `first_value` | float or str | First recorded value (or "NOT_DONE") | varies |
| `trend_slope` | float | Change per hour (last - first) / time_span | unit/hour |
| `mean_value` | float | Mean of all measurements | varies |
| `std_value` | float | Standard deviation | varies |
| `min_value` | float | Minimum value | varies |
| `max_value` | float | Maximum value | varies |
| `time_span_hours` | float | Time from first to last measurement | hours |
| `avg_time_between_measurements` | float | Average interval | hours |

### Priority Vitals (11 features)

| Feature Key | Description | Normal Range | Unit |
|-------------|-------------|--------------|------|
| `vital_temperature` | Body temperature | 36.1-37.2 | °C |
| `vital_heartrate` | Heart rate | 60-100 | bpm |
| `vital_resprate` | Respiratory rate | 12-20 | breaths/min |
| `vital_o2sat` | Oxygen saturation | 95-100 | % |
| `vital_sbp` | Systolic blood pressure | 90-120 | mmHg |
| `vital_dbp` | Diastolic blood pressure | 60-80 | mmHg |

### Priority Labs (11 features)

| Feature Key | Description | Normal Range | Unit |
|-------------|-------------|--------------|------|
| `lab_wbc` | White blood cell count | 4.5-11.0 | K/µL |
| `lab_hemoglobin` | Hemoglobin | 12-16 | g/dL |
| `lab_hematocrit` | Hematocrit | 36-48 | % |
| `lab_platelets` | Platelet count | 150-400 | K/µL |
| `lab_sodium` | Sodium | 136-145 | mEq/L |
| `lab_potassium` | Potassium | 3.5-5.0 | mEq/L |
| `lab_chloride` | Chloride | 96-106 | mEq/L |
| `lab_bicarbonate` | Bicarbonate | 22-28 | mEq/L |
| `lab_bun` | Blood urea nitrogen | 7-20 | mg/dL |
| `lab_creatinine` | Creatinine | 0.6-1.2 | mg/dL |
| `lab_glucose` | Glucose | 70-100 | mg/dL |

### NOT_DONE Token

When a measurement is missing:
```json
{
  "lab_hemoglobin": {
    "is_missing": true,
    "measurement_count": 0,
    "last_value": "NOT_DONE",
    "first_value": "NOT_DONE",
    "trend_slope": 0.0,
    "mean_value": 0.0,
    "std_value": 0.0,
    "min_value": 0.0,
    "max_value": 0.0,
    "time_span_hours": 0.0,
    "avg_time_between_measurements": 0.0
  }
}
```

**Important**: `last_value` and `first_value` are strings `"NOT_DONE"`, not numeric!

### Encoding Methods

**Aggregated** (default):
- Fixed-size representation per feature
- Summary statistics + temporal metadata
- Best for: Tabular models, autoencoders

**Sequential** (alternative):
```json
{
  "vital_heartrate": {
    "is_missing": false,
    "sequence": [
      {"value": 82.0, "time_since_start": 0.0, "ordinal_index": 1},
      {"value": 80.0, "time_since_start": 2.5, "ordinal_index": 2},
      {"value": 78.0, "time_since_start": 4.0, "ordinal_index": 3}
    ],
    "length": 3
  }
}
```

Best for: RNNs, Transformers, sequence models

### Loading Example

```python
import json
from pathlib import Path

# Load structured features
features_path = Path('output/train/structured_features/s10000032_study50414267.json')
with open(features_path, 'r') as f:
    features = json.load(f)

# Access vital signs
hr = features['vital_heartrate']
print(f"Heart rate: {hr['last_value']} bpm")
print(f"Trend: {hr['trend_slope']:.2f} bpm/hour")
print(f"Measurements: {hr['measurement_count']}")

# Check for missing labs
hgb = features['lab_hemoglobin']
if hgb['is_missing']:
    print("Hemoglobin: NOT_DONE")
else:
    print(f"Hemoglobin: {hgb['last_value']} g/dL")

# Count missing features
total_features = len(features)
missing_count = sum(1 for f in features.values() if f['is_missing'])
print(f"Missing: {missing_count}/{total_features} ({missing_count/total_features*100:.1f}%)")
```

---

## Text Features Schema

### File Format

- **Extension**: `.pt` (PyTorch tensor/dict)
- **Size**: ~4 KB per file
- **Structure**: Dictionary with summary, tokens, and metadata

### Schema Specification

```python
{
  'summary': str,                    # Claude-generated summary
  'tokens': {
    'input_ids': torch.LongTensor,   # ClinicalBERT token IDs
    'attention_mask': torch.LongTensor,  # Attention mask
    'num_tokens': int,               # Actual token count
    'is_truncated': bool             # True if text was truncated
  },
  'num_entities': int,               # Number of medical entities extracted
  'entities': List[str],             # Entity texts (first 20)
  'context_sentences': int           # Number of sentences retrieved
}
```

### Field Descriptions

| Field | Type | Description |
|-------|------|-------------|
| `summary` | str | Clinical summary (3-5 sentences, ~378 chars) |
| `tokens.input_ids` | LongTensor | ClinicalBERT token IDs, shape [512] |
| `tokens.attention_mask` | LongTensor | Attention mask (1=real, 0=padding), shape [512] |
| `tokens.num_tokens` | int | Actual tokens before padding |
| `tokens.is_truncated` | bool | Whether text exceeded 512 tokens |
| `num_entities` | int | Medical entities found by scispacy |
| `entities` | List[str] | Entity texts (max 20 saved) |
| `context_sentences` | int | Sentences used for summarization |

### Summary Format

Claude-generated summaries follow this structure:
1. Chief complaint and presenting symptoms
2. Relevant medical history (cardiopulmonary focus)
3. Physical exam findings related to chest/lungs
4. Vital signs and lab abnormalities
5. Working diagnosis or clinical concerns

**Example**:
```
Patient is a 65-year-old male presenting with mild shortness of breath for 2 days.
History includes hypertension and diabetes mellitus, both well-controlled on medications.
Physical exam shows clear lung fields bilaterally with normal cardiac exam.
Vital signs are within normal limits: HR 78, BP 120/80, SpO2 98% on room air.
No acute cardiopulmonary process suspected; likely viral upper respiratory infection.
```

### Token Specification

- **Model**: `emilyalsentzer/Bio_ClinicalBERT`
- **Max length**: 512 tokens
- **Padding**: Padded to 512 with zeros
- **Truncation**: Enabled (truncates at 512)

**Token IDs**:
- `0`: PAD token
- `101`: CLS token (start)
- `102`: SEP token (end)
- `103`: UNK token (unknown)

### Entity Types

scispacy `en_core_sci_md` extracts medical/scientific entities:
- Diseases: "pneumonia", "hypertension", "diabetes"
- Symptoms: "chest pain", "shortness of breath", "fever"
- Anatomy: "lungs", "heart", "chest"
- Medications: "aspirin", "lisinopril", "metformin"
- Procedures: "chest x-ray", "ecg", "blood draw"
- Lab values: "white blood cell count", "oxygen saturation"

### Loading Example

```python
import torch
from pathlib import Path

# Load text features
text_path = Path('output/train/text_features/s10000032_study50414267.pt')
text_data = torch.load(text_path)

# Summary
print("Summary:")
print(text_data['summary'])
print()

# Entities
print(f"Entities extracted: {text_data['num_entities']}")
print("Top entities:", text_data['entities'][:5])
print()

# Tokens
tokens = text_data['tokens']
print(f"Token count: {tokens['num_tokens']} / 512")
print(f"Truncated: {tokens['is_truncated']}")
print(f"Input IDs shape: {tokens['input_ids'].shape}")
print(f"Attention mask shape: {tokens['attention_mask'].shape}")

# Decode tokens (requires tokenizer)
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained('emilyalsentzer/Bio_ClinicalBERT')
decoded_text = tokenizer.decode(tokens['input_ids'], skip_special_tokens=True)
print("\nDecoded text:")
print(decoded_text[:200], "...")
```

### Empty Note Handling

When no clinical note is available:
```python
{
  'summary': "",
  'tokens': {
    'input_ids': tensor([101, 102, 0, 0, ..., 0]),  # Only CLS and SEP
    'attention_mask': tensor([1, 1, 0, 0, ..., 0]),
    'num_tokens': 2,
    'is_truncated': False
  },
  'num_entities': 0,
  'entities': [],
  'context_sentences': 0
}
```

---

## Metadata Schema

### File Format

- **Extension**: `.json` (JSON)
- **Size**: ~1 KB per file
- **Structure**: Dictionary with processing metadata

### Schema Specification

```python
{
  'subject_id': int,
  'study_id': int,
  'split': str,                      # 'train' or 'val'
  'study_datetime': str,             # ISO format
  'view_position': str,              # 'PA', 'AP', 'LATERAL'
  'image_count': int,
  'processing_timestamp': str,       # ISO format
  'processing_time_seconds': float,
  'modalities_processed': List[str], # ['image', 'structured', 'text']
  'errors': List[str]                # Any processing errors
}
```

### Loading Example

```python
import json
from pathlib import Path

# Load metadata
metadata_path = Path('output/train/metadata/s10000032_study50414267.json')
with open(metadata_path, 'r') as f:
    metadata = json.load(f)

print(f"Subject ID: {metadata['subject_id']}")
print(f"Study ID: {metadata['study_id']}")
print(f"Study date: {metadata['study_datetime']}")
print(f"View: {metadata['view_position']}")
print(f"Processing time: {metadata['processing_time_seconds']:.2f}s")
print(f"Modalities: {', '.join(metadata['modalities_processed'])}")

if metadata['errors']:
    print("Errors:")
    for error in metadata['errors']:
        print(f"  - {error}")
```

---

## File Relationships and Loading

### Sample ID Mapping

All files for a single sample share the same ID:
```
s{subject_id}_study{study_id}
```

**Example**: Subject 10000032, Study 50414267
- Image: `s10000032_study50414267.pt`
- Structured: `s10000032_study50414267.json`
- Text: `s10000032_study50414267.pt`
- Metadata: `s10000032_study50414267.json`

### Loading Complete Sample

```python
import torch
import json
from pathlib import Path

def load_sample(subject_id: int, study_id: int, split: str = 'train'):
    """Load all modalities for a sample."""
    base_dir = Path(f'output/{split}')
    sample_id = f's{subject_id}_study{study_id}'

    sample = {}

    # Load image
    image_path = base_dir / 'images' / f'{sample_id}.pt'
    if image_path.exists():
        sample['image'] = torch.load(image_path)

    # Load structured features
    structured_path = base_dir / 'structured_features' / f'{sample_id}.json'
    if structured_path.exists():
        with open(structured_path, 'r') as f:
            sample['structured'] = json.load(f)

    # Load text features
    text_path = base_dir / 'text_features' / f'{sample_id}.pt'
    if text_path.exists():
        sample['text'] = torch.load(text_path)

    # Load metadata
    metadata_path = base_dir / 'metadata' / f'{sample_id}.json'
    if metadata_path.exists():
        with open(metadata_path, 'r') as f:
            sample['metadata'] = json.load(f)

    return sample


# Usage
sample = load_sample(10000032, 50414267, split='train')

print(f"Image shape: {sample['image'].shape}")
print(f"Structured features: {len(sample['structured'])} features")
print(f"Text summary length: {len(sample['text']['summary'])} chars")
print(f"Processing time: {sample['metadata']['processing_time_seconds']:.2f}s")
```

### PyTorch Dataset Integration

```python
import pandas as pd
import torch
from torch.utils.data import Dataset

class MIMICDataset(Dataset):
    def __init__(self, cohort_csv: str, output_dir: str, split: str = 'train'):
        self.cohort = pd.read_csv(cohort_csv)
        self.output_dir = Path(output_dir) / split
        self.split = split

    def __len__(self):
        return len(self.cohort)

    def __getitem__(self, idx):
        row = self.cohort.iloc[idx]
        subject_id = int(row['subject_id'])
        study_id = int(row['study_id'])
        sample_id = f's{subject_id}_study{study_id}'

        # Load all modalities
        image = torch.load(self.output_dir / 'images' / f'{sample_id}.pt')

        with open(self.output_dir / 'structured_features' / f'{sample_id}.json') as f:
            structured = json.load(f)

        text = torch.load(self.output_dir / 'text_features' / f'{sample_id}.pt')

        return {
            'image': image,
            'structured': structured,
            'text': text['tokens']['input_ids'],
            'attention_mask': text['tokens']['attention_mask'],
            'metadata': {
                'subject_id': subject_id,
                'study_id': study_id,
                'summary': text['summary']
            }
        }

# Usage
dataset = MIMICDataset(
    cohort_csv='output/cohorts/normal_cohort_train.csv',
    output_dir='step2_preprocessing/output',
    split='train'
)

sample = dataset[0]
```

---

## Example Records

### Example 1: Complete Sample with All Modalities

**Cohort CSV Row**:
```csv
subject_id,study_id,study_datetime,ViewPosition,no_finding,disposition,anchor_age,gender
10000032,50414267,2180-04-15 03:26:00,PA,1.0,HOME,65,M
```

**Image** (`s10000032_study50414267.pt`):
```python
tensor_shape: torch.Size([1, 3056, 2544])
tensor_dtype: torch.float32
tensor_range: [0.0, 1.0]
tensor_mean: 0.487
tensor_std: 0.213
size_mb: 29.6
```

**Structured Features** (`s10000032_study50414267.json`):
```json
{
  "vital_heartrate": {
    "is_missing": false,
    "measurement_count": 3,
    "last_value": 78.0,
    "first_value": 82.0,
    "trend_slope": -1.0,
    "mean_value": 80.0,
    "std_value": 2.0,
    "min_value": 78.0,
    "max_value": 82.0,
    "time_span_hours": 4.0,
    "avg_time_between_measurements": 2.0
  },
  "vital_o2sat": {
    "is_missing": false,
    "measurement_count": 2,
    "last_value": 98.0,
    "first_value": 97.0,
    "trend_slope": 0.5,
    "mean_value": 97.5,
    "std_value": 0.5,
    "min_value": 97.0,
    "max_value": 98.0,
    "time_span_hours": 2.0,
    "avg_time_between_measurements": 2.0
  },
  "lab_hemoglobin": {
    "is_missing": true,
    "measurement_count": 0,
    "last_value": "NOT_DONE",
    "first_value": "NOT_DONE",
    "trend_slope": 0.0,
    "mean_value": 0.0,
    "std_value": 0.0,
    "min_value": 0.0,
    "max_value": 0.0,
    "time_span_hours": 0.0,
    "avg_time_between_measurements": 0.0
  }
}
```

**Text Features** (`s10000032_study50414267.pt`):
```python
{
  'summary': 'Patient is a 65-year-old male presenting with mild shortness of breath for 2 days. History includes hypertension and diabetes mellitus, both well-controlled on medications. Physical exam shows clear lung fields bilaterally with normal cardiac exam. Vital signs are within normal limits: HR 78, BP 120/80, SpO2 98% on room air. No acute cardiopulmonary process suspected; likely viral upper respiratory infection.',

  'tokens': {
    'input_ids': tensor([101, 5970, 2003, 1037, 3515, ..., 0, 0]),  # [512]
    'attention_mask': tensor([1, 1, 1, 1, 1, ..., 0, 0]),            # [512]
    'num_tokens': 87,
    'is_truncated': False
  },

  'num_entities': 13,
  'entities': [
    'shortness of breath',
    'hypertension',
    'diabetes mellitus',
    'lung fields',
    'cardiac exam',
    'vital signs',
    'heart rate',
    'blood pressure',
    'oxygen saturation',
    'cardiopulmonary',
    'viral',
    'upper respiratory infection',
    'physical exam'
  ],

  'context_sentences': 8
}
```

### Example 2: Sample with Missing Modalities

**Scenario**: ED visit without hospital admission (no labs)

**Cohort CSV Row**:
```csv
subject_id,study_id,hadm_id,disposition
10002764,51234567,,HOME
```

**Structured Features** (`s10002764_study51234567.json`):
```json
{
  "vital_heartrate": {
    "is_missing": false,
    "last_value": 85.0,
    // ... (vitals present)
  },
  "lab_wbc": {
    "is_missing": true,
    "last_value": "NOT_DONE",
    // ... (all labs NOT_DONE)
  },
  "lab_hemoglobin": {
    "is_missing": true,
    "last_value": "NOT_DONE"
  }
  // ... all 11 labs will be NOT_DONE
}
```

### Example 3: Empty Clinical Note

**Text Features** (`s10003456_study52345678.pt`):
```python
{
  'summary': "",
  'tokens': {
    'input_ids': tensor([101, 102, 0, 0, ..., 0]),  # Only CLS and SEP
    'attention_mask': tensor([1, 1, 0, 0, ..., 0]),
    'num_tokens': 2,
    'is_truncated': False
  },
  'num_entities': 0,
  'entities': [],
  'context_sentences': 0
}
```

---

## Data Statistics

### Step 1 Output Statistics

**Cohort Size**:
- Total normal cases: ~20,000 studies
- Training set: ~17,000 (85%)
- Validation set: ~3,000 (15%)

**Demographics**:
- Age range: 18-89 years (mean: 52 ± 18)
- Gender: 48% F, 52% M
- View position: 70% PA, 25% AP, 5% LATERAL

**Clinical Context**:
- ED disposition: 100% HOME (by design)
- Hospital admission: 0% (ED-only visits)
- Image count: 65% single view, 35% multiple views

### Step 2 Output Statistics

**Processing Success Rate**: 93.5%

**File Sizes**:
- Images: ~29 MB per file (total: ~580 GB for 20k samples)
- Structured: ~2 KB per file (total: ~40 MB)
- Text: ~4 KB per file (total: ~80 MB)
- Metadata: ~1 KB per file (total: ~20 MB)

**Feature Availability**:
- Images: 98% (2% missing/corrupted)
- Vitals: 60% complete (40% have ≥1 missing vital)
- Labs: 35% complete (65% have ≥1 missing lab, often all missing for ED-only)
- Text: 85% have clinical notes (15% empty)

**Text Statistics**:
- Summary length: 378 ± 120 characters
- Token count: 87 ± 35 tokens
- Entities extracted: 6 ± 3 entities
- Context sentences: 8 ± 4 sentences

---

## See Also

- [Architecture Documentation](ARCHITECTURE.md) - Technical architecture and design
- [Configuration Guide](CONFIGURATION_GUIDE.md) - Configuration options and tuning
- [Main README](../README.md) - Quick start and usage examples
