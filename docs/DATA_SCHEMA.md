# Preprocessed Data Schema

This document describes the output format of the MIMIC-CXR preprocessing pipeline. The preprocessing transforms raw MIMIC datasets into a unified, MAE-ready format optimized for multimodal machine learning.

## Output Directory Structure

```
output/preprocessed/{cohort_name}/
├── images.h5              # Chest X-ray images (HDF5)
├── structured.parquet     # Demographics, vitals, labs
├── text.parquet           # Reports and summaries
├── image_results.parquet  # Image processing status log
└── manifest.json          # Processing statistics
```

Where `{cohort_name}` is one of:
- `normal_train` - Training split of normal CXR studies
- `normal_val` - Validation split of normal CXR studies
- `anomalous_train` - Training split of anomalous CXR studies
- `anomalous_val` - Validation split of anomalous CXR studies

---

## File Formats

### 1. images.h5 (HDF5)

The primary image storage file using HDF5 format for efficient random access and compression.

#### Structure
```
/images/{idx}     - Image tensor [C, H, W] as float32
/metadata/{idx}   - JSON metadata string
/index            - Parquet bytes mapping study_id -> idx
```

#### Image Format
| Property | Value |
|----------|-------|
| Shape | `[1, H, W]` (grayscale with channel dim) |
| Dtype | `float32` |
| Normalization | Min-max scaled to [0, 1] |
| Compression | gzip |
| Resolution | Full resolution (variable, e.g., 2544x3056) |

#### Metadata (per image)
```json
{
  "study_id": 58119947,
  "subject_id": 16890260,
  "shape": [1, 2544, 3056],
  "image_path": "/path/to/original/jpeg"
}
```

#### Index
Embedded Parquet DataFrame mapping `study_id` to HDF5 index:
| Column | Type | Description |
|--------|------|-------------|
| `idx` | int64 | HDF5 dataset index |
| `study_id` | int64 | CXR study identifier |
| `subject_id` | int64 | Patient identifier |

#### Loading Example
```python
import h5py
import io
import pandas as pd

with h5py.File('images.h5', 'r') as f:
    # Load index
    index_bytes = f['index'][:]
    index_df = pd.read_parquet(io.BytesIO(bytes(index_bytes)))

    # Load specific image by study_id
    study_id = 58119947
    idx = index_df[index_df['study_id'] == study_id]['idx'].iloc[0]
    image = f['images'][str(idx)][:]  # Shape: [1, H, W]

    # Load metadata
    metadata = json.loads(f['metadata'][str(idx)][()])
```

---

### 2. structured.parquet

Clinical structured data including demographics, triage vitals, ED vitals, and lab values.

#### Schema

##### Keys
| Column | Type | Description |
|--------|------|-------------|
| `subject_id` | int32 | Patient identifier |
| `study_id` | int64 | CXR study identifier |

##### Demographics
| Column | Type | Description |
|--------|------|-------------|
| `age` | float64 | Patient age (anchor_age from MIMIC-IV) |
| `gender` | string | "M" or "F" |
| `gender_M` | int8 | Binary gender encoding (1=Male, 0=Female) |

##### Triage Vitals (ED arrival)
| Column | Type | Unit | Description |
|--------|------|------|-------------|
| `triage_temperature` | float32 | °F | Body temperature |
| `triage_heartrate` | float32 | bpm | Heart rate |
| `triage_resprate` | float32 | /min | Respiratory rate |
| `triage_o2sat` | float32 | % | Oxygen saturation |
| `triage_sbp` | float32 | mmHg | Systolic blood pressure |
| `triage_dbp` | float32 | mmHg | Diastolic blood pressure |
| `triage_pain` | string | 0-10 | Pain score (string for special values) |
| `triage_acuity` | int8 | 1-5 | ESI triage acuity level |

##### ED Vitals (aggregated over stay)
For each vital sign: `{vital}_mean`, `{vital}_min`, `{vital}_max`, `{vital}_std`, `{vital}_count`

Vitals: `temperature`, `heartrate`, `resprate`, `o2sat`, `sbp`, `dbp`

| Column Pattern | Type | Description |
|----------------|------|-------------|
| `{vital}_mean` | float32 | Mean value during ED stay |
| `{vital}_min` | float32 | Minimum value |
| `{vital}_max` | float32 | Maximum value |
| `{vital}_std` | float64 | Standard deviation |
| `{vital}_count` | float64 | Number of measurements |

##### Lab Values (within temporal window)
For each lab: `lab_{test}_mean`, `lab_{test}_min`, `lab_{test}_max`, `lab_{test}_count`

| Lab Test | Description |
|----------|-------------|
| `bicarbonate` | Serum bicarbonate |
| `bnp` | B-type natriuretic peptide |
| `bun` | Blood urea nitrogen |
| `calcium` | Serum calcium |
| `chloride` | Serum chloride |
| `creatinine` | Serum creatinine |
| `glucose` | Blood glucose |
| `hematocrit` | Hematocrit |
| `hemoglobin` | Hemoglobin |
| `lactate` | Lactate level |
| `magnesium` | Serum magnesium |
| `platelets` | Platelet count |
| `potassium` | Serum potassium |
| `procalcitonin` | Procalcitonin |
| `sodium` | Serum sodium |
| `troponin` | Troponin (cardiac marker) |
| `wbc` | White blood cell count |

##### Diagnosis/Procedure Counts
| Column | Type | Description |
|--------|------|-------------|
| `num_ed_diagnoses` | int64 | Count of ED ICD diagnoses |
| `num_hospital_diagnoses` | int64 | Count of hospital ICD diagnoses |
| `num_procedures` | int64 | Count of ICD procedures |

##### Availability Flags
| Column | Type | Description |
|--------|------|-------------|
| `has_triage` | bool | Has any triage vital data |
| `has_labs` | bool | Has any lab values |
| `has_ed_vitals` | bool | Has any ED vital sign data |

---

### 3. text.parquet

Radiology reports and Claude-generated summaries with clinical context.

#### Schema
| Column | Type | Description |
|--------|------|-------------|
| `subject_id` | int32 | Patient identifier |
| `study_id` | int64 | CXR study identifier |
| `report` | string | Original radiology report (from CXR-PRO) |
| `report_clean` | string | Cleaned/normalized report text |
| `clinical_context` | string | Formatted clinical context for summarization |
| `summary` | string | Claude-generated clinical summary |
| `tokens` | string | Comma-separated ClinicalBERT token IDs |
| `token_count` | int64 | Number of tokens |
| `has_report` | bool | Report availability flag |

#### Clinical Context Format
The `clinical_context` field contains a structured summary of clinical data:

```
Patient: 65 year old male
Chief complaint: chest pain
Vitals: Temp 98.6°F, HR 88bpm, RR 18/min, SpO2 98%, SBP 140mmHg, DBP 85mmHg
Triage acuity: 3 (Urgent)
Labs: WBC 8.5, Hgb 14.2, Cr 1.1, Glucose 110
ED diagnoses: R07.9, I25.10
Disposition: HOME
```

#### Summary Generation
Summaries are generated using Claude API with the following prompt structure:
- Includes clinical context (demographics, vitals, labs, diagnoses)
- Synthesizes imaging findings with clinical presentation
- Highlights correlations or discordances
- 2-3 sentences, clinically focused

---

### 4. image_results.parquet

Log of image processing outcomes for QA and debugging.

#### Schema
| Column | Type | Description |
|--------|------|-------------|
| `study_id` | int64 | CXR study identifier |
| `subject_id` | int64 | Patient identifier |
| `success` | bool | Processing succeeded |
| `error` | string | Error message (if failed) |
| `shape` | string | Image shape as string (e.g., "[1, 2544, 3056]") |

#### Common Error Types
- `"No image found"` - Study directory missing or empty
- `"Failed to load"` - Image file corrupted or unreadable

---

### 5. manifest.json

Processing statistics and metadata.

#### Schema
```json
{
  "cohort_name": "normal_val",
  "cohort_path": "/path/to/cohort.parquet",
  "total_samples": 5190,
  "start_time": "2025-11-25T04:10:43.959742",
  "end_time": "2025-11-25T13:27:02.043973",
  "duration_seconds": 33378.08,
  "samples_per_second": 0.155,
  "modalities": {
    "images": {
      "output_path": "/path/to/images.h5",
      "total": 5190,
      "success": 4285,
      "failed": 905,
      "success_rate": 0.826
    },
    "structured": {
      "output_path": "/path/to/structured.parquet",
      "total": 5190,
      "with_triage": 5083,
      "with_labs": 4787,
      "with_ed_vitals": 5040
    },
    "text": {
      "output_path": "/path/to/text.parquet",
      "total": 5190,
      "with_report": 5087,
      "avg_token_count": 129.08,
      "summarization_enabled": true,
      "include_context": true
    }
  }
}
```

---

## Data Linking

All files can be joined on `study_id` (primary key) or `subject_id` (patient key):

```python
import pandas as pd
import h5py
import io

# Load all modalities
structured = pd.read_parquet('structured.parquet')
text = pd.read_parquet('text.parquet')

# Join structured + text
combined = structured.merge(text, on=['study_id', 'subject_id'])

# Get image for specific study
with h5py.File('images.h5', 'r') as f:
    index_bytes = f['index'][:]
    index_df = pd.read_parquet(io.BytesIO(bytes(index_bytes)))

    for study_id in combined['study_id'].head(10):
        idx_row = index_df[index_df['study_id'] == study_id]
        if len(idx_row) > 0:
            idx = idx_row['idx'].iloc[0]
            image = f['images'][str(idx)][:]
```

---

## Dataset Statistics (Normal Validation Example)

| Metric | Value |
|--------|-------|
| Total samples | 5,190 |
| Successful images | 4,285 (82.6%) |
| With triage vitals | 5,083 (97.9%) |
| With lab values | 4,787 (92.2%) |
| With ED vitals | 5,040 (97.1%) |
| With radiology report | 5,087 (98.0%) |
| Average token count | 129 tokens |
| Total image file size | ~23 GB |

---

## MAE Training Integration

For Masked Autoencoder training, the preprocessed data integrates as follows:

1. **Image Loading**: Load from HDF5 with lazy access for memory efficiency
2. **Patch Embedding**: Images at full resolution, patch into 16x16 or 32x32 patches
3. **Clinical Features**: Concatenate normalized structured features as auxiliary input
4. **Text Encoding**: Use pre-tokenized ClinicalBERT tokens or summary embeddings

```python
class PreprocessedMAEDataset(torch.utils.data.Dataset):
    def __init__(self, preprocessed_dir):
        self.structured = pd.read_parquet(preprocessed_dir / 'structured.parquet')
        self.text = pd.read_parquet(preprocessed_dir / 'text.parquet')
        self.hdf5 = h5py.File(preprocessed_dir / 'images.h5', 'r')

        # Build study_id -> idx mapping
        index_bytes = self.hdf5['index'][:]
        self.index = pd.read_parquet(io.BytesIO(bytes(index_bytes)))
        self.index = self.index.set_index('study_id')

    def __getitem__(self, idx):
        row = self.structured.iloc[idx]
        study_id = row['study_id']

        # Get image
        hdf5_idx = self.index.loc[study_id, 'idx']
        image = self.hdf5['images'][str(hdf5_idx)][:]

        # Get text features
        text_row = self.text[self.text['study_id'] == study_id].iloc[0]

        return {
            'image': torch.from_numpy(image),
            'structured': self._extract_features(row),
            'text': text_row['summary'],
            'study_id': study_id,
        }
```

---

## Version History

| Date | Version | Changes |
|------|---------|---------|
| 2025-11-25 | 1.0 | Initial schema documentation |