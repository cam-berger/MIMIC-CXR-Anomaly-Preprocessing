# Pre-compiled Aggregate Dataset Guide

## Overview

The pre-compilation system transforms raw MIMIC data into an optimized, multimodal dataset format that balances:
- **Fast training iteration** (memory-mapped HDF5 for images)
- **Flexible analytics** (columnar Parquet for structured/text data)

### Architecture

```
Raw MIMIC Data Sources
├── MIMIC-CXR-JPG (images)
├── MIMIC-IV (demographics, labs, admissions)
├── MIMIC-IV-ED (vitals, triage)
├── CXR-PRO (radiology reports)
├── MIMIC-IV-Note (clinical notes, optional)
└── MIMIC-IV Medications (prescriptions, optional)
          ↓
    Pre-compilation
    Pipeline
          ↓
Optimized Storage
├── images.h5 (HDF5, ~30MB per image)
├── data.parquet (Parquet, ~5KB per sample)
└── manifest.json (metadata, checksums)
```

## Data Sources

| Source | Content | Size | Required Credentialing |
|--------|---------|------|------------------------|
| **MIMIC-CXR-JPG v2.1.0** | Full-resolution chest X-rays | ~500 GB | Standard |
| **MIMIC-IV v3.1** | Demographics, labs, admissions | ~50 GB | Standard |
| **MIMIC-IV-ED v2.2** | ED vitals, triage, diagnoses | ~5 GB | Standard |
| **CXR-PRO v1.0** | Radiology report impressions (no priors) | 66 MB | Standard |
| **MIMIC-IV-Note** | Clinical notes (discharge, radiology) | ~10 GB | **Additional** |
| **MIMIC-IV Prescriptions** | Medication administration | ~2 GB | Standard |

### Credential Auto-detection

The system automatically detects available data sources:
- If `mimic_iv_note: "auto"` → Uses if available, skips if not
- If `mimic_iv_note: true` → Requires availability (error if missing)
- If `mimic_iv_note: false` → Never uses

## Storage Format

### Hybrid Storage Strategy

**HDF5 for Images** (Memory-mapped, Chunked)
- Format: Hierarchical Data Format 5
- Compression: gzip (configurable)
- Access: Memory-mapped for lazy loading
- Size: ~29.4 MB per full-resolution image (3056×2544 pixels)

**Parquet for Structured/Text** (Columnar, Queryable)
- Format: Apache Parquet
- Compression: gzip/snappy (configurable)
- Access: Fast column projection and filtering
- Size: ~2-5 KB per sample

### Directory Structure

**Single-batch Mode** (local testing):
```
precompiled_dataset/
├── train/
│   ├── batch_0000/
│   │   ├── images.h5           # All training images (~500 GB)
│   │   ├── data.parquet        # All structured/text features (~100 MB)
│   │   └── metadata.json       # Batch metadata
│   └── manifest.json           # Dataset manifest
├── val/
│   └── batch_0000/
│       ├── images.h5           # All validation images
│       └── data.parquet        # All validation structured/text
└── build_summary.txt
```

**Multi-batch Mode** (Lambda deployment):
```
precompiled_dataset/
├── train/
│   ├── batch_0000/             # Samples 0-999
│   │   ├── images.h5           # ~30 GB (1000 images)
│   │   ├── data.parquet        # ~5 MB
│   │   └── metadata.json
│   ├── batch_0001/             # Samples 1000-1999
│   │   ├── images.h5           # ~30 GB
│   │   ├── data.parquet        # ~5 MB
│   │   └── metadata.json
│   ├── ...
│   └── manifest.json           # Global manifest with all batches
└── val/
    └── batch_0000/
        ├── images.h5
        └── data.parquet
```

## Data Schema

### HDF5 Schema (Images)

```
images.h5
├── images/
│   ├── study_50000014
│   │   ├── [dataset: (3, 3056, 2544) float32]
│   │   └── attrs: {shape, dtype, size_bytes, subject_id, study_id}
│   ├── study_50000028
│   └── ...
└── attrs: {batch_id, split, compression, chunk_size, total_samples}
```

### Parquet Schema (Structured/Text)

| Column | Type | Description | Example |
|--------|------|-------------|---------|
| **Identifiers** ||||
| `sample_id` | string | Unique identifier | "study_50000014" |
| `subject_id` | int32 | Patient ID | 11484195 |
| `study_id` | int32 | Study ID | 50000014 |
| `hadm_id` | int32/null | Hospital admission ID | 28293847 |
| `study_datetime` | datetime | CXR study time | "2151-08-23 14:32:00" |
| `ed_intime` | datetime | ED admission time | "2151-08-23 12:15:00" |
| **Vitals** ||||
| `vital_temperature_last` | float | Last temperature (°C) | 36.8 |
| `vital_temperature_mean` | float | Mean temperature | 36.7 |
| `vital_temperature_count` | int | Measurement count | 3 |
| `vital_heartrate_last` | float | Last heart rate (bpm) | 78 |
| `vital_resprate_last` | float | Respiratory rate (breaths/min) | 16 |
| `vital_o2sat_last` | float | Oxygen saturation (%) | 98 |
| `vital_sbp_last` | float | Systolic BP (mmHg) | 120 |
| `vital_dbp_last` | float | Diastolic BP (mmHg) | 80 |
| **Labs** ||||
| `lab_wbc_last` | float/"NOT_DONE" | White blood cell count | 8.5 |
| `lab_hemoglobin_last` | float/"NOT_DONE" | Hemoglobin (g/dL) | 13.2 |
| `lab_platelets_last` | float/"NOT_DONE" | Platelet count | 250 |
| `lab_sodium_last` | float/"NOT_DONE" | Sodium (mEq/L) | 140 |
| `lab_potassium_last` | float/"NOT_DONE" | Potassium (mEq/L) | 4.0 |
| `lab_creatinine_last` | float/"NOT_DONE" | Creatinine (mg/dL) | 0.9 |
| `lab_glucose_last` | float/"NOT_DONE" | Glucose (mg/dL) | 95 |
| **Medications** (if enabled) ||||
| `med_antibiotics_present` | bool | Antibiotics administered | true |
| `med_antibiotics_count` | int | Number of antibiotic orders | 2 |
| `med_diuretics_present` | bool | Diuretics administered | false |
| `med_bronchodilators_present` | bool | Bronchodilators administered | true |
| **Text** ||||
| `text_summary` | string | Claude-generated summary | "Normal chest x-ray..." |
| `text_entity_count` | int | Medical entity count | 5 |
| `text_has_content` | bool | Has text content | true |
| `text_token_ids` | string (JSON) | ClinicalBERT token IDs | "[101, 2345, ...]" |
| **Errors** ||||
| `has_errors` | bool | Processing errors occurred | false |
| `error_count` | int | Number of errors | 0 |
| `errors` | string (JSON) | Error messages | "[]" |

### Missing Value Handling

- **Vitals/Labs**: `"NOT_DONE"` token indicates measurement not performed
- **Medications**: `present=false, count=0` for not administered
- **Text**: Empty string for missing reports
- **Images**: Excluded from dataset if missing (sample not included)

## Configuration

### config.yaml

```yaml
# Pre-compilation settings
precompilation:
  enabled: true
  batch_size: null  # null = single batch, integer = multi-batch
  output_dir: "./precompiled_dataset"

  # Storage configuration
  storage:
    image_format: "hdf5"
    structured_format: "parquet"
    compression: "gzip"  # or "snappy", "lz4", "zstd"
    chunk_size: 100  # Samples per HDF5 chunk

  # Data source toggles
  data_sources:
    mimic_cxr_jpg: true
    mimic_iv: true
    mimic_ed: true
    cxr_pro: true
    mimic_iv_note: "auto"  # auto-detect
    mimic_iv_med: "auto"

  # Temporal window (hours)
  temporal_window:
    before_study_hours: 48  # Extract data from 48h before CXR
    after_study_hours: 24   # Extract data up to 24h after CXR

  # Checkpointing
  checkpoint:
    enabled: true
    save_every_n_samples: 100
    checkpoint_dir: "./checkpoints"

  # Manifest
  manifest:
    include_checksums: true
    include_stats: true
```

## Usage

### 1. Build Pre-compiled Dataset

**Single-batch Mode** (local testing):
```bash
# Build validation set (small, for testing)
python run_precompilation.py \
    --split val \
    --batch-size 0 \
    --output-dir ./precompiled_test

# Build training set (full dataset, single file)
python run_precompilation.py \
    --split train \
    --batch-size 0 \
    --output-dir ./precompiled_full
```

**Multi-batch Mode** (Lambda deployment):
```bash
# Build with 1000 samples per batch
python run_precompilation.py \
    --split train \
    --batch-size 1000 \
    --output-dir ./precompiled_batched

# Build with 5000 samples per batch (larger batches for bigger instances)
python run_precompilation.py \
    --split train \
    --batch-size 5000 \
    --output-dir ./precompiled_batched_5k
```

**Resume from Checkpoint**:
```bash
# Resume interrupted build
python run_precompilation.py \
    --split train \
    --batch-size 1000 \
    --resume
```

### 2. Load Pre-compiled Dataset (Training)

```python
from step2_preprocessing.src.integration.precompiled_dataset import (
    PrecompiledMultimodalDataset,
    precompiled_collate_fn
)
from torch.utils.data import DataLoader

# Initialize dataset
train_dataset = PrecompiledMultimodalDataset(
    data_dir="./precompiled_dataset",
    split="train",
    load_images=True,
    load_structured=True,
    load_text=True,
    cache_images=False  # Set True if enough RAM
)

print(f"Loaded {len(train_dataset)} training samples")

# Create DataLoader
train_loader = DataLoader(
    train_dataset,
    batch_size=4,
    shuffle=True,
    num_workers=4,
    collate_fn=precompiled_collate_fn,
    pin_memory=True
)

# Training loop
for batch in train_loader:
    images = batch['images']  # [B, C, H, W]
    structured = batch['structured']  # Dict of features
    text = batch['text']  # Dict with summaries, tokens

    # Your model training code...
```

### 3. Query Pre-compiled Dataset (Analytics)

```python
import pandas as pd
from step2_preprocessing.src.builders.parquet_writer import ParquetReader

# Initialize reader
reader = ParquetReader([
    "./precompiled_dataset/train/batch_0000/data.parquet",
    "./precompiled_dataset/train/batch_0001/data.parquet",
    # ... add all batches
])

# Query: Find all samples with high WBC
high_wbc = reader.query("lab_wbc_last > 15.0")
print(f"Found {len(high_wbc)} samples with high WBC")

# Query: Patients on antibiotics with pneumonia symptoms
antibiotics = reader.query(
    "med_antibiotics_present == True and vital_temperature_last > 38.0"
)

# Load specific sample
sample = reader.load_sample("study_50000014")
print(f"Sample: {sample['subject_id']}, WBC: {sample['lab_wbc_last']}")

# Or use DuckDB for SQL queries
import duckdb

con = duckdb.connect()
result = con.execute("""
    SELECT subject_id, study_id, vital_temperature_last, lab_wbc_last
    FROM read_parquet('./precompiled_dataset/train/*/data.parquet')
    WHERE vital_temperature_last > 38.0 AND lab_wbc_last > 12.0
    LIMIT 10
""").df()

print(result)
```

## Batching for Lambda Deployment

### Determining Optimal Batch Size

**Per-sample Size Estimates**:
- Image (full-res): ~30 MB
- Structured features: ~2 KB
- Text features: ~3 KB
- **Total per sample**: ~30 MB

**Lambda Instance Storage Examples**:
| Instance Storage | Recommended Batch Size | Total Size |
|------------------|------------------------|------------|
| 50 GB | 1,500 samples | ~45 GB |
| 100 GB | 3,000 samples | ~90 GB |
| 250 GB | 8,000 samples | ~240 GB |
| 500 GB | 16,000 samples | ~480 GB |

### Uploading to Lambda

**Option 1: Transfer Individual Batches**
```bash
# Sync each batch separately
rsync -avz --progress \
    ./precompiled_dataset/train/batch_0000/ \
    lambda:~/mimic_data/precompiled/train/batch_0000/

rsync -avz --progress \
    ./precompiled_dataset/train/batch_0001/ \
    lambda:~/mimic_data/precompiled/train/batch_0001/
```

**Option 2: Create Batch Archives**
```bash
# Create tar archives per batch (with compression)
cd precompiled_dataset/train/
tar -czf batch_0000.tar.gz batch_0000/
tar -czf batch_0001.tar.gz batch_0001/

# Upload archives
scp batch_0000.tar.gz lambda:~/mimic_data/
ssh lambda "cd ~/mimic_data && tar -xzf batch_0000.tar.gz"
```

## Performance Benchmarks

### Build Time

| Dataset | Samples | Mode | Time | Throughput |
|---------|---------|------|------|------------|
| Validation | 200 | Single-batch | ~5 min | 40 samples/min |
| Validation | 3,000 | Single-batch | ~2 hours | 25 samples/min |
| Training | 17,000 | Multi-batch (1000) | ~12 hours | 24 samples/min |

*Note: With text summarization enabled, add ~2-3 seconds per sample for API calls*

### Loading Speed

| Operation | Speed | Notes |
|-----------|-------|-------|
| **HDF5 Memory-mapped Load** | ~0.05s per image | No decompression |
| **Parquet Row Load** | ~0.001s per row | Columnar projection |
| **DataLoader (4 workers)** | ~100 samples/sec | Batched loading |
| **Full Cohort Analytics Query** | ~2-5s | 20K samples, Parquet |

## Testing

### Run Test Suite

```bash
# Test all components
python test_precompilation.py --test all --num-samples 10

# Test only single-batch build
python test_precompilation.py --test single --num-samples 20

# Test multi-batch mode
python test_precompilation.py --test multi --num-samples 50

# Clean up after tests
python test_precompilation.py --test all --cleanup
```

### Manual Validation

```bash
# 1. Verify CXR-PRO integration
python step2_preprocessing/test_cxr_pro_integration.py

# 2. Build small test dataset
python run_precompilation.py --split val --batch-size 0 --output-dir ./test_output

# 3. Load and inspect
python -c "
from step2_preprocessing.src.integration.precompiled_dataset import PrecompiledMultimodalDataset
ds = PrecompiledMultimodalDataset('./test_output', 'val')
print(f'Loaded {len(ds)} samples')
print(ds.get_statistics())
sample = ds[0]
print(f'Sample: {sample.keys()}')
"
```

## Troubleshooting

### Common Issues

**1. "Manifest not found"**
- Ensure build completed successfully
- Check `manifest.json` exists in split directory

**2. "HDF5 file not found"**
- Verify all batches were created during build
- Check file paths in manifest match actual files

**3. "Out of memory during build"**
- Reduce batch size
- Disable image caching
- Process in smaller chunks

**4. "Missing data sources"**
- Run credential check: `check_credential_availability()`
- Set `mimic_iv_note: "auto"` for optional sources

**5. "Slow loading during training"**
- Increase DataLoader `num_workers`
- Enable `pin_memory=True`
- Use larger batch sizes

## Future Enhancements

- [ ] Support for additional modalities (ECG, lab trends)
- [ ] On-the-fly augmentation hooks
- [ ] Streaming mode for very large datasets
- [ ] Integration with MLflow for experiment tracking
- [ ] Automatic batch size optimization based on instance type

## References

- [MIMIC-CXR-JPG Documentation](https://physionet.org/content/mimic-cxr-jpg/2.0.0/)
- [CXR-PRO Paper](https://physionet.org/content/cxr-pro/1.0.0/)
- [HDF5 Documentation](https://docs.h5py.org/)
- [Apache Parquet Documentation](https://parquet.apache.org/)
- [PyTorch DataLoader Guide](https://pytorch.org/tutorials/beginner/data_loading_tutorial.html)
