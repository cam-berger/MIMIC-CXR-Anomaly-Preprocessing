# Pre-Deployment Testing Checklist for Lambda

This checklist ensures the preprocessing pipeline is production-ready before deploying to Lambda GPU instances.

## ✅ Testing Phases

### Phase 1: Unit Testing (Individual Components) ⏱️ ~30 minutes

Test each processor in isolation to catch bugs early.

#### 1.1 Image Processing
```bash
cd step2_preprocessing
pytest tests/unit/test_image_loader.py -v

# What this validates:
# - Full-resolution loading works
# - Normalization is correct (minmax/z-score)
# - Image shapes are preserved
# - Memory usage is reasonable
```

**Expected Result**: All tests pass, images load at ~3000×2500 px

#### 1.2 Text Processing
```bash
pytest tests/unit/test_note_processor.py -v

# What this validates:
# - NER extracts medical entities correctly
# - Retrieval finds relevant sentences
# - Claude summarization works (requires API key)
# - ClinicalBERT tokenization works
# - Fallback behavior when API fails
```

**Expected Result**: All tests pass, summaries are coherent

#### 1.3 Structured Data Processing
```bash
pytest tests/unit/test_temporal_processor.py -v

# What this validates:
# - Labs/vitals extracted correctly
# - Temporal aggregation (last, mean, trend, count)
# - NOT_DONE token for missing values
# - Time window filtering (-48h to +24h)
```

**Expected Result**: All tests pass, temporal features computed correctly

#### 1.4 Multimodal Dataset
```bash
pytest tests/unit/test_multimodal_dataset.py -v

# What this validates:
# - PyTorch Dataset interface works
# - Batch collation is correct
# - All three modalities load together
```

**Expected Result**: All tests pass, batches are correctly formatted

---

### Phase 2: Integration Testing (Full Pipeline) ⏱️ ~1-2 hours

Test the complete pipeline with real data at small scale.

#### 2.1 Small-Scale End-to-End Test (10 samples)
```bash
# Test with 10 samples from validation cohort
cd step2_preprocessing
python main.py \
  --max-samples 10 \
  --output-dir ./output_test_10 \
  --val-only

# What this validates:
# - All three modalities process successfully
# - File outputs are created correctly
# - Error handling works
# - Processing time per sample
```

**Expected Outputs**:
```
output_test_10/
├── val/
│   ├── images/           # 10 .pt files (~294 MB)
│   ├── text_features/    # 10 .pt files (~100 KB)
│   ├── structured_features/  # 10 .json files (~20 KB)
│   └── metadata/         # 10 .json files (~10 KB)
└── preprocessing.log
```

**Manual Inspection**:
```bash
# Check image shape
python -c "
import torch
img = torch.load('output_test_10/val/images/[first_file].pt')
print(f'Image shape: {img.shape}')
print(f'Image dtype: {img.dtype}')
print(f'Image range: [{img.min():.3f}, {img.max():.3f}]')
"

# Check text features
python -c "
import torch
text = torch.load('output_test_10/val/text_features/[first_file].pt')
print(f'Summary: {text[\"summary\"][:200]}...')
print(f'Num entities: {text[\"num_entities\"]}')
print(f'Num tokens: {text[\"tokens\"][\"num_tokens\"]}')
"

# Check structured features
python -c "
import json
with open('output_test_10/val/structured_features/[first_file].json') as f:
    data = json.load(f)
print('Labs available:', [k for k, v in data['labs'].items() if not v['is_missing']])
print('Vitals available:', [k for k, v in data['vitals'].items() if not v['is_missing']])
"
```

**Success Criteria**:
- ✅ 10/10 samples process successfully
- ✅ Images are full resolution (~3000×2500)
- ✅ Text summaries are coherent and relevant
- ✅ Structured features have reasonable values
- ✅ Processing time: 1.5-6.5s per sample

#### 2.2 CXR-PRO Integration Test
```bash
# Test CXR-PRO radiology report loading
python step2_preprocessing/test_cxr_pro_integration.py

# What this validates:
# - CXR-PRO reports load correctly
# - Study-level aggregation works
# - Coverage is >95% on validation cohort
# - No hallucinated prior references
```

**Expected Result**:
- Coverage: >95% (e.g., 198/200 samples have reports)
- No errors or exceptions

#### 2.3 Pipeline Stages Test
```bash
# Test each stage of the pipeline individually
python test_pipeline_stages.py

# What this validates:
# - Stage 1: CXR-PRO loading
# - Stage 2: MIMIC-IV data loading
# - Stage 3: Image processing
# - Stage 4: Structured data processing
# - Stage 5: Text processing
# - Stage 6: HDF5 writing
# - Stage 7: Parquet writing
# - Stage 8: Full pipeline integration
```

**Expected Result**: All 8 stages complete successfully with detailed output

---

### Phase 3: Pre-compilation Testing ⏱️ ~2-4 hours

Test the optimized pre-compilation pipeline that will be used on Lambda.

#### 3.1 Small-Scale Pre-compilation (10 samples)
```bash
# Build pre-compiled dataset with 10 samples
python test_precompilation.py --num-samples 10

# Or use the full script:
python run_precompilation.py \
  --cohort step2_preprocessing/cohorts/validation_subset_200.csv \
  --output-dir ./precompiled_test \
  --split val \
  --batch-size 0 \
  --max-samples 10

# What this validates:
# - HDF5 writing works (memory-mapped images)
# - Parquet writing works (structured/text features)
# - Manifest generation is correct
# - Checksums are computed
```

**Expected Outputs**:
```
precompiled_test/
├── test/
│   ├── batch_0/
│   │   ├── images.h5          # ~300 MB for 10 samples
│   │   └── data.parquet        # ~50 KB
│   └── manifest.json
└── checkpoints/
    └── checkpoint_test.json
```

**Manual Validation**:
```bash
# Inspect HDF5 file
python -c "
import h5py
with h5py.File('precompiled_test/val/batch_0/images.h5', 'r') as f:
    print('Dataset shape:', f['images'].shape)
    print('Dataset dtype:', f['images'].dtype)
    print('Compression:', f['images'].compression)
    print('Chunk size:', f['images'].chunks)
"

# Inspect Parquet file
python -c "
import pandas as pd
df = pd.read_parquet('precompiled_test/val/batch_0/data.parquet')
print('Columns:', df.columns.tolist())
print('Shape:', df.shape)
print('Sample IDs:', df['sample_id'].tolist())
print('Completeness:', df['has_image'].sum(), 'images,', df['has_text'].sum(), 'text,', df['has_structured'].sum(), 'structured')
"
```

**Success Criteria**:
- ✅ HDF5 file created with correct shape
- ✅ Parquet file has all expected columns
- ✅ Manifest.json has correct metadata
- ✅ All 10 samples have complete data

#### 3.2 Pre-compiled Dataset Loading Test
```bash
# Test loading from pre-compiled dataset
python -c "
import sys
from pathlib import Path
sys.path.insert(0, str(Path('step2_preprocessing/src')))

from integration.precompiled_dataset import PrecompiledMultimodalDataset
from torch.utils.data import DataLoader

# Load dataset
dataset = PrecompiledMultimodalDataset(
    precompiled_dir='precompiled_test/val',
    config=None  # Will use default
)

print(f'Dataset length: {len(dataset)}')

# Test __getitem__
sample = dataset[0]
print(f'Sample keys: {sample.keys()}')
print(f'Image shape: {sample[\"image\"].shape}')
print(f'Text keys: {sample[\"text\"].keys()}')
print(f'Structured keys: {sample[\"structured\"].keys()}')

# Test DataLoader
loader = DataLoader(dataset, batch_size=4, shuffle=False)
batch = next(iter(loader))
print(f'Batch image shape: {batch[\"image\"].shape}')
print(f'Batch size: {len(batch[\"metadata\"][\"sample_id\"])}')
"

# What this validates:
# - Dataset can be instantiated
# - __getitem__ returns correct format
# - DataLoader batching works
# - Memory-mapped loading is fast
```

**Expected Result**: All operations succeed, loading is fast (<0.1s per sample)

#### 3.3 Medium-Scale Pre-compilation (100 samples)
```bash
# Build with 100 samples to test at scale
python run_precompilation.py \
  --cohort step2_preprocessing/cohorts/validation_subset_200.csv \
  --output-dir ./precompiled_100 \
  --split val \
  --batch-size 0 \
  --max-samples 100

# What this validates:
# - Performance at scale
# - Memory usage stays reasonable
# - No memory leaks
# - Checkpoint/resume works
```

**Success Criteria**:
- ✅ 100/100 samples process successfully
- ✅ Processing time: 150-650s total (1.5-6.5s per sample)
- ✅ Memory usage: <16 GB peak
- ✅ HDF5 file size: ~3 GB
- ✅ Parquet file size: ~500 KB

**Test Checkpoint/Resume**:
```bash
# Kill the process halfway through (Ctrl+C)
python run_precompilation.py \
  --cohort step2_preprocessing/cohorts/validation_subset_200.csv \
  --output-dir ./precompiled_100_resume \
  --split val \
  --batch-size 0 \
  --max-samples 100

# Resume from checkpoint
python run_precompilation.py \
  --cohort step2_preprocessing/cohorts/validation_subset_200.csv \
  --output-dir ./precompiled_100_resume \
  --split val \
  --batch-size 0 \
  --max-samples 100 \
  --resume
```

**Expected Result**: Resumes from last checkpoint, doesn't reprocess completed samples

---

### Phase 4: Data Quality Validation ⏱️ ~30 minutes

Validate the quality and completeness of processed data.

#### 4.1 Completeness Check
```bash
# Check how many samples have all three modalities
python -c "
import pandas as pd

# Load parquet
df = pd.read_parquet('precompiled_test/test/batch_0/data.parquet')

# Check completeness
has_all = df['has_image'] & df['has_text'] & df['has_structured']
print(f'Samples with all modalities: {has_all.sum()}/{len(df)} ({100*has_all.mean():.1f}%)')

# Check per-modality
print(f'Has image: {df[\"has_image\"].sum()}/{len(df)} ({100*df[\"has_image\"].mean():.1f}%)')
print(f'Has text: {df[\"has_text\"].sum()}/{len(df)} ({100*df[\"has_text\"].mean():.1f}%)')
print(f'Has structured: {df[\"has_structured\"].sum()}/{len(df)} ({100*df[\"has_structured\"].mean():.1f}%)')

# Check missing reasons
print('\\nMissing structured data reasons:')
print(df[~df['has_structured']].groupby('structured_missing_reason').size())
"
```

**Success Criteria**:
- ✅ >95% of samples have all three modalities
- ✅ 100% have images (should always be available)
- ✅ >95% have text (CXR-PRO coverage)
- ✅ >90% have structured data (vitals always available, labs may be missing)

#### 4.2 Data Range Validation
```bash
# Check that data values are in expected ranges
python -c "
import pandas as pd
import h5py
import numpy as np

# Check image ranges
with h5py.File('precompiled_test/val/batch_0/images.h5', 'r') as f:
    images = f['images'][:]
    print(f'Image range: [{images.min():.3f}, {images.max():.3f}]')
    print(f'Image mean: {images.mean():.3f}')
    print(f'Image std: {images.std():.3f}')
    # Should be [0, 1] for minmax normalization

# Check structured data ranges
df = pd.read_parquet('precompiled_test/val/batch_0/data.parquet')

# Heartrate should be 40-200 bpm
hr_vals = df[df['heartrate_last'] != 'NOT_DONE']['heartrate_last'].astype(float)
print(f'\\nHeartrate range: [{hr_vals.min():.1f}, {hr_vals.max():.1f}]')
if hr_vals.min() < 30 or hr_vals.max() > 250:
    print('⚠️  WARNING: Heartrate values out of expected range!')

# Temperature should be 95-105 F
temp_vals = df[df['temperature_last'] != 'NOT_DONE']['temperature_last'].astype(float)
print(f'Temperature range: [{temp_vals.min():.1f}, {temp_vals.max():.1f}]')
if temp_vals.min() < 90 or temp_vals.max() > 110:
    print('⚠️  WARNING: Temperature values out of expected range!')
"
```

**Success Criteria**:
- ✅ Images in [0, 1] range (minmax normalization)
- ✅ Vitals in physiologically reasonable ranges
- ✅ No NaN or Inf values
- ✅ No extreme outliers

#### 4.3 Text Quality Validation
```bash
# Manually review a few summaries
python -c "
import pandas as pd
df = pd.read_parquet('precompiled_test/val/batch_0/data.parquet')

print('=== Sample Text Summaries ===')
for i, row in df.head(5).iterrows():
    print(f'\\n--- Sample {i+1}: {row[\"sample_id\"]} ---')
    print(f'Summary ({row[\"summary_length\"]} chars):')
    print(row['summary'])
    print(f'Entities: {row[\"entity_count\"]}')
    print(f'Tokens: {row[\"token_count\"]}')
"
```

**Manual Checklist**:
- ✅ Summaries are coherent and medically accurate
- ✅ Summaries focus on chest/cardiopulmonary findings
- ✅ No hallucinated information
- ✅ Entities are extracted correctly
- ✅ Token counts are reasonable (not all CLS/SEP)

---

### Phase 5: Performance Benchmarking ⏱️ ~1 hour

Measure performance to estimate Lambda costs and timing.

#### 5.1 Data Loading Speed
```bash
# Benchmark pre-compiled vs on-demand loading
python -c "
import time
import sys
from pathlib import Path
sys.path.insert(0, str(Path('step2_preprocessing/src')))

from integration.precompiled_dataset import PrecompiledMultimodalDataset
from torch.utils.data import DataLoader

# Test pre-compiled loading
dataset = PrecompiledMultimodalDataset('precompiled_test/val')
loader = DataLoader(dataset, batch_size=4, num_workers=0)

# Time 10 batches
start = time.time()
for i, batch in enumerate(loader):
    if i >= 10:
        break
    pass
elapsed = time.time() - start

print(f'Pre-compiled loading: {elapsed:.2f}s for 10 batches')
print(f'Per-sample: {elapsed/(10*4):.3f}s')
print(f'Expected: 0.05-0.1s per sample (30-130x faster than on-demand)')
"
```

**Success Criteria**:
- ✅ Loading speed: 0.05-0.1s per sample
- ✅ At least 15x faster than on-demand (1.5-6.5s per sample)

#### 5.2 Memory Usage
```bash
# Monitor memory during loading
python -c "
import psutil
import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path('step2_preprocessing/src')))

from integration.precompiled_dataset import PrecompiledMultimodalDataset
from torch.utils.data import DataLoader

process = psutil.Process(os.getpid())
mem_before = process.memory_info().rss / 1024**3

# Load dataset
dataset = PrecompiledMultimodalDataset('precompiled_100/val')
loader = DataLoader(dataset, batch_size=4, num_workers=4)

# Iterate through all batches
for batch in loader:
    pass

mem_after = process.memory_info().rss / 1024**3

print(f'Memory before: {mem_before:.2f} GB')
print(f'Memory after: {mem_after:.2f} GB')
print(f'Memory increase: {mem_after - mem_before:.2f} GB')
print(f'Expected: <4 GB for 100 samples')
"
```

**Success Criteria**:
- ✅ Memory usage: <4 GB for 100 samples
- ✅ No memory leaks (memory stabilizes after first epoch)

---

### Phase 6: Lambda Deployment Readiness ⏱️ ~30 minutes

Final checks before deploying to Lambda.

#### 6.1 Multi-Batch Mode Test
```bash
# Test multi-batch mode (Lambda deployment scenario)
python run_precompilation.py \
  --cohort step2_preprocessing/cohorts/validation_subset_200.csv \
  --output-dir ./precompiled_batched \
  --split val \
  --batch-size 50 \
  --max-samples 200

# What this validates:
# - Multiple batches created correctly
# - Manifest includes all batches
# - Each batch is self-contained
# - Batch loading works independently
```

**Expected Outputs**:
```
precompiled_batched/
└── val/
    ├── batch_0/  # 50 samples
    ├── batch_1/  # 50 samples
    ├── batch_2/  # 50 samples
    ├── batch_3/  # 50 samples
    └── manifest.json
```

**Validate Manifest**:
```bash
python -c "
import json
with open('precompiled_batched/val/manifest.json') as f:
    manifest = json.load(f)

print(f'Total batches: {manifest[\"num_batches\"]}')
print(f'Total samples: {manifest[\"total_samples\"]}')
for batch in manifest['batches']:
    print(f'  Batch {batch[\"batch_id\"]}: {batch[\"num_samples\"]} samples')
    print(f'    HDF5: {batch[\"hdf5_size_mb\"]:.1f} MB')
    print(f'    Parquet: {batch[\"parquet_size_mb\"]:.2f} MB')
"
```

#### 6.2 Configuration Validation
```bash
# Verify configuration is production-ready
python -c "
import yaml
with open('step2_preprocessing/config/config.yaml') as f:
    config = yaml.safe_load(f)

print('=== Configuration Validation ===')
print(f'✓ Claude model: {config[\"text\"][\"summarization\"][\"model\"]}')
print(f'✓ Image resolution: {\"full\" if config[\"image\"][\"preserve_full_resolution\"] else \"downsampled\"}')
print(f'✓ Missing token: {config[\"structured\"][\"missing_token\"]}')
print(f'✓ Pre-compilation enabled: {config[\"precompilation\"][\"enabled\"]}')
print(f'✓ Storage format: HDF5={config[\"precompilation\"][\"storage\"][\"image_format\"]}, Parquet={config[\"precompilation\"][\"storage\"][\"structured_format\"]}')
print(f'✓ CXR-PRO enabled: {config[\"precompilation\"][\"data_sources\"][\"cxr_pro\"]}')
print(f'✓ Checkpoint enabled: {config[\"precompilation\"][\"checkpoint\"][\"enabled\"]}')
"
```

#### 6.3 Dependencies Check
```bash
# Verify all dependencies are installed
python -c "
import sys
missing = []

# Core dependencies
try:
    import torch
    print(f'✓ PyTorch: {torch.__version__}')
except ImportError:
    missing.append('torch')

try:
    import h5py
    print(f'✓ h5py: {h5py.__version__}')
except ImportError:
    missing.append('h5py')

try:
    import pyarrow
    print(f'✓ pyarrow: {pyarrow.__version__}')
except ImportError:
    missing.append('pyarrow')

try:
    import pandas as pd
    print(f'✓ pandas: {pd.__version__}')
except ImportError:
    missing.append('pandas')

try:
    import spacy
    print(f'✓ spacy: {spacy.__version__}')
except ImportError:
    missing.append('spacy')

try:
    from langchain_anthropic import ChatAnthropic
    print(f'✓ langchain-anthropic: installed')
except ImportError:
    missing.append('langchain-anthropic')

if missing:
    print(f'\\n❌ Missing dependencies: {missing}')
    sys.exit(1)
else:
    print(f'\\n✓ All dependencies installed')
"
```

---

## 📊 Success Criteria Summary

Before deploying to Lambda, ensure:

### Data Quality
- [ ] >95% of samples have all three modalities
- [ ] 100% of samples have images at full resolution
- [ ] Text summaries are coherent and medically accurate
- [ ] Structured features have reasonable value ranges
- [ ] No NaN, Inf, or extreme outliers

### Performance
- [ ] Data loading: 0.05-0.1s per sample (pre-compiled)
- [ ] Processing time: 1.5-6.5s per sample (on-demand)
- [ ] Memory usage: <16 GB peak during processing
- [ ] No memory leaks during iteration

### Infrastructure
- [ ] All unit tests pass
- [ ] All integration tests pass
- [ ] Checkpoint/resume works correctly
- [ ] Multi-batch mode works correctly
- [ ] Manifest generation is correct
- [ ] Pre-compiled dataset loading works

### Configuration
- [ ] Using Claude Sonnet 4.5
- [ ] Full-resolution images preserved
- [ ] CXR-PRO integration enabled
- [ ] Temporal window: -48h to +24h
- [ ] NOT_DONE token for missing values
- [ ] All dependencies installed

---

## 🚀 Ready for Lambda?

Once all checks pass, you're ready to:
1. Build full pre-compiled dataset (20k samples)
2. Deploy to Lambda GPU instances
3. Monitor performance and costs
4. Scale to production

**Estimated Build Time (Full Dataset)**:
- Single batch: 12-20 hours (1 instance)
- Multi-batch (20 batches of 1000): 12-20 hours (1 instance) or 36-60 minutes (20 instances in parallel)

**Estimated Lambda Costs** (per epoch):
- On-demand: ~$1,800 (120 hours at $15/hour)
- Pre-compiled: ~$375 (25 hours at $15/hour)
- **Savings**: ~$1,365 per epoch (76% reduction)
