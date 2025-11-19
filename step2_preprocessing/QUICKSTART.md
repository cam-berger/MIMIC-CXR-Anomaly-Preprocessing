# Quick Start Guide - Step 2 Preprocessing

Get started with Step 2 preprocessing in 5 minutes.

## Prerequisites

- ✅ Step 1 completed (cohort CSV files generated)
- ✅ MIMIC-CXR-JPG dataset downloaded
- ✅ MIMIC-IV and MIMIC-IV-ED datasets downloaded
- ✅ Conda/Miniconda installed

## Quick Setup

### Option 1: Automated Setup (Recommended)

```bash
cd step2_preprocessing
./setup.sh
```

This will:
- Create conda environment
- Install all dependencies
- Download scispacy model
- Download ClinicalBERT
- Verify installation

### Option 2: Manual Setup

```bash
# Create environment
conda create -n ml_env python=3.9
conda activate ml_env

# Install dependencies
pip install -r requirements.txt

# Install scispacy model
pip install scispacy
pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.1/en_core_sci_md-0.5.1.tar.gz

# Verify
python test_setup.py
```

## Configuration

Edit `config/config.yaml` to set your data paths:

```yaml
data:
  step1_cohort_train: "/path/to/step1/output/cohort_train.csv"
  step1_cohort_val: "/path/to/step1/output/cohort_val.csv"
  mimic_cxr_base: "/path/to/mimic-cxr-jpg"
  mimic_iv_base: "/path/to/mimiciv/3.1"
  mimic_ed_base: "/path/to/mimic-iv-ed/2.2"
  output_base: "output"
```

## Run Preprocessing

### Test on Small Sample

```bash
# Test single sample
python test_sample.py

# Test 10 samples
python main.py --max-samples 10
```

### Process Full Dataset

```bash
# Without Claude (uses fallback summarization)
python main.py --skip-text

# With Claude (requires API key)
export ANTHROPIC_API_KEY='your-key'
python main.py
```

## Check Outputs

```bash
# View processing summary
cat output/preprocessing_summary.json

# Check a processed sample
ls output/train/images/
ls output/train/structured_features/
ls output/train/text_features/
```

## Common Issues

### "scispacy model not found"
```bash
pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.1/en_core_sci_md-0.5.1.tar.gz
```

### "No Anthropic API key"
Either:
- Run with `--skip-text` flag
- Set environment variable: `export ANTHROPIC_API_KEY='your-key'`
- Pass via command line: `python main.py --anthropic-api-key YOUR_KEY`

### "CUDA out of memory"
```bash
# Use CPU only
CUDA_VISIBLE_DEVICES="" python main.py
```

## Next Steps

See [README.md](README.md) for:
- Detailed configuration options
- Output file formats
- Advanced usage
- Architecture documentation
- Performance tuning
