# Preprocessing Analysis Notebooks

## Lambda Pipeline Analysis (`lambda_pipeline_analysis.ipynb`)

**Purpose**: Comprehensive analysis of preprocessing pipeline outputs for validation and quality checks.

### What It Analyzes

1. **Overall Pipeline Metrics**
   - Success rates by modality
   - Error analysis and distribution
   - Processing completeness

2. **Text Features**
   - Token count distribution (should be 50-200 for valid text)
   - Claude summary length and quality
   - Named entity counts
   - Empty text detection

3. **Image Features**
   - Image shape distribution
   - Resolution analysis
   - File size distribution
   - Coverage metrics

4. **Structured Features**
   - DICOM metadata coverage (10 fields: view position, orientation, portable detection, etc.)
   - Vital signs availability (temperature, heart rate, BP, O2 sat, etc.)
   - Lab values coverage
   - Missing data patterns

5. **Multimodal Completeness**
   - Samples with all 3 modalities
   - Partial completeness analysis
   - Demographics correlation

6. **Quality Checks**
   - Empty text detection
   - Missing summaries
   - Outlier detection
   - Data validation

### Usage on Lambda GPU

#### 1. Transfer Notebook to Lambda
```bash
# From local machine
export LAMBDA_IP=192.222.59.237
export PEM_KEY=/home/dev/Downloads/berger-cm.pem

scp -i $PEM_KEY lambda_pipeline_analysis.ipynb ubuntu@$LAMBDA_IP:~/mimic-cxr-validation/step2_preprocessing/notebooks/
```

#### 2. Install Jupyter on Lambda
```bash
# SSH into Lambda
ssh -i $PEM_KEY ubuntu@$LAMBDA_IP

# Activate venv
cd ~/mimic-cxr-validation/step2_preprocessing
source venv/bin/activate

# Install Jupyter
pip install jupyter notebook ipykernel matplotlib seaborn

# Add kernel
python -m ipykernel install --user --name=preprocessing-env --display-name="Preprocessing Analysis"
```

#### 3. Start Jupyter Server
```bash
# Option A: With port forwarding (recommended)
jupyter notebook --no-browser --port=8888

# Then on local machine, create SSH tunnel:
ssh -i $PEM_KEY -N -L 8888:localhost:8888 ubuntu@$LAMBDA_IP

# Access in browser: http://localhost:8888
```

```bash
# Option B: Direct access (if Lambda has public IP)
jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser --allow-root

# Access in browser: http://<LAMBDA_IP>:8888
# WARNING: This is less secure, use port forwarding instead
```

#### 4. Run Analysis
1. Open `lambda_pipeline_analysis.ipynb` in Jupyter
2. Update paths in cell 2 if needed:
   ```python
   OUTPUT_DIR = Path('../output/validation_200/train')
   COHORT_PATH = Path('../cohorts/validation_subset_200_with_reports.csv')
   ```
3. Run all cells (Cell > Run All)
4. Review outputs and visualizations

### Key Outputs

The notebook generates:
1. **Comprehensive statistics** printed to output cells
2. **Visualizations** for all major metrics
3. **JSON report**: `pipeline_analysis_report.json`
4. **Dashboard image**: `pipeline_analysis_dashboard.png`

### Expected Results (After CXR-PRO Fix)

- **Text coverage**: 99.5% (199/200 samples with valid text)
- **Mean tokens**: 50-200 (vs 2 before fix)
- **DICOM metadata**: 100% coverage for all 10 fields
- **Multimodal completeness**: 95%+ samples with all 3 modalities
- **Error rate**: <5%

### Validation Criteria

✅ **PASS** if:
- Text coverage ≥95%
- Mean tokens ≥50
- DICOM coverage ≥90%
- Complete multimodal samples ≥90%

⚠️ **REVIEW** if:
- Text coverage 80-95%
- Mean tokens 20-50
- Any modality coverage <80%

❌ **FAIL** if:
- Text coverage <80%
- Mean tokens <20
- Complete multimodal samples <70%

### Troubleshooting

**Issue**: "Output directory not found"
- Check `OUTPUT_DIR` path in cell 2
- Ensure preprocessing has completed
- Verify outputs exist: `ls -lh output/validation_200/train/`

**Issue**: "Module not found"
- Install missing packages: `pip install <package>`
- Ensure venv is activated: `source venv/bin/activate`

**Issue**: "Kernel dies when loading data"
- Large dataset may exceed memory
- Reduce batch size or analyze subset
- Use Lambda instance with more RAM

### Alternative: Command-Line Analysis

If Jupyter doesn't work, convert to script:
```bash
# Convert notebook to Python script
jupyter nbconvert --to script lambda_pipeline_analysis.ipynb

# Run as script
python lambda_pipeline_analysis.py
```

---

**Created**: November 23, 2025
**For**: Lambda GPU preprocessing validation
**Compatible with**: Python 3.10+, PyTorch 2.0+
