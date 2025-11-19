# MIMIC-CXR Anomaly Detection: Normal Cohort Identification (Step 1)

This codebase implements **Step 1** of the unsupervised multimodal anomaly detection pipeline for chest X-rays: **Identifying and Filtering "Normal" Cases across MIMIC datasets**.

## Overview

The goal of Step 1 is to create a high-quality cohort of "normal" cases from MIMIC datasets by:

1. **Radiology Report Criteria**: Filtering chest X-ray studies with "No Finding" labels and no acute pathologies
2. **Clinical Context Criteria**: Ensuring patients had benign ED courses with no critical outcomes
3. **Data Integration**: Merging MIMIC-CXR-JPG, MIMIC-IV-ED, and MIMIC-IV data based on temporal and patient matching

The resulting cohort will be used to train an unsupervised anomaly detection model in subsequent steps.

## Project Structure

```
MIMIC-CXR-Anomaly-Preprocessing/
├── main.py                      # Main execution script
├── requirements.txt             # Python dependencies
├── README.md                    # This file
│
├── src/
│   ├── config/
│   │   ├── config.py           # Filter and processing configurations
│   │   └── paths.py            # Data path configurations
│   │
│   ├── data_loaders/
│   │   ├── cxr_loader.py       # MIMIC-CXR-JPG data loader
│   │   ├── ed_loader.py        # MIMIC-IV-ED data loader
│   │   └── iv_loader.py        # MIMIC-IV data loader
│   │
│   ├── filters/
│   │   ├── radiology_filter.py # Radiology report filtering
│   │   └── clinical_filter.py  # Clinical context filtering
│   │
│   ├── mergers/
│   │   └── cohort_builder.py   # Cohort building and merging
│   │
│   ├── validators/
│   │   ├── data_validator.py   # Data quality validation
│   │   └── sample_checker.py   # Manual review sampling
│   │
│   └── utils/
│       ├── logging_utils.py    # Logging utilities
│       └── data_utils.py       # Data utility functions
│
└── output/                      # Generated output (created on run)
    ├── cohorts/                 # Normal cohort CSV/Parquet files
    ├── manual_review/           # Sample cases for manual review
    ├── reports/                 # Validation and summary reports
    └── logs/                    # Execution logs
```

## Installation

### Prerequisites

- Python 3.8 or higher
- Access to MIMIC datasets:
  - MIMIC-CXR-JPG v2.1.0
  - MIMIC-IV-ED v2.2
  - MIMIC-IV v3.1

### Setup

1. Clone or navigate to this directory:
```bash
cd /home/dev/Documents/Portfolio/MIMIC/MIMIC-CXR-Anomaly-Preprocessing
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Verify data paths in `src/config/paths.py` match your local setup:
   - MIMIC-CXR-JPG: `/media/dev/MIMIC_DATA/mimic-cxr-jpg`
   - MIMIC-IV-ED: `/home/dev/Documents/Portfolio/MIMIC_Data/physionet.org/files/mimic-iv-ed/2.2`
   - MIMIC-IV: `/home/dev/Documents/Portfolio/MIMIC_Data/physionet.org/files/mimiciv/3.1`

## Usage

### Basic Usage

Run the main script to identify normal cases:

```bash
python main.py
```

This will:
- Load and filter MIMIC datasets
- Build a cohort of normal cases
- Split into training and validation sets
- Save results to `output/` directory
- Generate validation reports and samples for manual review

### Advanced Options

```bash
# Specify custom output directory
python main.py --output-dir my_output

# Skip hospital admission outcome filtering (faster, less strict)
python main.py --no-hospital-filter

# Adjust number of validation samples
python main.py --validation-samples 200

# Enable debug logging
python main.py --log-level DEBUG

# Skip validation step
python main.py --skip-validation

# Optimize memory usage
python main.py --optimize-memory
```

### Command-Line Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--output-dir` | Output directory for results | `output` |
| `--no-hospital-filter` | Skip hospital outcome filtering | False |
| `--validation-samples` | Number of samples for manual review | 100 |
| `--log-level` | Logging level (DEBUG/INFO/WARNING/ERROR) | INFO |
| `--skip-validation` | Skip data validation | False |
| `--optimize-memory` | Optimize memory usage | False |

## Configuration

### Filter Configuration

Edit `src/config/config.py` to customize filtering criteria:

**Radiology Report Criteria:**
- `require_no_finding`: Require "No Finding" = 1.0
- `exclude_pathology_labels`: List of pathologies to exclude

**Clinical Context Criteria:**
- `acceptable_dispositions`: ED dispositions to include (e.g., "HOME", "DISCHARGED")
- `excluded_dispositions`: ED dispositions to exclude (e.g., "ADMITTED", "EXPIRED")
- `critical_diagnosis_patterns`: ICD code patterns for critical diagnoses to exclude
- `time_window_hours`: Time window for matching CXR with ED stays (default: 24 hours)
- `min_age`: Minimum patient age (default: 18)

**Data Split:**
- `validation_fraction`: Fraction for validation set (default: 0.15)
- `random_seed`: Random seed for reproducibility (default: 42)

## Output Files

After running, the following files are generated in the `output/` directory:

### Cohorts
- `cohorts/normal_cohort_full.csv` - Complete normal cohort
- `cohorts/normal_cohort_full.parquet` - Complete cohort (Parquet format)
- `cohorts/normal_cohort_train.csv` - Training set
- `cohorts/normal_cohort_validation.csv` - Validation set

### Manual Review
- `manual_review/sample_for_review.csv` - Random sample for manual verification
- `manual_review/edge_cases_*.csv` - Edge cases requiring special attention

### Reports
- `reports/validation_report.txt` - Data quality validation report
- `reports/summary_statistics.json` - Comprehensive cohort statistics
- `reports/data_dictionary.csv` - Data dictionary of all columns

### Logs
- `logs/cohort_building_YYYYMMDD_HHMMSS.log` - Detailed execution log

## Normal Case Definition

A case is considered "normal" if it meets **ALL** of the following criteria:

### Radiology Criteria
1. CheXpert "No Finding" label = 1.0
2. All pathology labels (Pneumonia, Effusion, etc.) are NOT positive (1.0)
3. No acute or significant abnormalities mentioned in report

### Clinical Context Criteria
1. **ED Disposition**: Patient was discharged home (not admitted to hospital)
2. **No Critical Diagnoses**: No ICD codes for critical conditions like:
   - Sepsis, pneumonia, acute MI
   - Heart failure, respiratory failure
   - Pulmonary embolism, pneumothorax
   - ARDS, severe infections
3. **No ICU Admission**: If admitted, no ICU stay
4. **No In-Hospital Death**: If admitted, patient survived
5. **Temporal Consistency**: CXR performed during or near ED visit (within 24-hour window)
6. **Age**: Patient age ≥ 18 years

## Expected Results

Based on the MIMIC-CXR dataset (>200,000 studies), you can expect:

- **Normal cases identified**: ~20,000-40,000 studies (10-20% of total)
- **Unique subjects**: ~15,000-25,000 patients
- **Processing time**: 5-15 minutes (depending on system)
- **Memory usage**: 4-8 GB peak

## Validation and Quality Assurance

The pipeline includes multiple validation steps:

1. **Automatic Data Validation**
   - Checks for required columns
   - Identifies missing values
   - Detects duplicates
   - Validates data types and value ranges
   - Verifies temporal consistency

2. **Manual Review Samples**
   - Random sample of 100 cases for manual verification
   - Edge cases identified for special review
   - Stratified sampling by demographics

3. **Statistical Summaries**
   - Cohort characteristics (age, gender distribution)
   - Filtering statistics at each step
   - Data completeness metrics

## Next Steps

After generating the normal cohort:

1. **Manual Review**: Review `manual_review/sample_for_review.csv` to verify quality
2. **Check Edge Cases**: Examine `manual_review/edge_cases_*.csv` files
3. **Review Reports**: Check `reports/validation_report.txt` for data quality issues
4. **Proceed to Step 2**: Use the generated cohorts for data preprocessing (image and clinical data extraction)

## Troubleshooting

### Common Issues

**"Path not found" errors:**
- Verify data paths in `src/config/paths.py` match your local setup
- Ensure you have proper permissions to access the data directories

**"No normal cases found":**
- Check filtering criteria in `src/config/config.py` (may be too strict)
- Verify input data is properly formatted
- Check logs for specific filtering statistics

**Memory errors:**
- Use `--optimize-memory` flag
- Use `--no-hospital-filter` to skip hospital data loading
- Process on a machine with more RAM (>8GB recommended)

**Slow execution:**
- Use `--no-hospital-filter` for faster processing
- Consider using SSD for data storage
- Reduce `--validation-samples` count

## Data Requirements

This pipeline requires the following MIMIC data files:

**MIMIC-CXR-JPG:**
- `mimic-cxr-2.0.0-chexpert.csv.gz` (CheXpert labels)
- `mimic-cxr-2.0.0-metadata.csv.gz` (Image metadata)

**MIMIC-IV-ED:**
- `edstays.csv` (ED stay information)
- `diagnosis.csv` (ED diagnoses)
- `triage.csv` (Triage data)

**MIMIC-IV:**
- `patients.csv` (Patient demographics)
- `admissions.csv` (Hospital admissions)
- `transfers.csv` (Transfer events)

## License

This code is provided for research and educational purposes. Please ensure you have proper authorization and credentialing to access MIMIC datasets through PhysioNet.

## References

- Johnson, A., Pollard, T., Mark, R. et al. MIMIC-CXR-JPG, a large publicly available database of labeled chest radiographs. *Sci Data* 6, 317 (2019).
- Johnson, A., Bulgarelli, L., Pollard, T. et al. MIMIC-IV-ED. PhysioNet (2023).
- Johnson, A., Bulgarelli, L., Shen, L. et al. MIMIC-IV. PhysioNet (2024).

## Contact

For issues or questions about this pipeline, please open an issue in the project repository.