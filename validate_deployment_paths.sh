#!/bin/bash
# Deployment Path Validation Script
# Verifies all required data paths exist before running preprocessing
# This prevents the 93.5% failure rate caused by missing CXR-PRO configuration

set -e

echo "================================================================"
echo "MIMIC-CXR Preprocessing - Deployment Path Validation"
echo "================================================================"
echo ""

# Check if config file path is provided
if [ -z "$1" ]; then
    echo "Usage: $0 <path_to_config.yaml>"
    echo "Example: $0 step2_preprocessing/config/config_validation.yaml"
    exit 1
fi

CONFIG_FILE="$1"

if [ ! -f "$CONFIG_FILE" ]; then
    echo "ERROR: Config file not found: $CONFIG_FILE"
    exit 1
fi

echo "Validating paths from config: $CONFIG_FILE"
echo ""

# Extract paths from YAML config using grep and sed
echo "Extracting paths from config..."
MIMIC_CXR_BASE=$(grep "mimic_cxr_base:" "$CONFIG_FILE" | sed 's/.*mimic_cxr_base: *"\?\([^"]*\)"\?.*/\1/' | tr -d '"')
MIMIC_IV_BASE=$(grep "mimic_iv_base:" "$CONFIG_FILE" | sed 's/.*mimic_iv_base: *"\?\([^"]*\)"\?.*/\1/' | tr -d '"')
MIMIC_ED_BASE=$(grep "mimic_ed_base:" "$CONFIG_FILE" | sed 's/.*mimic_ed_base: *"\?\([^"]*\)"\?.*/\1/' | tr -d '"')
CXR_PRO_REPORTS=$(grep "cxr_pro_reports:" "$CONFIG_FILE" | sed 's/.*cxr_pro_reports: *"\?\([^"]*\)"\?.*/\1/' | tr -d '"')

echo "  mimic_cxr_base: $MIMIC_CXR_BASE"
echo "  mimic_iv_base: $MIMIC_IV_BASE"
echo "  mimic_ed_base: $MIMIC_ED_BASE"
echo "  cxr_pro_reports: $CXR_PRO_REPORTS"
echo ""

# Validation flags
ALL_VALID=true

echo "================================================================"
echo "Validation Results:"
echo "================================================================"
echo ""

# 1. Validate CXR images directory
echo "1. MIMIC-CXR Images"
echo "   Path: $MIMIC_CXR_BASE"
if [ -d "$MIMIC_CXR_BASE/files" ]; then
    NUM_IMAGES=$(find "$MIMIC_CXR_BASE/files" -name "*.jpg" 2>/dev/null | wc -l)
    echo "   Status: ✅ PASS"
    echo "   Found: $NUM_IMAGES JPG files"
else
    echo "   Status: ❌ FAIL - Directory not found"
    echo "   Impact: Image processing will fail completely"
    ALL_VALID=false
fi
echo ""

# 2. Validate MIMIC-IV structured data
echo "2. MIMIC-IV Structured Data"
echo "   Base: $MIMIC_IV_BASE"

# Check patients.csv
if [ -f "$MIMIC_IV_BASE/hosp/patients.csv" ]; then
    SIZE=$(du -h "$MIMIC_IV_BASE/hosp/patients.csv" | cut -f1)
    echo "   - patients.csv: ✅ PASS ($SIZE)"
else
    echo "   - patients.csv: ❌ FAIL - File not found"
    ALL_VALID=false
fi

# Check admissions.csv
if [ -f "$MIMIC_IV_BASE/hosp/admissions.csv" ]; then
    SIZE=$(du -h "$MIMIC_IV_BASE/hosp/admissions.csv" | cut -f1)
    echo "   - admissions.csv: ✅ PASS ($SIZE)"
else
    echo "   - admissions.csv: ❌ FAIL - File not found"
    ALL_VALID=false
fi

# Check labevents.csv (optional - may be large or filtered)
if [ -f "$MIMIC_IV_BASE/hosp/labevents.csv" ]; then
    SIZE=$(du -h "$MIMIC_IV_BASE/hosp/labevents.csv" | cut -f1)
    echo "   - labevents.csv: ✅ PASS ($SIZE)"
else
    echo "   - labevents.csv: ⚠️  WARN - File not found (optional for validation subset)"
fi
echo ""

# 3. Validate MIMIC-IV-ED data
echo "3. MIMIC-IV-ED Data"
echo "   Base: $MIMIC_ED_BASE"

# Check vitals
if [ -f "$MIMIC_ED_BASE/ed/vitalsign.csv" ]; then
    SIZE=$(du -h "$MIMIC_ED_BASE/ed/vitalsign.csv" | cut -f1)
    echo "   - vitalsign.csv: ✅ PASS ($SIZE)"
else
    echo "   - vitalsign.csv: ❌ FAIL - File not found"
    echo "   Impact: Structured features (vitals) will be empty (NOT_DONE)"
    ALL_VALID=false
fi

# Check triage
if [ -f "$MIMIC_ED_BASE/ed/triage.csv" ]; then
    SIZE=$(du -h "$MIMIC_ED_BASE/ed/triage.csv" | cut -f1)
    echo "   - triage.csv: ✅ PASS ($SIZE)"
else
    echo "   - triage.csv: ❌ FAIL - File not found"
    ALL_VALID=false
fi
echo ""

# 4. Validate CXR-PRO reports (CRITICAL - this was the missing piece!)
echo "4. CXR-PRO Radiology Reports (CRITICAL)"
echo "   Path: $CXR_PRO_REPORTS"

if [ -z "$CXR_PRO_REPORTS" ]; then
    echo "   Status: ❌ FAIL - cxr_pro_reports field NOT FOUND in config!"
    echo "   Impact: Text processing will fail completely (2-token empty sequences)"
    echo "   Fix: Add 'cxr_pro_reports' field to config file"
    ALL_VALID=false
elif [ -f "$CXR_PRO_REPORTS" ]; then
    SIZE=$(du -h "$CXR_PRO_REPORTS" | cut -f1)
    NUM_REPORTS=$(tail -n +2 "$CXR_PRO_REPORTS" | wc -l)
    echo "   Status: ✅ PASS"
    echo "   Size: $SIZE"
    echo "   Reports: $NUM_REPORTS"
else
    echo "   Status: ❌ FAIL - File not found"
    echo "   Impact: Text processing will fail completely (2-token empty sequences)"
    echo "   This caused 93.5% of samples to have empty structured data in Lambda validation"
    ALL_VALID=false
fi
echo ""

# 5. Validate DICOM metadata (for image acquisition context)
echo "5. DICOM Metadata (Image Acquisition Context)"
DICOM_METADATA=$(grep "dicom_metadata_path:" "$CONFIG_FILE" | sed 's/.*dicom_metadata_path: *"\?\([^"]*\)"\?.*/\1/' | tr -d '"')
echo "   Path: $DICOM_METADATA"

if [ -z "$DICOM_METADATA" ]; then
    echo "   Status: ⚠️  WARN - dicom_metadata_path field NOT FOUND in config"
    echo "   Impact: DICOM features (view position, orientation) will not be included"
    echo "   Note: Not critical but reduces model robustness (can't distinguish AP vs PA)"
    # Not setting ALL_VALID=false since this is optional
elif [ -f "$DICOM_METADATA" ]; then
    SIZE=$(du -h "$DICOM_METADATA" | cut -f1)
    # Count lines (gzipped file)
    if [[ "$DICOM_METADATA" == *.gz ]]; then
        NUM_RECORDS=$(zcat "$DICOM_METADATA" | wc -l)
    else
        NUM_RECORDS=$(wc -l < "$DICOM_METADATA")
    fi
    echo "   Status: ✅ PASS"
    echo "   Size: $SIZE"
    echo "   Records: $NUM_RECORDS (DICOM images)"
    echo "   Features: View position, patient orientation, image dimensions"
else
    echo "   Status: ⚠️  WARN - File not found"
    echo "   Impact: DICOM features will not be included (view position, orientation)"
    echo "   Note: Not critical but recommended for production"
fi
echo ""

# Final verdict
echo "================================================================"
if [ "$ALL_VALID" = true ]; then
    echo "✅ All paths validated successfully!"
    echo "================================================================"
    echo ""
    echo "You can proceed with preprocessing:"
    echo "  python3 main.py --config $CONFIG_FILE"
    echo ""
    exit 0
else
    echo "❌ Validation FAILED - Missing required data paths"
    echo "================================================================"
    echo ""
    echo "Please fix the issues above before running preprocessing."
    echo ""
    echo "Common fixes:"
    echo "  1. Update config file paths using sed commands (see LAMBDA_DEPLOYMENT.md)"
    echo "  2. Ensure validation_data_subset was extracted with all 4 components:"
    echo "     - CXR images"
    echo "     - MIMIC-IV structured data"
    echo "     - MIMIC-IV-ED vitals/triage"
    echo "     - CXR-PRO reports (CRITICAL - often forgotten!)"
    echo "  3. Re-run extract_validation_subset.sh if data is missing"
    echo ""
    exit 1
fi
