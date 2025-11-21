#!/bin/bash
# Extract only 200 validation samples from MIMIC datasets for Lambda GPU transfer
# This avoids transferring the full 500GB dataset - extracts only ~5-15GB needed

set -e

COHORT_FILE="/home/dev/Documents/Portfolio/MIMIC/MIMIC-CXR-Anomaly-Preprocessing/step2_preprocessing/cohorts/validation_subset_200.csv"
OUTPUT_DIR="/home/dev/Documents/Portfolio/MIMIC/MIMIC-CXR-Anomaly-Preprocessing/validation_data_subset"

echo "================================================================"
echo "Extracting 200-sample validation subset for Lambda GPU transfer"
echo "================================================================"
echo ""

# Check cohort file exists
if [ ! -f "$COHORT_FILE" ]; then
    echo "ERROR: Cohort file not found: $COHORT_FILE"
    exit 1
fi

echo "Cohort file: $COHORT_FILE"
echo "Output directory: $OUTPUT_DIR"
echo ""

# Create output directories
mkdir -p ${OUTPUT_DIR}/{cxr/files,mimic-iv/hosp,mimic-ed/ed}

# 1. Extract CXR images for 200 samples
echo "Step 1/3: Extracting CXR images..."
echo "--------------------------------------"
python3 << 'EOF'
import pandas as pd
import shutil
from pathlib import Path
from tqdm import tqdm

# Load 200-sample cohort
cohort_file = "/home/dev/Documents/Portfolio/MIMIC/MIMIC-CXR-Anomaly-Preprocessing/step2_preprocessing/cohorts/validation_subset_200.csv"
cohort = pd.read_csv(cohort_file)

cxr_base = Path("/media/dev/MIMIC_DATA/mimic-cxr-jpg")
output_base = Path("/home/dev/Documents/Portfolio/MIMIC/MIMIC-CXR-Anomaly-Preprocessing/validation_data_subset/cxr")

print(f"Processing {len(cohort)} samples...")
success_count = 0
missing_count = 0

for idx, row in tqdm(cohort.iterrows(), total=len(cohort), desc="Copying CXR images"):
    subject_id = str(row['subject_id'])
    study_id = str(row['study_id'])

    # MIMIC-CXR directory structure: p10/p10000032/s50414267/
    subject_prefix = f"p{subject_id[:2]}"
    subject_dir = f"p{subject_id}"
    study_dir = f"s{study_id}"

    # Source and destination
    src_dir = cxr_base / "files" / subject_prefix / subject_dir / study_dir
    dst_dir = output_base / "files" / subject_prefix / subject_dir / study_dir

    # Copy study directory (contains 1-3 JPG images)
    if src_dir.exists():
        dst_dir.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(src_dir, dst_dir, dirs_exist_ok=True)
        success_count += 1
    else:
        print(f"WARNING: Missing study directory: {src_dir}")
        missing_count += 1

print(f"\nCXR image extraction complete!")
print(f"  Successfully copied: {success_count} studies")
print(f"  Missing studies: {missing_count}")
EOF

echo ""

# 2. Copy MIMIC-IV structured data (labs/vitals)
echo "Step 2/3: Copying MIMIC-IV structured data..."
echo "----------------------------------------------"

# Copy dictionary files (small, always need these)
echo "  - Copying d_labitems.csv..."
cp /home/dev/Documents/Portfolio/MIMIC_Data/physionet.org/files/mimiciv/3.1/hosp/d_labitems.csv \
   ${OUTPUT_DIR}/mimic-iv/hosp/ 2>/dev/null || echo "    WARNING: d_labitems.csv not found"

# Copy full patients.csv (12MB - manageable)
echo "  - Copying patients.csv..."
cp /home/dev/Documents/Portfolio/MIMIC_Data/physionet.org/files/mimiciv/3.1/hosp/patients.csv \
   ${OUTPUT_DIR}/mimic-iv/hosp/ 2>/dev/null || echo "    WARNING: patients.csv not found"

# Copy full admissions.csv (90MB - manageable)
echo "  - Copying admissions.csv..."
cp /home/dev/Documents/Portfolio/MIMIC_Data/physionet.org/files/mimiciv/3.1/hosp/admissions.csv \
   ${OUTPUT_DIR}/mimic-iv/hosp/ 2>/dev/null || echo "    WARNING: admissions.csv not found"

# For labevents.csv (18GB), copy only relevant subject_ids
echo "  - Extracting labevents for 200 samples (this may take a few minutes)..."
python3 << 'EOFLAB'
import pandas as pd
from pathlib import Path

cohort = pd.read_csv("/home/dev/Documents/Portfolio/MIMIC/MIMIC-CXR-Anomaly-Preprocessing/step2_preprocessing/cohorts/validation_subset_200.csv")
subject_ids = set(cohort['subject_id'].unique())

# Read labevents in chunks and filter
output_path = Path("/home/dev/Documents/Portfolio/MIMIC/MIMIC-CXR-Anomaly-Preprocessing/validation_data_subset/mimic-iv/hosp/labevents.csv")
labevents_path = Path("/home/dev/Documents/Portfolio/MIMIC_Data/physionet.org/files/mimiciv/3.1/hosp/labevents.csv")

if labevents_path.exists():
    print(f"    Filtering {len(subject_ids)} subjects from labevents.csv...")
    first_chunk = True
    for chunk in pd.read_csv(labevents_path, chunksize=1000000):
        # Filter for our subjects
        filtered = chunk[chunk['subject_id'].isin(subject_ids)]
        if len(filtered) > 0:
            filtered.to_csv(output_path, mode='w' if first_chunk else 'a', header=first_chunk, index=False)
            first_chunk = False
    print(f"    Extracted labevents for {len(subject_ids)} subjects")
else:
    print("    WARNING: labevents.csv not found")
EOFLAB

echo ""

# 3. Copy MIMIC-IV-ED data (notes + vitals)
echo "Step 3/3: Copying MIMIC-IV-ED data..."
echo "--------------------------------------"

# Copy ED directory recursively
if [ -d "/home/dev/Documents/Portfolio/MIMIC_Data/physionet.org/files/mimic-iv-ed/2.2/ed" ]; then
    echo "  - Copying ED data..."
    cp -r /home/dev/Documents/Portfolio/MIMIC_Data/physionet.org/files/mimic-iv-ed/2.2/ed/* \
       ${OUTPUT_DIR}/mimic-ed/ed/
    echo "  - ED data copied successfully"
else
    echo "  WARNING: MIMIC-IV-ED directory not found"
fi

echo ""

# Calculate final size
echo "================================================================"
echo "Extraction complete!"
echo "================================================================"
echo ""
echo "Output directory size:"
du -sh ${OUTPUT_DIR}
echo ""
echo "Breakdown by component:"
du -sh ${OUTPUT_DIR}/cxr/
du -sh ${OUTPUT_DIR}/mimic-iv/
du -sh ${OUTPUT_DIR}/mimic-ed/
echo ""
echo "Next steps:"
echo "  1. Compress for transfer: tar -czf validation_data_subset.tar.gz validation_data_subset/"
echo "  2. Compress code: tar -czf step2_preprocessing.tar.gz step2_preprocessing/"
echo "  3. Transfer to Lambda GPU: rsync -avz *.tar.gz ubuntu@<LAMBDA_IP>:~/mimic-cxr-validation/"
echo ""
echo "See deployment plan for full instructions."
echo "================================================================"
