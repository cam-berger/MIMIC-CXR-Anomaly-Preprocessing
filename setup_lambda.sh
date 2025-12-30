#!/bin/bash
# Lambda GPU environment setup script
# Run this on the Lambda instance after transferring files

set -e

echo "============================================================"
echo "Lambda GH200 Environment Setup"
echo "============================================================"

cd ~/mimic-training

# Step 1: Extract archives
echo ""
echo "[1/5] Extracting archives..."
echo "------------------------------------------------------------"
tar -xzf ~/validation_data_subset.tar.gz -C data/
tar -xzf ~/code.tar.gz
tar -xzf ~/cohorts.tar.gz
mv ~/mae_pretrained.pt output/mae_best.pt

echo "✓ Extracted validation_data_subset/"
echo "✓ Extracted source code"
echo "✓ Extracted cohorts"
echo "✓ Moved mae_best.pt"

# Step 2: Verify extracted data
echo ""
echo "[2/5] Verifying extracted data..."
echo "------------------------------------------------------------"
ls -lh data/validation_data_subset/
ls -lh output/cohorts/
ls -lh output/mae_best.pt

# Step 3: Create Python virtual environment
echo ""
echo "[3/5] Creating Python environment..."
echo "------------------------------------------------------------"
python3 -m venv venv
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Step 4: Install dependencies
echo ""
echo "[4/5] Installing dependencies..."
echo "------------------------------------------------------------"

# Install PyTorch with CUDA 12.6 (CRITICAL for GH200)
echo "Installing PyTorch with CUDA 12.6..."
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu126

# Install main requirements
echo "Installing requirements..."
pip install -r requirements.txt

# Install scispacy and medical NLP models
echo "Installing scispacy..."
pip install scispacy
python3 -m spacy download en_core_sci_md

echo "✓ All dependencies installed"

# Step 5: Configure environment
echo ""
echo "[5/5] Configuring environment..."
echo "------------------------------------------------------------"

# Create .env file
cat > .env << 'EOF'
# MIMIC Dataset Paths (Lambda filesystem)
MIMIC_CXR_JPG_PATH=/home/ubuntu/mimic-training/data/validation_data_subset/mimic-cxr-jpg
MIMIC_IV_PATH=/home/ubuntu/mimic-training/data/validation_data_subset/mimic-iv/3.1
MIMIC_IV_ED_PATH=/home/ubuntu/mimic-training/data/validation_data_subset/mimic-iv-ed/2.2
CXR_PRO_PATH=/home/ubuntu/mimic-training/data/validation_data_subset/cxr-pro/1.0.0
OUTPUT_PATH=/home/ubuntu/mimic-training/output

# Disable Claude API (save costs, use clinical context only)
ANTHROPIC_API_KEY=""
EOF

echo "✓ Created .env file"

# Verify GPU access
echo ""
echo "Verifying GPU access..."
source venv/bin/activate
python3 << 'PYEOF'
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
else:
    print("WARNING: CUDA not available!")
PYEOF

# Verify data paths
echo ""
echo "Verifying data paths..."
source .env
for path in "$MIMIC_CXR_JPG_PATH" "$MIMIC_IV_PATH" "$MIMIC_IV_ED_PATH" "$CXR_PRO_PATH"; do
    if [ -d "$path" ]; then
        echo "✓ $path exists"
    else
        echo "✗ $path missing!"
    fi
done

# Summary
echo ""
echo "============================================================"
echo "Setup Complete!"
echo "============================================================"
echo ""
echo "Next steps:"
echo "  1. Activate environment: source venv/bin/activate"
echo "  2. Load env vars: source .env"
echo "  3. Run preprocessing: see instructions below"
echo ""
echo "Preprocessing command:"
echo "  nohup python3 preprocess.py \\"
echo "    --cohort output/cohorts/cohort.parquet \\"
echo "    --output-dir output/preprocessed/anomalous_train \\"
echo "    --leak-free \\"
echo "    --no-summarization \\"
echo "    --workers 16 \\"
echo "    --img-size 1024 \\"
echo "    > preprocessing.log 2>&1 &"
echo ""
echo "Monitor with: tail -f preprocessing.log"
echo ""
