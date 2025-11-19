#!/bin/bash
# Setup script for Step 2 preprocessing

set -e  # Exit on error

echo "========================================"
echo "Step 2 Preprocessing - Setup Script"
echo "========================================"
echo ""

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo "Error: conda not found. Please install Miniconda or Anaconda first."
    exit 1
fi

# Environment name
ENV_NAME="ml_env"

# Check if environment exists
if conda env list | grep -q "^${ENV_NAME} "; then
    echo "Environment '${ENV_NAME}' already exists."
    read -p "Do you want to recreate it? (y/n): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "Removing existing environment..."
        conda env remove -n ${ENV_NAME} -y
    else
        echo "Using existing environment."
        eval "$(conda shell.bash hook)"
        conda activate ${ENV_NAME}
    fi
else
    echo "Creating conda environment: ${ENV_NAME}"
    conda create -n ${ENV_NAME} python=3.9 -y
    eval "$(conda shell.bash hook)"
    conda activate ${ENV_NAME}
fi

# Activate environment
eval "$(conda shell.bash hook)"
conda activate ${ENV_NAME}

echo ""
echo "Installing Python packages..."
pip install -r requirements.txt

echo ""
echo "Installing scispacy model..."
pip install scispacy
pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.1/en_core_sci_md-0.5.1.tar.gz

echo ""
echo "Downloading ClinicalBERT tokenizer..."
python -c "from transformers import AutoTokenizer; print('Downloading...'); AutoTokenizer.from_pretrained('emilyalsentzer/Bio_ClinicalBERT'); print('Done!')"

echo ""
echo "Verifying installation..."
python test_setup.py

echo ""
echo "========================================"
echo "Setup Complete!"
echo "========================================"
echo ""
echo "Next steps:"
echo "  1. Update config/config.yaml with your data paths"
echo "  2. Set ANTHROPIC_API_KEY environment variable (optional)"
echo "  3. Test with: python test_sample.py"
echo "  4. Run full preprocessing: python main.py"
echo ""
