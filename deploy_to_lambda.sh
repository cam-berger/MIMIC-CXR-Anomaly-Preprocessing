#!/bin/bash
# Automated deployment script for Lambda GPU instance
# Deploys 200-sample validation preprocessing pipeline to Lambda GH200 instance

set -e

# Configuration
LAMBDA_IP="192.222.58.66"
LAMBDA_USER="ubuntu"
SSH_KEY="/home/dev/Downloads/berger-cm.pem"
LOCAL_BASE="/home/dev/Documents/Portfolio/MIMIC/MIMIC-CXR-Anomaly-Preprocessing"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo "================================================================"
echo -e "${BLUE}Lambda GPU Deployment Script${NC}"
echo "================================================================"
echo ""
echo "Target: ${LAMBDA_USER}@${LAMBDA_IP}"
echo "SSH Key: ${SSH_KEY}"
echo ""

# Verify SSH key exists
if [ ! -f "$SSH_KEY" ]; then
    echo -e "${RED}ERROR: SSH key not found: $SSH_KEY${NC}"
    exit 1
fi

# Set correct permissions on SSH key
chmod 400 "$SSH_KEY"

# Function to run commands on Lambda
run_remote() {
    ssh -i "$SSH_KEY" -o StrictHostKeyChecking=no "${LAMBDA_USER}@${LAMBDA_IP}" "$@"
}

# Function to transfer files to Lambda
transfer_file() {
    local src="$1"
    local dst="$2"
    echo -e "${YELLOW}Transferring: $(basename $src)${NC}"
    rsync -avz --progress -e "ssh -i $SSH_KEY -o StrictHostKeyChecking=no" "$src" "${LAMBDA_USER}@${LAMBDA_IP}:${dst}"
}

# ========================================
# Phase 1: Verify Prerequisites
# ========================================
echo -e "${BLUE}Phase 1: Verifying prerequisites...${NC}"
echo "-----------------------------------"

# Check if validation subset exists
if [ ! -d "${LOCAL_BASE}/validation_data_subset" ]; then
    echo -e "${RED}ERROR: Validation data subset not found!${NC}"
    echo "Please run: ./extract_validation_subset.sh"
    exit 1
fi

# Check if compressed archives exist or create them
echo "Checking/creating compressed archives..."

if [ ! -f "${LOCAL_BASE}/validation_data_subset.tar.gz" ]; then
    echo -e "${YELLOW}Creating validation_data_subset.tar.gz...${NC}"
    cd "$LOCAL_BASE"
    tar -czf validation_data_subset.tar.gz validation_data_subset/
    echo -e "${GREEN}✓ Created validation_data_subset.tar.gz${NC}"
else
    echo -e "${GREEN}✓ validation_data_subset.tar.gz exists${NC}"
fi

if [ ! -f "${LOCAL_BASE}/step2_preprocessing.tar.gz" ]; then
    echo -e "${YELLOW}Creating step2_preprocessing.tar.gz...${NC}"
    cd "$LOCAL_BASE"
    tar -czf step2_preprocessing.tar.gz \
        step2_preprocessing/ \
        run_precompilation.py \
        requirements.txt \
        README.md
    echo -e "${GREEN}✓ Created step2_preprocessing.tar.gz${NC}"
else
    echo -e "${GREEN}✓ step2_preprocessing.tar.gz exists${NC}"
fi

echo ""

# ========================================
# Phase 2: Test Lambda Connection
# ========================================
echo -e "${BLUE}Phase 2: Testing Lambda connection...${NC}"
echo "---------------------------------------"

if run_remote "echo 'Connection successful'"; then
    echo -e "${GREEN}✓ Lambda connection verified${NC}"
else
    echo -e "${RED}ERROR: Cannot connect to Lambda instance${NC}"
    exit 1
fi

# Check GPU availability
echo "Checking GPU availability..."
if run_remote "nvidia-smi --query-gpu=name --format=csv,noheader" 2>/dev/null; then
    GPU_NAME=$(run_remote "nvidia-smi --query-gpu=name --format=csv,noheader")
    echo -e "${GREEN}✓ GPU detected: $GPU_NAME${NC}"
else
    echo -e "${YELLOW}WARNING: nvidia-smi not available or GPU not detected${NC}"
fi

echo ""

# ========================================
# Phase 3: Create Directory Structure on Lambda
# ========================================
echo -e "${BLUE}Phase 3: Creating directory structure on Lambda...${NC}"
echo "--------------------------------------------------"

run_remote "mkdir -p ~/data ~/output ~/code ~/output/checkpoints"
echo -e "${GREEN}✓ Directory structure created${NC}"
echo ""

# ========================================
# Phase 4: Transfer Data to Lambda
# ========================================
echo -e "${BLUE}Phase 4: Transferring data to Lambda...${NC}"
echo "---------------------------------------"
echo "This may take 10-30 minutes depending on bandwidth..."
echo ""

# Transfer validation data subset (~3 GB)
transfer_file "${LOCAL_BASE}/validation_data_subset.tar.gz" "~/"

# Transfer code (~100 MB)
transfer_file "${LOCAL_BASE}/step2_preprocessing.tar.gz" "~/"

# Transfer Lambda config
transfer_file "${LOCAL_BASE}/step2_preprocessing/config/config_lambda.yaml" "~/"

echo -e "${GREEN}✓ All files transferred${NC}"
echo ""

# ========================================
# Phase 5: Extract Files on Lambda
# ========================================
echo -e "${BLUE}Phase 5: Extracting files on Lambda...${NC}"
echo "---------------------------------------"

run_remote "
cd ~ && \
echo 'Extracting validation data...' && \
tar -xzf validation_data_subset.tar.gz && \
mv validation_data_subset/* data/ && \
rmdir validation_data_subset && \
echo 'Extracting code...' && \
tar -xzf step2_preprocessing.tar.gz -C code/ && \
echo 'Extraction complete'
"

echo -e "${GREEN}✓ Files extracted${NC}"
echo ""

# ========================================
# Phase 6: Setup Python Environment
# ========================================
echo -e "${BLUE}Phase 6: Setting up Python environment...${NC}"
echo "----------------------------------------"

run_remote "
cd ~/code && \
python3 -m venv venv && \
source venv/bin/activate && \
pip install --upgrade pip && \
echo 'Installing dependencies (this may take 5-10 minutes)...' && \
pip install -r requirements.txt && \
echo 'Installing scispacy model...' && \
pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.1/en_core_sci_md-0.5.1.tar.gz && \
echo 'Python environment setup complete'
"

echo -e "${GREEN}✓ Python environment ready${NC}"
echo ""

# ========================================
# Phase 7: Configure Lambda Paths
# ========================================
echo -e "${BLUE}Phase 7: Configuring paths...${NC}"
echo "-----------------------------"

run_remote "
cp ~/config_lambda.yaml ~/code/step2_preprocessing/config/config.yaml && \
echo 'Configuration updated'
"

echo -e "${GREEN}✓ Configuration complete${NC}"
echo ""

# ========================================
# Phase 8: Verify Installation
# ========================================
echo -e "${BLUE}Phase 8: Verifying installation...${NC}"
echo "-----------------------------------"

run_remote "
source ~/code/venv/bin/activate && \
python3 -c 'import torch; print(f\"PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}\")' && \
python3 -c 'import spacy; nlp = spacy.load(\"en_core_sci_md\"); print(\"scispacy loaded\")' && \
python3 -c 'import h5py; import pyarrow; print(\"h5py and pyarrow loaded\")' && \
echo 'All dependencies verified'
"

echo -e "${GREEN}✓ Installation verified${NC}"
echo ""

# ========================================
# Phase 9: Create Run Script on Lambda
# ========================================
echo -e "${BLUE}Phase 9: Creating run script on Lambda...${NC}"
echo "-----------------------------------------"

run_remote "cat > ~/run_precompilation.sh << 'EOFSCRIPT'
#!/bin/bash
# Pre-compilation script for Lambda GPU instance

set -e

cd ~/code
source venv/bin/activate

# Set environment variables
export CUDA_VISIBLE_DEVICES=0
export ANTHROPIC_API_KEY='${ANTHROPIC_API_KEY:-your-api-key-here}'

# Verify API key is set
if [ \"\$ANTHROPIC_API_KEY\" = \"your-api-key-here\" ]; then
    echo \"WARNING: ANTHROPIC_API_KEY not set! Please set it in your environment.\"
    echo \"Run: export ANTHROPIC_API_KEY='your-key'\"
    exit 1
fi

echo \"=======================================\"
echo \"Starting Pre-compilation Pipeline\"
echo \"=======================================\"
echo \"\"
echo \"Configuration:\"
echo \"  - Samples: 200 (validation subset)\"
echo \"  - Mode: Pre-compilation (Step 2.5)\"
echo \"  - Features: CXR-PRO reports + Claude summarization\"
echo \"  - Output: /home/ubuntu/output/val/batch_0/\"
echo \"\"

# Run pre-compilation
python run_precompilation.py \\
    --config step2_preprocessing/config/config.yaml \\
    --split val \\
    --max-samples 200 \\
    2>&1 | tee ~/output/precompilation.log

echo \"\"
echo \"=======================================\"
echo \"Pre-compilation Complete!\"
echo \"=======================================\"
echo \"\"
echo \"Output location: ~/output/val/batch_0/\"
echo \"Log file: ~/output/precompilation.log\"
echo \"\"
echo \"Next steps:\"
echo \"  1. Run validation: python step2_preprocessing/validate_mae_readiness.py --output-dir ~/output/val/batch_0\"
echo \"  2. Download results: See deployment guide\"
echo \"\"
EOFSCRIPT

chmod +x ~/run_precompilation.sh
echo 'Run script created'
"

echo -e "${GREEN}✓ Run script ready${NC}"
echo ""

# ========================================
# Deployment Complete
# ========================================
echo "================================================================"
echo -e "${GREEN}Deployment Complete!${NC}"
echo "================================================================"
echo ""
echo "Next steps:"
echo ""
echo "1. Set your Anthropic API key on Lambda:"
echo "   ssh -i $SSH_KEY ${LAMBDA_USER}@${LAMBDA_IP}"
echo "   export ANTHROPIC_API_KEY='your-api-key-here'"
echo ""
echo "2. Run pre-compilation (3-5 hours):"
echo "   ~/run_precompilation.sh"
echo ""
echo "3. Monitor progress:"
echo "   tail -f ~/output/precompilation.log"
echo ""
echo "4. Validate results:"
echo "   source ~/code/venv/bin/activate"
echo "   python ~/code/step2_preprocessing/validate_mae_readiness.py --output-dir ~/output/val/batch_0"
echo ""
echo "5. Download results to local machine:"
echo "   scp -i $SSH_KEY ${LAMBDA_USER}@${LAMBDA_IP}:~/output/mae_readiness_report.json ./"
echo ""
echo "================================================================"
echo "Estimated costs: ~\$40-50 (Lambda GPU + Claude API)"
echo "Estimated time: ~6-8 hours total"
echo "================================================================"