#!/bin/bash
# SAM3 Web Tracker — Setup Script
# Run this once to install all dependencies

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=========================================="
echo "  SAM3 Web Tracker — Environment Setup"
echo "=========================================="

# ── Activate sam3 conda env ────────────────────────────────────────────────
if ! command -v conda >/dev/null 2>&1; then
    echo "ERROR: conda not found on PATH. Install Miniconda/Miniforge first."
    exit 1
fi
source "$(conda info --base)/etc/profile.d/conda.sh"
if ! conda env list | awk '{print $1}' | grep -qx sam3; then
    echo "Creating conda env 'sam3' (Python 3.12)..."
    conda create -y -n sam3 python=3.12
fi
conda activate sam3

echo ""
echo "Step 1/4: Installing Python backend dependencies..."
pip install -r "$SCRIPT_DIR/backend/requirements.txt"

echo ""
echo "Step 2/4: Installing SAM3 from GitHub..."
SAM3_DIR="$SCRIPT_DIR/.sam3_src"
if [ -d "$SAM3_DIR" ]; then
    echo "  SAM3 source already cloned, pulling latest..."
    git -C "$SAM3_DIR" pull
else
    echo "  Cloning SAM3 repository..."
    git clone https://github.com/facebookresearch/sam3.git "$SAM3_DIR"
fi
pip install -e "$SAM3_DIR"

echo ""
echo "Step 3/4: Installing Node.js frontend dependencies..."
cd "$SCRIPT_DIR/frontend"
npm install
cd "$SCRIPT_DIR"

# Point git at tracked hooks (.githooks/prepare-commit-msg strips Cursor commit trailers).
if git -C "$SCRIPT_DIR" rev-parse --git-dir >/dev/null 2>&1; then
  git -C "$SCRIPT_DIR" config core.hooksPath .githooks
  echo ""
  echo "Git hooks registered (see .githooks/; strips Made-with: Cursor trailers on commit)."
fi

echo ""
echo "Step 4/4: Downloading SAM3 model weights..."
echo "  NOTE: The SAM3 model is gated on HuggingFace."
echo "  You need to:"
echo "    1. Request access at: https://huggingface.co/facebook/sam3"
echo "    2. Create a token at: https://huggingface.co/settings/tokens"
echo "    3. Set: export HF_TOKEN=hf_your_token_here"
echo "    4. Run: conda run -n sam3 python scripts/download_model.py"
echo ""
if [ -n "$HF_TOKEN" ]; then
    echo "  HF_TOKEN found, attempting download..."
    conda run -n sam3 python "$SCRIPT_DIR/scripts/download_model.py"
else
    echo "  Skipping model download (HF_TOKEN not set)."
fi

echo ""
echo "=========================================="
echo "  Setup complete!"
echo ""
echo "  To download the model later:"
echo "    export HF_TOKEN=hf_your_token"
echo "    conda run -n sam3 python scripts/download_model.py"
echo ""
echo "  To start the app:"
echo "    bash start_backend.sh   # in one terminal"
echo "    bash start_frontend.sh  # in another terminal"
echo ""
echo "  Then open: http://localhost:5173"
echo "=========================================="
