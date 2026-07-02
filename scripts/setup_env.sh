#!/usr/bin/env bash
set -euo pipefail

# setup_env.sh
# Bootstraps the development environment for Bitcoin Sentiment Trader Performance.
#
# Usage:
#   ./scripts/setup_env.sh               # Interactive (asks about conda vs pip)
#   ./scripts/setup_env.sh --conda        # Use conda environment
#   ./scripts/setup_env.sh --venv         # Use Python venv
#
# Prerequisites:
#   - conda (if --conda) or python3.11+ (if --venv)
#   - pip

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

# --- Parse flags ---
USE_CONDA=false
USE_VENV=false

case "${1:-}" in
    --conda) USE_CONDA=true ;;
    --venv)  USE_VENV=true ;;
    "")
        echo "Select environment manager:"
        select choice in conda venv quit; do
            case $choice in
                conda) USE_CONDA=true; break;;
                venv)  USE_VENV=true; break;;
                quit)  exit 0;;
            esac
        done
        ;;
    *)
        echo "Usage: $0 [--conda | --venv]"
        exit 1
        ;;
esac

# --- Conda setup ---
if [ "$USE_CONDA" = true ]; then
    echo "Setting up conda environment from environment/conda.yaml..."
    if ! command -v conda &> /dev/null; then
        echo "ERROR: conda not found. Install Miniconda or Anaconda first."
        echo "  https://docs.conda.io/en/latest/miniconda.html"
        exit 1
    fi
    conda env create -f environment/conda.yaml --yes
    echo ""
    echo "Conda environment 'bst' created."
    echo "Activate with: conda activate bst"
fi

# --- venv setup ---
if [ "$USE_VENV" = true ]; then
    echo "Setting up Python virtual environment..."
    python3 -m venv .venv
    source .venv/bin/activate
    pip install --upgrade pip
    pip install -r requirements.txt
    pip install -r requirements-dev.txt
    echo ""
    echo "Virtual environment created at .venv/"
    echo "Activate with: source .venv/bin/activate"
fi

# --- .env file (if missing) ---
if [ ! -f .env ]; then
    echo ""
    echo "Creating .env from .env.example..."
    cp .env.example .env
    echo "Edit .env with your local paths and credentials."
fi

# --- Pre-commit hooks (optional) ---
if command -v pre-commit &> /dev/null; then
    echo ""
    echo "Installing pre-commit hooks..."
    pre-commit install
fi

echo ""
echo "Environment setup complete."
echo "Next: python pipelines/run_full_pipeline.py --config configs/base.yaml"
