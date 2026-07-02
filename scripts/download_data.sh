#!/usr/bin/env bash
set -euo pipefail

# download_data.sh
# Downloads raw data files for the Bitcoin Sentiment Trader Performance project.
# Place Fear & Greed Index and Hyperliquid Trader History CSVs into data/raw/.
#
# Usage:
#   ./scripts/download_data.sh [--force]
#
# Note: Hyperliquid data must be exported manually from the exchange UI.
# This script only handles the Fear & Greed Index (publicly available).

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
FORCE="${1:-}"

# --- Fear & Greed Index ---
FG_DIR="$PROJECT_ROOT/data/raw/fear_greed"
FG_FILE="$FG_DIR/fear_greed_index.csv"
FG_URL="https://raw.githubusercontent.com/your-org/fear-greed-data/main/fear_greed_index.csv"

mkdir -p "$FG_DIR"

if [ -f "$FG_FILE" ] && [ "$FORCE" != "--force" ]; then
    echo "Fear & Greed data already exists at $FG_FILE"
    echo "Use --force to re-download."
else
    echo "Downloading Fear & Greed Index..."
    if command -v curl &> /dev/null; then
        curl -sL "$FG_URL" -o "$FG_FILE"
    elif command -v wget &> /dev/null; then
        wget -q "$FG_URL" -O "$FG_FILE"
    else
        echo "ERROR: Neither curl nor wget found. Install one and retry."
        exit 1
    fi
    echo "Saved to $FG_FILE"
fi

# --- Trader History ---
TH_DIR="$PROJECT_ROOT/data/raw/trader_history"
TH_FILE="$TH_DIR/historical_data.csv"

mkdir -p "$TH_DIR"

if [ -f "$TH_FILE" ]; then
    echo "Trader history data already exists at $TH_FILE"
else
    echo ""
    echo "=== ACTION REQUIRED ==="
    echo "Trader history data must be exported manually from the Hyperliquid exchange."
    echo "Export your CSV and place it at:"
    echo "  $TH_FILE"
    echo ""
    echo "Expected columns: Trade ID, Account, Timestamp, Side, Direction,"
    echo "                  Size USD, Execution Price, Closed PnL, Fee"
fi

echo ""
echo "Setup complete."
echo "Next step: conda activate bst && python pipelines/run_full_pipeline.py --config configs/base.yaml"
