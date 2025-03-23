#!/bin/bash
set -euo pipefail

# Default values for arguments
DEFAULT_DATA_DIR="example/data"
DEFAULT_DATASET="nearest-neighbors-datasets"
HF_USERNAME="habedi"

# Use provided arguments or defaults
DATA_DIR="${1:-$DEFAULT_DATA_DIR}"
DATASET="${2:-$DEFAULT_DATASET}"

echo "Downloading datasets to $DATA_DIR/$DATASET"

# Create the path if it doesn't exist
mkdir -p "$DATA_DIR/$DATASET"

# Download the datasets from the Hugging Face Hub
huggingface-cli download $HF_USERNAME/$DATASET --repo-type dataset --local-dir "$DATA_DIR/$DATASET"

echo "Download complete!"
