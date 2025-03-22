#!/bin/bash
set -euo pipefail

# Download directory (relative to the script)
DATA_DIR="$(dirname "$0")"
echo "Downloading datasets to $DATA_DIR"

SUBDIR="nearest-neighbors-datasets"

# Create the path if it doesn't exist
mkdir -p "$DATA_DIR/$SUBDIR"

# Download the datasets from the Hugging Face Hub
huggingface-cli download habedi/nearest-neighbors-datasets --repo-type dataset --local-dir "$DATA_DIR/$SUBDIR"

echo "Download complete!"
