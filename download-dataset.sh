#!/bin/bash
# Download the references dataset from the official Rinha 2026 repository
set -e

RESOURCES_DIR="$(cd "$(dirname "$0")" && pwd)/resources"
OUTPUT="$RESOURCES_DIR/references.json.gz"

if [ -f "$OUTPUT" ]; then
    echo "references.json.gz already exists, skipping download."
    exit 0
fi

echo "Downloading references.json.gz (~16MB)..."
curl -L -o "$OUTPUT" \
    "https://github.com/zanfranceschi/rinha-de-backend-2026/raw/main/resources/references.json.gz"

echo "Done! File saved to $OUTPUT"
ls -lh "$OUTPUT"
