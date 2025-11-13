#!/bin/bash
# Smoke test for training pipeline

set -e

echo "=== LADOS Training Smoke Test ==="

# Download small sample
echo "Downloading small sample dataset..."
python scripts/download_lados.py \
    --output-dir data/raw \
    --processed-dir data/processed \
    --small-sample \
    --skip-conversion || echo "Download skipped (using existing data)"

# Run one epoch training
echo "Running one epoch training..."
python scripts/train.py \
    --config configs/train_resnet50.yaml \
    --dry-run || python scripts/train.py --config configs/train_resnet50.yaml

# Check output
if [ -f "runs/*/metrics.json" ]; then
    echo "✓ Training completed, metrics.json found"
else
    echo "✗ Training failed or metrics.json not found"
    exit 1
fi

echo "=== Smoke test passed ==="

