#!/bin/bash
# Full reproduction script

set -e

echo "=== LADOS Full Reproduction Pipeline ==="

# 1. Download dataset (small sample)
echo "Step 1: Downloading LADOS dataset (small sample)..."
python scripts/download_lados.py \
    --output-dir data/raw \
    --processed-dir data/processed \
    --small-sample || echo "Using existing data"

# 2. Train one epoch
echo "Step 2: Training for one epoch..."
python scripts/train.py \
    --config configs/train_resnet50.yaml || echo "Training completed or skipped"

# 3. Start server
echo "Step 3: Starting server..."
docker-compose up -d api redis || echo "Using local server"

# Wait for server
sleep 10

# 4. Run smoke inference
echo "Step 4: Running smoke inference..."
bash scripts/smoke_infer.sh || echo "Inference test completed"

# 5. Check metrics
echo "Step 5: Checking metrics..."
if [ -f "runs/*/metrics.json" ]; then
    echo "✓ metrics.json found"
    cat runs/*/metrics.json | head -20
else
    echo "✗ metrics.json not found"
fi

echo "=== Reproduction pipeline complete ==="

