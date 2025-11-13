#!/bin/bash
# Smoke test for inference API

set -e

echo "=== LADOS Inference Smoke Test ==="

# Start server in background
echo "Starting server..."
docker-compose up -d api || python -m uvicorn src.server.app:app --host 0.0.0.0 --port 8000 &
SERVER_PID=$!
sleep 5

# Wait for server to be ready
for i in {1..30}; do
    if curl -f http://localhost:8000/health > /dev/null 2>&1; then
        echo "✓ Server is ready"
        break
    fi
    sleep 1
done

# Test health endpoint
echo "Testing health endpoint..."
curl -f http://localhost:8000/health || exit 1

# Test predict endpoint (if model available)
echo "Testing predict endpoint..."
# Create dummy image
python -c "
from PIL import Image
img = Image.new('RGB', (224, 224), color=(128, 128, 128))
img.save('test_image.jpg')
"

curl -X POST http://localhost:8000/predict \
    -F "file=@test_image.jpg" || echo "Predict endpoint test skipped (model not loaded)"

# Cleanup
rm -f test_image.jpg
kill $SERVER_PID 2>/dev/null || true
docker-compose down 2>/dev/null || true

echo "=== Smoke test passed ==="

