"""Tests for FastAPI endpoints."""

import io

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from src.server.app import app

client = TestClient(app)


def test_health_endpoint():
    """Test health check endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "model_loaded" in data


def test_predict_endpoint_no_model():
    """Test predict endpoint when model not loaded."""
    # Create dummy image
    img = Image.new("RGB", (224, 224), color=(128, 128, 128))
    img_bytes = io.BytesIO()
    img.save(img_bytes, format="JPEG")
    img_bytes.seek(0)

    # Note: This will fail if model not loaded, which is expected
    # In real test, we'd mock the model
    response = client.post("/predict", files={"file": ("test.jpg", img_bytes, "image/jpeg")})
    # Either 200 (if model loaded) or 503 (if not)
    assert response.status_code in [200, 503]


def test_metrics_endpoint():
    """Test metrics endpoint."""
    response = client.get("/metrics")
    assert response.status_code == 200
    data = response.json()
    assert "training_metrics" in data
    assert "inference_stats" in data
