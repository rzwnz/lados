"""Tests for training pipeline."""

import tempfile
from pathlib import Path

import pytest
import torch

from src.models import create_model


def test_model_creation():
    """Test model creation."""
    model = create_model(backbone="resnet50", num_classes=6, pretrained=False)
    assert model is not None

    # Test forward pass
    dummy_input = torch.randn(1, 3, 224, 224)
    output = model(dummy_input)
    assert output.shape == (1, 6)


def test_one_batch_training():
    """Test one batch training step."""
    model = create_model(backbone="resnet50", num_classes=6, pretrained=False)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    # Dummy batch
    batch_size = 4
    images = torch.randn(batch_size, 3, 224, 224)
    labels = torch.randint(0, 6, (batch_size,))

    # Forward
    outputs = model(images)
    loss = criterion(outputs, labels)

    # Backward
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    assert loss.item() > 0

