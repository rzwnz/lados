"""Model registry for easy backbone swapping."""

from typing import Optional

import torch
import torch.nn as nn
import torchvision.models as tv_models

try:
    import timm
except ImportError:
    timm = None


def create_model(
    backbone: str = "resnet50",
    num_classes: int = 6,
    pretrained: bool = True,
    freeze_backbone: bool = False,
) -> nn.Module:
    """
    Create a classification model with specified backbone.

    Args:
        backbone: Model backbone ('resnet50', 'efficientnet_b0', 'vit_b_16')
        num_classes: Number of output classes
        pretrained: Use ImageNet pretrained weights
        freeze_backbone: Freeze backbone parameters for transfer learning

    Returns:
        PyTorch model
    """
    if backbone == "resnet50":
        model = tv_models.resnet50(pretrained=pretrained)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        if freeze_backbone:
            for param in model.parameters():
                param.requires_grad = False
            model.fc.requires_grad_(True)

    elif backbone == "efficientnet_b0":
        if timm:
            model = timm.create_model(
                "efficientnet_b0", pretrained=pretrained, num_classes=num_classes
            )
        else:
            model = tv_models.efficientnet_b0(pretrained=pretrained)
            model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
        if freeze_backbone:
            for name, param in model.named_parameters():
                if "classifier" not in name:
                    param.requires_grad = False

    elif backbone == "vit_b_16":
        if timm:
            model = timm.create_model(
                "vit_base_patch16_224", pretrained=pretrained, num_classes=num_classes
            )
        else:
            model = tv_models.vit_b_16(pretrained=pretrained)
            model.heads.head = nn.Linear(model.heads.head.in_features, num_classes)
        if freeze_backbone:
            for name, param in model.named_parameters():
                if "heads" not in name:
                    param.requires_grad = False

    else:
        raise ValueError(
            f"Unknown backbone: {backbone}. "
            f"Available: {list_available_models()}"
        )

    return model


def list_available_models() -> list[str]:
    """List available model backbones."""
    return ["resnet50", "efficientnet_b0", "vit_b_16"]


def unfreeze_last_n_layers(model: nn.Module, n: int = 2) -> None:
    """
    Unfreeze last N layers for fine-tuning.

    Args:
        model: PyTorch model
        n: Number of layers to unfreeze (from end)
    """
    # Get all parameters
    params = list(model.named_parameters())
    # Unfreeze last n layers
    for name, param in params[-n:]:
        param.requires_grad = True

