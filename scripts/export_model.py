#!/usr/bin/env python3
"""Export trained model to TorchScript and ONNX formats."""

import argparse
from pathlib import Path

import torch

from src.models import create_model


def export_torchscript(model: torch.nn.Module, output_path: Path, img_size: int = 224) -> None:
    """Export model to TorchScript."""
    model.eval()
    dummy_input = torch.randn(1, 3, img_size, img_size)

    try:
        traced_model = torch.jit.trace(model, dummy_input)
        traced_model.save(str(output_path))
        print(f"✓ Exported TorchScript to: {output_path}")
    except Exception as e:
        print(f"✗ TorchScript export failed: {e}")


def export_onnx(model: torch.nn.Module, output_path: Path, img_size: int = 224) -> None:
    """Export model to ONNX."""
    model.eval()
    dummy_input = torch.randn(1, 3, img_size, img_size)

    try:
        torch.onnx.export(
            model,
            dummy_input,
            str(output_path),
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
            opset_version=13,
        )
        print(f"✓ Exported ONNX to: {output_path}")
    except Exception as e:
        print(f"✗ ONNX export failed: {e}")


def main():
    parser = argparse.ArgumentParser(description="Export model to TorchScript/ONNX")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to checkpoint")
    parser.add_argument("--output-dir", type=Path, required=True, help="Output directory")
    parser.add_argument("--backbone", type=str, default="resnet50", help="Model backbone")
    parser.add_argument("--num-classes", type=int, default=6, help="Number of classes")
    parser.add_argument("--img-size", type=int, default=224, help="Image size")
    parser.add_argument("--formats", nargs="+", default=["torchscript", "onnx"], help="Export formats")

    args = parser.parse_args()

    # Load checkpoint
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    class_names = checkpoint.get("class_names", [f"class_{i}" for i in range(args.num_classes)])

    # Create model
    model = create_model(
        backbone=args.backbone, num_classes=args.num_classes, pretrained=False
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # Export
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if "torchscript" in args.formats:
        export_torchscript(model, args.output_dir / "model.pt", args.img_size)

    if "onnx" in args.formats:
        export_onnx(model, args.output_dir / "model.onnx", args.img_size)

    print("Export complete!")


if __name__ == "__main__":
    main()

