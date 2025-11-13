"""Celery task definitions."""

import os
import subprocess
from pathlib import Path
from typing import Dict, List

from .celery_app import celery_app


@celery_app.task(bind=True, name="task_train_job")
def task_train_job(self, job_id: str, config_path: str = "configs/train_resnet50.yaml") -> Dict:
    """
    Async training task.

    Args:
        job_id: Unique job identifier
        config_path: Path to training config YAML

    Returns:
        Job result dictionary
    """
    self.update_state(state="PROGRESS", meta={"progress": 0, "message": "Starting training"})

    try:
        # Run training script
        cmd = ["python", "scripts/train.py", "--config", config_path]
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=Path.cwd())

        if result.returncode == 0:
            return {
                "job_id": job_id,
                "status": "completed",
                "message": "Training completed successfully",
                "stdout": result.stdout,
            }
        else:
            return {
                "job_id": job_id,
                "status": "failed",
                "message": "Training failed",
                "stderr": result.stderr,
            }
    except Exception as e:
        return {
            "job_id": job_id,
            "status": "failed",
            "message": f"Training error: {str(e)}",
        }


@celery_app.task(bind=True, name="task_infer_batch")
def task_infer_batch(self, job_id: str, image_paths: List[str]) -> Dict:
    """
    Async batch inference task.

    Args:
        job_id: Unique job identifier
        image_paths: List of image file paths

    Returns:
        Batch prediction results
    """
    self.update_state(state="PROGRESS", meta={"progress": 0, "message": "Processing batch"})

    # Import here to avoid circular imports
    import torch
    from PIL import Image
    from torchvision import transforms

    from src.models import create_model

    # Load model
    model_path = os.getenv("MODEL_PATH", "runs/latest/checkpoint.pt")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(model_path, map_location=device)
    model = create_model(
        backbone="resnet50", num_classes=checkpoint.get("num_classes", 6), pretrained=False
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    # Preprocess
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    results = []
    total = len(image_paths)

    for idx, img_path in enumerate(image_paths):
        try:
            image = Image.open(img_path).convert("RGB")
            image_tensor = transform(image).unsqueeze(0).to(device)

            with torch.no_grad():
                outputs = model(image_tensor)
                probs = torch.softmax(outputs, dim=1)[0]
                pred_idx = probs.argmax().item()

            results.append(
                {
                    "filename": Path(img_path).name,
                    "predicted_class": pred_idx,
                    "confidence": float(probs[pred_idx].item()),
                }
            )

            # Update progress
            progress = int((idx + 1) / total * 100)
            self.update_state(
                state="PROGRESS",
                meta={"progress": progress, "message": f"Processed {idx+1}/{total}"},
            )
        except Exception as e:
            results.append({"filename": Path(img_path).name, "error": str(e)})

    return {
        "job_id": job_id,
        "status": "completed",
        "results": results,
        "count": len(results),
    }


@celery_app.task(bind=True, name="task_export_artifact")
def task_export_artifact(
    self, job_id: str, checkpoint_path: str, output_dir: str, formats: List[str] = None
) -> Dict:
    """
    Async model export task.

    Args:
        job_id: Unique job identifier
        checkpoint_path: Path to model checkpoint
        output_dir: Output directory for exports
        formats: Export formats (torchscript, onnx)

    Returns:
        Export result dictionary
    """
    if formats is None:
        formats = ["torchscript", "onnx"]

    try:
        cmd = [
            "python",
            "scripts/export_model.py",
            "--checkpoint",
            checkpoint_path,
            "--output-dir",
            output_dir,
            "--formats",
        ] + formats

        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode == 0:
            return {
                "job_id": job_id,
                "status": "completed",
                "output_dir": output_dir,
                "formats": formats,
            }
        else:
            return {
                "job_id": job_id,
                "status": "failed",
                "message": result.stderr,
            }
    except Exception as e:
        return {
            "job_id": job_id,
            "status": "failed",
            "message": str(e),
        }
