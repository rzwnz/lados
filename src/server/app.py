"""FastAPI application with inference and training endpoints."""

import io
import os
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import torch
from fastapi import FastAPI, File, HTTPException, Header, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image
from pydantic import BaseModel
from prometheus_client import Counter, Histogram, generate_latest

from src.models import create_model

# Prometheus metrics
inference_counter = Counter("inference_requests_total", "Total inference requests")
inference_latency = Histogram("inference_latency_seconds", "Inference latency")
batch_job_counter = Counter("batch_jobs_total", "Total batch jobs")


class PredictionResponse(BaseModel):
    """Prediction response model."""

    model_version: str
    predictions: List[dict]
    timestamp: str
    inference_time_ms: float


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    model_loaded: bool
    device: str


class MetricsResponse(BaseModel):
    """Metrics response model."""

    training_metrics: Optional[dict] = None
    inference_stats: dict


# Global model state
model = None
model_version = os.getenv("MODEL_VERSION", "v1.0")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class_names = []


def load_model(model_path: str, backbone: str = "resnet50", num_classes: int = 6) -> None:
    """Load model from checkpoint."""
    global model, class_names

    checkpoint_path = Path(model_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    class_names = checkpoint.get("class_names", [f"class_{i}" for i in range(num_classes)])

    # Use actual number of classes from checkpoint
    actual_num_classes = len(class_names) if class_names else num_classes
    # Also check the checkpoint state dict to determine num_classes
    if "model_state_dict" in checkpoint:
        # Check the last layer size
        state_dict = checkpoint["model_state_dict"]
        for key in ["fc.weight", "classifier.1.weight", "heads.head.weight"]:
            if key in state_dict:
                actual_num_classes = state_dict[key].shape[0]
                break

    model = create_model(backbone=backbone, num_classes=actual_num_classes, pretrained=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    print(f"✓ Model loaded from {model_path} on {device} ({actual_num_classes} classes)")


def predict_image(image: Image.Image) -> dict:
    """Run inference on a single image."""
    from torchvision import transforms

    # Preprocess
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    image_tensor = transform(image).unsqueeze(0).to(device)

    # Inference
    start_time = time.time()
    with torch.no_grad():
        outputs = model(image_tensor)
        probs = torch.softmax(outputs, dim=1)[0]
        pred_idx = probs.argmax().item()

    inference_time = (time.time() - start_time) * 1000  # ms

    # Format predictions
    predictions = []
    for idx, prob in enumerate(probs.cpu().numpy()):
        predictions.append({"class": class_names[idx], "score": float(prob)})

    # Sort by score
    predictions.sort(key=lambda x: x["score"], reverse=True)

    return {
        "predictions": predictions,
        "top_class": class_names[pred_idx],
        "top_score": float(probs[pred_idx].item()),
        "inference_time_ms": inference_time,
    }


# Initialize FastAPI app
app = FastAPI(title="LADOS Classification API", version="1.0.0")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup_event():
    """Load model on startup."""
    model_path = os.getenv("MODEL_PATH", "runs/latest/checkpoint.pt")
    backbone = os.getenv("DEFAULT_BACKBONE", "resnet50")
    num_classes = int(os.getenv("NUM_CLASSES", "6"))

    try:
        load_model(model_path, backbone, num_classes)
    except Exception as e:
        print(f"Warning: Could not load model: {e}")
        print("Model will need to be loaded manually via /load_model endpoint")


@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "device": str(device),
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    """
    Predict class for a single image.

    Returns:
        Prediction with class probabilities
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    # Validate file
    if file.size > int(os.getenv("MAX_UPLOAD_SIZE_MB", "10")) * 1024 * 1024:
        raise HTTPException(status_code=400, detail="File too large")

    # Read and validate image
    try:
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image: {e}")

    # Predict
    inference_counter.inc()
    with inference_latency.time():
        result = predict_image(image)

    return PredictionResponse(
        model_version=model_version,
        predictions=result["predictions"],
        timestamp=datetime.utcnow().isoformat() + "Z",
        inference_time_ms=result["inference_time_ms"],
    )


@app.post("/predict_batch")
async def predict_batch(files: List[UploadFile] = File(...)):
    """
    Predict classes for multiple images.

    For batches > threshold, queues a Celery job.
    """
    batch_threshold = int(os.getenv("BATCH_JOB_THRESHOLD", "16"))

    if len(files) > batch_threshold:
        # Queue Celery job
        from src.tasks import task_infer_batch

        job_id = str(uuid.uuid4())
        # In real implementation, save files and enqueue task
        batch_job_counter.inc()
        return {
            "job_id": job_id,
            "status": "queued",
            "message": f"Batch job queued ({len(files)} images)",
        }
    else:
        # Process synchronously
        results = []
        for file in files:
            try:
                image_bytes = await file.read()
                image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
                result = predict_image(image)
                results.append(
                    {
                        "filename": file.filename,
                        "predictions": result["predictions"],
                        "top_class": result["top_class"],
                        "top_score": result["top_score"],
                    }
                )
            except Exception as e:
                results.append({"filename": file.filename, "error": str(e)})

        return {"results": results, "count": len(results)}


@app.post("/train")
async def trigger_training(
    x_api_key: Optional[str] = Header(None),
):
    """
    Trigger model retraining (protected endpoint).

    Requires API key authentication.
    """
    api_key = os.getenv("API_KEY")
    if api_key and x_api_key != api_key:
        raise HTTPException(status_code=401, detail="Invalid API key")

    from src.tasks import task_train_job

    job_id = str(uuid.uuid4())
    # Enqueue training task
    # task_train_job.delay(job_id)

    return {"job_id": job_id, "status": "queued", "message": "Training job queued"}


@app.get("/metrics", response_model=MetricsResponse)
async def get_metrics():
    """Get training and inference metrics."""
    # Load latest metrics from runs directory
    metrics = {}
    runs_dir = Path("runs")
    if runs_dir.exists():
        # Find latest run
        run_dirs = sorted(runs_dir.glob("*"), key=lambda x: x.stat().st_mtime, reverse=True)
        if run_dirs:
            metrics_path = run_dirs[0] / "metrics.json"
            if metrics_path.exists():
                import json

                with open(metrics_path) as f:
                    metrics = json.load(f)

    # Get Prometheus metrics
    from prometheus_client import REGISTRY

    total_requests = 0
    avg_latency_ms = 0.0

    # Try to get metrics from registry
    for collector in REGISTRY._collector_to_names:
        if hasattr(collector, "_value"):
            if "inference_requests_total" in str(collector):
                total_requests = int(
                    collector._value.get() if hasattr(collector._value, "get") else 0
                )

    return MetricsResponse(
        training_metrics=metrics.get("final_metrics"),
        inference_stats={
            "total_requests": total_requests,
            "avg_latency_ms": avg_latency_ms,
        },
    )


@app.get("/metrics/prometheus")
async def prometheus_metrics():
    """Prometheus metrics endpoint."""
    return generate_latest()


@app.get("/job/{job_id}")
async def get_job_status(job_id: str):
    """Get status of async job."""
    # In real implementation, query Celery/Redis for job status
    return {"job_id": job_id, "status": "unknown", "message": "Job status not implemented"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
