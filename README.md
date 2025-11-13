# LADOS Classification Project

End-to-end PyTorch image classification system for the **LADOS** (Large-scale Aerial Dataset for Oil Spill detection) dataset. This project provides training, fine-tuning, evaluation, inference serving, and a complete observability stack.

## 🎯 Features

- **Multi-backbone support**: ResNet50, EfficientNet-B0, ViT-B/16
- **Training pipeline**: Mixed precision, gradient accumulation, early stopping
- **FastAPI backend**: RESTful API for inference and training
- **Async processing**: Celery + Redis for batch jobs
- **Observability**: Prometheus, Grafana, ELK stack
- **Frontend options**: Flutter web app and Streamlit app
- **Docker Compose**: Complete local development environment
- **Health checks**: Automated service verification scripts

## 📋 Requirements

- Python ≥3.11
- CUDA-capable GPU (RTX 4060 8GB VRAM recommended) or CPU
- Docker & Docker Compose
- Flutter SDK (for Flutter frontend, optional)
- 1GB+ free disk space for dataset

## 🚀 Quick Start

### 1. Clone and Setup

```bash
git clone <repository-url>
cd lados
```

### 2. Install Dependencies

**Option A: Using Poetry (recommended)**
```bash
poetry install
poetry shell
```

**Option B: Using pip**
```bash
pip install -r requirements.txt
```

### 3. Configure Environment

```bash
cp env.example .env
# Edit .env with your settings (API keys, paths, etc.)
```

### 4. Download Dataset

**Small sample mode (for quick testing):**
```bash
python scripts/download_lados.py \
    --output-dir data/raw \
    --processed-dir data/processed \
    --small-sample
```

**Full dataset:**
```bash
# Set ROBOFLOW_API_KEY environment variable if using Roboflow
export ROBOFLOW_API_KEY=your_api_key
python scripts/download_lados.py \
    --output-dir data/raw \
    --processed-dir data/processed
```

### 5. Train Model

```bash
# Train ResNet50 (default)
python scripts/train.py --config configs/train_resnet50.yaml

# Train EfficientNet-B0
python scripts/train.py --config configs/train_efficientnet_b0.yaml

# Train ViT-B/16
python scripts/train.py --config configs/train_vit.yaml

# Dry run (one batch)
python scripts/train.py --config configs/train_resnet50.yaml --dry-run
```

### 6. Start Services

**Using Docker Compose (recommended):**
```bash
docker compose up -d
```

This starts:
- FastAPI server (http://localhost:8000)
- Celery worker
- Redis
- Elasticsearch
- Kibana (http://localhost:5601)
- Grafana (http://localhost:3000)
- Prometheus (http://localhost:9090)

**Verify all services are running:**
```bash
python scripts/check_services.py
```

**Or run locally:**
```bash
# Start FastAPI server
uvicorn src.server.app:app --host 0.0.0.0 --port 8000

# Start Celery worker (in another terminal)
celery -A src.tasks.celery_app worker --loglevel=info
```

### 7. Test Inference

```bash
# Health check
curl http://localhost:8000/health

# Single image prediction
curl -X POST http://localhost:8000/predict \
  -F "file=@path/to/image.jpg"

# Get metrics
curl http://localhost:8000/metrics
```

### 8. Run Frontend

**Option 1: Flutter Web App (Recommended)**

See [flutter/sample_app/README.md](flutter/sample_app/README.md) for detailed setup instructions.

```bash
cd flutter/sample_app
source ~/.zshrc  # Load Flutter PATH if needed
flutter pub get
flutter run -d chrome --web-port 8080
```

**Option 2: Streamlit App (Backup)**

Simple Python-based frontend. Requires `streamlit` and `plotly`:

```bash
# Install dependencies (if not already installed)
pip install streamlit plotly requests

# Make sure FastAPI server is running
streamlit run streamlit_app.py
# App opens at http://localhost:8501
```

Features:
- Single and batch image upload
- Interactive Plotly charts
- Real-time metrics display
- Health status monitoring

To change the API URL, edit `API_URL` in `streamlit_app.py`.

## 📁 Project Structure

```
lados/
├── configs/              # Training configurations (YAML)
├── data/                 # Dataset (raw & processed)
│   └── README.md        # Dataset documentation
├── manifests/            # Train/val/test CSV manifests
├── runs/                 # Training outputs & checkpoints
├── scripts/              # Utility scripts
│   ├── download_lados.py
│   ├── train.py
│   ├── export_model.py
│   ├── check_services.py  # Health check script
│   └── check_services.sh  # Health check script (bash)
├── src/
│   ├── data/            # Dataset & transforms
│   ├── models/           # Model definitions
│   ├── eval/             # Metrics & plotting
│   ├── server/           # FastAPI application
│   └── tasks/            # Celery tasks
├── tests/                # Unit tests
├── infra/                # Airflow DAGs
├── observability/        # Prometheus, Grafana, ELK configs
├── flutter/              # Flutter web app
│   └── sample_app/       # Flutter application
├── ops/                  # CI/CD workflows
├── docker-compose.yml    # Docker services
├── Dockerfile            # API container
├── requirements.txt      # Python dependencies
├── streamlit_app.py     # Streamlit frontend
└── README.md
```

## 🔧 Configuration

### Training Config

Edit `configs/train_resnet50.yaml` (or other backbone configs):

```yaml
data:
  root: "data/processed"
  img_size: 224

model:
  backbone: "resnet50"
  num_classes: 6
  pretrained: true

training:
  epochs: 50
  batch_size: 16
  mixed_precision: true
  gradient_accumulation_steps: 2
  early_stopping:
    enabled: true
    patience: 10
    metric: "val_macro_f1"
```

### Environment Variables

Key variables in `.env`:

- `MODEL_PATH`: Path to model checkpoint (default: `runs/latest/checkpoint.pt`)
- `API_KEY`: API key for `/train` endpoint
- `CELERY_BROKER_URL`: Redis broker URL
- `MAX_UPLOAD_SIZE_MB`: Max upload size (default: 10MB)
- `BATCH_JOB_THRESHOLD`: Batch size threshold for async jobs (default: 16)

## 📊 Dataset

The LADOS dataset contains **3,388** pixel-level annotated aerial images with **6 classification classes**:
- `oil` - Pure oil spills
- `emulsion` - Oil emulsions
- `oil_platform` - Oil platforms
- `sheen` - Oil sheen
- `ship` - Ships/vessels
- `background` - Background/no annotations

For detailed dataset information, see [data/README.md](data/README.md).

## 🧪 Testing

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ -v --cov=src --cov-report=html

# Run specific test
pytest tests/test_data.py -v
```

## 🔍 Smoke Tests

```bash
# Training smoke test
bash scripts/smoke_train.sh

# Inference smoke test
bash scripts/smoke_infer.sh
```

## 📈 Monitoring & Health Checks

### Automated Health Check

```bash
# Python script (recommended)
python scripts/check_services.py

# Bash script
./scripts/check_services.sh
```

For detailed health check information, see [SERVICE_HEALTH_CHECK.md](SERVICE_HEALTH_CHECK.md).

### Prometheus Metrics

- `inference_requests_total`: Total inference requests
- `inference_latency_seconds`: Inference latency histogram
- `batch_jobs_total`: Total batch jobs

Access: http://localhost:8000/metrics/prometheus

### Grafana Dashboard

Import dashboard from `observability/grafana/dashboard.json`:
- Inference latency & throughput
- GPU utilization
- Per-class accuracy
- Training loss & validation F1

Access: http://localhost:3000 (default: admin/admin)

### Kibana Logs

Structured JSON logs are sent to Elasticsearch. Access Kibana at http://localhost:5601

## 📚 API Documentation

Once server is running, access interactive docs:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

### Key Endpoints

- `GET /health` - Health check
- `POST /predict` - Single image prediction
- `POST /predict_batch` - Batch prediction (queues if > threshold)
- `POST /train` - Trigger retraining (requires API key)
- `GET /metrics` - Get training/inference metrics
- `GET /metrics/prometheus` - Prometheus metrics
- `GET /job/{job_id}` - Get async job status

## 🐳 Docker

### Build

```bash
docker compose build
```

### Start Services

```bash
docker compose up -d
```

### View Logs

```bash
docker compose logs -f api
docker compose logs -f celery-worker
```

### Stop Services

```bash
docker compose down
```

## 🚢 Model Export

```bash
python scripts/export_model.py \
    --checkpoint runs/20240101_120000/checkpoint.pt \
    --output-dir runs/20240101_120000/exports \
    --formats torchscript onnx
```

## 🔐 Security

- API key authentication for `/train` endpoint
- File upload size limits (10MB default)
- Input validation and sanitization
- Secrets via environment variables (never hardcoded)

## 📝 Development

### Code Quality

```bash
# Format
black src/ tests/ scripts/

# Lint
ruff check src/ tests/

# Type check
mypy src/ --ignore-missing-imports

# Sort imports
isort src/ tests/ scripts/
```

## 🐛 Troubleshooting

### GPU Out of Memory

- Reduce batch size in config
- Enable batch size tuner (automatic)
- Use gradient accumulation
- Enable mixed precision

### Model Not Loading

- Check `MODEL_PATH` in `.env`
- Ensure checkpoint exists at specified path
- Verify model backbone matches checkpoint

### Celery Tasks Not Running

- Check Redis connection: `docker compose ps redis`
- Verify `CELERY_BROKER_URL` in `.env`
- Check worker logs: `docker compose logs celery-worker`

### Services Not Starting

- Run health check: `python scripts/check_services.py`
- Check logs: `docker compose logs [service-name]`
- See [SERVICE_HEALTH_CHECK.md](SERVICE_HEALTH_CHECK.md) for details

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make changes with tests
4. Run linting and tests
5. Submit a pull request

## 🙏 Acknowledgments

- LADOS dataset creators
- PyTorch, FastAPI, Celery communities
- All open-source contributors
- PMIFI supremacy xdxdxd
