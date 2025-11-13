# Service Health Check Guide

This guide explains how to verify that all services in the LADOS Docker Compose setup are working correctly.

## Quick Health Check

### Option 1: Python Script (Recommended)
```bash
# Make sure you have requests installed
pip install requests

# Run the health check
python scripts/check_services.py
```

### Option 2: Bash Script
```bash
./scripts/check_services.sh
```

### Option 3: Manual Checks

#### 1. Check Container Status
```bash
docker compose ps
```
All services should show "Up" status.

#### 2. Check FastAPI (Port 8000)
```bash
# Health check
curl http://localhost:8000/health

# Expected response:
# {"status":"healthy","model_loaded":true,"device":"cpu"}

# API Documentation
curl http://localhost:8000/docs
# Should return HTML page

# Metrics endpoint
curl http://localhost:8000/metrics
```

#### 3. Check Redis (Port 6379)
```bash
docker compose exec redis redis-cli ping
# Should return: PONG
```

#### 4. Check Elasticsearch (Port 9200)
```bash
curl http://localhost:9200/_cluster/health
# Should return JSON with "status": "green" or "yellow"
```

#### 5. Check Kibana (Port 5601)
```bash
curl http://localhost:5601
# Should return HTML (status 200)
```

#### 6. Check Grafana (Port 3000)
```bash
curl http://localhost:3000/api/health
# Should return JSON with status
```

#### 7. Check Prometheus (Port 9090)
```bash
curl http://localhost:9090/-/healthy
# Should return: Prometheus is Healthy.
```

#### 8. Check Celery Worker
```bash
# Check logs for "ready" message
docker compose logs celery-worker | grep -i ready

# Or check recent logs
docker compose logs --tail 50 celery-worker
```

## Service URLs

Once all services are running, you can access:

- **FastAPI**: http://localhost:8000
  - API Docs: http://localhost:8000/docs
  - Health: http://localhost:8000/health
  - Metrics: http://localhost:8000/metrics

- **Kibana**: http://localhost:5601

- **Grafana**: http://localhost:3000
  - Default credentials: admin/admin

- **Prometheus**: http://localhost:9090

## Troubleshooting

### Service Not Starting

1. **Check logs:**
   ```bash
   docker compose logs [service-name]
   ```

2. **Restart a specific service:**
   ```bash
   docker compose restart [service-name]
   ```

3. **Rebuild and restart:**
   ```bash
   docker compose up -d --build [service-name]
   ```

### Common Issues

#### FastAPI Model Not Loaded
- Check if model file exists: `ls -la runs/latest/checkpoint.pt`
- Check API logs: `docker compose logs api`
- Model will show as not loaded if file doesn't exist (this is OK for testing)

#### Redis Connection Issues
- Verify Redis is running: `docker compose ps redis`
- Check Redis logs: `docker compose logs redis`

#### Elasticsearch Not Healthy
- Check if it's still starting (can take 30-60 seconds)
- Check logs: `docker compose logs elasticsearch`
- Verify memory: Elasticsearch needs at least 512MB

#### Celery Worker Not Processing
- Check if Redis is accessible from worker
- Check worker logs: `docker compose logs celery-worker`
- Verify worker is registered: Check Redis for Celery keys

## Testing API Endpoints

### Health Check
```bash
curl http://localhost:8000/health
```

### Single Image Prediction
```bash
curl -X POST http://localhost:8000/predict \
  -F "file=@path/to/image.jpg"
```

### Batch Prediction
```bash
curl -X POST http://localhost:8000/predict_batch \
  -F "files=@image1.jpg" \
  -F "files=@image2.jpg"
```

### Get Metrics
```bash
curl http://localhost:8000/metrics
```

## Monitoring

### View All Logs
```bash
docker compose logs -f
```

### View Specific Service Logs
```bash
docker compose logs -f [service-name]
```

### Resource Usage
```bash
docker stats
```

## Quick Status Command

Create an alias for quick checks:
```bash
alias check-lados='python scripts/check_services.py'
```

Then simply run:
```bash
check-lados
```

