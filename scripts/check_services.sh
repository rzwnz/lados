#!/bin/bash

# Health check script for LADOS Docker Compose services
# Usage: ./scripts/check_services.sh

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}=== LADOS Services Health Check ===${NC}\n"

# Function to check if a service is running
check_container() {
    local service=$1
    if docker compose ps | grep -q "${service}.*Up"; then
        echo -e "${GREEN}✓${NC} ${service} container is running"
        return 0
    else
        echo -e "${RED}✗${NC} ${service} container is not running"
        return 1
    fi
}

# Function to check HTTP endpoint
check_http() {
    local url=$1
    local name=$2
    local expected_status=${3:-200}
    
    if response=$(curl -s -w "\n%{http_code}" -o /tmp/response.json "$url" 2>/dev/null); then
        status_code=$(echo "$response" | tail -n1)
        if [ "$status_code" = "$expected_status" ]; then
            echo -e "${GREEN}✓${NC} ${name} is accessible (HTTP ${status_code})"
            return 0
        else
            echo -e "${YELLOW}⚠${NC} ${name} returned HTTP ${status_code} (expected ${expected_status})"
            return 1
        fi
    else
        echo -e "${RED}✗${NC} ${name} is not accessible"
        return 1
    fi
}

# Function to check Redis
check_redis() {
    if docker compose exec -T redis redis-cli ping > /dev/null 2>&1; then
        echo -e "${GREEN}✓${NC} Redis is responding to PING"
        return 0
    else
        echo -e "${RED}✗${NC} Redis is not responding"
        return 1
    fi
}

# Function to check Elasticsearch
check_elasticsearch() {
    if curl -s http://localhost:9200/_cluster/health > /dev/null 2>&1; then
        health=$(curl -s http://localhost:9200/_cluster/health | grep -o '"status":"[^"]*"' | cut -d'"' -f4)
        echo -e "${GREEN}✓${NC} Elasticsearch is healthy (status: ${health})"
        return 0
    else
        echo -e "${RED}✗${NC} Elasticsearch is not accessible"
        return 1
    fi
}

# Check all containers
echo -e "${BLUE}Checking containers...${NC}"
check_container api
check_container celery-worker
check_container redis
check_container elasticsearch
check_container kibana
check_container grafana
check_container prometheus
echo ""

# Check API endpoints
echo -e "${BLUE}Checking API endpoints...${NC}"
check_http "http://localhost:8000/health" "FastAPI Health"
check_http "http://localhost:8000/docs" "FastAPI Docs"
check_http "http://localhost:8000/metrics" "FastAPI Metrics"
echo ""

# Check Redis
echo -e "${BLUE}Checking Redis...${NC}"
check_redis
echo ""

# Check Elasticsearch
echo -e "${BLUE}Checking Elasticsearch...${NC}"
check_elasticsearch
echo ""

# Check Kibana
echo -e "${BLUE}Checking Kibana...${NC}"
check_http "http://localhost:5601" "Kibana" "200"
echo ""

# Check Grafana
echo -e "${BLUE}Checking Grafana...${NC}"
check_http "http://localhost:3000/api/health" "Grafana" "200"
echo ""

# Check Prometheus
echo -e "${BLUE}Checking Prometheus...${NC}"
check_http "http://localhost:9090/-/healthy" "Prometheus" "200"
echo ""

# Check Celery worker (via logs)
echo -e "${BLUE}Checking Celery worker...${NC}"
if docker compose logs celery-worker --tail 10 | grep -q "ready"; then
    echo -e "${GREEN}✓${NC} Celery worker appears to be ready"
else
    echo -e "${YELLOW}⚠${NC} Could not confirm Celery worker status (check logs)"
fi
echo ""

# Summary
echo -e "${BLUE}=== Summary ===${NC}"
echo "To view detailed logs: docker compose logs [service-name]"
echo "To view all logs: docker compose logs"
echo "To restart a service: docker compose restart [service-name]"
echo "To check service status: docker compose ps"

