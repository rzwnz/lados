#!/usr/bin/env python3
"""
Comprehensive health check script for LADOS Docker Compose services.
Tests all services and provides detailed status information.
"""

import sys
import json
import time
import subprocess
from typing import Dict, Tuple, Optional
import requests
from requests.exceptions import RequestException

# Color codes
class Colors:
    RED = '\033[0;31m'
    GREEN = '\033[0;32m'
    YELLOW = '\033[1;33m'
    BLUE = '\033[0;34m'
    CYAN = '\033[0;36m'
    NC = '\033[0m'  # No Color

def print_header(text: str):
    print(f"\n{Colors.BLUE}{'=' * 60}{Colors.NC}")
    print(f"{Colors.BLUE}{text}{Colors.NC}")
    print(f"{Colors.BLUE}{'=' * 60}{Colors.NC}\n")

def check_container(service: str) -> Tuple[bool, str]:
    """Check if a Docker container is running."""
    try:
        result = subprocess.run(
            ['docker', 'compose', 'ps', '--format', 'json'],
            capture_output=True,
            text=True,
            check=True
        )
        containers = [json.loads(line) for line in result.stdout.strip().split('\n') if line]
        for container in containers:
            if container.get('Service') == service:
                status = container.get('Status', '')
                if 'Up' in status:
                    return True, status
                return False, status
        return False, "Not found"
    except Exception as e:
        return False, str(e)

def check_http(url: str, name: str, timeout: int = 5, expected_status: int = 200) -> Tuple[bool, str]:
    """Check if an HTTP endpoint is accessible."""
    try:
        response = requests.get(url, timeout=timeout)
        if response.status_code == expected_status:
            return True, f"HTTP {response.status_code}"
        return False, f"HTTP {response.status_code} (expected {expected_status})"
    except RequestException as e:
        return False, str(e)

def check_redis() -> Tuple[bool, str]:
    """Check Redis connection."""
    try:
        result = subprocess.run(
            ['docker', 'compose', 'exec', '-T', 'redis', 'redis-cli', 'ping'],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0 and 'PONG' in result.stdout:
            return True, "PONG"
        return False, result.stderr or "No response"
    except Exception as e:
        return False, str(e)

def check_elasticsearch() -> Tuple[bool, str]:
    """Check Elasticsearch health."""
    try:
        response = requests.get('http://localhost:9200/_cluster/health', timeout=5)
        if response.status_code == 200:
            data = response.json()
            status = data.get('status', 'unknown')
            return True, f"Status: {status}"
        return False, f"HTTP {response.status_code}"
    except RequestException as e:
        return False, str(e)

def check_api_health() -> Tuple[bool, Dict]:
    """Check FastAPI health endpoint."""
    try:
        response = requests.get('http://localhost:8000/health', timeout=5)
        if response.status_code == 200:
            data = response.json()
            return True, data
        return False, {"error": f"HTTP {response.status_code}"}
    except RequestException as e:
        return False, {"error": str(e)}

def check_celery_worker() -> Tuple[bool, str]:
    """Check Celery worker status via logs."""
    try:
        result = subprocess.run(
            ['docker', 'compose', 'logs', '--tail', '20', 'celery-worker'],
            capture_output=True,
            text=True,
            timeout=5
        )
        logs = result.stdout.lower()
        if 'ready' in logs or 'celery@' in logs:
            return True, "Worker appears active"
        return False, "No activity detected"
    except Exception as e:
        return False, str(e)

def print_result(name: str, success: bool, details: str = ""):
    """Print a formatted result."""
    status = f"{Colors.GREEN}✓{Colors.NC}" if success else f"{Colors.RED}✗{Colors.NC}"
    print(f"{status} {name}", end="")
    if details:
        print(f" - {details}")
    else:
        print()

def main():
    print_header("LADOS Services Health Check")
    
    results = {
        'containers': {},
        'services': {},
        'overall': True
    }
    
    # Check containers
    print(f"{Colors.CYAN}Checking Docker Containers...{Colors.NC}")
    services = ['api', 'celery-worker', 'redis', 'elasticsearch', 'kibana', 'grafana', 'prometheus']
    
    for service in services:
        success, status = check_container(service)
        print_result(service, success, status)
        results['containers'][service] = {'success': success, 'status': status}
        if not success:
            results['overall'] = False
    
    # Check API endpoints
    print(f"\n{Colors.CYAN}Checking API Endpoints...{Colors.NC}")
    api_endpoints = [
        ('http://localhost:8000/health', 'FastAPI Health'),
        ('http://localhost:8000/docs', 'FastAPI Docs'),
        ('http://localhost:8000/metrics', 'FastAPI Metrics'),
    ]
    
    for url, name in api_endpoints:
        success, details = check_http(url, name)
        print_result(name, success, details)
        results['services'][name] = {'success': success, 'details': details}
        if not success:
            results['overall'] = False
    
    # Check API health details
    success, health_data = check_api_health()
    if success:
        print(f"\n{Colors.CYAN}API Health Details:{Colors.NC}")
        print(f"  Status: {health_data.get('status', 'unknown')}")
        print(f"  Model Loaded: {health_data.get('model_loaded', False)}")
        print(f"  Device: {health_data.get('device', 'unknown')}")
    
    # Check Redis
    print(f"\n{Colors.CYAN}Checking Redis...{Colors.NC}")
    success, details = check_redis()
    print_result('Redis', success, details)
    results['services']['Redis'] = {'success': success, 'details': details}
    if not success:
        results['overall'] = False
    
    # Check Elasticsearch
    print(f"\n{Colors.CYAN}Checking Elasticsearch...{Colors.NC}")
    success, details = check_elasticsearch()
    print_result('Elasticsearch', success, details)
    results['services']['Elasticsearch'] = {'success': success, 'details': details}
    if not success:
        results['overall'] = False
    
    # Check Kibana
    print(f"\n{Colors.CYAN}Checking Kibana...{Colors.NC}")
    success, details = check_http('http://localhost:5601', 'Kibana')
    print_result('Kibana', success, details)
    results['services']['Kibana'] = {'success': success, 'details': details}
    if not success:
        results['overall'] = False
    
    # Check Grafana
    print(f"\n{Colors.CYAN}Checking Grafana...{Colors.NC}")
    success, details = check_http('http://localhost:3000/api/health', 'Grafana')
    print_result('Grafana', success, details)
    results['services']['Grafana'] = {'success': success, 'details': details}
    if not success:
        results['overall'] = False
    
    # Check Prometheus
    print(f"\n{Colors.CYAN}Checking Prometheus...{Colors.NC}")
    success, details = check_http('http://localhost:9090/-/healthy', 'Prometheus')
    print_result('Prometheus', success, details)
    results['services']['Prometheus'] = {'success': success, 'details': details}
    if not success:
        results['overall'] = False
    
    # Check Celery worker
    print(f"\n{Colors.CYAN}Checking Celery Worker...{Colors.NC}")
    success, details = check_celery_worker()
    print_result('Celery Worker', success, details)
    results['services']['Celery Worker'] = {'success': success, 'details': details}
    if not success:
        results['overall'] = False
    
    # Summary
    print_header("Summary")
    if results['overall']:
        print(f"{Colors.GREEN}All services are healthy!{Colors.NC}\n")
    else:
        print(f"{Colors.RED}Some services have issues. Check details above.{Colors.NC}\n")
    
    print(f"{Colors.CYAN}Useful commands:{Colors.NC}")
    print("  View logs:        docker compose logs [service-name]")
    print("  View all logs:   docker compose logs")
    print("  Restart service: docker compose restart [service-name]")
    print("  Service status:  docker compose ps")
    print("  Stop all:        docker compose down")
    print("  Start all:       docker compose up -d")
    
    return 0 if results['overall'] else 1

if __name__ == '__main__':
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print(f"\n{Colors.YELLOW}Interrupted by user{Colors.NC}")
        sys.exit(1)
    except Exception as e:
        print(f"\n{Colors.RED}Error: {e}{Colors.NC}")
        sys.exit(1)

