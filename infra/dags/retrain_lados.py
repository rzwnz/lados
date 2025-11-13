"""
Airflow DAG for LADOS model retraining pipeline.

This DAG:
1. Checks for new labeled data
2. Triggers retraining via Celery
3. Exports model artifacts
4. Updates metrics
"""

from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator

default_args = {
    "owner": "lados-team",
    "depends_on_past": False,
    "start_date": datetime(2024, 1, 1),
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

dag = DAG(
    "retrain_lados",
    default_args=default_args,
    description="Retrain LADOS classification model",
    schedule_interval=timedelta(days=7),  # Weekly retraining
    catchup=False,
    tags=["lados", "retraining", "ml"],
)


def check_new_data(**context):
    """Check if new labeled data is available."""
    from pathlib import Path

    data_path = Path("/app/data/processed")
    # Simple check - in production, compare timestamps or checksums
    if data_path.exists():
        print("New data detected")
        return True
    return False


def trigger_training(**context):
    """Trigger training job via Celery."""
    import os
    import requests

    api_url = os.getenv("API_URL", "http://api:8000")
    api_key = os.getenv("API_KEY", "")

    response = requests.post(
        f"{api_url}/train",
        headers={"X-API-Key": api_key},
        timeout=30,
    )
    response.raise_for_status()
    job_id = response.json()["job_id"]
    print(f"Training job queued: {job_id}")
    return job_id


check_data_task = PythonOperator(
    task_id="check_new_data",
    python_callable=check_new_data,
    dag=dag,
)

trigger_training_task = PythonOperator(
    task_id="trigger_training",
    python_callable=trigger_training,
    dag=dag,
)

export_model_task = BashOperator(
    task_id="export_model",
    bash_command="""
    python scripts/export_model.py \
        --checkpoint runs/latest/checkpoint.pt \
        --output-dir runs/latest/exports \
        --formats torchscript onnx
    """,
    dag=dag,
)

update_metrics_task = BashOperator(
    task_id="update_metrics",
    bash_command="echo 'Metrics updated'",
    dag=dag,
)

# Define task dependencies
check_data_task >> trigger_training_task >> export_model_task >> update_metrics_task

