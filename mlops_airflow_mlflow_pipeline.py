from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator
from datetime import datetime, timedelta
import os
import sys

# Ensure the DAGs folder is on sys.path
sys.path.insert(0, os.path.dirname(__file__))

from tasks.data_ingestion import ingest_data
from tasks.data_validation import validate_data
from tasks.data_preprocessing import handle_missing, feature_engineering
from tasks.data_encoding import encode_features
from tasks.model_training import train_model
from tasks.model_evaluation import evaluate_model
from tasks.branching import branch_model
from tasks.model_registration import register_model, reject_model

default_args = {
    "owner": "airflow",
    "retries": 2,
    "retry_delay": timedelta(seconds=10)
}

with DAG(
    "mlops_pipeline",
    start_date=datetime(2024, 1, 1),
    schedule_interval=None,
    catchup=False,
    default_args=default_args,
    description="End-to-end Titanic ML Pipeline with Airflow & MLflow"
) as dag:

    # --- Data ingestion & validation ---
    ingest = PythonOperator(
        task_id="ingest_data",
        python_callable=ingest_data
    )

    validate = PythonOperator(
        task_id="validate_data",
        python_callable=validate_data
    )

    # --- Parallel preprocessing ---
    missing = PythonOperator(
        task_id="handle_missing",
        python_callable=handle_missing
    )

    feature = PythonOperator(
        task_id="feature_engineering",
        python_callable=feature_engineering
    )

    # --- Encoding ---
    encode = PythonOperator(
        task_id="encode_features",
        python_callable=encode_features
    )

    # --- Model training ---
    train = PythonOperator(
        task_id="train_model",
        python_callable=train_model,
        op_kwargs={"max_iter": 150, "C": 0.8}
        #op_kwargs={"max_iter": 200, "C": 1.0}
        #op_kwargs={"max_iter": 300, "C": 0.5}
    )

    # --- Model evaluation using the trained model ---
    evaluate = PythonOperator(
        task_id="evaluate_model",
        python_callable=evaluate_model
    )

    # --- Branching based on accuracy ---
    branch = BranchPythonOperator(
        task_id="branching",
        python_callable=branch_model
    )

    # --- Model registration / rejection ---
    register = PythonOperator(
        task_id="register_model",
        python_callable=register_model
    )

    reject = PythonOperator(
        task_id="reject_model",
        python_callable=reject_model
    )

    # ---------------- Task dependencies ----------------
    ingest >> validate >> [missing, feature]
    [missing, feature] >> encode >> train >> evaluate >> branch
    branch >> register
    branch >> reject