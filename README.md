**MLOps Airflow + MLflow Pipeline**

Overview
-: End-to-end example Airflow DAG that trains and registers a Titanic classifier using MLflow.
-: Uses Docker Compose to run Airflow (Celery executor) with Redis and PostgreSQL, and an MLflow UI service.

Repository structure
-: **docker-compose.yaml**: Compose file that starts Airflow services, Redis, Postgres and an `mlflow` UI service. ([docker-compose.yaml](docker-compose.yaml))
-: **requirements.txt**: Python dependencies for local development. ([requirements.txt](requirements.txt))
-: **dags/**: Airflow DAG definitions. Main pipeline: [dags/mlops_airflow_mlflow_pipeline.py](dags/mlops_airflow_mlflow_pipeline.py)
-: **dags/tasks/**: Task implementations used by the DAG (data ingestion, validation, preprocessing, encoding, training, evaluation, branching, registration). ([dags/tasks](dags/tasks))
-: **data/**: Example input dataset and outputs written by the pipeline (`titanic.csv`, `processed.csv`, `final.csv`).
-: **logs/**: Airflow task logs (mounted into containers).
-: **config/**, **plugins/**: Airflow configuration and plugins mount points.

Key files and DAG
-: DAG name: `mlops_pipeline` defined in [dags/mlops_airflow_mlflow_pipeline.py](dags/mlops_airflow_mlflow_pipeline.py).
-: High-level task flow:
	- `ingest_data` -> `validate_data`
	- parallel: `handle_missing` and `feature_engineering`
	- `encode_features` -> `train_model` -> `evaluate_model` -> `branching` -> (`register_model` or `reject_model`)

Task summaries
-: `ingest_data` (dags/tasks/data_ingestion.py): loads `titanic.csv` and pushes dataset path to XCom.
-: `validate_data` (dags/tasks/data_validation.py): basic missing-value checks; raises if too many missing values.
-: `handle_missing`, `feature_engineering` (dags/tasks/data_preprocessing.py): fill missing values and add simple features.
-: `encode_features` (dags/tasks/data_encoding.py): label/one-hot encode categorical fields and drop unused columns.
-: `train_model` (dags/tasks/model_training.py): trains a Logistic Regression, logs model + params to MLflow, pushes `run_id` to XCom.
-: `evaluate_model` (dags/tasks/model_evaluation.py): loads the run model from MLflow, computes metrics, logs them to the same run, pushes `accuracy` to XCom.
-: `branch_model` (dags/tasks/branching.py): branches to `register_model` if accuracy >= 0.80 else `reject_model`.
-: `register_model` / `reject_model` (dags/tasks/model_registration.py): registers model in MLflow Model Registry or logs rejection.

Requirements
-: Python packages listed in `requirements.txt` (pandas, numpy, scikit-learn, apache-airflow, mlflow). Containers in `docker-compose.yaml` install `mlflow`, `scikit-learn`, and `pandas` at startup via `_PIP_ADDITIONAL_REQUIREMENTS`.

Quick start (Docker Compose)
1. Ensure Docker and Docker Compose are installed and available.
2. From the project root, initialize Airflow DB and users:

```bash
docker compose up airflow-init
```

3. Start the services (detach to run in background):

```bash
docker compose up -d
```

4. Open the Airflow UI: http://localhost:8080 (default credentials: `airflow` / `airflow` if not changed).
5. Open the MLflow UI: http://localhost:5000

Triggering the DAG
-: Via UI: in Airflow webserver, enable and trigger `mlops_pipeline`.
-: Via CLI (from host):

```bash
docker compose run --rm airflow-cli airflow dags trigger mlops_pipeline
```

Data locations
-: Input dataset (mounted): `data/titanic.csv` -> available inside containers at `/mnt/ml-data/data/titanic.csv`.
-: Intermediate/outputs: `data/processed.csv`, `data/final.csv` (written to the same mounted folder by tasks).

Development notes
-: Adjust training hyperparameters by editing `op_kwargs` in [dags/mlops_airflow_mlflow_pipeline.py](dags/mlops_airflow_mlflow_pipeline.py) for `train_model`.
-: MLflow artifacts and runs are available in the MLflow UI; the DAG writes run metrics to the same run used for training.
-: For production usage, build a custom Airflow image and install pinned provider packages instead of using `_PIP_ADDITIONAL_REQUIREMENTS` at container start.

Troubleshooting
-: If XCom keys (e.g., `run_id`, `accuracy`) are missing, inspect task logs in the Airflow UI or `logs/` to find errors.
-: If containers fail to start, check Docker resource limits (memory/CPU) and the `airflow-init` output for warnings.

