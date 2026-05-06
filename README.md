# MLOps Pipeline using Apache Airflow and MLflow for Titanic Survival Prediction

## Overview

This project implements an **end-to-end machine learning pipeline** that automates the complete ML lifecycle for Titanic survival prediction. The pipeline integrates:

- **Apache Airflow**: Orchestrates workflow tasks and manages dependencies
- **MLflow**: Tracks experiments, logs metrics, and manages model versions
- **Docker Compose**: Containerizes Airflow services, MLflow UI, PostgreSQL, and Redis for local development

The pipeline demonstrates key MLOps concepts including:
- Automated data ingestion, validation, and preprocessing
- Parallel task execution for improved efficiency
- Model training with multiple hyperparameter configurations
- Experiment tracking and comparison
- Conditional branching logic for model approval/rejection
- Model registration in MLflow Model Registry
- Fault tolerance with automatic retry mechanisms

---

## System Architecture (Airflow + MLflow Interaction)

### Architecture Overview

The system integrates workflow orchestration with experiment tracking:

1. **Airflow DAG** triggers the pipeline execution
2. **Data ingestion** loads the Titanic dataset from CSV
3. **Data validation** ensures dataset quality (checks missing value thresholds)
4. **Data preprocessing** tasks run in parallel:
   - Handling missing values in Age and Embarked columns
   - Feature engineering (creating FamilySize, IsAlone features)
5. **Data encoding** converts categorical variables (Sex, Embarked) to numerical values
6. **Model training** uses Logistic Regression or Random Forest
   - MLflow logs: model type, hyperparameters, dataset size, model artifacts
7. **Model evaluation** computes metrics (Accuracy, Precision, Recall, F1 Score)
8. **Branching logic** uses BranchPythonOperator to decide model approval
   - If accuracy ≥ 0.80: model is approved for registration
   - If accuracy < 0.80: model is rejected
9. **Model registration** stores approved models in MLflow Model Registry

This architecture ensures **automation, reproducibility, and traceability** of machine learning experiments.

---

## DAG Structure and Task Dependencies

### DAG Name: `titanic_ml_pipeline`

#### Task Flow Diagram
```
ingest_data
    ↓
validate_data (with retry mechanism)
    ↓
[handle_missing_values ⟷ engineer_features] (parallel execution)
    ↓
encode_and_clean
    ↓
train_model (logs to MLflow)
    ↓
evaluate_model (logs metrics to MLflow)
    ↓
branch_model_decision
    ├→ register_model (if accuracy ≥ 0.80)
    └→ reject_model (if accuracy < 0.80)
```

### Task Descriptions

| Task | Purpose | Key Features |
|------|---------|--------------|
| **ingest_data** | Load Titanic CSV, log dataset shape and missing values | Pushes dataset path to XCom |
| **validate_data** | Check missing value percentages in Age and Embarked columns; threshold: 30% | **Demonstrates retry mechanism** with intentional first-attempt failure |
| **handle_missing_values** | Fill missing values in Age (median) and Embarked (mode) | Runs in parallel with feature_engineering |
| **engineer_features** | Create new features: FamilySize, IsAlone | Runs in parallel with handle_missing_values |
| **encode_and_clean** | Encode categorical variables (Sex, Embarked); drop irrelevant columns (PassengerId, Name, Ticket) | Joins parallel tasks |
| **train_model** | Train Logistic Regression or Random Forest model | Logs model type, hyperparameters, dataset size, and artifacts to MLflow |
| **evaluate_model** | Compute Accuracy, Precision, Recall, F1 Score | Logs metrics to MLflow; pushes accuracy to XCom for branching |
| **branch_model_decision** | Conditional branching based on model accuracy | Threshold: 0.80 |
| **register_model** | Register approved models in MLflow Model Registry | Enables model versioning and deployment |
| **reject_model** | Log rejection reason for models below threshold | Logs rejection details for analysis |

---

## Experiment Comparison Analysis

The pipeline was executed three times with different hyperparameters to analyze model performance:

### Experimental Results

| Run | Model Type | Hyperparameters | Accuracy | Status |
|-----|------------|-----------------|----------|--------|
| **Run 1** | Logistic Regression | max_iter=1000, C=1.0 | 0.79 | ❌ Rejected |
| **Run 2** | Random Forest | n_estimators=100, max_depth=5 | 0.80 | ✓ Approved |
| **Run 3** | Random Forest | n_estimators=200, max_depth=10 | 0.83 | ✓ Approved (Best) |

**Key Findings:**
- Run 3 achieved the highest accuracy (0.83) and was selected as the best-performing model
- Random Forest with more estimators and depth outperformed Logistic Regression
- MLflow UI provided easy visualization and comparison of performance metrics across runs
- This experiment tracking capability is essential for selecting production-ready models

### Tracking in MLflow

All runs are tracked in the MLflow UI where you can:
- Compare metrics across different model types and hyperparameters
- View logged artifacts and model signatures
- Monitor parameter impact on model performance
- Access the best model from the model registry

---

## Failure Handling and Retry Mechanism

### Retry Demonstration

The **validate_data** task is configured with the following retry parameters:

```python
retries=2                                    # Maximum 2 retry attempts
retry_delay=timedelta(seconds=10)            # 10-second delay between retries
```

#### Mechanism:
1. **First Attempt**: Task fails intentionally (uses a marker file to track attempts)
2. **Retry 1**: Detects marker file, removes it, and proceeds successfully
3. **Success**: Task completes on the first retry

#### Evidence:
Retry attempts can be observed in the **Airflow UI logs** with:
- Timestamp of each attempt
- Error messages from failed attempts
- Final successful completion
- Task state transitions visible in the DAG graph

#### Benefits:
- Temporary failures do not stop the entire pipeline
- Improves system reliability and robustness
- Essential for production deployments where transient issues may occur

---

## Repository Structure

```
.
├── docker-compose.yaml              # Docker services: Airflow, MLflow, PostgreSQL, Redis
├── Dockerfile                       # Custom Airflow image with dependencies
├── requirements.txt                 # Python dependencies (pandas, sklearn, mlflow, airflow)
├── mlops_airflow_mlflow_pipeline.py # Main DAG definition
├── README.md                        # This file
├── data/                            # Data directory (mounted volume)
│   ├── titanic.csv                 # Input Titanic dataset
│   ├── processed.csv               # Preprocessed data (output)
│   └── final.csv                   # Final encoded data (output)
├── logs/                           # Airflow task execution logs (mounted volume)
├── config/                         # Airflow configuration mount point
└── screenshots/                    # Evidence screenshots (retry logs, metrics, etc.)
```

---

## Quick Start (Docker Compose)

### Prerequisites
- Docker and Docker Compose installed
- At least 4GB RAM available for containers
- Titanic dataset (`data/titanic.csv`) in place

### Setup Steps

1. **Initialize Airflow Database:**
   ```bash
   docker compose up airflow-init
   ```

2. **Start All Services:**
   ```bash
   docker compose up -d
   ```
   This starts:
   - Airflow Scheduler
   - Airflow Webserver
   - Airflow Workers (Celery)
   - PostgreSQL (metadata store)
   - Redis (task broker)
   - MLflow (experiment tracking & UI)

3. **Access the UIs:**
   - **Airflow UI**: http://localhost:8080
     - Default credentials: `airflow` / `airflow`
   - **MLflow UI**: http://localhost:5000
     - Browse experiments, runs, and models

### Triggering the DAG

#### Option 1: Via Airflow WebUI
1. Navigate to http://localhost:8080
2. Find `titanic_ml_pipeline` in the DAG list
3. Click the toggle to enable it
4. Click "Trigger DAG" button

#### Option 2: Via CLI
```bash
docker compose exec airflow-cli airflow dags trigger titanic_ml_pipeline
```

#### Option 3: With Configuration (Custom Hyperparameters)
```bash
docker compose exec airflow-cli airflow dags trigger titanic_ml_pipeline \
  --conf '{"model_type": "RandomForest", "n_estimators": 200, "max_depth": 10}'
```

### Monitoring Execution

1. **Airflow UI**: View DAG graph, task status, and logs
2. **MLflow UI**: Track experiment runs, metrics, and artifacts
3. **Logs**: Check `logs/` directory for detailed task output

---

## Default DAG Configuration

```python
default_args = {
    "owner": "student",
    "depends_on_past": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=1),
}

dag = DAG(
    dag_id="titanic_ml_pipeline",
    schedule=None,              # Manual trigger only
    start_date=datetime(2024, 1, 1),
    catchup=False,
    max_active_runs=1,
    tags=["mlops", "titanic"],
)
```

---

## Data Flow

### Input Data
- **Source**: `data/titanic.csv` (mounted to containers)
- **Container Path**: `/opt/airflow/data/titanic.csv`
- **Features**: PassengerId, Pclass, Name, Sex, Age, SibSp, Parch, Ticket, Fare, Cabin, Embarked, Survived

### Processing Pipeline
1. **Ingestion** → Load CSV
2. **Validation** → Check data quality (missing values < 30%)
3. **Preprocessing** → Handle missing values, engineer features
4. **Encoding** → Convert categorical to numerical
5. **Training** → Fit ML model
6. **Evaluation** → Compute metrics
7. **Registration** → Store in MLflow if accuracy ≥ 0.80

### Output Data
- **processed.csv**: Preprocessed data with feature engineering
- **final.csv**: Final encoded data ready for model training
- **MLflow Artifacts**: Trained model, feature names, preprocessing info

---

## Development and Customization

### Adjusting Model Hyperparameters

Edit the `train_model` task in [mlops_airflow_mlflow_pipeline.py](mlops_airflow_mlflow_pipeline.py):

```python
train_model_task = PythonOperator(
    task_id="train_model",
    python_callable=train_model,
    op_kwargs={
        "model_type": "RandomForest",
        "n_estimators": 200,      # Adjust here
        "max_depth": 10,          # Adjust here
    },
    dag=dag,
)
```

### Adding New Features

Modify the `engineer_features` function to create additional features beyond FamilySize and IsAlone.

### Changing the Model Type

Options in the pipeline:
- `LogisticRegression`: Fast, interpretable linear model
- `RandomForest`: Ensemble method with better non-linear performance

Pass via DAG configuration or edit `op_kwargs`.

---

## Reflection on Production Deployment

While this project demonstrates a local development setup, deploying to production requires:

### 1. **Scalable Infrastructure**
- Deploy Airflow scheduler and workers on **Kubernetes** or cloud-managed services
- Use horizontal scaling for higher workloads and parallel task execution

### 2. **Remote Storage & Logging**
- Configure remote artifact storage: **AWS S3**, **Google Cloud Storage**, or **Azure Blob**
- Use remote logging services for centralized log aggregation
- Store MLflow tracking server in production database (PostgreSQL/MySQL)

### 3. **Monitoring & Alerting**
- Integrate monitoring systems (Prometheus, Datadog, CloudWatch)
- Set up alerts for pipeline failures, SLA violations, and performance degradation
- Implement data quality monitoring for drift detection

### 4. **Model Deployment**
- Integrate MLflow Model Registry with **CI/CD pipelines**
- Automatically deploy approved models to:
  - REST APIs (Flask, FastAPI)
  - Batch prediction systems
  - Serving platforms (Seldon, KServe)
- Implement model versioning and rollback capabilities

### 5. **Security & Governance**
- Implement RBAC and authentication for Airflow and MLflow
- Audit logging for compliance
- Model registry approval workflows
- Data encryption in transit and at rest

---

## Requirements

### Python Packages
Installed from `requirements.txt`:
- **pandas** ≥ 1.3.0 — Data manipulation
- **numpy** ≥ 1.20.0 — Numerical computing
- **scikit-learn** ≥ 0.24.0 — Machine learning algorithms
- **apache-airflow** ≥ 2.5.0 — Workflow orchestration
- **mlflow** ≥ 1.30.0 — Experiment tracking and model registry
- **python-dateutil** — Date utilities

### Docker Services
From `docker-compose.yaml`:
- **Apache Airflow** (Celery executor)
- **PostgreSQL** (metadata database)
- **Redis** (task broker)
- **MLflow** (tracking and model registry server)

---

## Troubleshooting

### Common Issues & Solutions

| Issue | Solution |
|-------|----------|
| **XCom keys missing** (e.g., `dataset_path`, `run_id`, `accuracy`) | Check task logs in Airflow UI; ensure tasks complete successfully and push to XCom |
| **Containers fail to start** | Verify Docker resource limits (4GB+ RAM); check `docker compose logs` for errors |
| **MLflow connection error** | Ensure `MLFLOW_TRACKING_URI` is set correctly to `http://mlflow:5000` |
| **Data file not found** | Verify `data/titanic.csv` exists; check mounted volume paths in `docker-compose.yaml` |
| **Tasks timeout during execution** | Increase task timeout values; check for slow preprocessing or model training |
| **Model not registered** | Verify accuracy threshold (0.80) is met; check branching logic and registration logs |

### Viewing Logs

```bash
# Airflow logs
docker compose logs -f airflow-scheduler
docker compose logs -f airflow-worker

# Specific task logs
docker compose exec airflow-cli airflow logs titanic_ml_pipeline ingest_data

# MLflow logs
docker compose logs -f mlflow
```

### Stopping Services

```bash
docker compose down
```

---

### Additional Details : 
Name: MLOps Pipeline - Titanic Survival Prediction
Topics: mlops, apache-airflow, mlflow, machine-learning, workflow-orchestration, experiment-tracking, docker, data-pipeline, model-registry

---

---

### Contact :  
For questions, feedback, or issues, feel free to contact me at: nooranishtiaq@gmail.com
---

## References

- [Apache Airflow Documentation](https://airflow.apache.org/)
- [MLflow Documentation](https://mlflow.org/docs/latest/index.html)
- [Titanic Dataset](https://www.kaggle.com/c/titanic)
- [Docker Compose Reference](https://docs.docker.com/compose/)

