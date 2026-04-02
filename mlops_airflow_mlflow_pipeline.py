"""
Titanic Survival Prediction – End-to-End MLOps Pipeline
=======================================================
Orchestrated by Apache Airflow | Tracked by MLflow
        Run 1: {"model_type": "LogisticRegression", "max_iter": 1000, "C": 1.0}
        Run 2: {"model_type": "RandomForest", "n_estimators": 100, "max_depth": 5}
        Run 3: {"model_type": "RandomForest", "n_estimators": 200, "max_depth": 10}
"""

from airflow import DAG

from airflow.providers.standard.operators.python import BranchPythonOperator, PythonOperator
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import os
import tempfile
import joblib

# ── Paths ──────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.environ.get("TITANIC_CSV_PATH", os.path.join(BASE_DIR, "data", "titanic.csv"))
if not os.path.exists(DATA_PATH):
    alt_path = "/opt/airflow/data/titanic.csv"
    if os.path.exists(alt_path):
        DATA_PATH = alt_path
    else:
        raise FileNotFoundError(
            f"Unable to locate titanic.csv. Tried: {DATA_PATH} and {alt_path}"
        )

ARTIFACTS_DIR = os.path.join(tempfile.gettempdir(), "titanic_pipeline")
os.makedirs(ARTIFACTS_DIR, exist_ok=True)

# ── MLflow config ──────────────────────────────────
MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://mlflow:5000")
MLFLOW_EXPERIMENT = "Titanic_Survival_Prediction"

os.environ["MLFLOW_TRACKING_URI"] = MLFLOW_TRACKING_URI

# ── Default DAG args ──────────────────────────────
default_args = {
    "owner": "student",
    "depends_on_past": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=1),
}

dag = DAG(
    dag_id="titanic_ml_pipeline",
    default_args=default_args,
    description="End-to-end Titanic survival prediction pipeline with Airflow & MLflow",
    schedule=None,                       # Manual trigger only
    start_date=datetime(2024, 1, 1),
    catchup=False,
    max_active_runs=1,
    tags=["mlops", "titanic"],
)


# ═══════════════════════════════════════════════════
# Task 2 – Data Ingestion 
# ═══════════════════════════════════════════════════
def ingest_data(**context):
    """Load Titanic CSV, print shape, log missing values, push path via XCom."""
    df = pd.read_csv(DATA_PATH)

    print(f"Dataset shape: {df.shape}")
    print(f"Missing values per column:\n{df.isnull().sum()}")

    context["ti"].xcom_push(key="dataset_path", value=DATA_PATH)
    print(f"Dataset path pushed to XCom: {DATA_PATH}")


ingest_task = PythonOperator(
    task_id="ingest_data",
    python_callable=ingest_data,
    dag=dag,
)


# ═══════════════════════════════════════════════════
# Task 3 – Data Validation 
# ═══════════════════════════════════════════════════
def validate_data(**context):
    """Check missing % in Age and Embarked. Raise if > 30%.
    Also demonstrates retry: on the very first attempt the task
    raises intentionally, succeeding on retry.
    """
    ti = context["ti"]
    path = ti.xcom_pull(key="dataset_path", task_ids="ingest_data")
    df = pd.read_csv(path)

    missing_age_pct = df["Age"].isnull().mean() * 100
    missing_embarked_pct = df["Embarked"].isnull().mean() * 100

    print(f"Missing Age: {missing_age_pct:.2f}%")
    print(f"Missing Embarked: {missing_embarked_pct:.2f}%")

    # ── Retry demonstration ──
    # Use a marker file: first attempt creates it and fails;
    # the retry sees it, deletes it, and continues.
    retry_marker = os.path.join(ARTIFACTS_DIR, "retry_marker.flag")
    if not os.path.exists(retry_marker):
        with open(retry_marker, "w") as f:
            f.write("triggered")
        raise RuntimeError(
            "Intentional failure to demonstrate retry behaviour! "
            "This task will succeed on the next attempt."
        )
    else:
        os.remove(retry_marker)
        print("Retry detected — proceeding after intentional first-attempt failure.")

    # ── Actual threshold check ──
    if missing_age_pct > 30 or missing_embarked_pct > 30:
        raise ValueError(
            f"Missing values exceed 30% threshold! "
            f"Age={missing_age_pct:.1f}%, Embarked={missing_embarked_pct:.1f}%"
        )

    print("Data validation PASSED")


validate_task = PythonOperator(
    task_id="validate_data",
    python_callable=validate_data,
    retries=2,           # allows automatic retry after intentional failure
    retry_delay=timedelta(seconds=10),
    dag=dag,
)


# ═══════════════════════════════════════════════════
# Task 4 – Parallel Processing 
#   Both tasks read the RAW data independently so they
#   can genuinely execute in parallel.
# ═══════════════════════════════════════════════════
def handle_missing_values(**context):
    """Fill missing Age (median) and Embarked (mode). Save cleaned CSV."""
    ti = context["ti"]
    path = ti.xcom_pull(key="dataset_path", task_ids="ingest_data")
    df = pd.read_csv(path)

    df["Age"] = df["Age"].fillna(df["Age"].median())
    df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])

    clean_path = os.path.join(ARTIFACTS_DIR, "titanic_clean.csv")
    df.to_csv(clean_path, index=False)
    ti.xcom_push(key="clean_path", value=clean_path)
    print(f"Missing values handled → saved to {clean_path}")


def engineer_features(**context):
    """Create FamilySize and IsAlone from raw data (independent of cleaning)."""
    ti = context["ti"]
    path = ti.xcom_pull(key="dataset_path", task_ids="ingest_data")
    df = pd.read_csv(path)

    df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
    df["IsAlone"] = (df["FamilySize"] == 1).astype(int)

    feat_path = os.path.join(ARTIFACTS_DIR, "titanic_features.csv")
    df.to_csv(feat_path, index=False)
    ti.xcom_push(key="featured_path", value=feat_path)
    print(f"Feature engineering done → saved to {feat_path}")


handle_missing_task = PythonOperator(
    task_id="handle_missing_values",
    python_callable=handle_missing_values,
    dag=dag,
)

engineer_features_task = PythonOperator(
    task_id="engineer_features",
    python_callable=engineer_features,
    dag=dag,
)


# ═══════════════════════════════════════════════════
# Task 5 – Data Encoding 
# ═══════════════════════════════════════════════════
def encode_and_clean(**context):
    """Merge cleaned + featured data, encode categoricals, drop irrelevant cols."""
    ti = context["ti"]
    clean_path = ti.xcom_pull(key="clean_path", task_ids="handle_missing_values")
    feat_path = ti.xcom_pull(key="featured_path", task_ids="engineer_features")

    df_clean = pd.read_csv(clean_path)
    df_feat = pd.read_csv(feat_path)

    # Take cleaned columns (Age, Embarked) + engineered columns (FamilySize, IsAlone)
    df = df_clean.copy()
    df["FamilySize"] = df_feat["FamilySize"]
    df["IsAlone"] = df_feat["IsAlone"]

    # Encode categorical variables
    df = pd.get_dummies(df, columns=["Sex", "Embarked"], drop_first=True)

    # Drop irrelevant columns
    df = df.drop(["Name", "Ticket", "Cabin", "PassengerId"], axis=1, errors="ignore")

    # Ensure boolean dummy columns are int
    for col in df.columns:
        if df[col].dtype == "bool":
            df[col] = df[col].astype(int)

    final_path = os.path.join(ARTIFACTS_DIR, "titanic_final.csv")
    df.to_csv(final_path, index=False)
    ti.xcom_push(key="final_path", value=final_path)
    print(f"Encoding complete → {final_path}")
    print(f"Final columns: {list(df.columns)}")


encode_task = PythonOperator(
    task_id="encode_and_clean",
    python_callable=encode_and_clean,
    dag=dag,
)


# ═══════════════════════════════════════════════════
# Task 6 – Model Training with MLflow 
# ═══════════════════════════════════════════════════
def train_model(**context):
    """Train LogisticRegression or RandomForest based on DAG trigger config.
    Logs params, model artifact, and dataset size to MLflow.
    Artifacts are uploaded via HTTP proxy through the MLflow tracking server,
    so no local /mlflow filesystem access is needed from Airflow containers.
    """
    ti = context["ti"]
    dag_conf = context.get("dag_run").conf or {}

    path = ti.xcom_pull(key="final_path", task_ids="encode_and_clean")
    df = pd.read_csv(path)

    X = df.drop("Survived", axis=1)
    y = df["Survived"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model_type = dag_conf.get("model_type", "LogisticRegression")

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    with mlflow.start_run(run_name=f"{model_type}_run") as run:
        if model_type == "RandomForest":
            n_estimators = int(dag_conf.get("n_estimators", 100))
            max_depth = dag_conf.get("max_depth", None)
            if max_depth is not None:
                max_depth = int(max_depth)

            model = RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                random_state=42,
            )
            mlflow.log_param("model_type", "RandomForest")
            mlflow.log_param("n_estimators", n_estimators)
            mlflow.log_param("max_depth", max_depth)
        else:
            
            max_iter = int(dag_conf.get("max_iter", 1000))
            C = float(dag_conf.get("C", 1.0))

            model = LogisticRegression(max_iter=max_iter, C=C, random_state=42)
            mlflow.log_param("model_type", "LogisticRegression")
            mlflow.log_param("max_iter", max_iter)
            mlflow.log_param("C", C)

        model.fit(X_train, y_train)

        mlflow.log_param("random_state", 42)
        mlflow.log_param("test_size", 0.2)
        mlflow.log_metric("dataset_rows", len(df))
        mlflow.log_metric("train_rows", len(X_train))
        mlflow.log_metric("test_rows", len(X_test))

        mlflow.sklearn.log_model(
            sk_model=model,
            name="model",
        )
        print(f"Model trained & logged to MLflow. Run ID: {run.info.run_id}")

        # Save model and test data to shared temp dir so downstream tasks
        # (evaluate_model) can load them. XCom only stores the file paths.
        model_path = os.path.join(ARTIFACTS_DIR, "model.joblib")
        joblib.dump(model, model_path)

        test_path = os.path.join(ARTIFACTS_DIR, "test_data.csv")
        test_df = X_test.copy()
        test_df["Survived"] = y_test.values
        test_df.to_csv(test_path, index=False)

        ti.xcom_push(key="run_id", value=run.info.run_id)
        ti.xcom_push(key="model_path", value=model_path)
        ti.xcom_push(key="test_data_path", value=test_path)


train_task = PythonOperator(
    task_id="train_model",
    python_callable=train_model,
    dag=dag,
)


# ═══════════════════════════════════════════════════
# Task 7 – Model Evaluation 
# ═══════════════════════════════════════════════════
def evaluate_model(**context):
    """Compute Accuracy, Precision, Recall, F1 and log to MLflow."""
    ti = context["ti"]

    model_path = ti.xcom_pull(key="model_path", task_ids="train_model")
    test_data_path = ti.xcom_pull(key="test_data_path", task_ids="train_model")
    run_id = ti.xcom_pull(key="run_id", task_ids="train_model")

    model = joblib.load(model_path)
    test_df = pd.read_csv(test_data_path)

    y_test = test_df["Survived"]
    X_test = test_df.drop("Survived", axis=1)

    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1-score:  {f1:.4f}")

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    with mlflow.start_run(run_id=run_id):
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("precision", prec)
        mlflow.log_metric("recall", rec)
        mlflow.log_metric("f1_score", f1)

    ti.xcom_push(key="accuracy", value=float(acc))


evaluate_task = PythonOperator(
    task_id="evaluate_model",
    python_callable=evaluate_model,
    dag=dag,
)


# ═══════════════════════════════════════════════════
# Task 8 – Branching Logic 
# ═══════════════════════════════════════════════════
def decide_model_fate(**context):
    """BranchPythonOperator: route to register or reject based on accuracy."""
    ti = context["ti"]
    acc = ti.xcom_pull(key="accuracy", task_ids="evaluate_model")
    print(f"Accuracy for branching decision: {acc:.4f}")

    if acc is not None and float(acc) >= 0.80:
        print("→ Routing to: register_model")
        return "register_model"
    else:
        print("→ Routing to: reject_model")
        return "reject_model"


branch_task = BranchPythonOperator(
    task_id="branch_model_decision",
    python_callable=decide_model_fate,
    dag=dag,
)


# ═══════════════════════════════════════════════════
# Task 9 – Model Registration / Rejection 
# ═══════════════════════════════════════════════════
def register_model_fn(**context):
    """Register the approved model in MLflow Model Registry."""
    ti = context["ti"]
    run_id = ti.xcom_pull(key="run_id", task_ids="train_model")
    acc = ti.xcom_pull(key="accuracy", task_ids="evaluate_model")

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    # ✅ artifact_path must match what was used in log_model above
    model_uri = f"runs:/{run_id}/model"
    result = mlflow.register_model(model_uri, "TitanicSurvivalModel")

    print(f"Model REGISTERED in MLflow Model Registry!")
    print(f"  Name:    {result.name}")
    print(f"  Version: {result.version}")
    print(f"  Run ID:  {run_id}")
    print(f"  Accuracy: {acc:.4f}")


def reject_model_fn(**context):
    """Log rejection reason when accuracy is below threshold."""
    ti = context["ti"]
    acc = ti.xcom_pull(key="accuracy", task_ids="evaluate_model")
    run_id = ti.xcom_pull(key="run_id", task_ids="train_model")

    reason = f"Model REJECTED — accuracy {acc:.4f} < 0.80 threshold"
    print(reason)

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    with mlflow.start_run(run_id=run_id):
        mlflow.set_tag("model_status", "REJECTED")
        mlflow.set_tag("rejection_reason", reason)


register_task = PythonOperator(
    task_id="register_model",
    python_callable=register_model_fn,
    dag=dag,
)

reject_task = PythonOperator(
    task_id="reject_model",
    python_callable=reject_model_fn,
    dag=dag,
)


# ═══════════════════════════════════════════════════
# Task Dependencies (DAG Graph)
# ═══════════════════════════════════════════════════
# Linear: ingest → validate
ingest_task >> validate_task

# Parallel: validate → [handle_missing, engineer_features]
validate_task >> [handle_missing_task, engineer_features_task]

# Join: both parallel tasks → encode
[handle_missing_task, engineer_features_task] >> encode_task

# Linear: encode → train → evaluate → branch
encode_task >> train_task >> evaluate_task >> branch_task

# Branch: register OR reject
branch_task >> [register_task, reject_task]