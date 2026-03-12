# train_model.py
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

def train_model(ti, max_iter=200, C=1.0):
    """
    Trains a Logistic Regression model and logs it to MLflow.
    Pushes MLflow run_id to XCom for downstream registration.
    """
    # Read processed data
    df = pd.read_csv("/mnt/ml-data/data/final.csv")  # Make sure this path exists in Docker
    X = df.drop("Survived", axis=1)
    y = df["Survived"]

    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # Start MLflow run
    with mlflow.start_run() as run:
        model = LogisticRegression(max_iter=max_iter, C=C)
        model.fit(X_train, y_train)

        # Log parameters and model
        mlflow.log_param("model_type", "LogisticRegression")
        mlflow.log_param("max_iter", max_iter)
        mlflow.log_param("C", C)
        mlflow.log_param("dataset_size", len(df))

        mlflow.sklearn.log_model(model, "model")

        # Push run_id to XCom for register_model task
        ti.xcom_push(key="run_id", value=run.info.run_id)

    print(f"Training complete. MLflow run_id={run.info.run_id}")