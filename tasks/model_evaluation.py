import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def evaluate_model(ti):
    """
    Evaluates the trained model logged in MLflow (from train_model task)
    without retraining. Logs metrics and pushes accuracy to XCom for branching.
    """
    # Pull the MLflow run_id from train_model task
    run_id = ti.xcom_pull(task_ids="train_model", key="run_id")
    if run_id is None:
        raise ValueError("run_id not found in XCom. Cannot evaluate model.")

    # Load the trained model from MLflow
    model_uri = f"runs:/{run_id}/model"
    model = mlflow.sklearn.load_model(model_uri)

    # Load the dataset
    df = pd.read_csv("/mnt/ml-data/data/final.csv")
    X = df.drop("Survived", axis=1)
    y = df["Survived"]

    # Split dataset the same way as training
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # Predict using the trained model
    preds = model.predict(X_test)

    # Compute metrics
    acc = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds)
    rec = recall_score(y_test, preds)
    f1 = f1_score(y_test, preds)

    # Log metrics to MLflow under the same run
    with mlflow.start_run(run_id=run_id):
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("precision", prec)
        mlflow.log_metric("recall", rec)
        mlflow.log_metric("f1", f1)

    # Push accuracy to XCom for branching
    ti.xcom_push(key="accuracy", value=acc)

    print(f"Evaluation complete. Accuracy: {acc}, Precision: {prec}, Recall: {rec}, F1: {f1}")