import mlflow
from mlflow.tracking import MlflowClient

def register_model(ti):
    """
    Registers the trained MLflow model in the Model Registry.
    Uses the run_id from train_model task via XCom.
    """
    # Pull the MLflow run_id from train_model task
    run_id = ti.xcom_pull(task_ids="train_model", key="run_id")
    if not run_id:
        raise ValueError("run_id not found in XCom. Cannot register model.")

    # Model URI in MLflow
    model_uri = f"runs:/{run_id}/model"
    model_name = "TitanicModel"

    client = MlflowClient()

    # Create registered model if it doesn't exist
    try:
        client.create_registered_model(model_name)
        print(f"Registered new model: {model_name}")
    except Exception:
        # Model already exists
        print(f"Model '{model_name}' already exists in the registry.")

    # Create a new model version
    model_version = client.create_model_version(
        name=model_name,
        source=model_uri,
        run_id=run_id
    )

    print(f"Model version {model_version.version} registered successfully from run_id={run_id}")


def reject_model(ti):
    """
    Logs rejection reason if model accuracy is low.
    """
    # Pull accuracy from evaluate_model task
    acc = ti.xcom_pull(task_ids="evaluate_model", key="accuracy")
    print(f"Model rejected due to low accuracy: {acc}")