# scripts/register_model.py
import mlflow
from mlflow.tracking import MlflowClient

def register_best_model():
    """Finds the best run and registers its model in the MLflow Model Registry."""
    MLFLOW_TRACKING_URI = "http://127.0.0.1:5000"
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = MlflowClient()

    # Get the experiment ID (assuming default experiment "0")
    experiment = client.get_experiment_by_name("Default")
    experiment_id = experiment.experiment_id

    # Search for the best run within the experiment
    runs = mlflow.search_runs(
        experiment_ids=[experiment_id],
        order_by=["metrics.accuracy DESC"],
        max_results=1
    )

    if not runs.empty:
        best_run_id = runs.iloc[0].run_id
        model_uri = f"runs:/{best_run_id}/model"
        model_name = "IrisClassifier" # The name for our registered model

        print(f"Best run ID: {best_run_id}")
        print(f"Registering model from URI: {model_uri}")

        # Register the model
        model_version = mlflow.register_model(model_uri=model_uri, name=model_name)
        print(f"Model '{model_name}' registered with version {model_version.version}")

        # Optionally, transition the model to "Staging"
        client.transition_model_version_stage(
            name=model_name,
            version=model_version.version,
            stage="Staging"
        )
        print(f"Model version {model_version.version} transitioned to Staging.")
    else:
        print("No runs found.")

if __name__ == "__main__":
    register_best_model()