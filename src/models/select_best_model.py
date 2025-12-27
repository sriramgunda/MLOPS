import os
import mlflow
from mlflow.tracking import MlflowClient
import logging

logging.basicConfig(level=logging.INFO)

EXPERIMENT_NAME = "heart-disease"
MODEL_NAME = "HeartDiseaseClassifier"
METRIC = "roc_auc"

def select_best_model(metric=METRIC, experiment_name=EXPERIMENT_NAME, model_name=MODEL_NAME):
    mlflow_uri = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
    mlflow.set_tracking_uri(mlflow_uri)
    client = MlflowClient()
    
    logging.info(f"Using MLflow tracking URI: {mlflow_uri}")

    # -----------------------------
    # Get experiment
    # -----------------------------
    exp = client.get_experiment_by_name(experiment_name)
    if exp is None:
        raise ValueError(f"Experiment '{experiment_name}' not found")

    # -----------------------------
    # Get best run
    # -----------------------------
    runs = client.search_runs(
        experiment_ids=[exp.experiment_id],
        order_by=[f"metrics.{metric} DESC"],
        max_results=1
    )
    
    if not runs:
        raise ValueError(f"No runs found for experiment '{experiment_name}'")

    best_run = runs[0]
    logging.info(f"Best run found: {best_run.info.run_id}, {metric}={best_run.data.metrics[metric]}")

    # -----------------------------
    # Register model
    # -----------------------------
    model_uri = f"runs:/{best_run.info.run_id}/model"

    logging.info(f"Registering model '{model_name}' from run {best_run.info.run_id}...")
    registered = mlflow.register_model(model_uri=model_uri, name=model_name)

    # -----------------------------
    # Promote to Production
    # -----------------------------
    client.transition_model_version_stage(
        name=model_name,
        version=registered.version,
        stage="Production",
        archive_existing_versions=True
    )

    logging.info(
        f"Model '{model_name}' v{registered.version} promoted to Production from run {best_run.info.run_id}"
    )

    return best_run.info.run_id, registered.version


# -----------------------------
# Script entry point
# -----------------------------
if __name__ == "__main__":
    try:
        select_best_model()
    except Exception as e:
        logging.error(f"Failed to select and register best model: {e}")
        exit(1)
