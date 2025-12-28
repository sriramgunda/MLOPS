import mlflow.sklearn
import pandas as pd
import numpy as np
import os
from data_loader import load_data

def predict_pipeline():
    """
    Demonstrates reproducibility by loading the saved Inference Pipeline
    DIRECTLY from MLflow and making predictions on raw data.
    """
    experiment_name = "Heart_Disease_Prediction_AutoML"
    
    # 1. Get the latest main run from MLflow
    print(f"Searching for latest main run in experiment: {experiment_name}...")
    try:
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment is None:
            print(f"Experiment '{experiment_name}' not found.")
            return
        
        # Filter for the main run by checking the custom tag
        runs = mlflow.search_runs(
            experiment_ids=[experiment.experiment_id],
            filter_string="tags.run_type = 'parent'",
            order_by=["start_time DESC"],
            max_results=1
        )
        
        if runs.empty:
            print("No main runs found. Falling back to simple search...")
            runs = mlflow.search_runs(
                experiment_ids=[experiment.experiment_id],
                order_by=["start_time DESC"],
                max_results=5
            )
            # Try to find a run that isn't a child run (doesn't have parent_run_id)
            for _, run_row in runs.iterrows():
                if "tags.mlflow.parentRunId" not in run_row or pd.isna(run_row["tags.mlflow.parentRunId"]):
                    runs = pd.DataFrame([run_row])
                    break
        
        if runs.empty:
             print("Could not find a suitable parent run.")
             return

        latest_run_id = runs.iloc[0]["run_id"]
        run_name = runs.iloc[0]["tags.mlflow.runName"]
        print(f"Using Run: {run_name} (ID: {latest_run_id})")
        
    except Exception as e:
        print(f"Error connecting to MLflow: {e}")
        return

    # 2. Load the model from MLflow Registry or Artifacts
    registry_uri = "models:/Heart_Disease_Prediction_Pipeline/latest"
    model_uri = f"runs:/{latest_run_id}/inference_pipeline"
    
    print(f"Attempting to load model from Registry: {registry_uri} ...")
    try:
        pipeline = mlflow.sklearn.load_model(registry_uri)
        print("Pipeline loaded successfully from MLflow Model Registry.")
    except Exception as registry_e:
        print(f"Registry load failed (Model may not be registered yet): {registry_e}")
        print(f"Falling back to Run-based loading from: {model_uri} ...")
        try:
            pipeline = mlflow.sklearn.load_model(model_uri)
            print("Pipeline loaded successfully from MLflow Run artifacts.")
        except Exception as run_e:
            print(f"Failed to load model from both Registry and Runs: {run_e}")
            return

    # Load Sample Raw Data (Simulating new incoming data)
    print("Loading raw data for inference test...")
    df = load_data()
    if df is None: return
    
    # Take 5 random samples
    sample_data = df.sample(5, random_state=42)
    X_sample = sample_data.drop("target", axis=1)
    y_true = sample_data["target"]
    
    print("\n--- Input Data (Raw) ---")
    print(X_sample)
    
    # Predict using the loaded pipeline
    # The pipeline handles scaling and encoding automatically!
    try:
        predictions = pipeline.predict(X_sample)
        probabilities = pipeline.predict_proba(X_sample)[:, 1]
        
        print("\n--- Predictions ---")
        results = pd.DataFrame({
            'Actual': y_true.values,
            'Predicted': predictions,
            'Prob_Disease': probabilities.round(4)
        })
        print(results)
        
        print("\nSuccess! The pipeline correctly processed raw data and generated predictions.")
        
    except Exception as e:
        print(f"Inference failed: {e}")

if __name__ == "__main__":
    predict_pipeline()
