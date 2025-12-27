import os
import numpy as np
import mlflow
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from src.data.load_data import load_data
from src.data.preprocess import build_preprocessor

# -----------------------------
# MLflow configuration
# -----------------------------
MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
mlflow.set_tracking_uri(MLFLOW_URI)
mlflow.set_experiment("heart-disease-eval")

# -----------------------------
# Load data
# -----------------------------
df = load_data()
X = df.drop("target", axis=1)
y = df["target"]

# -----------------------------
# Define models
# -----------------------------
models = {
    "LogisticRegression": LogisticRegression(max_iter=1000),
    "RandomForest": RandomForestClassifier(
        n_estimators=200,
        random_state=42
    )
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# -----------------------------
# Evaluate each model
# -----------------------------
for name, model in models.items():
    pipe = Pipeline([
        ("prep", build_preprocessor()),
        ("clf", model)
    ])

    scores = cross_validate(
        pipe,
        X,
        y,
        cv=cv,
        scoring=["accuracy", "precision", "recall", "roc_auc"]
    )

    mean_scores = {
        metric: np.mean(values)
        for metric, values in scores.items()
    }

    print(f"{name} cross-validated metrics:", mean_scores)

    with mlflow.start_run(run_name=name):
        for metric, value in mean_scores.items():
            mlflow.log_metric(metric, value)
