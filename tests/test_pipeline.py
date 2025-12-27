import pytest
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_validate
import mlflow
from mlflow.tracking import MlflowClient

from src.data.load_data import load_data
from src.data.preprocess import build_preprocessor


# -----------------------------
# 1. Test data loading
# -----------------------------
def test_load_data():
    df = load_data()
    assert isinstance(df, pd.DataFrame)

    expected_cols = [
        "age", "sex", "cp", "trestbps", "chol",
        "fbs", "restecg", "thalach", "exang",
        "oldpeak", "slope", "ca", "thal", "target"
    ]
    assert list(df.columns) == expected_cols
    assert set(df["target"].unique()).issubset({0, 1})
    assert df.isna().sum().max() < len(df)


# -----------------------------
# 2. Test preprocessor
# -----------------------------
def test_preprocessor():
    df = load_data()
    X = df.drop("target", axis=1)

    preprocessor = build_preprocessor()
    X_transformed = preprocessor.fit_transform(X)

    assert isinstance(X_transformed, np.ndarray)
    assert X_transformed.shape[0] == X.shape[0]


# -----------------------------
# 3. Test training pipeline
# -----------------------------
@pytest.mark.parametrize(
    "model",
    [
        LogisticRegression(max_iter=500),
        RandomForestClassifier(n_estimators=10, random_state=42),
    ],
)
def test_training_pipeline(model):
    df = load_data().iloc[:50]
    X = df.drop("target", axis=1)
    y = df["target"]

    pipe = Pipeline([
        ("prep", build_preprocessor()),
        ("clf", model)
    ])

    pipe.fit(X, y)
    preds = pipe.predict(X)

    assert len(preds) == len(y)


# -----------------------------
# 4. Test cross-validation
# -----------------------------
def test_cross_validation_metrics():
    df = load_data().iloc[:50]
    X = df.drop("target", axis=1)
    y = df["target"]

    pipe = Pipeline([
        ("prep", build_preprocessor()),
        ("clf", LogisticRegression(max_iter=500))
    ])

    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

    scores = cross_validate(
        pipe,
        X,
        y,
        cv=cv,
        scoring=["accuracy", "precision", "recall", "roc_auc"]
    )

    for metric in ["test_accuracy", "test_precision", "test_recall", "test_roc_auc"]:
        assert metric in scores
        assert np.all((scores[metric] >= 0.0) & (scores[metric] <= 1.0))


# -----------------------------
# 5. Test MLflow logging (isolated)
# -----------------------------
def test_mlflow_logging(tmp_path):
    mlflow_uri = f"file:///{tmp_path}/mlruns"
    mlflow.set_tracking_uri(mlflow_uri)

    mlflow.set_experiment("test-exp")

    with mlflow.start_run():
        mlflow.log_metric("accuracy", 0.9)

    client = MlflowClient()
    exp = client.get_experiment_by_name("test-exp")

    runs = client.search_runs([exp.experiment_id])
    assert len(runs) == 1
    assert "accuracy" in runs[0].data.metrics
