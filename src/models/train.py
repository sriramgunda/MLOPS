import os
import mlflow
import mlflow.sklearn
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
from src.data.load_data import load_data
from src.data.preprocess import build_preprocessor

# ---------------------------
# MLflow configuration
# ---------------------------
MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
mlflow.set_tracking_uri(MLFLOW_URI)
mlflow.set_experiment("heart-disease")

# ---------------------------
# Load data
# ---------------------------
df = load_data()
X = df.drop("target", axis=1)
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y, test_size=0.2, random_state=42
)

# ---------------------------
# Models & hyperparameters
# ---------------------------
models = {
    "LogisticRegression": (
        LogisticRegression(max_iter=1000),
        {"classifier__C": [0.01, 0.1, 1, 10]}
    ),
    "RandomForest": (
        RandomForestClassifier(random_state=42),
        {"classifier__n_estimators": [100, 200],
         "classifier__max_depth": [None, 5, 10]}
    )
}

# ---------------------------
# Train & log models
# ---------------------------
for name, (model, params) in models.items():
    pipe = Pipeline([
        ("prep", build_preprocessor()),
        ("classifier", model)
    ])

    grid = GridSearchCV(pipe, params, cv=5, scoring="roc_auc")

    with mlflow.start_run(run_name=name):
        grid.fit(X_train, y_train)

        y_pred = grid.predict(X_test)
        y_prob = grid.predict_proba(X_test)[:, 1]

        mlflow.log_params(grid.best_params_)
        mlflow.log_metric("accuracy", accuracy_score(y_test, y_pred))
        mlflow.log_metric("precision", precision_score(y_test, y_pred))
        mlflow.log_metric("recall", recall_score(y_test, y_pred))
        mlflow.log_metric("roc_auc", roc_auc_score(y_test, y_prob))

        # Log model with **relative artifact path**
        mlflow.sklearn.log_model(grid.best_estimator_, name="model")

print("All models logged successfully!")
