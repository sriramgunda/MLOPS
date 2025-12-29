import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow
import mlflow.sklearn
import os

from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# ===============================
# 1. DATA ACQUISITION
# ===============================

heart_disease = fetch_ucirepo(id=45)

X = heart_disease.data.features
y = heart_disease.data.targets

# Convert multi-class target to binary
y = y["num"].apply(lambda x: 1 if x > 0 else 0)

# ===============================
# 2. DATA CLEANING
# ===============================
os.makedirs("plots", exist_ok=True)

# Missing values check
print("Missing values:\n", X.isnull().sum())

# ===============================
# 3. EXPLORATORY DATA ANALYSIS
# ===============================

# Class balance
plt.figure(figsize=(4, 4))
sns.countplot(x=y)
plt.title("Class Distribution (Heart Disease)")
plt.savefig("plots/class_balance.png")
plt.close()

# Feature distributions
X.hist(figsize=(15, 10), bins=20)
plt.suptitle("Feature Distributions")
plt.savefig("plots/feature_distributions.png")
plt.close()

# Correlation heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(pd.concat([X, y], axis=1).corr(), cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.savefig("plots/correlation_heatmap.png")
plt.close()

# ===============================
# 4. FEATURE ENGINEERING
# ===============================

numeric_features = X.columns

preprocessor = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ]
)

# ===============================
# 5. MODELS
# ===============================

models = {
    "Logistic_Regression": LogisticRegression(max_iter=1000),
    "Random_Forest": RandomForestClassifier(
        n_estimators=200,
        max_depth=6,
        random_state=42
    )
}

# ===============================
# 6. MLflow EXPERIMENT TRACKING
# ===============================

mlflow.set_experiment("Heart Disease Risk Prediction")

scoring = {
    "accuracy": "accuracy",
    "precision": "precision",
    "recall": "recall",
    "roc_auc": "roc_auc"
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for model_name, model in models.items():

    with mlflow.start_run(run_name=model_name):

        pipeline = Pipeline(
            steps=[
                ("preprocessing", preprocessor),
                ("classifier", model)
            ]
        )

        cv_results = cross_validate(
            pipeline,
            X,
            y,
            cv=cv,
            scoring=scoring,
            return_train_score=False
        )

        # Log metrics
        for metric in scoring.keys():
            mlflow.log_metric(
                metric,
                np.mean(cv_results[f"test_{metric}"])
            )

        # Log model
        pipeline.fit(X, y)
        mlflow.sklearn.log_model(pipeline, model_name)

        # Log parameters
        mlflow.log_params(model.get_params())

        print(f"\n{model_name} Results:")
        for metric in scoring.keys():
            print(
                f"{metric}: "
                f"{np.mean(cv_results[f'test_{metric}']):.4f}"
            )

print("\nTraining complete. Launch MLflow UI to view results.")
