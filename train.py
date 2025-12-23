import pandas as pd
import numpy as np
import pickle
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


def main():
    # 1. Configure MLflow (local, free)
    mlflow.set_tracking_uri("file:/content/mlruns")
    mlflow.set_experiment("heart-disease-poc")

    with mlflow.start_run():

        # 2. Download dataset
        url = (
                "https://archive.ics.uci.edu/ml/"
                "machine-learning-databases/"
                "heart-disease/processed.cleveland.data"
        )

        columns = [
            "age", "sex", "cp", "trestbps", "chol",
            "fbs", "restecg", "thalach", "exang",
            "oldpeak", "slope", "ca", "thal", "target"
        ]

        df = pd.read_csv(url, header=None, names=columns)

        # 3. Minimal preprocessing
        df.replace("?", np.nan, inplace=True)
        df = df.apply(pd.to_numeric)
        df.dropna(inplace=True)
        df["target"] = (df["target"] > 0).astype(int)

        X = df.drop("target", axis=1)
        y = df["target"]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # 4. Train model
        max_iter = 1000
        model = LogisticRegression(max_iter=max_iter)
        model.fit(X_train, y_train)

        # 5. Evaluate
        accuracy = accuracy_score(y_test, model.predict(X_test))

        # 6. Log to MLflow
        mlflow.log_param("model_type", "LogisticRegression")
        mlflow.log_param("max_iter", max_iter)
        mlflow.log_metric("accuracy", accuracy)

        # 7. Save & log model artifact
        with open("heart_model.pkl", "wb") as f:
            pickle.dump(model, f)

        mlflow.log_artifact("heart_model.pkl")

        print("Training complete. Accuracy:", accuracy)


if __name__ == "__main__":
    main()
