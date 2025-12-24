import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier

def test_model_training_and_prediction():
    X = pd.DataFrame({
        "age": [50, 60, 45, 70],
        "chol": [200, 240, 180, 260]
    })

    y = [0, 1, 0, 1]

    model_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("classifier", RandomForestClassifier(
                n_estimators=10,
                random_state=42
            ))
        ]
    )

    model_pipeline.fit(X, y)
    preds = model_pipeline.predict(X)

    assert len(preds) == len(y), "Prediction size mismatch"
    assert set(preds).issubset({0, 1}), "Invalid prediction values"
