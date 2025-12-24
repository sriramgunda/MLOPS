import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

def test_preprocessing_pipeline():
    # Dummy input with missing values
    X = pd.DataFrame({
        "age": [60, np.nan, 45],
        "chol": [240, 180, np.nan]
    })

    pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ]
    )

    X_transformed = pipeline.fit_transform(X)

    # Assertions
    assert not np.isnan(X_transformed).any(), "Missing values not handled"
    assert X_transformed.shape == (3, 2), "Incorrect output shape"
