import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import sys
import os

# Add src to path to import from train_automl
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from train_automl import custom_metric

def test_custom_metric():
    # Create dummy data
    X_val = np.random.rand(100, 5)
    y_val = np.random.randint(0, 2, 100)
    X_train = np.random.rand(200, 5)
    y_train = np.random.randint(0, 2, 200)

    # Create a dummy estimator
    estimator = RandomForestClassifier(n_estimators=10, random_state=42)
    estimator.fit(X_train, y_train)

    # Test the custom_metric function
    val_loss, metrics_dict = custom_metric(X_val, y_val, estimator, labels=None, X_train=X_train, y_train=y_train)

    # Assertions
    assert isinstance(val_loss, float), "val_loss should be a float"
    assert val_loss >= 0 and val_loss <= 1, "val_loss should be between 0 and 1"
    assert isinstance(metrics_dict, dict), "metrics_dict should be a dict"
    assert 'val_accuracy' in metrics_dict, "metrics_dict should contain val_accuracy"
    assert 'val_precision' in metrics_dict, "metrics_dict should contain val_precision"
    assert 'val_recall' in metrics_dict, "metrics_dict should contain val_recall"
    assert 'val_f1' in metrics_dict, "metrics_dict should contain val_f1"
    assert 'val_roc_auc' in metrics_dict, "metrics_dict should contain val_roc_auc"
    assert 'pred_time' in metrics_dict, "metrics_dict should contain pred_time"

    # Check that metrics are reasonable
    for key in ['val_accuracy', 'val_precision', 'val_recall', 'val_f1', 'val_roc_auc']:
        assert 0 <= metrics_dict[key] <= 1, f"{key} should be between 0 and 1"

    print("test_custom_metric passed")

def test_learner_map():
    # Test the learner_map dictionary from train_automl
    from train_automl import learner_map

    assert isinstance(learner_map, dict), "learner_map should be a dict"
    assert 'lgbm' in learner_map, "learner_map should contain 'lgbm'"
    assert learner_map['lgbm'] == 'LightGBM', "lgbm should map to LightGBM"
    assert learner_map['xgboost'] == 'XGBoost', "xgboost should map to XGBoost"
    assert learner_map['rf'] == 'Random Forest', "rf should map to Random Forest"

    print("test_learner_map passed")

def test_preprocessing_pipeline():
    # Test the preprocessing part
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler, OneHotEncoder
    from sklearn.compose import ColumnTransformer

    # Create dummy data similar to heart disease dataset
    data = {
        'age': np.random.randint(29, 78, 100),
        'sex': np.random.randint(0, 2, 100),
        'cp': np.random.randint(0, 4, 100),
        'trestbps': np.random.randint(94, 201, 100),
        'chol': np.random.randint(126, 565, 100),
        'fbs': np.random.randint(0, 2, 100),
        'restecg': np.random.randint(0, 3, 100),
        'thalach': np.random.randint(71, 203, 100),
        'exang': np.random.randint(0, 2, 100),
        'oldpeak': np.random.uniform(0, 6.2, 100),
        'slope': np.random.randint(0, 3, 100),
        'ca': np.random.randint(0, 4, 100),
        'thal': np.random.randint(0, 3, 100),
        'target': np.random.randint(0, 2, 100)
    }
    df = pd.DataFrame(data)

    X = df.drop("target", axis=1)
    y = df["target"]

    categorical_features = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]
    numerical_features = ["age", "trestbps", "chol", "thalach", "oldpeak"]

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    # Assertions
    assert X_train_processed.shape[0] == X_train.shape[0], "Training data shape mismatch"
    assert X_test_processed.shape[0] == X_test.shape[0], "Test data shape mismatch"
    assert X_train_processed.shape[1] > X_train.shape[1], "Processed data should have more columns due to OHE"

    print("test_preprocessing_pipeline passed")

if __name__ == "__main__":
    test_custom_metric()
    test_learner_map()
    test_preprocessing_pipeline()
    print("All tests passed!")