from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

NUM_FEATURES = ["age", "trestbps", "chol", "thalach", "oldpeak"]
CAT_FEATURES = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]

def build_preprocessor():
    num_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    cat_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    return ColumnTransformer(transformers=[
        ("num", num_pipeline, NUM_FEATURES),
        ("cat", cat_pipeline, CAT_FEATURES)
    ])

