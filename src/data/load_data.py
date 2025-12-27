import pandas as pd
from pathlib import Path

DATA_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"

COLUMNS = [
    "age", "sex", "cp", "trestbps", "chol",
    "fbs", "restecg", "thalach", "exang",
    "oldpeak", "slope", "ca", "thal", "target"
]

DATA_PATH = Path("data/raw/heart.csv")


def load_data():
    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(
        DATA_URL,
        header=None,
        names=COLUMNS,
        na_values="?"
    )

    # Convert target to binary
    df["target"] = (df["target"] > 0).astype(int)

    df.to_csv(DATA_PATH, index=False)

    print(f"Dataset saved to {DATA_PATH}")
    return df


if __name__ == "__main__":
    load_data()
