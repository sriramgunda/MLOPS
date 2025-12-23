import pickle
import numpy as np
import os

MODEL_PATH = os.path.join(
    os.path.dirname(__file__),
    "..",
    "heart_model.pkl"
)

def test_model_load_and_predict():
    # Load model
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)

    # Sample input (same feature order as training)
    sample_input = np.array([[
        55,   # age
        1,    # sex
        0,    # cp
        140,  # trestbps
        250,  # chol
        0,    # fbs
        1,    # restecg
        150,  # thalach
        0,    # exang
        1.0,  # oldpeak
        1,    # slope
        0,    # ca
        2     # thal
    ]])

    prediction = model.predict(sample_input)

    # Assertions
    assert prediction.shape == (1,)
    assert prediction[0] in [0, 1]
