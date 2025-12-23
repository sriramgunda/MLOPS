import pickle
import numpy as np
from flask import Flask, request, jsonify

# Load trained model
with open("heart_model.pkl", "rb") as f:
    model = pickle.load(f)

app = Flask(__name__)


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    # Expecting: {"features": [f1, f2, ..., f13]}
    features = np.array([data["features"]])

    prediction = int(model.predict(features)[0])
    confidence = float(model.predict_proba(features)[0][prediction])

    return jsonify(
        {
            "prediction": prediction,
            "confidence": confidence,
        }
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001)


"""
Test the code using this call:
curl -X POST http://127.0.0.1:5001/predict \
-H "Content-Type: application/json" \
-d '{"features":[55,1,0,140,250,0,1,150,0,1.0,1,0,2]}'
"""
