# test_request.py
import requests
import json

# URL of your running FastAPI app
# URL = "http://127.0.0.1:8000/predict"

# Docker
# URL = "http://localhost:8000/predict"

URL = "http://api:8000/predict"

# Example input features (replace with your actual feature names)
payload = {
    "age": 63,
    "sex": 1,
    "cp": 3,
    "trestbps": 145,
    "chol": 233,
    "fbs": 1,
    "restecg": 0,
    "thalach": 150,
    "exang": 0,
    "oldpeak": 2.3,
    "slope": 0,
    "ca": 0,
    "thal": 1
}

# Send POST request
response = requests.post(URL, json=payload)

# Print response
print("Status code:", response.status_code)
print("Response JSON:", response.json())
