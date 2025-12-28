import pytest
from fastapi.testclient import TestClient
import pandas as pd
from unittest.mock import patch, mock_open
import pickle
import sys
import os

# Add app to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'app'))

from main import app

client = TestClient(app)

# Mock model for testing
mock_model = type('MockModel', (), {
    'predict': lambda self, df: [1],
    'predict_proba': lambda self, df: [[0.3, 0.7]]
})()

def test_health_check():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"status": "API is running"}

def test_predict_heart_disease_valid_input():
    # Mock the model loading
    with patch('builtins.open', mock_open()) as mock_file:
        with patch('pickle.load', return_value=mock_model):
            # Reload the app to use the mock
            import importlib
            import main
            importlib.reload(main)
            client = TestClient(main.app)

            data = {
                "age": 51.0,
                "sex": 1.0,
                "cp": 2.0,
                "trestbps": 130.0,
                "chol": 250.0,
                "fbs": 0.0,
                "restecg": 1.0,
                "thalach": 150.0,
                "exang": 0.0,
                "oldpeak": 1.5,
                "slope": 2.0,
                "ca": 0.0,
                "thal": 2.0
            }
            response = client.post("/predict", json=data)
            assert response.status_code == 200
            result = response.json()
            assert "prediction" in result
            assert "confidence" in result
            assert isinstance(result["prediction"], int)
            assert isinstance(result["confidence"], float)
            assert 0 <= result["confidence"] <= 1

def test_predict_heart_disease_missing_field():
    data = {
        "age": 50.0,
        "sex": 1.0,
        # Missing cp
        "trestbps": 130.0,
        "chol": 250.0,
        "fbs": 0.0,
        "restecg": 1.0,
        "thalach": 150.0,
        "exang": 0.0,
        "oldpeak": 1.5,
        "slope": 2.0,
        "ca": 0.0,
        "thal": 2.0
    }
    response = client.post("/predict", json=data)
    assert response.status_code == 422  # Validation error

def test_predict_heart_disease_invalid_type():
    data = {
        "age": "fifty",  # Invalid type
        "sex": 1.0,
        "cp": 2.0,
        "trestbps": 130.0,
        "chol": 250.0,
        "fbs": 0.0,
        "restecg": 1.0,
        "thalach": 150.0,
        "exang": 0.0,
        "oldpeak": 1.5,
        "slope": 2.0,
        "ca": 0.0,
        "thal": 2.0
    }
    response = client.post("/predict", json=data)
    assert response.status_code == 422  # Validation error

def test_metrics_endpoint():
    response = client.get("/metrics")
    assert response.status_code == 200
    assert "text/plain" in response.headers["content-type"]
    # Check if it contains some expected metrics
    content = response.text
    assert "api_requests_total" in content
    assert "api_request_latency_seconds" in content

def test_request_logging():
    # Test that logging works (we can't easily test the logs, but ensure no errors)
    response = client.get("/")
    assert response.status_code == 200

def test_prometheus_metrics_increment():
    # Test that metrics are incremented
    initial_response = client.get("/metrics")
    initial_count = initial_response.text.count('api_requests_total')

    client.get("/")

    after_response = client.get("/metrics")
    after_count = after_response.text.count('api_requests_total')

    assert after_count > initial_count

if __name__ == "__main__":
    pytest.main([__file__])