# src/api/app.py
import os
from fastapi import FastAPI
import mlflow
import mlflow.sklearn
import logging
import time
import sys
import pandas as pd
from contextlib import asynccontextmanager
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from fastapi.responses import Response
from fastapi import FastAPI, Request

# -----------------------
# Logging configuration
# -----------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# -----------------------------
# MLflow config
# -----------------------------
MODEL_NAME = "HeartDiseaseClassifier"
MODEL_STAGE = "Production"
MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
mlflow.set_tracking_uri(MLFLOW_URI)

# -----------------------------
# Lifespan context for startup/shutdown
# -----------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    global model
    try:
        logger.info("Loading model from MLflow Registry...")
        model = mlflow.sklearn.load_model(f"models:/{MODEL_NAME}/{MODEL_STAGE}")
        logger.info("Model loaded successfully")
    except Exception as e:
        logger.exception("Failed to load model")
        raise RuntimeError("Model loading failed") from e
    yield
    logger.info("Application shutdown")

# -----------------------------
# FastAPI app
# -----------------------------
app = FastAPI(title="Heart Disease Prediction API", lifespan=lifespan)

# -----------------------
# Prometheus metrics
# -----------------------
REQUEST_COUNT = Counter(
    "http_requests_total", "Total HTTP requests", ["method", "endpoint", "status"]
)

REQUEST_LATENCY = Histogram(
    "http_request_duration_seconds", "HTTP request latency", ["method", "endpoint"]
)

# -----------------------
# Middleware for logging & metrics
# -----------------------
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()
    endpoint = request.url.path
    method = request.method
    logger.info(f"Start request: {method} {endpoint}")

    try:
        response = await call_next(request)
        status_code = response.status_code
    except Exception as e:
        status_code = 500
        raise e
    finally:
        duration = time.time() - start_time
        REQUEST_COUNT.labels(method=method, endpoint=endpoint, status=status_code).inc()
        REQUEST_LATENCY.labels(method=method, endpoint=endpoint).observe(duration)
        logger.info(f"End request: {method} {endpoint} status={status_code} duration={duration:.3f}s")

    return response

# -----------------------
# Metrics endpoint for Prometheus
# -----------------------
@app.get("/metrics")
async def metrics():
    data = generate_latest()
    return Response(content=data, media_type=CONTENT_TYPE_LATEST)


# -----------------------
# Health check endpoint
# -----------------------
@app.get("/health")
async def health():
    """
    Health check endpoint for Kubernetes & monitoring.
    """
    try:
        # Check if model is loaded
        if model is None:
            return {"status": "unhealthy", "reason": "model not loaded"}
        return {"status": "healthy"}
    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}


FEATURE_COLUMNS = [
    "age", "sex", "cp", "trestbps", "chol",
    "fbs", "restecg", "thalach", "exang",
    "oldpeak", "slope", "ca", "thal"
]

# -----------------------------
# Prediction endpoint
# -----------------------------
@app.post("/predict")
def predict(features: dict):
    logger.info(f"Request received: {features}")

    # Validate input keys
    missing = set(FEATURE_COLUMNS) - set(features.keys())
    if missing:
        from fastapi import HTTPException
        raise HTTPException(
            status_code=400,
            detail=f"Missing features: {missing}"
        )

    X = pd.DataFrame([features], columns=FEATURE_COLUMNS)
    prob = model.predict_proba(X)[0][1]

    return {"prediction": int(prob >= 0.5), "confidence": float(prob)}
