import logging
import time
from fastapi import FastAPI, Request
from pydantic import BaseModel
import pickle
import pandas as pd
import os

from prometheus_client import Counter, Histogram, generate_latest
from fastapi.responses import Response

# ------------------------
# Logging Configuration
# ------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)

# Metrics
REQUEST_COUNT = Counter(
    "api_requests_total",
    "Total API requests",
    ["method", "endpoint"]
)

REQUEST_LATENCY = Histogram(
    "api_request_latency_seconds",
    "Request latency"
)

# Load model
model_path = os.path.join(os.path.dirname(__file__), "model.pkl")
with open(model_path, "rb") as f:
    model = pickle.load(f)

app = FastAPI(title="Heart Disease Prediction API")

# ------------------------
# Request Logging Middleware
# ------------------------
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    duration = time.time() - start_time

    logger.info(
        f"{request.method} {request.url.path} "
        f"status={response.status_code} "
        f"duration={duration:.3f}s"
    )
    return response

# Input schema
class PatientInput(BaseModel):
    age: float
    sex: float
    cp: float
    trestbps: float
    chol: float
    fbs: float
    restecg: float
    thalach: float
    exang: float
    oldpeak: float
    slope: float
    ca: float
    thal: float

@app.get("/")
def health_check():
    return {"status": "API is running"}

@app.post("/predict")
def predict_heart_disease(data: PatientInput):

    df = pd.DataFrame([data.dict()])
    
    prediction = int(model.predict(df)[0])
    probability = float(model.predict_proba(df)[0][1])

    logger.info(
        f"Prediction={prediction}, Confidence={probability:.4f}"
    )

    return {
        "prediction": prediction,
        "confidence": round(probability, 4)
    }

@app.middleware("http")
async def prometheus_metrics(request: Request, call_next):
    REQUEST_COUNT.labels(
        method=request.method,
        endpoint=request.url.path
    ).inc()

    with REQUEST_LATENCY.time():
        response = await call_next(request)

    return response

@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type="text/plain")