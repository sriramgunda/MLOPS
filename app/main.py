from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import pandas as pd

# Load model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

app = FastAPI(title="Heart Disease Prediction API")

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
    
    prediction = model.predict(df)[0]
    probability = model.predict_proba(df)[0][1]

    return {
        "prediction": int(prediction),
        "confidence": round(probability, 4)
    }
