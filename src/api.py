from fastapi import FastAPI
import joblib
import pandas as pd
from pathlib import Path

app = FastAPI(title="Fraud Risk Scoring API")

MODEL_PATH = Path("artifacts/fraud_model.joblib")
THRESHOLD_PATH = Path("artifacts/threshold.joblib")

model = joblib.load(MODEL_PATH)
threshold = joblib.load(THRESHOLD_PATH)

@app.get("/")
def root():
    return {"message": "Fraud Detection API is running"}

@app.post("/predict")
def predict(transaction: dict):
    df = pd.DataFrame([transaction])
    proba = model.predict_proba(df)[0, 1]
    fraud = proba >= threshold

    return {
        "fraud_probability": float(proba),
        "fraud_prediction": bool(fraud),
        "threshold": float(threshold)
    } 

from fastapi.responses import HTMLResponse
from pathlib import Path

@app.get("/ui", response_class=HTMLResponse)
def ui():
    html_path = Path(__file__).parent / "ui.html"
    return html_path.read_text(encoding="utf-8")