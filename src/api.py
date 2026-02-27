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

    from fastapi.responses import HTMLResponse

UI_HTML = """
<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>Fraud Risk Scoring UI</title>
</head>
<body style="font-family: Arial; max-width:900px; margin:40px auto;">

<h1>Fraud Risk Scoring UI</h1>

<button onclick="fill()">Fill Example</button>
<button onclick="predict()">Predict</button>

<h3>Request JSON</h3>
<textarea id="payload" style="width:100%; height:260px;"></textarea>

<h3>Result</h3>
<pre id="result">{}</pre>

<script>
function fill() {
  const ex = {
    "Time":0,
    "Amount":149.62,
    "V1":-1.35,"V2":-0.07,"V3":2.53,"V4":1.37,"V5":-0.33,
    "V6":0.46,"V7":0.23,"V8":0.09,"V9":0.36,"V10":0.09,
    "V11":-0.55,"V12":-0.61,"V13":-0.99,"V14":-0.31,
    "V15":1.46,"V16":-0.47,"V17":0.20,"V18":0.02,
    "V19":0.40,"V20":0.25,"V21":-0.01,"V22":0.27,
    "V23":-0.11,"V24":0.06,"V25":0.12,"V26":-0.18,
    "V27":0.13,"V28":-0.02
  };
  document.getElementById("payload").value =
    JSON.stringify(ex, null, 2);
}

async function predict() {
  const payload =
    JSON.parse(document.getElementById("payload").value);

  const res = await fetch("/predict", {
    method:"POST",
    headers:{"Content-Type":"application/json"},
    body:JSON.stringify(payload)
  });

  document.getElementById("result").textContent =
    await res.text();
}
</script>

</body>
</html>
"""

@app.get("/ui", response_class=HTMLResponse)
def ui():
    return UI_HTML 