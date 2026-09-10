from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

app = FastAPI(title="Fraud Detection API")

model = joblib.load("models/final_model.pkl")
scaler = joblib.load("data/scaler.pkl")

FEATURE_ORDER = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]


class Transaction(BaseModel):
    Time: float
    V1: float; V2: float; V3: float; V4: float; V5: float
    V6: float; V7: float; V8: float; V9: float; V10: float
    V11: float; V12: float; V13: float; V14: float; V15: float
    V16: float; V17: float; V18: float; V19: float; V20: float
    V21: float; V22: float; V23: float; V24: float; V25: float
    V26: float; V27: float; V28: float
    Amount: float


@app.get("/")
def root():
    return {"status": "Fraud Detection API is running"}


@app.post("/predict")
def predict(transaction: Transaction):
    data = transaction.dict()
    df = pd.DataFrame([data])[FEATURE_ORDER]

    # Scale Time and Amount using the SAME scaler fitted during training
    df[["Time", "Amount"]] = scaler.transform(df[["Time", "Amount"]])

    prediction = model.predict(df)[0]
    probability = model.predict_proba(df)[0][1]

    return {
        "is_fraud": bool(prediction),
        "fraud_probability": round(float(probability), 4),
    }