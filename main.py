"""
FastAPI application for the Customer Churn Prediction System.

Exposes a REST API so the model can be called over HTTP from anywhere
(a web form, another service, curl, Postman, etc.).

Run locally:
    uvicorn main:app --reload

Then visit http://127.0.0.1:8000/docs for interactive Swagger UI.

Requires the artifacts produced by the training notebook to be present
in the same directory:
    - final_churn_model.h5
    - preprocessor.pkl
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import pandas as pd
import joblib
import tensorflow as tf
import os

app = FastAPI(
    title="Customer Churn Prediction API",
    description="Predicts whether a telecom customer will churn based on account and service attributes.",
    version="1.0.0",
)

MODEL_PATH = "final_churn_model.h5"
PREPROCESSOR_PATH = "preprocessor.pkl"

model = None
preprocessor = None


class CustomerData(BaseModel):
    gender: str = Field(..., example="Female")
    SeniorCitizen: int = Field(..., example=0, description="0 = No, 1 = Yes")
    Partner: str = Field(..., example="Yes")
    Dependents: str = Field(..., example="No")
    tenure: int = Field(..., example=2, description="Months with the company")
    PhoneService: str = Field(..., example="Yes")
    MultipleLines: str = Field(..., example="No")
    InternetService: str = Field(..., example="Fiber optic")
    OnlineSecurity: str = Field(..., example="No")
    OnlineBackup: str = Field(..., example="No")
    DeviceProtection: str = Field(..., example="No")
    TechSupport: str = Field(..., example="No")
    StreamingTV: str = Field(..., example="Yes")
    StreamingMovies: str = Field(..., example="Yes")
    Contract: str = Field(..., example="Month-to-month")
    PaperlessBilling: str = Field(..., example="Yes")
    PaymentMethod: str = Field(..., example="Electronic check")
    MonthlyCharges: float = Field(..., example=85.70)
    TotalCharges: float = Field(..., example=171.40)


class PredictionResponse(BaseModel):
    churn_probability: float
    churn_prediction: int
    risk_label: str


@app.on_event("startup")
def load_artifacts():
    """Load the model and preprocessor once, when the server starts."""
    global model, preprocessor

    if not os.path.exists(MODEL_PATH) or not os.path.exists(PREPROCESSOR_PATH):
        print(
            f"WARNING: '{MODEL_PATH}' or '{PREPROCESSOR_PATH}' not found. "
            "Run the training notebook first and place both files next to main.py."
        )
        return

    model = tf.keras.models.load_model(MODEL_PATH)
    preprocessor = joblib.load(PREPROCESSOR_PATH)
    print("Model and preprocessor loaded successfully.")


@app.get("/")
def root():
    return {
        "message": "Customer Churn Prediction API is running.",
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health")
def health_check():
    ready = model is not None and preprocessor is not None
    return {"status": "ok" if ready else "model_not_loaded", "model_ready": ready}


@app.post("/predict", response_model=PredictionResponse)
def predict(customer: CustomerData):
    if model is None or preprocessor is None:
        raise HTTPException(
            status_code=503,
            detail="Model is not loaded. Ensure final_churn_model.h5 and preprocessor.pkl are present.",
        )

    df_customer = pd.DataFrame([customer.dict()])
    processed = preprocessor.transform(df_customer)

    prob = float(model.predict(processed, verbose=0)[0][0])
    pred = int(prob > 0.5)
    label = "High Risk (Likely to Churn)" if pred == 1 else "Low Risk (Likely to Stay)"

    return PredictionResponse(
        churn_probability=round(prob, 4),
        churn_prediction=pred,
        risk_label=label,
    )
