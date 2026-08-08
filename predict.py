"""
Standalone inference script for the Customer Churn Prediction System.

Loads the final tuned model and the fitted preprocessor, then predicts
churn probability for a single customer record.

Usage:
    from predict import predict_customer_churn

    prob, pred = predict_customer_churn(sample_customer)

Requires the artifacts produced by the training notebook to be present
in the same directory:
    - final_churn_model.h5
    - preprocessor.pkl
"""

import pandas as pd
import joblib
import tensorflow as tf


def predict_customer_churn(
    customer_data: dict,
    model_path: str = "final_churn_model.h5",
    preprocessor_path: str = "preprocessor.pkl",
):
    """
    Predict churn probability and label for a single customer.

    Parameters
    ----------
    customer_data : dict
        Raw customer attributes, matching the columns used during training
        (gender, SeniorCitizen, Partner, Dependents, tenure, PhoneService,
        MultipleLines, InternetService, OnlineSecurity, OnlineBackup,
        DeviceProtection, TechSupport, StreamingTV, StreamingMovies,
        Contract, PaperlessBilling, PaymentMethod, MonthlyCharges,
        TotalCharges).
    model_path : str
        Path to the saved Keras model (.h5).
    preprocessor_path : str
        Path to the saved scikit-learn ColumnTransformer (.pkl).

    Returns
    -------
    churn_prob : float
        Predicted probability of churn (0-1).
    churn_pred : int
        Binary prediction: 1 = Churn, 0 = No Churn (threshold = 0.5).
    """
    model = tf.keras.models.load_model(model_path)
    preprocessor = joblib.load(preprocessor_path)

    df_customer = pd.DataFrame([customer_data])
    processed_data = preprocessor.transform(df_customer)

    churn_prob = float(model.predict(processed_data, verbose=0)[0][0])
    churn_pred = int(churn_prob > 0.5)

    return churn_prob, churn_pred


if __name__ == "__main__":
    sample_customer = {
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": 2,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "Yes",
        "StreamingMovies": "Yes",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 85.70,
        "TotalCharges": 171.40,
    }

    prob, pred = predict_customer_churn(sample_customer)
    print("=== Sample Customer Prediction Result ===")
    print(f"Churn Probability: {prob * 100:.2f}%")
    print(f"Churn Risk Status: {'High Risk (Will Churn)' if pred == 1 else 'Low Risk (Retained)'}")
