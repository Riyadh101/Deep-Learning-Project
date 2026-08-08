# Customer Churn Prediction System

An end-to-end deep learning pipeline that predicts whether a telecom customer will churn, built with TensorFlow/Keras. The project covers data preprocessing, ANN model design, regularization, hyperparameter tuning, evaluation, and a deployable inference pipeline.

## Business Problem

In the telecom industry, acquiring a new customer costs **5–25x more** than retaining an existing one. This project predicts customer churn (`Yes` / `No`) from demographic, account, and subscription data, allowing retention teams to proactively target at-risk customers.

## Dataset

- **Source:** [IBM Telco Customer Churn Dataset (Kaggle)](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
- **Size:** 7,043 customers, 21 features
- **Target:** `Churn` (`Yes` / `No`) — imbalanced (~73% No, ~27% Yes)
- **Features:** demographics, contract type, payment method, monthly/total charges, subscribed services (internet, security, streaming, etc.)

## Pipeline Overview

1. **EDA** — class imbalance, contract-type impact, tenure vs. churn behavior
2. **Data Cleaning** — drop `customerID`, fix `TotalCharges` type/missing values, encode target
3. **Preprocessing** — `StandardScaler` for numeric features, `OneHotEncoder` for categorical features, fitted **only on the training split** to prevent data leakage
4. **Train / Validation / Test Split** — 70% / 15% / 15%, stratified on the target
5. **Model** — Artificial Neural Network (ANN) with:
   - L2 kernel regularization
   - Dropout layers
   - Early stopping on validation loss
   - Class weighting to handle target imbalance
6. **Hyperparameter Tuning** — learning-rate sweep, K-Fold cross-validation, and Keras Tuner random search over layer sizes, dropout rate, and learning rate
7. **Evaluation** — classification report, confusion matrix, ROC-AUC on the held-out test set
8. **Baseline Comparison** — Random Forest classifier for context
9. **Deployment** — trained model and preprocessor saved to disk with a reusable inference function

## Results (Test Set)

| Metric | No Churn | Churn |
|---|---|---|
| Precision | 0.90 | 0.54 |
| Recall | 0.76 | 0.78 |
| F1-score | 0.82 | 0.63 |

- **Overall Accuracy:** ~76%
- **ROC-AUC:** ~0.84
- Class weighting was used to prioritize **recall on the churn class**, since missing an at-risk customer is costlier to the business than a false alarm.

## Project Structure

```
.
├── Customer_Churn_Prediction_System.ipynb   # Full end-to-end pipeline
├── predict.py                               # Standalone inference script
├── README.md                                # This file
├── requirements.txt                         # Python dependencies
├── best_churn_model.h5                      # Initial model checkpoint (pre-tuning)
├── final_churn_model.h5                     # Final tuned model (used for inference)
└── preprocessor.pkl                         # Fitted scaler + encoder for inference
```

## Setup

```bash
git clone <repo-url>
cd customer-churn-prediction
pip install -r requirements.txt
```

Download `Telco-Customer-Churn.csv` from the [Kaggle dataset page](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) and place it in the project root, then run the notebook top to bottom.

## Inference Example

```python
from predict import predict_customer_churn

sample_customer = {
    'gender': 'Female', 'SeniorCitizen': 0, 'Partner': 'Yes', 'Dependents': 'No',
    'tenure': 2, 'PhoneService': 'Yes', 'MultipleLines': 'No',
    'InternetService': 'Fiber optic', 'OnlineSecurity': 'No', 'OnlineBackup': 'No',
    'DeviceProtection': 'No', 'TechSupport': 'No', 'StreamingTV': 'Yes',
    'StreamingMovies': 'Yes', 'Contract': 'Month-to-month', 'PaperlessBilling': 'Yes',
    'PaymentMethod': 'Electronic check', 'MonthlyCharges': 85.70, 'TotalCharges': 171.40
}

prob, pred = predict_customer_churn(sample_customer)
print(f"Churn probability: {prob:.2%}  |  Prediction: {'Churn' if pred else 'No Churn'}")
```

## Limitations

- Trained on a single snapshot of one telecom provider's data — may not generalize to other markets or time periods without retraining.
- Precision on the churn class (~0.54) is moderate; roughly half of customers flagged as "at risk" will not actually churn, a deliberate trade-off given the class weighting toward recall.
- No temporal/behavioral features (e.g., usage trends over time) — only static account attributes are used.

## Tech Stack

Python, TensorFlow/Keras, scikit-learn, Keras Tuner, pandas, matplotlib/seaborn
