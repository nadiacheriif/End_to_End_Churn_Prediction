import json
import joblib
import numpy as np
import mlflow
import pandas as pd
import os

# Compute project-root `artifacts` path (repo-root/artifacts)
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
ARTIFACTS_DIR = os.path.join(project_root, "artifacts")

# Preprocessing artifact
preprocess_path = os.path.join(ARTIFACTS_DIR, "preprocessing.pkl")
if not os.path.exists(preprocess_path):
    raise FileNotFoundError(
        f"Preprocessing artifact not found at {preprocess_path}.\n"
        "Run the training pipeline (e.g. `Scripts/run.py`) to generate artifacts/preprocessing.pkl"
    )

try:
    preprocessing = joblib.load(preprocess_path)
except Exception as e:
    raise RuntimeError(f"Failed to load preprocessing artifact: {e}")

feature_columns = preprocessing.get("feature_columns")
if not feature_columns:
    raise RuntimeError("`feature_columns` not found inside preprocessing artifact")

# Load model from artifacts/model (saved during training)
model_path = os.path.join(ARTIFACTS_DIR, "model")
if not os.path.exists(model_path):
    raise FileNotFoundError(
        f"Model artifact not found at {model_path}.\n"
        "Ensure the training pipeline logged a model under artifacts/model or MLflow and copy it to artifacts/model"
    )

try:
    model = mlflow.sklearn.load_model(model_path)
except Exception as e:
    raise RuntimeError(f"Failed to load model from {model_path}: {e}")

def predict_single(input_dict: dict):
    """
    Applies training-time feature transformations and prediction.
    """
    df = pd.DataFrame([input_dict])

    # Add missing columns
    for col in feature_columns:
        if col not in df.columns:
            df[col] = 0

    df = df[feature_columns]  # keep order

    proba = model.predict_proba(df)[0][1]
    pred = int(proba >= 0.35)

    return {
        "probability_churn": float(proba),
        "prediction": pred
    }

def predict_batch(input_df: pd.DataFrame):
    """
    Applies training-time feature transformations and prediction.
    """
    df = input_df.copy()

    # Add missing columns
    for col in feature_columns:
        if col not in df.columns:
            df[col] = 0

    df = df[feature_columns]  # keep order

    proba = model.predict_proba(df)[:, 1]
    pred = (proba >= 0.35).astype(int)

    return pd.DataFrame({
        "probability_churn": proba,
        "prediction": pred
    })
    

