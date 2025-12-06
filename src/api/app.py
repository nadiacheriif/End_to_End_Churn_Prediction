from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import pandas as pd
import os
import sys

# Ensure project root is on sys.path so `src` package imports work regardless of CWD
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.data.predict.predict import predict_single, predict_batch
# ---------------------------
# Initialize App
# ---------------------------
app = FastAPI(title="Telco Customer Churn Prediction API")

# ---------------------------
# Enable CORS (frontend / Gradio / dashboards)
# ---------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------
# Request Models
# ---------------------------
class SinglePredictionRequest(BaseModel):
    data: dict

class BatchPredictionRequest(BaseModel):
    data: list

# ---------------------------
# Endpoints
# ---------------------------
@app.post("/predict/single")
def predict_single_endpoint(request: SinglePredictionRequest):
    try:
        return predict_single(request.data)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/predict/batch")
def predict_batch_endpoint(request: BatchPredictionRequest):
    try:
        df = pd.DataFrame(request.data)
        result = predict_batch(df)
        return result.to_dict(orient="records")
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/")
def read_root():
    return {
        "message": "Welcome to the Telco Customer Churn Prediction API.",
        "usage": {
            "single": "/predict/single",
            "batch": "/predict/batch",
            "health": "/health"
        }
    }

@app.get("/health")
def health_check():
    return {"status": "ok"}

# ---------------------------
# Global Error Handler
# ---------------------------
@app.exception_handler(HTTPException)
def http_exception_handler(request, exc):
    return JSONResponse(
        status_code=exc.status_code,
        content={"message": exc.detail},
    )
