"""
Auto-generated FastAPI server for model Species__Iris-setosa_LinearRegression
Run: uvicorn app:app --host 0.0.0.0 --port 8080
"""
import joblib
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Any, Dict, List

app = FastAPI(title="Model API - Species__Iris-setosa_LinearRegression")

# Load model (adjust path if needed)
model = joblib.load("Species__Iris-setosa_LinearRegression.joblib")

class PredictPayload(BaseModel):
    inputs: List[Dict[str, Any]]

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict(payload: PredictPayload):
    try:
        # expect inputs: list of feature dicts
        import pandas as pd
        X = pd.DataFrame(payload.inputs)
        preds = model.predict(X)
        return {"predictions": preds.tolist()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)
