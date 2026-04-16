import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn

# 1. Initialize FastAPI
app = FastAPI(title="MLOps Prediction Service")


model = joblib.load("models/model.pkl")
preprocessor = joblib.load("models/preprocessor.pkl")


class PredictionInput(BaseModel):
    feature1: float
    feature2: float
    category_col: str

@app.get("/")
def home():
    return {"health_check": "OK", "model_version": "1.0.0"}

@app.post("/predict")
def predict(input_data: PredictionInput):
    # Convert input to DataFrame
    df = pd.DataFrame([input_data.dict()])
    
    # Apply the modular preprocessing artifact
    processed_data = preprocessor.transform(df)
    
    # Make prediction
    prediction = model.predict(processed_data)
    
    return {"prediction": int(prediction[0])}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
