import os
import pandas as pd
import joblib
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.normpath(os.path.join(BASE_DIR, "models", "rent_predictor.pkl"))

bundle = joblib.load(MODEL_PATH)
model = bundle["model"]
FEATURES = bundle["features"]
scaler = bundle["scaler"]
numeric_cols = bundle["numeric_cols"]

location_features = [col for col in FEATURES if col.startswith("Location_")]
locations = sorted([col.replace("Location_", "") for col in location_features])

app = FastAPI(title="Rental Price Predictor API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class PredictRequest(BaseModel):
    bedrooms: int
    washrooms: int
    marla: float
    location: str


class PredictResponse(BaseModel):
    predicted_price_pkr: float
    formatted: str
    bedrooms: int
    washrooms: int
    marla: float
    location: str


@app.get("/api/locations")
def get_locations():
    return {"locations": locations}


@app.post("/api/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    input_data = {col: 0 for col in FEATURES}
    input_data["Bedrooms"] = req.bedrooms
    input_data["Washrooms"] = req.washrooms
    input_data["Marla"] = req.marla

    loc_col = f"Location_{req.location}"
    if loc_col in input_data:
        input_data[loc_col] = 1

    input_df = pd.DataFrame([input_data])
    input_df[numeric_cols] = scaler.transform(input_df[numeric_cols])

    prediction = model.predict(input_df)[0]

    if prediction >= 100_000:
        formatted = f"{prediction / 100_000:.2f} Lakh"
    else:
        formatted = f"{prediction / 1_000:.1f} Thousand"

    return PredictResponse(
        predicted_price_pkr=round(prediction),
        formatted=formatted,
        bedrooms=req.bedrooms,
        washrooms=req.washrooms,
        marla=req.marla,
        location=req.location,
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
