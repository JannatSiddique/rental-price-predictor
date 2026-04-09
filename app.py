import os
import sys
import pandas as pd
import joblib
import streamlit as st

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(CURRENT_DIR)

from src.preprocessing import preprocess

MODEL_PATH = os.path.join(CURRENT_DIR, "..", "models", "rent_predictor.pkl")

bundle = joblib.load(MODEL_PATH)

model = bundle["model"]
FEATURES = bundle["features"]
scaler = bundle["scaler"]
numeric_cols = bundle["numeric_cols"]

st.title("Rental Price Predictor")

bedrooms = st.number_input("Bedrooms", 0, 10, 3)
washrooms = st.number_input("Washrooms", 0, 10, 2)
marla = st.number_input("Marla", 1.0, 50.0, 5.0)

location_features = [
    col for col in FEATURES if col.startswith("Location_")
]

locations = [col.replace("Location_", "") for col in location_features]

location = st.selectbox("Location", locations)

currency = "PKR"
if st.button("Predict Rent"):

    input_df = pd.DataFrame([{
        "Bedrooms": bedrooms,
        "Washrooms": washrooms,
        "Marla": marla,
        "Location": location,
        "Currency": currency,
        "Details": "" 
    }])

    processed, _ = preprocess(input_df)

    processed[numeric_cols] = scaler.transform(processed[numeric_cols])

    processed = processed.reindex(columns=FEATURES, fill_value=0)

    prediction = model.predict(processed)[0]

    st.success(f"💰 Estimated Rent: PKR {prediction:,.0f}")
