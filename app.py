import os
import pandas as pd
import joblib
import streamlit as st

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.normpath(os.path.join(BASE_DIR, "models", "rent_predictor.pkl"))

bundle = joblib.load(MODEL_PATH)
model = bundle["model"]
FEATURES = bundle["features"]
scaler = bundle["scaler"]
numeric_cols = bundle["numeric_cols"]

location_features = [col for col in FEATURES if col.startswith("Location_")]
locations = sorted([col.replace("Location_", "") for col in location_features])

# --- UI ---
st.set_page_config(page_title="Rental Price Predictor", page_icon="🏠")
st.title(" Rental Price Predictor")
st.markdown("Estimate monthly rent based on property details.")

col1, col2, col3 = st.columns(3)
with col1:
    bedrooms = st.number_input("Bedrooms", 0, 10, 2)
with col2:
    washrooms = st.number_input("Washrooms", 0, 10, 2)
with col3:
    marla = st.number_input("Marla", 1.0, 50.0, 5.0)

location = st.selectbox("Location", locations)

if st.button("Predict Rent", key="predict_btn", use_container_width=True):

    input_data = {col: 0 for col in FEATURES}
    input_data["Bedrooms"] = bedrooms
    input_data["Washrooms"] = washrooms
    input_data["Marla"] = marla

    loc_col = f"Location_{location}"
    if loc_col in input_data:
        input_data[loc_col] = 1
    else:
        st.warning(f"Location '{location}' not found in model. Predicting without location.")

    input_df = pd.DataFrame([input_data])
    input_df[numeric_cols] = scaler.transform(input_df[numeric_cols])

    prediction = model.predict(input_df)[0]

    if prediction >= 100_000:
        formatted = f"{prediction / 100_000:.2f} Lakh"
    else:
        formatted = f"{prediction / 1_000:.1f} Thousand"

    st.success(f"💰 Estimated Rent: PKR {prediction:,.0f} ({formatted})")
    st.caption(f"Location: {location} | {bedrooms} bed | {washrooms} bath | {marla} Marla")