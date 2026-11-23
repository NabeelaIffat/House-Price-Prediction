from pathlib import Path
import json
import joblib
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "house_price_model.joblib"
METRICS_PATH = ROOT / "metrics.json"

st.set_page_config(page_title="House Price Predictor", page_icon="🏠", layout="centered")
st.title("🏠 House Price Prediction")
st.write("Estimate a house sale price using regression algorithms trained on the Kaggle Ames Housing dataset.")

if not MODEL_PATH.exists():
    st.error("The trained model is missing. Run `python train_model.py` in the VS Code terminal first.")
    st.stop()

model = joblib.load(MODEL_PATH)

with st.form("prediction_form"):
    st.subheader("Enter house details")
    col1, col2 = st.columns(2)
    with col1:
        area = st.number_input("Living area (square feet)", min_value=300, max_value=10000, value=1500, step=50)
        quality = st.slider("Overall quality (1–10)", min_value=1, max_value=10, value=6)
        garage = st.number_input("Garage capacity (cars)", min_value=0, max_value=5, value=2)
        basement = st.number_input("Basement area (square feet)", min_value=0, max_value=5000, value=800, step=50)
    with col2:
        bathrooms = st.number_input("Full bathrooms", min_value=0, max_value=5, value=2)
        year = st.number_input("Year built", min_value=1870, max_value=2026, value=2000)
        bedrooms = st.number_input("Bedrooms above ground", min_value=0, max_value=10, value=3)
    submitted = st.form_submit_button("Predict house price")

if submitted:
    sample = pd.DataFrame([{
        "GrLivArea": area,
        "OverallQual": quality,
        "GarageCars": garage,
        "TotalBsmtSF": basement,
        "FullBath": bathrooms,
        "YearBuilt": year,
        "BedroomAbvGr": bedrooms
    }])
    prediction = float(model.predict(sample)[0])
    st.success(f"Estimated sale price: ${prediction:,.0f}")
    st.caption("This is an estimate based on historical training data, not a formal property valuation.")

if METRICS_PATH.exists():
    with open(METRICS_PATH, encoding="utf-8") as f:
        metrics = json.load(f)
    st.divider()
    st.subheader("Model comparison")
    st.write(f"Best model selected using lowest validation RMSE: **{metrics['best_model']}**")
    st.dataframe(pd.DataFrame(metrics["results"]).T, use_container_width=True)
