import streamlit as st
import numpy as np
import pandas as pd
import pickle
import tensorflow as tf
from pathlib import Path

# -----------------------------
# Constants
# -----------------------------
FEATURES = [
    "Relative_Compactness",
    "Surface_Area",
    "Wall_Area",
    "Roof_Area",
    "Overall_Height",
    "Glazing_Area",
    "Orientation",
    "Glazing_Area_Distribution",
]

BASE_DIR = Path(__file__).resolve().parent
SCALER_PATH = BASE_DIR / "scaler_2.pkl"
MODEL_PATH = BASE_DIR / "best_model.keras"

# -----------------------------
# Load model + scaler (once)
# -----------------------------
@st.cache_resource
def load_model_and_scaler():
    with open(SCALER_PATH, "rb") as f:
        scaler = pickle.load(f)
    model = tf.keras.models.load_model(MODEL_PATH)
    return model, scaler

model, scaler = load_model_and_scaler()

# -----------------------------
# UI
# -----------------------------
st.title("Energy Efficiency – Violeta's Model")
st.write("Enter building parameters to predict **Heating Load** and **Cooling Load**.")

with st.form("inputs"):
    st.subheader("Input features")

    relative_compactness = st.slider("Relative Compactness", 0.62, 0.98, 0.80)
    surface_area = st.slider("Surface Area", 500.0, 900.0, 650.0)
    wall_area = st.slider("Wall Area", 200.0, 420.0, 300.0)
    roof_area = st.slider("Roof Area", 150.0, 350.0, 250.0)
    overall_height = st.selectbox("Overall Height", [3.5, 7.0])
    glazing_area = st.selectbox("Glazing Area", [0.0, 0.1, 0.25, 0.4])
    orientation = st.selectbox("Orientation", [2, 3, 4, 5])
    glazing_area_distribution = st.selectbox(
        "Glazing Area Distribution", [0, 1, 2, 3, 4, 5]
    )

    submitted = st.form_submit_button("Predict")

# -----------------------------
# Prediction
# -----------------------------
if submitted:
    input_df = pd.DataFrame(
        [[
            relative_compactness,
            surface_area,
            wall_area,
            roof_area,
            overall_height,
            glazing_area,
            orientation,
            glazing_area_distribution,
        ]],
        columns=FEATURES
    )

    X_scaled = scaler.transform(input_df)
    preds = model.predict(X_scaled)

    heating, cooling = float(preds[0, 0]), float(preds[0, 1])

    st.subheader("Predicted loads")
    st.metric("Heating Load", f"{heating:.2f}")
    st.metric("Cooling Load", f"{cooling:.2f}")
