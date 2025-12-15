import streamlit as st
import numpy as np
import pandas as pd
import pickle
import tensorflow as tf
from pathlib import Path

# -----------------------------
# Paths (same folder as app.py)
# -----------------------------
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
# Figure out expected columns
# -----------------------------
def get_expected_feature_names(scaler_obj):
    # If scaler was fit on a DataFrame, sklearn stores feature_names_in_
    names = list(getattr(scaler_obj, "feature_names_in_", []))
    if names:
        return names

    # Fallback if feature_names_in_ is missing
    return [
        "Relative_Compactness",
        "Surface_Area",
        "Wall_Area",
        "Roof_Area",
        "Overall_Height",
        "Glazing_Area",
        "Orientation",
        "Glazing_Area_Distribution",
    ]

EXPECTED_FEATURES = get_expected_feature_names(scaler)

# -----------------------------
# UI
# -----------------------------
st.title("Energy Efficiency – Violeta's Model")
st.write("Enter building parameters to predict **Heating Load** and **Cooling Load**.")

# Optional debug (toggle)
with st.expander("Debug info (optional)"):
    st.write("App folder:", str(BASE_DIR))
    st.write("Model file exists:", MODEL_PATH.exists())
    st.write("Scaler file exists:", SCALER_PATH.exists())
    st.write("Scaler expects n features:", getattr(scaler, "n_features_in_", "unknown"))
    st.write("Scaler expected feature names:", EXPECTED_FEATURES)

with st.form("inputs"):
    st.subheader("Input features")

    relative_compactness = st.slider("Relative Compactness", 0.62, 0.98, 0.80)
    surface_area = st.slider("Surface Area", 500.0, 900.0, 650.0)
    wall_area = st.slider("Wall Area", 200.0, 420.0, 300.0)
    roof_area = st.slider("Roof Area", 150.0, 350.0, 250.0)
    overall_height = st.selectbox("Overall Height", [3.5, 7.0])
    glazing_area = st.selectbox("Glazing Area", [0.0, 0.1, 0.25, 0.4])
    orientation = st.selectbox("Orientation", [2, 3, 4, 5])
    glazing_area_distribution = st.selectbox("Glazing Area Distribution", [0, 1, 2, 3, 4, 5])

    submitted = st.form_submit_button("Predict")

# -----------------------------
# Prediction
# -----------------------------
if submitted:
    # Create a mapping from "canonical" names -> values
    values_by_canonical = {
        "Relative_Compactness": relative_compactness,
        "Surface_Area": surface_area,
        "Wall_Area": wall_area,
        "Roof_Area": roof_area,
        "Overall_Height": overall_height,
        "Glazing_Area": glazing_area,
        "Orientation": orientation,
        "Glazing_Area_Distribution": glazing_area_distribution,
    }

    # Build one row in the exact feature order the scaler expects.
    # If the scaler uses space-names (e.g. "Relative Compactness"), map them.
    def canonicalize(name: str) -> str:
        # Converts "Relative Compactness" -> "Relative_Compactness"
        return name.strip().replace(" ", "_")

    try:
        row = []
        for col in EXPECTED_FEATURES:
            canon = canonicalize(col)
            if canon not in values_by_canonical:
                raise ValueError(
                    f"Scaler expects feature '{col}' (canonical '{canon}'), but the app doesn't provide it."
                )
            row.append(values_by_canonical[canon])

        input_df = pd.DataFrame([row], columns=EXPECTED_FEATURES)

        # Transform and predict
        X_scaled = scaler.transform(input_df)
        preds = model.predict(X_scaled)

        heating = float(preds[0, 0])
        cooling = float(preds[0, 1])

        st.subheader("Predicted loads")
        st.metric("Heating Load", f"{heating:.2f}")
        st.metric("Cooling Load", f"{cooling:.2f}")

    except Exception as e:
        st.error("Prediction failed due to a preprocessing mismatch.")
        st.exception(e)
        st.info(
            "Most common cause: the scaler was fit on different feature names or extra preprocessing (e.g., one-hot encoding). "
            "Check the 'Debug info' expander above to see what the scaler expects."
        )
