# =====================================================
# 📱 APP STORE CATEGORY PREDICTOR (Streamlit + Google Drive)
# =====================================================

import streamlit as st
import pandas as pd
import joblib
import requests
from pathlib import Path

# -----------------------------------------------------
# 🔹 MODEL DOWNLOAD FUNCTION
# -----------------------------------------------------
def download_model_from_gdrive():
    file_id = "1sFiXnwDupqkWBweyu2wbjxH9YkGMf_kv"
    url = f"https://drive.google.com/uc?export=download&id={file_id}"
    output = "catboost_app_category_model.pkl"

    if not Path(output).exists():
        st.info("📥 Downloading model from Google Drive...")
        try:
            response = requests.get(url)
            response.raise_for_status()
            with open(output, "wb") as f:
                f.write(response.content)
            st.success("✅ Model downloaded successfully!")
        except Exception as e:
            st.error(f"❌ Failed to download model: {e}")

# -----------------------------------------------------
# 🧠 LOAD MODEL
# -----------------------------------------------------
try:
    if not Path("catboost_app_category_model.pkl").exists():
        download_model_from_gdrive()

    model = joblib.load("catboost_app_category_model.pkl")
    st.success("✅ Model loaded successfully!")

except Exception as e:
    st.error(f"⚠️ Model could not be loaded: {e}")
    model = None

# -----------------------------------------------------
# 🎯 APP UI
# -----------------------------------------------------
st.title("📱 App Store Category Predictor")
st.write("Predict the category of an iOS app using a trained CatBoost model.")

# User Inputs
developer_id = st.number_input("Developer ID", min_value=0)
app_size = st.number_input("App Size (MB)", min_value=0.0)
average_rating = st.slider("Average User Rating", 0.0, 5.0, 4.0)
ios_version = st.number_input("Required iOS Version", min_value=1.0)
time_gap = st.number_input("Time Gap (Days)", min_value=0)

if st.button("🔮 Predict"):
    if model is not None:
        input_df = pd.DataFrame({
            "DeveloperId": [developer_id],
            "Size_MB": [app_size],
            "Average_User_Rating": [average_rating],
            "Required_IOS_Version": [ios_version],
            "Time_Gap_Days": [time_gap]
        })

        prediction = model.predict(input_df)[0]
        st.success(f"🎯 Predicted App Category: {prediction}")
    else:
        st.warning("⚠️ Model not available. Please check your Drive link or internet connection.")
