# =====================================================
# 📱 APP STORE CATEGORY PREDICTOR — FINAL STREAMLIT APP
# =====================================================

import streamlit as st
import pandas as pd
import joblib
import requests
from pathlib import Path

# -----------------------------------------------------
# 🔹 MODEL DOWNLOAD FUNCTION (GitHub Release)
# -----------------------------------------------------
def download_model_from_github():
    url = "https://github.com/Abdulqadir05/app-store-category-predictor/releases/download/v1.0/catboost_app_category_model.pkl"
    output = "catboost_app_category_model.pkl"

    if Path(output).exists():
        return

    st.info("📥 Downloading model from GitHub release... (please wait ~30s)")
    try:
        with requests.get(url, stream=True, timeout=120) as r:
            r.raise_for_status()
            with open(output, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
        st.success("✅ Model downloaded successfully!")
    except Exception as e:
        st.error(f"❌ Failed to download model: {e}")

# -----------------------------------------------------
# 🧠 LOAD MODEL
# -----------------------------------------------------
try:
    if not Path("catboost_app_category_model.pkl").exists():
        download_model_from_github()

    model = joblib.load("catboost_app_category_model.pkl")
    st.success("✅ Model loaded successfully!")

except Exception as e:
    st.error(f"⚠️ Model could not be loaded: {e}")
    model = None

# -----------------------------------------------------
# 🎯 STREAMLIT APP UI
# -----------------------------------------------------
st.title("📱 App Store Category Predictor")
st.markdown("""
This app predicts the **Category of an iOS App** based on its metadata using a trained **CatBoost Classifier**.  
Enter the following details to get the predicted app category ⤵️
""")

# Input fields
developer_id = st.number_input("👨‍💻 Developer ID", min_value=0, step=1)
app_size = st.number_input("💾 App Size (MB)", min_value=0.0, step=0.1)
average_rating = st.slider("⭐ Average User Rating", 0.0, 5.0, 4.0)
ios_version = st.number_input("📱 Required iOS Version", min_value=1.0, step=0.1)
time_gap = st.number_input("⏱️ Time Gap (Days)", min_value=0, step=1)

# -----------------------------------------------------
# 🧩 MATCH TRAINING FEATURE ORDER (with defaults)
# -----------------------------------------------------
TRAINING_FEATURES = [
    "DeveloperId", "Size_MB", "Average_User_Rating", "Required_IOS_Version",
    "Time_Gap_Days", "Content_Rating", "Release_Year",
    "Updated_Year", "Updated_Month", "Release_Month"
]

# Default values for non-user features
default_values = {
    "Content_Rating": "Everyone",
    "Release_Year": 2023,
    "Updated_Year": 2024,
    "Updated_Month": 5,
    "Release_Month": 8
}

# -----------------------------------------------------
# 🔮 PREDICTION SECTION
# -----------------------------------------------------
if st.button("🔮 Predict Category"):
    if model is not None:
        # Prepare input DataFrame
        input_data = {
            "DeveloperId": developer_id,
            "Size_MB": app_size,
            "Average_User_Rating": average_rating,
            "Required_IOS_Version": ios_version,
            "Time_Gap_Days": time_gap,
            **default_values
        }
        input_df = pd.DataFrame([input_data])[TRAINING_FEATURES]

        st.write("🧩 **Input to model:**")
        st.dataframe(input_df)

        try:
            with st.spinner("🤖 Predicting category..."):
                prediction = model.predict(input_df)[0]

            st.success(f"🎯 **Predicted App Category:** {prediction}")
            st.balloons()

        except Exception as e:
            st.error(f"⚠️ Prediction failed: {e}")
    else:
        st.warning("⚠️ Model not available. Please check your internet connection or model link.")

# -----------------------------------------------------
# 🧾 FOOTER
# -----------------------------------------------------
st.markdown("---")
st.caption("🚀 Powered by CatBoost & Streamlit | Deployed by Abdul Qadir")
