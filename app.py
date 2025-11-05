# =====================================================
# 📱 APP STORE CATEGORY PREDICTOR (Streamlit + GitHub)
# =====================================================

import streamlit as st
import pandas as pd
import joblib
import requests
from pathlib import Path
import time

# -----------------------------------------------------
# 🔹 MODEL DOWNLOAD FUNCTION (from GitHub Releases)
# -----------------------------------------------------
def download_model_from_github():
    url = "https://github.com/Abdulqadir05/app-store-category-predictor/releases/download/v1.0/catboost_app_category_model.pkl"
    output = "catboost_app_category_model.pkl"

    if Path(output).exists():
        return

    with st.spinner("📥 Downloading model from GitHub release... (please wait 30–40s)"):
        try:
            with requests.get(url, stream=True, timeout=180) as r:
                r.raise_for_status()
                with open(output, "wb") as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
            st.success("✅ Model downloaded successfully!")
            st.balloons()
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
# 🎯 APP UI
# -----------------------------------------------------
st.title("📱 App Store Category Predictor")
st.caption("Made with ❤️ by Abdul Qadir | IIT Jodhpur")
st.write("Predict the **category of an iOS app** using a trained CatBoost model.")

st.markdown("---")

# 🧾 User Inputs
developer_id = st.number_input("🧑‍💻 Developer ID", min_value=0)
app_size = st.number_input("💾 App Size (MB)", min_value=0.0)
average_rating = st.slider("⭐ Average User Rating", 0.0, 5.0, 4.0)
ios_version = st.number_input("📱 Required iOS Version", min_value=1.0)
time_gap = st.number_input("⏱️ Time Gap (Days)", min_value=0)

# -----------------------------------------------------
# 🧩 Helper: Ensure same feature order as training
# -----------------------------------------------------
EXPECTED_FEATURES = [
    "DeveloperId",
    "Size_MB",
    "Average_User_Rating",
    "Required_IOS_Version",
    "Time_Gap_Days"
]


if st.button("🔮 Predict Category"):
    if model is not None:
        input_df = pd.DataFrame({
            "DeveloperId": [developer_id],
            "Size_MB": [app_size],
            "Average_User_Rating": [average_rating],
            "Required_IOS_Version": [ios_version],
            "Time_Gap_Days": [time_gap]
        })

        # Reorder columns to match model training schema
        input_df = input_df[EXPECTED_FEATURES]

        with st.spinner("🤖 Predicting category..."):
            prediction = model.predict(input_df)
        st.success(f"🎯 **Predicted App Category:** {prediction[0]}")
        st.balloons()
    else:
        st.warning("⚠️ Model not available. Please check your GitHub release link or internet connection.")


st.markdown("---")
st.caption("🚀 Powered by CatBoost & Streamlit | Deployed by Abdul Qadir")
