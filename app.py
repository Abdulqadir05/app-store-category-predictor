
# 📱 APP STORE CATEGORY PREDICTOR — Streamlit Deployment
# =====================================================

import streamlit as st
import pandas as pd
import joblib
import requests
from pathlib import Path

# -----------------------------------------------------
# 🔹 GitHub Release URLs (MODEL + SCHEMA)
# -----------------------------------------------------
MODEL_URL = "https://github.com/Abdulqadir05/app-store-category-predictor/releases/download/v1.0/catboost_app_category_model.pkl"
SCHEMA_URL = "https://github.com/Abdulqadir05/app-store-category-predictor/releases/download/v1.0/feature_schema.pkl"

MODEL_PATH = "catboost_app_category_model.pkl"
SCHEMA_PATH = "feature_schema.pkl"

# -----------------------------------------------------
# 🔹 Utility: Download file if missing
# -----------------------------------------------------
def download_file(url, out_path, label):
    """Downloads large files from GitHub release"""
    if Path(out_path).exists():
        return
    with st.spinner(f"📥 Downloading {label}... please wait"):
        try:
            r = requests.get(url, stream=True, timeout=180)
            r.raise_for_status()
            with open(out_path, "wb") as f:
                for chunk in r.iter_content(8192):
                    if chunk:
                        f.write(chunk)
            st.success(f"✅ {label} downloaded successfully!")
        except Exception as e:
            st.error(f"❌ Failed to download {label}: {e}")

# -----------------------------------------------------
# 🔹 Load model & schema
# -----------------------------------------------------
try:
    if not Path(MODEL_PATH).exists():
        download_file(MODEL_URL, MODEL_PATH, "Model")
    if not Path(SCHEMA_PATH).exists():
        download_file(SCHEMA_URL, SCHEMA_PATH, "Schema")

    model = joblib.load(MODEL_PATH)
    schema = joblib.load(SCHEMA_PATH)
    st.success("✅ Model and schema loaded successfully!")

except Exception as e:
    st.error(f"⚠️ Model loading failed: {e}")
    model, schema = None, None

# -----------------------------------------------------
# 🔹 Category mapping (for human-readable output)
# -----------------------------------------------------
category_map = {
    0: 'Book', 1: 'Business', 2: 'Catalogs', 3: 'Education', 4: 'Entertainment',
    5: 'Finance', 6: 'Food & Drink', 7: 'Games', 8: 'Health & Fitness', 9: 'Lifestyle',
    10: 'Medical', 11: 'Music', 12: 'Navigation', 13: 'News', 14: 'Photo & Video',
    15: 'Productivity', 16: 'Reference', 17: 'Shopping', 18: 'Social Networking',
    19: 'Sports', 20: 'Travel', 21: 'Utilities', 22: 'Weather', 23: 'Kids',
    24: 'Graphics & Design', 25: 'AR & VR'
}

# -----------------------------------------------------
# 🎨 Streamlit UI
# -----------------------------------------------------
st.title("📱 App Store Category Predictor")
st.markdown("Predict the **category of an iOS app** using a trained CatBoost model.")

st.divider()

# 🔹 User Inputs
developer_id  = st.number_input("👨‍💻 Developer ID", min_value=0, step=1, value=500000000)
app_size_mb   = st.number_input("💾 App Size (MB)", min_value=0.0, step=0.1, value=120.5)
avg_rating    = st.slider("⭐ Average User Rating", 0.0, 5.0, 4.3, 0.1)
ios_version   = st.text_input("📱 Required iOS Version (e.g., 13.0)", value="13.0")
time_gap_days = st.number_input("⏱️ Time Gap (Days)", min_value=0, step=1, value=150)

st.divider()

# 🔹 Prediction button
if st.button("🔮 Predict App Category"):
    if model is None or schema is None:
        st.error("⚠️ Model not loaded. Please refresh and try again.")
    else:
        try:
            # Prepare input
            features = schema["features"]
            input_data = pd.DataFrame([{
                "DeveloperId": developer_id,
                "Size_MB": app_size_mb,
                "Average_User_Rating": avg_rating,
                "Required_IOS_Version": ios_version,
                "Time_Gap_Days": time_gap_days,
                "Content_Rating": "4+",
                "Release_Year": 2023,
                "Updated_Year": 2024,
                "Updated_Month": 6,
                "Release_Month": 8
            }])[features]

            st.caption("🧩 Final input sample sent to model:")
            st.dataframe(input_data)

            # Predict
            y_pred_num = model.predict(input_data)
            y_pred_label = category_map.get(int(y_pred_num[0]), "Unknown")

            st.success(f"🎯 **Predicted App Category:** {y_pred_label}")

        except Exception as e:
            st.error(f"⚠️ Prediction failed: {e}")

# -----------------------------------------------------
# 🧾 FOOTER
# -----------------------------------------------------
st.markdown("---")
st.caption("🚀 Built & Deployed by **Abdul Qadir** | IIT Jodhpur | Powered by CatBoost & Streamlit")
