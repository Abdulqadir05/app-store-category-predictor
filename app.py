# ============================================================
# 📱 APP STORE CATEGORY PREDICTOR — Streamlit + CatBoost (v1.0)
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import requests
from pathlib import Path

# ============================================================
# ⚙️ PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="App Store Category Predictor",
    page_icon="📱",
    layout="centered"
)

st.markdown(
    """
    <style>
        body {background-color: #0e1117; color: #fafafa;}
        .stButton>button {
            background: linear-gradient(90deg, #00c853, #b2ff59);
            color: black;
            border-radius: 8px;
            font-weight: 600;
            padding: 0.6em 1.2em;
        }
        .stButton>button:hover {
            background: linear-gradient(90deg, #69f0ae, #76ff03);
            color: black;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# ============================================================
# 🧠 LOAD MODEL + ENCODER (FROM GITHUB RELEASE)
# ============================================================
@st.cache_resource
def load_model():
    """Download and load model + encoder from GitHub release."""
    model_path = Path("catboost_app_category_model.pkl")
    encoder_path = Path("category_label_encoder.pkl")

    model_url = "https://github.com/Abdulqadir05/app-store-category-predictor/releases/download/v1.0/catboost_app_category_model.pkl"
    encoder_url = "https://github.com/Abdulqadir05/app-store-category-predictor/releases/download/v1.0/category_label_encoder.pkl"

    def download_file(url, out_path, label):
        st.info(f"📦 Downloading {label} from GitHub (please wait 20–30s)...")
        with requests.get(url, stream=True, timeout=180) as r:
            r.raise_for_status()
            with open(out_path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
        st.success(f"✅ {label} downloaded successfully!")

    if not model_path.exists():
        download_file(model_url, model_path, "model")

    if not encoder_path.exists():
        download_file(encoder_url, encoder_path, "label encoder")

    try:
        model = joblib.load(model_path)
        le_target = joblib.load(encoder_path)
        st.success("✅ Model and encoder loaded successfully!")
        return model, le_target
    except Exception as e:
        st.error(f"⚠️ Failed to load model or encoder: {e}")
        return None, None


# ------------------------------------------------------------
# 🔹 Load model globally
# ------------------------------------------------------------
model, le_target = load_model()

# ============================================================
# 🎨 APP UI
# ============================================================
st.title("📱 App Store Category Predictor")
st.caption("Predict the most likely **App Store Category** using your trained CatBoost AI model.")

st.markdown("---")

# --- Input fields ---
developer_id = st.number_input("👨‍💻 Developer ID", min_value=0, step=1, value=500000000)
app_size_mb = st.number_input("💾 App Size (MB)", min_value=0.0, step=0.1, value=150.0)
avg_rating = st.slider("⭐ Average User Rating", 0.0, 5.0, 4.5, 0.1)
ios_version = st.number_input("📱 Required iOS Version", min_value=1.0, step=0.1, value=13.0)
time_gap = st.number_input("⏱️ Time Gap (Days)", min_value=0, step=1, value=120)

st.caption("🧩 *Default values for Content Rating and Release Info are used based on historical app data.*")

# ------------------------------------------------------------
# 🧮 PREDICTION
# ------------------------------------------------------------
if st.button("🔮 Predict App Category"):
    if model is None or le_target is None:
        st.error("⚠️ Model or encoder not available. Please reload the app.")
    else:
        # Prepare input DataFrame
        DEFAULTS = {
            "Content_Rating": "4+",
            "Release_Year": 2023,
            "Updated_Year": 2024,
            "Updated_Month": 6,
            "Release_Month": 8
        }

        row = {
            "DeveloperId": developer_id,
            "Size_MB": app_size_mb,
            "Average_User_Rating": avg_rating,
            "Required_IOS_Version": str(ios_version),  # keep as string (categorical)
            "Time_Gap_Days": time_gap,
            **DEFAULTS
        }

        features = [
            "DeveloperId", "Size_MB", "Average_User_Rating", "Required_IOS_Version",
            "Time_Gap_Days", "Content_Rating", "Release_Year",
            "Updated_Year", "Updated_Month", "Release_Month"
        ]

        X = pd.DataFrame([row])[features]

        st.write("📊 **Input Data Preview:**")
        st.dataframe(X, use_container_width=True)

        # Predict
        try:
            # Make Prediction
            y_pred_num = model.predict(X)
            y_pred_num = np.array(y_pred_num, dtype=int).flatten()
            # 🔹 Convert numeric prediction to label using LabelEncoder
            y_pred_label = le_target.inverse_transform([int(y_pred_num[0])])[0]

            st.success(f"🎯 **Predicted App Category:** {y_pred_label}")
       except Exception as e:
            st.error(f"⚠️ Prediction failed: {e}")

# ============================================================
# 🧾 FOOTER
# ============================================================
st.markdown("---")
st.caption(
    """
    💼 **Developed by Abdul Qadir**  
    🎓 *BS in Applied AI & Data Science, IIT Jodhpur*  
    🚀 *Powered by CatBoost + Streamlit*  
    🌐 [GitHub Repository](https://github.com/Abdulqadir05/app-store-category-predictor)
    """
)

