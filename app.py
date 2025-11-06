import streamlit as st
import pandas as pd
import joblib
import requests
from pathlib import Path

MODEL_URL = "https://github.com/Abdulqadir05/app-store-category-predictor/releases/download/v1.0/catboost_app_category_model.pkl"
ENC_URL   = "https://github.com/Abdulqadir05/app-store-category-predictor/releases/download/v1.0/ios_version_labelencoder.pkl"

MODEL_PATH = "catboost_app_category_model.pkl"
ENC_PATH   = "ios_version_labelencoder.pkl"

def download_file(url, out_path, label):
    if Path(out_path).exists():
        return
    with st.spinner(f"📥 Downloading {label}..."):
        r = requests.get(url, stream=True, timeout=180)
        r.raise_for_status()
        with open(out_path, "wb") as f:
            for chunk in r.iter_content(8192):
                if chunk:
                    f.write(chunk)
    st.success(f"✅ {label} downloaded!")

# ---------- load artifacts ----------
try:
    if not Path(MODEL_PATH).exists():
        download_file(MODEL_URL, MODEL_PATH, "model")
    if not Path(ENC_PATH).exists():
        download_file(ENC_URL, ENC_PATH, "iOS version encoder")

    model = joblib.load(MODEL_PATH)
    le_ios = joblib.load(ENC_PATH)
    st.success("✅ Model & encoder loaded!")
except Exception as e:
    st.error(f"⚠️ Artifacts load failed: {e}")
    model, le_ios = None, None

st.title("📱 App Store Category Predictor")
st.write("Enter raw/human values; app will encode iOS version correctly.")

# ---- Inputs (human-friendly) ----
developer_id  = st.number_input("👨‍💻 Developer ID", min_value=0, step=1, value=500000000)
app_size_mb   = st.number_input("💾 App Size (MB)", min_value=0.0, step=0.1, value=150.0)
avg_rating    = st.slider("⭐ Average User Rating", 0.0, 5.0, 4.5, 0.1)
ios_version_f = st.number_input("📱 Required iOS Version (e.g., 13.0)", min_value=1.0, step=0.1, value=13.0)
time_gap_days = st.number_input("⏱️ Time Gap (Days)", min_value=0, step=1, value=120)

# ---- Training feature order (your CatBoost expects this) ----
TRAINING_FEATURES = [
    "DeveloperId", "Size_MB", "Average_User_Rating", "Required_IOS_Version",
    "Time_Gap_Days", "Content_Rating", "Release_Year",
    "Updated_Year", "Updated_Month", "Release_Month"
]

DEFAULTS = {
    "Content_Rating": "4+",   # ya jo training me tha
    "Release_Year": 2023,
    "Updated_Year": 2024,
    "Updated_Month": 5,
    "Release_Month": 8
}

if st.button("🔮 Predict Category"):
    if (model is None) or (le_ios is None):
        st.error("Artifacts not loaded.")
    else:
        # 1) iOS version ko string banayein EXACTLY waise jaise training me tha
        ios_str = f"{ios_version_f:.1f}"  # "13.0" format

        # 2) Encoder se transform karein (agar value unseen ho to nearest fallback)
        try:
            ios_encoded = int(le_ios.transform([ios_str])[0])
        except Exception:
            # fallback: agar unseen ho, to sabse kareeb class choose
            classes = list(le_ios.classes_)
            # very simple nearest fallback by string presence
            if ios_str in classes:
                ios_encoded = int(le_ios.transform([ios_str])[0])
            else:
                # fallback to most common / median class
                ios_encoded = int(le_ios.transform([max(classes)])[0])

        # 3) Input DF banayein EXACT training order me
        row = {
            "DeveloperId": developer_id,
            "Size_MB": app_size_mb,
            "Average_User_Rating": avg_rating,
            "Required_IOS_Version": ios_encoded,  # ENCODED VALUE PASSED
            "Time_Gap_Days": time_gap_days,
            **DEFAULTS
        }
        X = pd.DataFrame([row])[TRAINING_FEATURES]

        st.caption("🧩 Final features sent to model (matching training schema):")
        st.dataframe(X)

        # 4) Predict
        try:
            y_pred = model.predict(X)[0]
            st.success(f"🎯 Predicted App Category: {y_pred}")
        except Exception as e:
            st.error(f"⚠️ Prediction failed: {e}")

# -----------------------------------------------------
# 🧾 FOOTER
# -----------------------------------------------------
st.markdown("---")
st.caption("🚀 Powered by CatBoost & Streamlit | Deployed by Abdul Qadir")
