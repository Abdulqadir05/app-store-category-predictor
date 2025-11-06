import streamlit as st
import pandas as pd
import joblib
import requests
from pathlib import Path

# -----------------------------------------------------
# 🔗 Model & Encoder URLs (from GitHub Release)
# -----------------------------------------------------
MODEL_URL = "https://github.com/Abdulqadir05/app-store-category-predictor/releases/download/v1.0/catboost_app_category_model.pkl"
ENC_URL   = "https://github.com/Abdulqadir05/app-store-category-predictor/releases/download/v1.0/ios_version_labelencoder.pkl"

MODEL_PATH = "catboost_app_category_model.pkl"
ENC_PATH   = "ios_version_labelencoder.pkl"

# -----------------------------------------------------
# 📦 Function to Download Files from GitHub
# -----------------------------------------------------
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
    st.success(f"✅ {label} downloaded successfully!")

# -----------------------------------------------------
# 🧠 Load Model and Encoder
# -----------------------------------------------------
try:
    if not Path(MODEL_PATH).exists():
        download_file(MODEL_URL, MODEL_PATH, "CatBoost Model")
    if not Path(ENC_PATH).exists():
        download_file(ENC_URL, ENC_PATH, "iOS Version Encoder")

    model = joblib.load(MODEL_PATH)
    le_ios = joblib.load(ENC_PATH)
    st.success("✅ Model & Encoder loaded successfully!")

except Exception as e:
    st.error(f"⚠️ Model or Encoder load failed: {e}")
    model, le_ios = None, None

# -----------------------------------------------------
# 🧩 Streamlit App UI
# -----------------------------------------------------
st.title("📱 App Store Category Predictor")
st.markdown("Predict the category of an iOS app using a trained **CatBoost model**. Enter app details below 👇")

# ---- User Inputs ----
developer_id  = st.number_input("👨‍💻 Developer ID", min_value=0, step=1, value=500000000)
app_size_mb   = st.number_input("💾 App Size (MB)", min_value=0.0, step=0.1, value=150.0)
avg_rating    = st.slider("⭐ Average User Rating", 0.0, 5.0, 4.5, 0.1)
ios_version_f = st.text_input("📱 Required iOS Version (e.g., 13.0 or 4+)", value="13.0")
time_gap_days = st.number_input("⏱️ Time Gap (Days)", min_value=0, step=1, value=120)

# ---- Fixed Default Columns (used during training) ----
TRAINING_FEATURES = [
    "DeveloperId", "Size_MB", "Average_User_Rating", "Required_IOS_Version",
    "Time_Gap_Days", "Content_Rating", "Release_Year",
    "Updated_Year", "Updated_Month", "Release_Month"
]

DEFAULTS = {
    "Content_Rating": "4+",   # based on training defaults
    "Release_Year": 2023,
    "Updated_Year": 2024,
    "Updated_Month": 5,
    "Release_Month": 8
}

# -----------------------------------------------------
# 🔮 Prediction Section
# -----------------------------------------------------
if st.button("🔮 Predict Category"):
    if (model is None) or (le_ios is None):
        st.error("⚠️ Model or encoder not loaded.")
    else:
        try:
            # Clean iOS version input (remove '+')
            ios_str = str(ios_version_f).replace("+", "").strip()

            # Encode using saved LabelEncoder
            try:
                ios_encoded = int(le_ios.transform([ios_str])[0])
            except Exception:
                # Fallback: unseen iOS versions handled safely
                classes = list(le_ios.classes_)
                if ios_str in classes:
                    ios_encoded = int(le_ios.transform([ios_str])[0])
                else:
                    ios_encoded = int(le_ios.transform([max(classes)])[0])

            # Prepare feature DataFrame (matching training order)
            row = {
                "DeveloperId": developer_id,
                "Size_MB": app_size_mb,
                "Average_User_Rating": avg_rating,
                "Required_IOS_Version": ios_encoded,
                "Time_Gap_Days": time_gap_days,
                **DEFAULTS
            }

            X = pd.DataFrame([row])[TRAINING_FEATURES]

            st.caption("🧩 Final features sent to model (training schema):")
            st.dataframe(X)

            # Predict category
            y_pred = model.predict(X)[0]
            st.success(f"🎯 Predicted App Category: {y_pred}")

        except Exception as e:
            st.error(f"⚠️ Prediction failed: {e}")

# -----------------------------------------------------
# 🧾 Footer
# -----------------------------------------------------
st.markdown("---")
st.caption("🚀 Powered by CatBoost & Streamlit | Developed by **Abdul Qadir (IIT Jodhpur)**")

