
# =============================================================
# 📱 APP STORE CATEGORY PREDICTOR — Professional Streamlit App
# =============================================================

import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# =============================================================
# 🎯 Page Configuration
# =============================================================
st.set_page_config(
    page_title="App Store Category Predictor",
    page_icon="📱",
    layout="centered",
    initial_sidebar_state="expanded"
)

# =============================================================
# 🎨 Custom CSS Styling
# =============================================================
st.markdown("""
    <style>
        body { background-color: #f5f7fa; }
        .main-title {
            font-size: 36px !important;
            color: #2b8a3e;
            text-align: center;
            font-weight: bold;
        }
        .sub-title {
            font-size: 18px !important;
            color: #495057;
            text-align: center;
        }
        .stButton>button {
            background-color: #2b8a3e;
            color: white;
            font-weight: bold;
            border-radius: 12px;
            padding: 0.6em 1.2em;
        }
        .stButton>button:hover {
            background-color: #218838;
            color: #f8f9fa;
        }
        .prediction-box {
            background-color: #e9f5ee;
            padding: 20px;
            border-radius: 10px;
            border-left: 6px solid #2b8a3e;
        }
    </style>
""", unsafe_allow_html=True)

# =============================================================
# 🧠 Load Model + Label Encoder with Cache
# =============================================================
@st.cache_resource
def load_model():
    import requests
    from pathlib import Path

    model_path = Path("catboost_app_category_model.pkl")
    encoder_path = Path("category_label_encoder.pkl")

    # 🔗 Direct GitHub release URLs
    model_url = "https://github.com/Abdulqadir05/app-store-category-predictor/releases/download/v1.0/catboost_app_category_model.pkl"
    encoder_url = "https://github.com/Abdulqadir05/app-store-category-predictor/releases/download/v1.0/category_label_encoder.pkl"

    # 📥 Download model if not exists
    if not model_path.exists():
        st.info("📦 Downloading model from GitHub release (please wait 20–30 seconds)...")
        with requests.get(model_url, stream=True, timeout=180) as r:
            r.raise_for_status()
            with open(model_path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)

    # 📥 Download encoder if not exists
    if not encoder_path.exists():
        st.info("📦 Downloading label encoder from GitHub release...")
        with requests.get(encoder_url, stream=True, timeout=180) as r:
            r.raise_for_status()
            with open(encoder_path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)

    try:
        model = joblib.load(model_path)
        le_target = joblib.load(encoder_path)
        st.success("✅ Model and encoder loaded successfully!")
        return model, le_target
    except Exception as e:
        st.error(f"⚠️ Failed to load model or encoder: {e}")
        return None, None


# =============================================================
# 🏷️ App Header
# =============================================================
st.markdown('<p class="main-title">📱 App Store Category Predictor</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">An AI-powered model to predict the iOS app category using CatBoost</p>', unsafe_allow_html=True)
st.divider()

# =============================================================
# 🧩 Input Fields (UI Layout)
# =============================================================
st.header("🔧 Input Parameters")

col1, col2 = st.columns(2)

with col1:
    developer_id = st.number_input("👨‍💻 Developer ID", min_value=100000000, max_value=2000000000, value=500000000, step=1, key="dev_id")
    app_size_mb = st.number_input("💾 App Size (MB)", min_value=1.0, max_value=5000.0, value=150.0, step=0.1, key="size_mb")
    avg_rating = st.slider("⭐ Average User Rating", 0.0, 5.0, 4.5, 0.1, key="rating")

with col2:
    ios_version = st.number_input("📱 Required iOS Version", min_value=1.0, max_value=20.0, value=13.0, step=0.1, key="ios_version")
    time_gap = st.number_input("⏱️ Time Gap (Days)", min_value=0, max_value=1000, value=120, step=1, key="time_gap")

st.markdown("🧩 *Default values for Content Rating and Release Info are used based on historical app data.*")

# =============================================================
# 🎯 Prediction Logic
# =============================================================
if st.button("🔮 Predict App Category", use_container_width=True, key="predict_btn"):
    if model is None or le_target is None:
        st.error("⚠️ Model or encoder not loaded. Please verify your setup.")
    else:
        try:
            # Input as DataFrame (matching training schema)
            input_df = pd.DataFrame([{
                "DeveloperId": developer_id,
                "Size_MB": app_size_mb,
                "Average_User_Rating": avg_rating,
                "Required_IOS_Version": str(ios_version),  # keep as string if categorical
                "Time_Gap_Days": time_gap,
                "Content_Rating": "4+",
                "Release_Year": 2023,
                "Updated_Year": 2024,
                "Updated_Month": 6,
                "Release_Month": 8
            }])

            # Prediction
            pred_num = model.predict(input_df)[0]
            pred_label = le_target.inverse_transform([int(pred_num)])[0]

            # Result Box
            st.markdown('<div class="prediction-box">', unsafe_allow_html=True)
            st.subheader("🎯 Prediction Result:")
            st.write(f"**Predicted App Category:** `{pred_label}`")
            st.markdown("</div>", unsafe_allow_html=True)

        except Exception as e:
            st.error(f"⚠️ Prediction failed: {e}")

# =============================================================
# 🧾 Footer
# =============================================================
st.divider()
st.markdown("""
**👨‍💻 Developed by:** Abdul Qadir  
🎓 *BS in Applied AI & Data Science, IIT Jodhpur*  
💼 *Aspiring Data Scientist | Machine Learning Engineer*  
📧 **b24bs1012@iitj.ac.in**  
🌍 *Powered by CatBoost + Streamlit*
""")

