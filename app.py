# ============================================================
# 📱 APP STORE CATEGORY PREDICTOR — Streamlit + CatBoost (v4.0 Pro)
# Auto-predict • Top-5 probabilities • Animated theme
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import requests
from pathlib import Path

# =========================
# ⚙️ PAGE CONFIG + THEME
# =========================
st.set_page_config(
    page_title="App Store Category Predictor",
    page_icon="📱",
    layout="centered"
)

st.markdown(
    """
    <style>
        :root {
            --bg0: #0e1117;
            --bg1: #121722;
            --fg:  #e9f1f7;
            --acc1:#00e676;
            --acc2:#b2ff59;
            --card:#1b2030;
        }
        .stApp { background: radial-gradient(1200px 600px at 10% -10%, #1a2030 0%, var(--bg0) 40%),
                                    radial-gradient(1200px 600px at 110% 10%, #141a26 0%, var(--bg0) 30%);}
        /* Animated gradient banner */
        .banner {
            margin: 6px 0 18px 0;
            padding: 14px 16px;
            border-radius: 14px;
            color: var(--fg);
            background: linear-gradient(120deg, rgba(0,230,118,.15), rgba(178,255,89,.10), rgba(0,230,118,.15));
            background-size: 400% 400%;
            animation: flow 12s ease infinite;
            border: 1px solid rgba(255,255,255,.06);
            box-shadow: 0 8px 28px rgba(0,0,0,.25), inset 0 0 24px rgba(0, 230, 118, .08);
        }
        @keyframes flow {
            0% { background-position: 0% 50%; }
            50% { background-position: 100% 50%; }
            100% { background-position: 0% 50%; }
        }
        .result-card {
            background: var(--card);
            border: 1px solid rgba(255,255,255,.06);
            border-radius: 14px;
            padding: 16px;
            color: var(--fg);
            box-shadow: 0 10px 40px rgba(0,0,0,.35);
        }
        .metric-badge {
            display: inline-block;
            padding: 6px 10px;
            border-radius: 999px;
            background: rgba(0,230,118,.12);
            border: 1px solid rgba(0,230,118,.35);
            color: var(--fg);
            font-weight: 600;
        }
        .subtle {
            color: rgba(233,241,247,.7);
            font-size: 0.9rem;
        }
        /* Buttons */
        .stButton>button {
            background: linear-gradient(90deg, var(--acc1), var(--acc2));
            color: #04120a;
            border-radius: 10px;
            font-weight: 700;
            padding: 0.6em 1.2em;
            border: 0;
        }
        .stButton>button:hover { filter: brightness(1.05); }
        /* Tables */
        .stDataFrame { border-radius: 12px; overflow:hidden; border:1px solid rgba(255,255,255,.06); }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown("<div class='banner'><b>App Store Category Predictor</b> · Real-time inference · Balanced CatBoost GPU · Top-5 confidences</div>", unsafe_allow_html=True)


# ===========================================
# 🗂️ CATEGORY MAP (fallback if needed)
# ===========================================
CATEGORY_MAP_FALLBACK = {
    0: "Books",
    1: "Business",
    2: "Catalogs",
    3: "Education",
    4: "Entertainment",
    5: "Finance",
    6: "Food & Drink",
    7: "Games",
    8: "Health & Fitness",
    9: "Lifestyle",
    10: "Magazines & Newspapers",
    11: "Medical",
    12: "Music",
    13: "Navigation",
    14: "News",
    15: "Photo & Video",
    16: "Productivity",
    17: "Reference",
    18: "Shopping",
    19: "Social Networking",
    20: "Sports",
    21: "Travel",
    22: "Utilities",
    23: "Weather",
    24: "Graphics & Design",
    25: "Developer Tools"
}


# ===========================================
# 🧠 LOAD MODEL + ENCODER FROM GITHUB RELEASE
# ===========================================
@st.cache_resource
def load_artifacts():
    """
    Downloads model + label encoder from GitHub Releases (v1.1: balanced model),
    then loads them with joblib. Cached for app session.
    """
    model_path = Path("catboost_app_category_model.pkl")
    encoder_path = Path("category_label_encoder.pkl")

    # 🔗 update these if you create another tag later (v1.2, etc.)
    model_url = "https://github.com/Abdulqadir05/app-store-category-predictor/releases/download/v1.1/catboost_app_category_model.pkl"
    encoder_url = "https://github.com/Abdulqadir05/app-store-category-predictor/releases/download/v1.1/category_label_encoder.pkl"

    def download(url: str, out_path: Path, label: str):
        if out_path.exists():  # avoid re-download
            return
        with requests.get(url, stream=True, timeout=180) as r:
            r.raise_for_status()
            with open(out_path, "wb") as f:
                for chunk in r.iter_content(8192):
                    if chunk:
                        f.write(chunk)

    try:
        download(model_url, model_path, "model")
        download(encoder_url, encoder_path, "label encoder")
        model = joblib.load(model_path)
        le_target = joblib.load(encoder_path)
        return model, le_target
    except Exception as e:
        st.error(f"⚠️ Failed to load artifacts: {e}")
        return None, None

model, le_target = load_artifacts()


# ===========================================
# 🧩 INPUTS (Auto-Predict — no button)
# ===========================================
st.write("### Provide App Attributes")

c1, c2 = st.columns(2)
with c1:
    developer_id = st.number_input("👨‍💻 Developer ID", min_value=0, step=1, value=500_000_000)
    avg_rating   = st.slider("⭐ Average User Rating", 0.0, 5.0, 4.5, 0.1)
    time_gap     = st.number_input("⏱️ Time Gap (Days)", min_value=0, step=1, value=120)

with c2:
    app_size_mb  = st.number_input("💾 App Size (MB)", min_value=0.0, step=0.1, value=150.0)
    ios_version  = st.number_input("📱 Required iOS Version (e.g., 13.0)", min_value=1.0, step=0.1, value=13.0)

with st.expander("Advanced (defaults mirror training schema)"):
    content_rating = st.selectbox("Content Rating", options=["4+", "9+", "12+", "17+"], index=0)
    release_year   = st.number_input("Release Year", min_value=2008, max_value=2025, value=2023)
    updated_year   = st.number_input("Updated Year", min_value=2008, max_value=2025, value=2024)
    updated_month  = st.number_input("Updated Month", min_value=1, max_value=12, value=6)
    release_month  = st.number_input("Release Month", min_value=1, max_value=12, value=8)


# ===========================================
# 🔮 PREDICT (Auto on every input change)
# ===========================================
def build_feature_row():
    defaults = {
        "Content_Rating": content_rating,   # keep as string
        "Release_Year": int(release_year),
        "Updated_Year": int(updated_year),
        "Updated_Month": int(updated_month),
        "Release_Month": int(release_month)
    }

    row = {
        "DeveloperId": int(developer_id),
        "Size_MB": float(app_size_mb),
        "Average_User_Rating": float(avg_rating),
        "Required_IOS_Version": str(ios_version),  # 🔑 CatBoost categorical as string
        "Time_Gap_Days": int(time_gap),
        **defaults
    }

    feature_order = [
        "DeveloperId", "Size_MB", "Average_User_Rating", "Required_IOS_Version",
        "Time_Gap_Days", "Content_Rating", "Release_Year",
        "Updated_Year", "Updated_Month", "Release_Month"
    ]
    return pd.DataFrame([row])[feature_order]


def decode_label(ids: np.ndarray) -> str:
    """
    Convert numeric class id to human-readable label.
    Priority:
      1) use saved LabelEncoder (if classes_ are strings),
      2) else fallback CATEGORY_MAP_FALLBACK,
      3) else show raw id.
    """
    try:
        arr = np.array(ids).ravel().astype(int)
        # try label encoder
        label = le_target.inverse_transform(arr)[0]
        # if encoder contains numeric strings (e.g., "7"), coerce to nice fallback
        if isinstance(label, (int, np.integer)) or (isinstance(label, str) and label.isdigit()):
            idx = int(label)
            return CATEGORY_MAP_FALLBACK.get(idx, f"Class {idx}")
        return str(label)
    except Exception:
        idx = int(np.array(ids).ravel()[0])
        return CATEGORY_MAP_FALLBACK.get(idx, f"Class {idx}")


def top_k_from_proba(proba_row: np.ndarray, k: int = 5):
    """Return top-k (label, prob) pairs, decoded."""
    idxs = np.argsort(proba_row)[-k:][::-1]
    labels = []
    for i in idxs:
        try:
            decoded = le_target.inverse_transform([i])[0]
            if isinstance(decoded, (int, np.integer)) or (isinstance(decoded, str) and decoded.isdigit()):
                decoded = CATEGORY_MAP_FALLBACK.get(int(decoded), f"Class {int(decoded)}")
        except Exception:
            decoded = CATEGORY_MAP_FALLBACK.get(int(i), f"Class {int(i)}")
        labels.append((decoded, float(proba_row[i])))
    return labels


if (model is None) or (le_target is None):
    st.error("❌ Model artifacts not loaded. Please refresh the app.")
else:
    X = build_feature_row()

    st.write("#### 🔎 Inference Input")
    st.dataframe(X, use_container_width=True)

    try:
        # predict class
        pred_raw = model.predict(X)
        pred_ids = np.array(pred_raw).ravel().astype(int)
        pred_label = decode_label(pred_ids)

        # predict proba
        proba = model.predict_proba(X)
        p_row = np.array(proba).reshape(-1, proba.shape[-1])[0]
        confidence = float(np.max(p_row)) * 100.0

        # top-5
        top5 = top_k_from_proba(p_row, k=5)

        # ========= RESULT CARD =========
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(
            f"""
            <div class="result-card">
                <div style="display:flex;align-items:center;gap:10px;flex-wrap:wrap;">
                    <div class="metric-badge">Predicted</div>
                    <h3 style="margin:0;color:var(--acc2)">{pred_label}</h3>
                </div>
                <div class="subtle" style="margin-top:6px;">Model confidence: <b>{confidence:.2f}%</b></div>
            </div>
            """,
            unsafe_allow_html=True
        )

        # ========= TOP-5 TABLE =========
        st.write("##### 🔝 Top-5 probable categories")
        top5_df = pd.DataFrame(
            [{"Category": lbl, "Confidence %": round(prob * 100.0, 2)} for lbl, prob in top5]
        )
        st.dataframe(top5_df, use_container_width=True)

        # simple bar chart (no extra deps)
        st.bar_chart(
            pd.DataFrame(
                {"Confidence": [p for _, p in top5]},
                index=[lbl for lbl, _ in top5]
            )
        )

    except Exception as e:
        st.error(f"⚠️ Prediction failed: {e}")


# =========================
# 🧾 FOOTER
# =========================
st.markdown("---")
st.caption(
    """
    💼 Developed by **Abdul Qadir** · 🎓 BS in Applied AI & Data Science, IIT Jodhpur  
    ⚡ Balanced CatBoost (GPU) · Auto-Predict · Top-5 Confidence  
    🌐 GitHub: https://github.com/Abdulqadir05/app-store-category-predictor
    """
)

