# 📱 App Store Category Predictor  
### _End-to-End AI Project by Abdul Qadir_

> 🚀 A complete Machine Learning system that predicts an iOS app’s **App Store category** from its metadata using a **CatBoost (GPU)** model, fully deployed with **Streamlit Cloud**.

---

## 🧠 Overview

This project demonstrates the **full AI lifecycle** — from **EDA → preprocessing → model training → balancing → GPU optimization → deployment**.  
It leverages a **CatBoost classifier** trained on over **1.2M App Store apps** to classify apps into 26 real categories (e.g., Games, Finance, Music, Education, etc.).

---

## 🎬 Demo

### 🌐 **[🚀 Live Web App](https://app-store-category-predictor-awihqbixm6tru4s9ktvrnr.streamlit.app/)**  
*(Hosted on Streamlit Cloud)*

| 🎥 **Demo Preview** | 🧩 **Animated Theme** |
|:-------------------:|:--------------------:|
| ![Demo](https://raw.githubusercontent.com/Abdulqadir05/app-store-category-predictor/main/assets/demo.gif) | ![UI](https://raw.githubusercontent.com/Abdulqadir05/app-store-category-predictor/main/assets/theme-preview.gif) |

> 💡 _The UI features smooth gradient animations, glass-morphism cards, and auto-prediction powered by CatBoost._

---

## 🧩 Features

✅ **EDA & Preprocessing** — Full data cleaning, encoding, outlier treatment, and feature engineering.  
✅ **Class Balancing** — Applied `RandomOverSampler` + `auto_class_weights='Balanced'`.  
✅ **GPU-Accelerated CatBoost** — 700+ iterations on NVIDIA GPU via Google Colab.  
✅ **Confidence Scoring** — Displays top-5 probable app categories with confidence %.  
✅ **Dynamic Streamlit UI** — Auto-prediction, animated gradient theme, dark aesthetic.  
✅ **Deployed via GitHub + Streamlit Cloud** — Model loaded dynamically from GitHub Releases (v1.1).  

---

## 📊 Model Performance

| Metric | Value | Note |
|:--------|:------:|:-----|
| **Train Accuracy** | 0.804 | Balanced training on 26 categories |
| **Test Accuracy** | 0.221 | Realistic generalization (balanced data) |
| **Framework** | CatBoost GPU | Multiclass, balanced, early stopping |
| **Dataset Size** | 1.2M+ rows | Cleaned & feature-engineered |

---

## ⚙️ Tech Stack

| Layer | Tools / Libraries |
|:------|:------------------|
| **Data Cleaning & EDA** | `pandas`, `matplotlib`, `seaborn`, `ydata_profiling` |
| **Feature Engineering** | `LabelEncoder`, `OneHotEncoder`, custom binning |
| **Model Training** | `CatBoostClassifier (GPU)`, `sklearn`, `imbalanced-learn` |
| **Model Storage** | `joblib`, GitHub Releases |
| **Deployment** | `Streamlit`, `Python 3.10+`, `requests`, `pandas` |
| **Hosting** | Streamlit Cloud |

---

## 🔍 Example Predictions

| Developer ID | Size_MB | Rating | iOS | Time_Gap | Predicted Category | Confidence |
|---------------|----------|--------|------|-----------|--------------------|-------------|
| 300000000 | 200 | 4.8 | 15.0 | 100 | 🎮 Games | 91.8% |
| 1100000000 | 130 | 4.1 | 13.0 | 180 | 💰 Finance | 82.3% |
| 100000000 | 45 | 4.2 | 12.0 | 240 | 🎵 Music | 78.4% |
| 800000000 | 50 | 4.6 | 14.0 | 250 | 🧩 Education | 75.6% |

> 🧠 _Model dynamically ranks top-5 probable categories with associated confidence levels._

---


---

## 🧠 Model Files

| File | Description | Source |
|:------|:-------------|:--------|
| `catboost_app_category_model.pkl` | Trained CatBoost classifier | [GitHub Release v1.1](https://github.com/Abdulqadir05/app-store-category-predictor/releases/tag/v1.1) |
| `category_label_encoder.pkl` | Encoded label map for 26 categories | [GitHub Release v1.1](https://github.com/Abdulqadir05/app-store-category-predictor/releases/tag/v1.1) |
| `feature_schema.pkl` | Training schema info (columns, types) | [GitHub Release v1.1](https://github.com/Abdulqadir05/app-store-category-predictor/releases/tag/v1.1) |

---

## 🚀 Deployment Pipeline

```mermaid
graph TD
A[🧹 Clean Dataset] --> B[🧠 Train CatBoost GPU Model]
B --> C[💾 Save Artifacts (.pkl)]
C --> D[☁️ Upload to GitHub Release]
D --> E[🌐 Streamlit App Fetches Model]
E --> F[⚡ Real-Time Predictions]

👨‍💻 **Author**

**Abdul Qadir** <br>
🎓 BS in Applied AI & Data Science, IIT Jodhpur <br>
💼 Aspiring Data Scientist | Machine Learning Engineer <br>
🌍 Passionate about End-to-End AI Solutions, EDA, and Model Deployment <br>
📧 Email: b24bs1012@iitj.ac.in <br>
🔗 GitHub: Abdulqadir05  <br>
🌐 Portfolio (Coming soon...)

🧩 **License**

This project is licensed under the MIT License — free to use, modify, and share for learning or production.

If you like this project, please ⭐ star the repo — it helps others find it and supports continued open-source work!
