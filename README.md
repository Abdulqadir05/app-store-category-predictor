# 📱 App Store Category Predictor (CatBoost + Streamlit)

An interactive **Machine Learning web app** built using **CatBoost** and **Streamlit** that predicts the **category of iOS apps** based on metadata like size, user rating, version, and release information.

---

## 🚀 Live Demo
🌐 **Try it now:** [https://abdulqadir05-app-store-category-predictor.streamlit.app](https://app-store-category-predictor-8amufwrfzumupubzqo6tdx.streamlit.app/)

---

## 🧩 Features
✅ Predicts app category using trained CatBoost ML model  
✅ Clean, interactive Streamlit UI  
✅ Optimized for large-scale dataset (1.2M+ records)  
✅ Ready for Streamlit Cloud deployment  

---

## ⚙️ Tech Stack

| Component | Technology |
|------------|-------------|
| ML Model | CatBoost Classifier |
| Web Framework | Streamlit |
| Data | Apple App Store Dataset |
| Preprocessing | pandas, scikit-learn |
| Deployment | Streamlit Cloud |

---

## 🧠 Project Workflow

1. **EDA & Preprocessing**  
   - Outlier removal, encoding, scaling, log transform  
   - Feature engineering (Release gap, iOS version, etc.)

2. **Modeling**  
   - Compared LightGBM & CatBoost  
   - Final: CatBoostClassifier (GPU Accelerated)  
   - Accuracy: ~36.5% on 1.23M rows  

3. **Deployment**  
   - Model serialized via joblib  
   - Streamlit web app  
   - Hosted on Streamlit Cloud  

---

## 🧰 Installation

### Clone the Repository
```bash
git clone https://github.com/Abdulqadir05/app-store-category-predictor.git
cd app-store-category-predictor
