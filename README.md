## 👨‍💻 Author  
**Abdul Qadir** <br>
🎓 BS in Applied AI & Data Science, IIT Jodhpur <br>
💼 Aspiring Data Scientist | Machine Learning Engineer <br>
🌍 Passion: End-to-End AI Solutions, EDA, Deployment <br>
📧 b24bs1012@iitj.ac.in

# 📱 App Store Category Predictor

> *An end-to-end Machine Learning project that predicts iOS App Categories using CatBoost (GPU accelerated).*

---

## 🚀 Project Overview

The **App Store Category Predictor** is a complete end-to-end ML solution that:
- Cleans and preprocesses raw App Store data 🧹  
- Performs **Exploratory Data Analysis (EDA)** to uncover trends 📊  
- Trains a **CatBoost GPU-powered classifier** for category prediction ⚙️  
- Deploys an interactive **Streamlit web app** for real-time predictions 🌐  

This project demonstrates mastery in **Applied AI**, **EDA**, **Model Building**, and **Deployment** — essential for data science industry roles.

---

## 🔍 Workflow Summary

| Stage | Description |
|:------|:-------------|
| 🧹 **Data Cleaning** | Handled missing values, encoding, scaling, and outlier correction |
| 📊 **EDA** | Used Matplotlib & Seaborn to visualize app trends and patterns |
| ⚙️ **Feature Engineering** | Created features like `Time_Gap_Days`, `App_Size_Bins`, etc. |
| 🧠 **Model Building** | Trained and optimized multiple ML models, finalized CatBoost |
| ⚡ **GPU Training** | Leveraged **Google Colab GPU** for accelerated model training |
| ✅ **Evaluation** | Compared metrics — Accuracy, Precision, Recall, and F1-score |
| 🌐 **Deployment** | Hosted on **Streamlit Cloud** for public access |

---

## 🧩 Tech Stack

| Domain | Tools / Libraries |
|:-------|:------------------|
| **Data Analysis** | Pandas, NumPy, Matplotlib, Seaborn |
| **Modeling** | CatBoost, Scikit-learn |
| **Feature Engineering** | Label Encoding, Data Transformation |
| **Deployment** | Streamlit, GitHub Releases, Streamlit Cloud |
| **Environment** | Google Colab (GPU) + VS Code |

---

## ⚙️ Model Details

- **Algorithm:** CatBoost Classifier  
- **Mode:** GPU Accelerated  
- **Loss Function:** MultiClass  
- **Training Accuracy:** ~51.4%  
- **Test Accuracy:** ~44.4%  
- **Top Influential Features:**
  - DeveloperId  
  - Size_MB  
  - Average_User_Rating  
  - Required_IOS_Version  
  - Time_Gap_Days  

---

## 🧠 How It Works

1. The user provides app details such as Developer ID, Size, Rating, iOS Version, etc.  
2. The model (CatBoost) processes the data and predicts the **App Category**.  
3. The model and schema are automatically loaded from **GitHub Releases** during deployment.  
4. The Streamlit app displays human-readable predictions like **“Games”**, **“Education”**, etc.

---

## 🧮 Example Input

| Feature | Example Value |
|:---------|:---------------|
| DeveloperId | 500000000 |
| Size_MB | 150.0 |
| Average_User_Rating | 4.3 |
| Required_IOS_Version | 13.0 |
| Time_Gap_Days | 120 |

**🎯 Predicted Output:** `Games`

---

## 🌍 Live Demo

🔗 **[View Deployed App on Streamlit Cloud](https://share.streamlit.io/Abdulqadir05/app-store-category-predictor/main/app.py)**  
*(Ensure GitHub release files are public for successful model loading.)*

---

💡 **Key Highlights**

✅ End-to-End ML Pipeline (EDA → Feature Engineering → Model → Deployment)
⚡ GPU Accelerated CatBoost Model
🌐 Auto-loads model from GitHub Releases
🧠 Human-readable category names (Games, Education, etc.)
🎨 Beautiful Streamlit UI with live interaction


**Future Enhancements**

🔄 Integrate Real-time App Data via API <br>
🧩 Add Explainability using SHAP / LIME <br>
🐳 Containerize with Docker for Cloud Deployment <br>
🧠 Experiment with Deep Learning models (Transformers, XGBoost hybrid)

🧾**License**

This project is released under the MIT License — free to use, modify, and distribute with attribution.

🌟**Show Your Support**

If you liked this project, give it a ⭐ on GitHub and share it with others!
Let’s build open, explainable, and scalable AI together 🚀

