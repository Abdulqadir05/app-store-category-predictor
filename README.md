## 👨‍💻 Author  
**Abdul Qadir** <br>
🎓 BS in Applied AI & Data Science, IIT Jodhpur <br>
💼 Aspiring Data Scientist | Machine Learning Engineer <br>
🌍 Passion: End-to-End AI Solutions, EDA, Deployment <br>
📧 b24bs1012@iitj.ac.in

📱 App Store Category Predictor — End-to-End ML Project
🧠 Predict the Category of iOS Apps using Machine Learning

This project is a complete end-to-end data science pipeline, built on the Apple App Store dataset.
It covers everything — from Exploratory Data Analysis (EDA) to model deployment using Streamlit Cloud.

🚀 Project Overview

The goal of this project is to build a classification model that predicts an app’s category based on its metadata such as rating, size, iOS version requirement, developer, and release/update patterns.

The final model is trained using CatBoostClassifier, optimized for handling categorical and large-scale data efficiently.

🧩 End-to-End Workflow
1️⃣ Data Collection & Understanding

Source: Apple App Store dataset (1.2M+ apps)

Columns:
App_Name, Category, Content_Rating, Size_MB, Required_IOS_Version,
Average_User_Rating, Price, DeveloperId, Release_Year, Updated_Year, etc.

2️⃣ Data Cleaning & Preprocessing

✅ Handled missing values
✅ Dropped duplicates
✅ Fixed inconsistent formats in Size_MB, Price, Required_IOS_Version
✅ Converted data types and extracted time-based features (Release_Month, Updated_Month)
✅ Handled non-ASCII text & Unicode developer names

3️⃣ Feature Engineering

⚙️ Created new features:

Time_Gap_Days (difference between release and update)

App_Type (Free vs Paid)

Encoded categorical variables using LabelEncoder / OneHotEncoder

Converted numerical outliers using log10 transformation and RobustScaler

4️⃣ Exploratory Data Analysis (EDA)

📊 Conducted using Matplotlib, Seaborn, Plotly

Key Insights Visualized:

Distribution of iOS version requirements

Rating trends vs app size

Most common release months

Free vs Paid ratio

Correlation heatmap

Top 10 features affecting app ratings

5️⃣ Outlier Detection & Transformation

Detected using IQR & Z-score methods, then fixed using:

Log transformation for skewed columns (Reviews, Price)

RobustScaler for Size_MB, Time_Gap_Days, Current_Version_Reviews

6️⃣ Feature Selection & Multicollinearity Check (VIF)

✅ Removed multicollinear features with high VIF
✅ Retained important predictors such as:
DeveloperId, Size_MB, Average_User_Rating, Required_IOS_Version, Time_Gap_Days, Release_Year, etc.

7️⃣ Model Building

Models Tested:

Decision Tree Classifier 🌳

Random Forest Classifier 🌲

Gradient Boosting Classifier 🚀

LightGBM ⚡

CatBoost Classifier (Final) 🏆

Why CatBoost?

Handles categorical data automatically

Efficient on large datasets

Less overfitting

GPU acceleration support

Final Metrics:

Metric	Value
Accuracy	0.365
Balanced Accuracy	0.247
Weighted F1	0.34
Best Iteration	697
8️⃣ Feature Importance (CatBoost)

Top 10 Features impacting prediction:

DeveloperId

Time_Gap_Days

Size_MB

Updated_Month

Required_IOS_Version

Release_Year

Content_Rating

Updated_Year

Release_Month

Average_User_Rating

9️⃣ Model Serialization

✅ Model saved using joblib as:

catboost_app_category_model.pkl


✅ Uploaded to GitHub Releases for Streamlit app download.

🔟 Model Deployment — Streamlit Web App

Deployed the final CatBoost model via Streamlit Cloud
🌐 Live App: 🌐 **Try it now:** [https://abdulqadir05-app-store-category-predictor.streamlit.app](https://app-store-category-predictor-8amufwrfzumupubzqo6tdx.streamlit.app/)

App Features:

Input developer, size, rating, iOS version, and time gap

Model auto-downloads from GitHub release

Predicts real-time category (e.g., Games, Music, Finance)

Modern dark-mode UI with icons and styling

🧰 Tech Stack
Category |	Tools Used
Language |	Python
Data Handling|Pandas, NumPy
Visualization |Matplotlib, Seaborn, Plotly
Modeling	Scikit-learn,| CatBoost, LightGBM
Deployment |Streamlit, GitHub
Version Control |	Git + GitHub
Storage | GitHub Releases / Google Drive (for model)

📁 Project Structure
📦 App_Store_Category_Predictor
 ┣ 📜 app.py
 ┣ 📜 requirements.txt
 ┣ 📜 README.md
 ┣ 📜 catboost_app_category_model.pkl  (stored via GitHub release)
 ┗ 📂 dataset/
     ┗ 📜 appleAppData.csv
     
🏁 Future Improvements

Integrate preprocessing pipeline directly into app (auto scaling & encoding).

Add SHAP explainability dashboard.

Migrate to FastAPI + Docker for API-based deployment.

🌟 If you liked this project

Give a ⭐ on GitHub to support more open-source AI projects like this!


