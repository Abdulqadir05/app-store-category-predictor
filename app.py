import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="App Category Predictor", page_icon="📱", layout="centered")


st.title("📱 App Store Category Predictor")
st.write("Predict the category of an iOS app using a trained CatBoost model.")

# Load model safely
try:
    model = joblib.load('catboost_app_category_model.pkl')
except Exception as e:
    st.error(f"⚠️ Model could not be loaded: {e}")
    st.stop()

# Example category mapping
mapping = {
    0:'Business',1:'Education',2:'Entertainment',3:'Games',4:'Lifestyle',
    5:'Finance',6:'Health & Fitness',7:'Music'
}

# Input
dev_id = st.text_input("Developer ID","12345")
size = st.number_input("App Size (MB)",1.0,5000.0,250.0)
rating = st.slider("Average User Rating",0.0,5.0,4.0)
ios = st.number_input("Required iOS Version",8.0,18.0,13.0,step=0.1)
gap = st.number_input("Time Gap (Days)",0,5000,365)

if st.button("🔮 Predict"):
    sample = pd.DataFrame({
        'DeveloperId':[dev_id],
        'Time_Gap_Days':[gap],
        'Size_MB':[size],
        'Updated_Month':['6'],
        'Required_IOS_Version':[ios],
        'Release_Year':[2023],
        'Content_Rating':['4+'],
        'Updated_Year':[2024],
        'Release_Month':['2'],
        'Average_User_Rating':[rating]
    })
    for c in ['DeveloperId','Content_Rating','Release_Month','Updated_Month']:
        sample[c]=sample[c].astype(str)

    pred = int(model.predict(sample)[0])
    label = mapping.get(pred,f"Unknown ({pred})")
    st.success(f"🎯 Predicted Category: {label}")
