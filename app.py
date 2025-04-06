
import streamlit as st
import numpy as np
import joblib

# Load model and scaler
model = joblib.load("heart_model.pkl")
scaler = joblib.load("scaler.pkl")

st.set_page_config(page_title="Heart Disease Prediction", layout="centered")

st.title("🫀 Heart Disease Prediction App")
st.write("Enter the patient details below to check the risk of heart disease.")

# Input fields
age = st.number_input("Age", min_value=1, max_value=120, value=50)
sex = st.selectbox("Sex", options=[("Male", 1), ("Female", 0)])
cp = st.selectbox("Chest Pain Type (cp)", options=[(0, 0), (1, 1), (2, 2), (3, 3)])
trestbps = st.number_input("Resting Blood Pressure (trestbps)", min_value=80, max_value=200, value=120)
chol = st.number_input("Serum Cholesterol in mg/dl (chol)", min_value=100, max_value=600, value=200)
fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl (fbs)", options=[("Yes", 1), ("No", 0)])
restecg = st.selectbox("Resting ECG results (restecg)", options=[0, 1, 2])
thalach = st.number_input("Max Heart Rate Achieved (thalach)", min_value=50, max_value=250, value=150)
exang = st.selectbox("Exercise Induced Angina (exang)", options=[("Yes", 1), ("No", 0)])
oldpeak = st.number_input("Oldpeak", min_value=0.0, max_value=10.0, step=0.1, value=1.0)
slope = st.selectbox("Slope of peak exercise ST segment", options=[0, 1, 2])
ca = st.selectbox("Number of major vessels (0-3) colored by fluoroscopy", options=[0, 1, 2, 3])
thal = st.selectbox("Thalassemia (thal)", options=[1, 2, 3])

if st.button("Predict"):
    input_data = np.array([
        age, sex[1], cp[1], trestbps, chol, fbs[1], restecg,
        thalach, exang[1], oldpeak, slope, ca, thal
    ]).reshape(1, -1)

    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)
    proba = model.predict_proba(input_scaled)
    st.write(f"Prediction Probability: {proba}")

    if prediction[0] == 1:
        st.error("⚠️ The patient is likely to have heart disease.")
    else:
        st.success("✅ The patient is unlikely to have heart disease.")
