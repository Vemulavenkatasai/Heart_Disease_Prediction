# 🫀 Heart Disease Prediction Web App

A machine learning-based web app to predict the risk of heart disease using patient medical data. Built with **scikit-learn** and deployed using **Streamlit Cloud**.

live app -->(https://vemulavenkatasai-heart-disease-prediction-app-o7bku0.streamlit.app/)

---

## 🚀 Features

- Predicts heart disease risk based on user input
- Uses a trained **Random Forest Classifier**
- Scales input data using **StandardScaler**
- User-friendly web interface built with **Streamlit**
- Live deployment on Streamlit Cloud

---

## 📊 Dataset

- Source: [UCI Heart Disease Dataset](https://www.kaggle.com/datasets)
- 13 medical input features like age, cholesterol, chest pain, etc.
- Binary target: 0 = No Heart Disease, 1 = Has Heart Disease

---

## 🛠️ Tech Stack

- **Python**
- **Pandas & NumPy**
- **scikit-learn**
- **Streamlit**
- **Jupyter Notebook**
- **Joblib** (for saving the model and scaler)

## 📁 Project Structure

```text
heart-disease-prediction/
│
├── .ipynb_checkpoints/         # Auto-generated Jupyter backups
├── Heart_Prediction.ipynb      # Jupyter notebook for training and evaluation
├── app.py                      # Streamlit web app to run the prediction
├── heart.csv                   # Dataset used for training and testing
├── heart_model.pkl             # Trained Random Forest model
├── scaler.pkl                  # Preprocessing scaler (StandardScaler)
├── requirements.txt            # All required Python dependencies
└── README.md                   # Project overview and documentation


