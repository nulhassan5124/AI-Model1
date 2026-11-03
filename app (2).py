import streamlit as st
try:
    import joblib
except ImportError:
    from sklearn.utils import _joblib as joblib
import pandas as pd
import numpy as np

# Set Page configuration
st.set_page_config(
    page_title="Car Fuel Type Predictor",
    page_icon="🚗",
    layout="centered"
)

# App title and description
st.title("🚗 Car Fuel Type Classifier")
st.write("This app uses a Decision Tree Classifier (balanced using SMOTE) to predict the Fuel Type of a car based on its features.")

# Load the pre-trained model
@st.cache_resource
def load_model():
    return joblib.load('decision_tree_smote_model.joblib')

try:
    model = load_model()
except Exception as e:
    st.error(f"Error loading the model. Please ensure 'decision_tree_smote_model.joblib' is in the same directory. Details: {e}")
    st.stop()

# User Inputs Sidebar / Form
st.subheader("Enter Car Specifications")

col1, col2 = st.columns(2)

with col1:
    price = st.number_input("Price (in Euros)", min_value=0, value=10000, step=100)
    age = st.number_input("Age of the Car (in years)", min_value=0.0, max_value=20.0, value=5.0, step=0.1)
    km = st.number_input("KM (Kilometers Driven)", min_value=0, value=50000, step=1000)
    cc = st.number_input("Engine CC", min_value=500, max_value=6000, value=1600, step=50)

with col2:
    weight = st.number_input("Weight (in kg)", min_value=500, max_value=3000, value=1100, step=10)
    met_color = st.selectbox("Metallic Color", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    automatic = st.selectbox("Transmission Type", options=[0, 1], format_func=lambda x: "Automatic" if x == 1 else "Manual")

# Predict Button
if st.button("🎯 Predict Fuel Type"):
    # Create a DataFrame matching the training features
    input_data = pd.DataFrame([[
        price,
        age,
        km,
        met_color,
        automatic,
        cc,
        weight
    ]], columns=['Price', 'Age', 'KM', 'MetColor', 'Automatic', 'CC', 'Weight'])

    # Make prediction
    prediction = model.predict(input_data)[0]

    # Display result
    st.success(f"Predicted Fuel Type: **{prediction}**")
