import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Load the trained model
model_filename = 'UPI_fraud_detection_Vo2_using_XGBoost.pkl'
model = pickle.load(open(model_filename, 'rb'))

# Load the scaler used during training
scaler = StandardScaler()

# Define categorical features and label encoders
categorical_features = ['MerchantCategory', 'TransactionType', 'IPAddress', 'UnusualLocation', 'UnusualAmount', 'NewDevice', 'BankName']
label_encoders = {feature: LabelEncoder() for feature in categorical_features}

# Load training data to fit encoders and scaler
training_data = pd.read_csv('Upi_fraud_dataset.csv')
for feature in categorical_features:
    label_encoders[feature].fit(training_data[feature])

# Preprocess the 'TransactionFrequency' column to extract numeric values
training_data['TransactionFrequency'] = training_data['TransactionFrequency'].str.extract(r'(\d+)').astype(float)

# Fit the scaler on the numerical features
numerical_features = ['TransactionFrequency']  # Add other numerical features if applicable
scaler.fit(training_data[numerical_features])

# Streamlit app
st.title("UPI Fraud Detection")

# Input fields for user data
st.header("Enter Transaction Details")
user_input = {}
for feature in categorical_features:
    user_input[feature] = st.selectbox(f"{feature}:", label_encoders[feature].classes_)

for feature in numerical_features:
    user_input[feature] = st.number_input(f"{feature}:", min_value=0.0, step=0.1)

# Convert user input into a DataFrame
input_df = pd.DataFrame([user_input])

# Add missing features with default values
input_df['UserID'] = 0  # Default value for UserID
input_df['DeviceID'] = 0  # Default value for DeviceID
input_df['Timestamp'] = 0  # Default value for Timestamp

# Reorder columns to match the model's expected feature order
input_df = input_df[['UserID', 'Timestamp', 'MerchantCategory', 'TransactionType', 'DeviceID', 
                     'IPAddress', 'TransactionFrequency', 'UnusualLocation', 'UnusualAmount', 
                     'NewDevice', 'BankName']]

# Encode categorical features
for feature in categorical_features:
    input_df[feature] = label_encoders[feature].transform(input_df[feature])

# Scale numerical features
input_df[numerical_features] = scaler.transform(input_df[numerical_features])

# Debugging: Log the input data
st.write("Input Data for Prediction:")
st.write(input_df)

# Make predictions
if st.button("Predict"):
    prediction_proba = model.predict_proba(input_df)[:, 1]
    
    # Allow user to adjust the threshold dynamically
    threshold = st.slider("Set Fraud Detection Threshold", min_value=0.0, max_value=1.0, value=0.5, step=0.01)

    # Debugging: Log the prediction probabilities and threshold
    st.write("Prediction Probability:", prediction_proba[0])
    st.write("Current Threshold:", threshold)

    if prediction_proba[0] >= threshold:
        st.error(f"Fraudulent Transaction Detected! Probability: {prediction_proba[0]:.2f}")
    else:
        st.success(f"Transaction is Legitimate. Probability: {1 - prediction_proba[0]:.2f}")