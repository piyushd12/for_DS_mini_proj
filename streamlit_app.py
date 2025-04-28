
import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import OneHotEncoder
from imblearn.over_sampling import SMOTE

st.title("UPI Fraud Detection")



@st.cache_data
def load_and_preprocess_data():
    df = pd.read_csv("Upi_fraud_dataset.csv")
    
    drop_cols = ['TransactionID', 'UserID', 'Timestamp', 'DeviceID', 'IPAddress', 'PhoneNumber', 'BankName']
    df = df.drop(columns=drop_cols)
    
    df['FraudFlag'] = df['FraudFlag'].apply(lambda x: 1 if x in [True, 'True', 'true'] else 0)
    
    df['TransactionFrequency'] = df['TransactionFrequency'].apply(lambda x: float(str(x).split('/')[0]))
    
    df.reset_index(drop=True, inplace=True)
    
    return df

df = load_and_preprocess_data()
st.write("### Dataset Preview", df.head())


target_column = "FraudFlag"
feature_columns = [
    'Amount', 'MerchantCategory', 'TransactionType', 'Latitude', 'Longitude', 
    'AvgTransactionAmount', 'TransactionFrequency', 'UnusualLocation', 
    'UnusualAmount', 'NewDevice', 'FailedAttempts'
]

X = df[feature_columns].copy()
y = df[target_column]

bool_cols = ['UnusualLocation', 'UnusualAmount', 'NewDevice']
for col in bool_cols:
    X[col] = X[col].apply(lambda x: 1 if x in [True, 'True', 'true'] else 0)

categorical_cols = ['MerchantCategory', 'TransactionType']
X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)

# st.write("### Processed Features Preview", X.head())


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

sm = SMOTE(random_state=42)
X_train_sm, y_train_sm = sm.fit_resample(X_train, y_train)


model = xgb.XGBClassifier(
    objective='binary:logistic',
    eval_metric='auc',
    use_label_encoder=False,
    random_state=42
)
model.fit(X_train_sm, y_train_sm)

y_pred = model.predict(X_test)
# st.write("### Model Evaluation on Test Set")
# st.text(classification_report(y_test, y_pred))
# st.write("Confusion Matrix:")
# st.write(confusion_matrix(y_test, y_pred))



# st.write("Enter transaction details below to predict if it is fraudulent:")

st.sidebar.header("Input Transaction Features")

amount = st.sidebar.number_input("Amount", min_value=0.0, value=0.0)
avg_transaction_amount = st.sidebar.number_input("Average Transaction Amount", min_value=0.0, value=0.0)
transaction_frequency = st.sidebar.number_input("Transaction Frequency (numeric value)", min_value=0.0, value=0.0)
latitude = st.sidebar.number_input("Latitude", value=0.0)
longitude = st.sidebar.number_input("Longitude", value=0.0)
failed_attempts = st.sidebar.number_input("Failed Attempts", min_value=0, value=0)

merchant_category = st.sidebar.selectbox("Merchant Category", df['MerchantCategory'].unique())
transaction_type = st.sidebar.selectbox("Transaction Type", df['TransactionType'].unique())

unusual_location = st.sidebar.checkbox("Unusual Location", value=False)
unusual_amount = st.sidebar.checkbox("Unusual Amount", value=False)
new_device = st.sidebar.checkbox("New Device", value=False)

input_dict = {
    'Amount': amount,
    'Latitude': latitude,
    'Longitude': longitude,
    'AvgTransactionAmount': avg_transaction_amount,
    'TransactionFrequency': transaction_frequency,
    'UnusualLocation': 1 if unusual_location else 0,
    'UnusualAmount': 1 if unusual_amount else 0,
    'NewDevice': 1 if new_device else 0,
    'FailedAttempts': failed_attempts,
    'MerchantCategory': merchant_category,
    'TransactionType': transaction_type
}

input_df = pd.DataFrame([input_dict])

input_df_processed = pd.get_dummies(input_df, columns=categorical_cols, drop_first=True)

for col in X.columns:
    if col not in input_df_processed.columns:
        input_df_processed[col] = 0

input_df_processed = input_df_processed[X.columns]

# st.write("### Processed Input Data")
# st.write(input_df_processed)


prediction_probability = model.predict_proba(input_df_processed)[0][1]

threshold = st.sidebar.slider("Decision Threshold", min_value=0.0, max_value=1.0, value=0.5)

if prediction_probability > threshold:
    prediction = "Fraudulent"
else:
    prediction = "Non Fraudulent"

st.write("### Prediction Result")
st.write(f"**Fraud Prediction Probability:** {prediction_probability*100:.2f}%")
st.write(f"**Transaction is predicted as:** {prediction}")
