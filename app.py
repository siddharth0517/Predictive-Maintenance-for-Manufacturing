# app.py
import streamlit as st
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

# Load the models
xgb_target = joblib.load('xgb_target_model.pkl')
xgb_failure_type = joblib.load('xgb_failure_type_model.pkl')

# Create a StandardScaler instance for feature scaling (same as in your training)
sc = StandardScaler()

# Title of the app
st.title('Predictive Maintenance for Manufacturing')

# Input form
st.header('Enter Machine Parameters')
param1 = st.number_input('Parameter 1 (Numeric Feature)', value=0.0)
param2 = st.number_input('Parameter 2 (Numeric Feature)', value=0.0)
param3 = st.number_input('Parameter 3 (Numeric Feature)', value=0.0)
param4 = st.number_input('Parameter 4 (Numeric Feature)', value=0.0)
param5 = st.number_input('Parameter 5 (Numeric Feature)', value=0.0)
param6 = st.number_input('Parameter 6 (Numeric Feature)', value=0.0)

# Example: Handling categorical input (if applicable, depending on dataset)
# Assuming first feature is categorical and needs encoding
categorical_feature = st.selectbox('Select Grade of Product', options=['LOW', 'MEDIUM', 'HIGH'])

# Apply same encoding as during preprocessing
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [0])], remainder='passthrough')

# Prepare input features for prediction
input_data = np.array([[categorical_feature, param1, param2, param3, param4, param5, param6]])
input_data = ct.transform(input_data)  # Apply OneHotEncoding
input_data[:, 1:] = sc.transform(input_data[:, 1:])  # Apply feature scaling to numerical columns

# Predictions
if st.button('Predict'):
    # Prediction for binary classification (Target)
    target_prediction = xgb_target.predict(input_data)
    failure_type_prediction = xgb_failure_type.predict(input_data)

    # Display predictions
    st.subheader('Prediction Results')
    st.write(f"Target Failure Prediction: {'Yes' if target_prediction[0] == 1 else 'No'}")
    st.write(f"Failure Type Prediction: {failure_type_prediction[0]}")
