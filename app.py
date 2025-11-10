import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load the trained model and scaler
try:
    gbc_model = joblib.load('gradient_boosting_model.pkl')
    scaler = joblib.load('scaler.pkl')
except FileNotFoundError:
    st.error("Model or scaler file not found. Please make sure 'gradient_boosting_model.pkl' and 'scaler.pkl' are in the same directory.")
    st.stop() # Stop the app if files are not found

st.title('Breast Cancer Diagnosis Predictor')

st.write("""
Enter the measurements of the breast mass to predict if it is Benign (1) or Malignant (0).
""")

# Create input fields for each feature
# You can customize the input type based on the feature (e.g., st.number_input, st.slider)
input_data = {}
feature_names = ['radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean', 'smoothness_mean',
                 'compactness_mean', 'concavity_mean', 'concave_points_mean', 'concave_points_mean', 'symmetry_mean',
                 'fractal_dimension_mean', 'radius_se', 'texture_se', 'perimeter_se', 'area_se',
                 'smoothness_se', 'compactness_se', 'concavity_se', 'concave_points_se',
                 'symmetry_se', 'fractal_dimension_se', 'radius_worst', 'texture_worst',
                 'perimeter_worst', 'area_worst', 'smoothness_worst', 'compactness_worst',
                 'concavity_worst', 'concave_points_worst', 'symmetry_worst',
                 'fractal_dimension_worst']

for feature in feature_names:
    # Added a check to avoid duplicate keys if 'concave_points_mean' appears twice
    if feature not in input_data:
        input_data[feature] = st.number_input(f'Enter {feature}', value=0.0)


# Convert input data to DataFrame
input_df = pd.DataFrame([input_data])

# Scale the input data
input_scaled = scaler.transform(input_df)

# Make prediction
if st.button('Predict'):
    prediction = gbc_model.predict(input_scaled)

    if prediction[0] == 1:
        st.success('Prediction: Benign')
    else:
        st.error('Prediction: Malignant')