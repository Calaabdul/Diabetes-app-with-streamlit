import streamlit as st
import pickle
import numpy as np

# Load the model and scaler
model_path = 'diabetes_model.pkl'
scaler_path = 'scaler.pkl'

with open(model_path, 'rb') as model_file:
    model = pickle.load(model_file)

with open(scaler_path, 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)

# Title and description
st.title('Diabetes Prediction App')
st.write("""
# Diabetes Prediction
This app predicts whether a person has diabetes or not based on certain input parameters.
""")

# Input fields
st.sidebar.header('User Input Parameters')

def user_input_features():
    pregnancies = st.sidebar.number_input('Pregnancies', min_value=0, max_value=20, value=1)
    glucose = st.sidebar.number_input('Glucose', min_value=0, max_value=200, value=120)
    blood_pressure = st.sidebar.number_input('Blood Pressure', min_value=0, max_value=122, value=70)
    skin_thickness = st.sidebar.number_input('Skin Thickness', min_value=0, max_value=99, value=20)
    insulin = st.sidebar.number_input('Insulin', min_value=0, max_value=846, value=79)
    bmi = st.sidebar.number_input('BMI', min_value=0.0, max_value=67.1, value=25.0)
    dpf = st.sidebar.number_input('Diabetes Pedigree Function', min_value=0.0, max_value=2.42, value=0.5)
    age = st.sidebar.number_input('Age', min_value=21, max_value=81, value=33)
    
    data = {
        'Pregnancies': pregnancies,
        'Glucose': glucose,
        'BloodPressure': blood_pressure,
        'SkinThickness': skin_thickness,
        'Insulin': insulin,
        'BMI': bmi,
        'DiabetesPedigreeFunction': dpf,
        'Age': age
    }
    
    features = np.array([list(data.values())])
    return features

input_df = user_input_features()

# Scale the input features
scaled_input = scaler.transform(input_df)

# Predict the outcome
prediction = model.predict(scaled_input)
prediction_proba = model.predict_proba(scaled_input)

# Display the prediction
st.subheader('Prediction')
diabetes_status = np.array(['Non-diabetic', 'Diabetic'])
st.write(diabetes_status[prediction][0])

st.subheader('Prediction Probability')
st.write(prediction_proba)
