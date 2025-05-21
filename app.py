import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import pickle

## Load the model, scaler, and encoders

model = tf.keras.models.load_model('model.keras')

with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

with open('geography_OHE.pkl', 'rb') as file:
    geography_OHE = pickle.load(file)

with open('gender_LE.pkl', 'rb') as file:
    gender_LE = pickle.load(file)


## StreamLit App

st.title('Customer Churn Prediction')


## User Input

geography = st.selectbox('Geography', geography_OHE.categories_[0])
gender = st.selectbox('Gender', gender_LE.classes_)
age = st.slider('Age', 16, 100)
balance = st.number_input('Balance')
credit_score = st.number_input('Credit Score')
estimated_salary = st.number_input('Estimated Salary')
tenure = st.slider('Tenure', 0, 10)
num_of_products = st.slider('Number of Products', 1, 4)
has_credit_card = st.selectbox('Has Credit Card', ['Yes', 'No'])
is_active_member = st.selectbox('Is Active Member', ['Yes', 'No'])


## We do not include geography, as we still need to encode it

input_data = pd.DataFrame({
    'CreditScore': [credit_score],
    'Gender': [gender_LE.transform([gender])[0]],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [1 if has_credit_card == 'Yes' else 0],
    'IsActiveMember': [1 if is_active_member=='Yes' else 0],
    'EstimatedSalary': [estimated_salary]
})

## Encode Geography columnns
geography_encoded = geography_OHE.transform([[geography]]).toarray()
geography_encoded_df = pd.DataFrame(geography_encoded, columns=geography_OHE.get_feature_names_out(['Geography']))

## Convert input data into dataframe, then concatenate with encoded Geography
input_data_df = pd.concat([pd.DataFrame(input_data), geography_encoded_df], axis=1)

## Scale data
input_data_scaled = scaler.transform(input_data_df)

## Predict using model and extract probability
prediction = model.predict(input_data_scaled)
prediction_probability = prediction[0][0]

st.write(f'Churning Probability: {prediction_probability}')

## Less than 0.45 --> Not likely to Churn
## Between 0.45 and 0.55 --> Unpredictable
## Above 0.55 --> Likely to Churn

if prediction_probability < 0.45:
    st.write('Customer is not likely to churn')
elif prediction_probability < 0.55:
    st.write('Customer behavior is unpredictable. Customer may or may not churn')
else:
    st.write('Customer is likely to churn')