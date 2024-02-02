# app.py
import streamlit as st
import pandas as pd
import joblib

def load_model(filepath):
    model = joblib.load(filepath)
    return model

def main():
    st.title('Diabetes Prediction App')
    st.write('This app predicts whether a person has diabetes or not based on input features.')

    age = st.slider('Age', 0, 120, 25)
    bmi = st.slider('BMI', 0.0, 70.0, 25.0)
    bp = st.slider('Blood Pressure', 0, 200, 100)
    s1 = st.slider('S1', 0, 300, 150)
    s2 = st.slider('S2', 0.0, 2.0, 1.0)
    s3 = st.slider('S3', 0.0, 300.0, 150.0)
    s4 = st.slider('S4', 0.0, 5.0, 2.5)
    s5 = st.slider('S5', 0, 400, 200)
    s6 = st.slider('S6', 0.0, 3.0, 1.5)

    user_data = [[age, bmi, bp, s1, s2, s3, s4, s5, s6]]

    if st.button('Predict'):
        model = load_model('diabetes_prediction_model.pkl')
        prediction = model.predict(user_data)
        if prediction[0] == 0:
            st.write('The model predicts that the person does not have diabetes.')
        else:
            st.write('The model predicts that the person has diabetes.')

if __name__ == '__main__':
    main()
