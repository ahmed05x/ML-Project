import streamlit as st
import joblib
import numpy as np

st.title("Random Forest Prediction")

model = joblib.load("model.pkl")

f1 = st.number_input("Feature 1")
f2 = st.number_input("Feature 2")
f3 = st.number_input("Feature 3")

if st.button("Predict"):
    prediction = model.predict([[f1,f2,f3]])
    st.write("Prediction:",prediction)