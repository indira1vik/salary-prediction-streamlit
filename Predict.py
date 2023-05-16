import streamlit as st
import pickle
import numpy as np

def load_model():
    with open('Model_Pickle_File.pkl','rb') as file:
        data = pickle.load(file)
    return data

data = load_model()

model_load = data["model"]
le_country = data["le_country"]
le_education = data["le_education"]

def show_predict_page():
    st.title("Salary Prediction")
    st.write("### Please fill the details to predict the salary")

    countries = (
        "United States",
        "India",
        "United Kingdom",
        "Germany",
        "Canada",
        "Brazil",
        "France",
        "Netherlands",
        "Poland",
        "Australia",
        "Spain",
        "Italy",
        "Russian Federation",
        "Sweden"
    )

    education = (
        "Bachelor's degree", 
        "Master's degree", 
        "Less than a Bachelors",
        "Post grad"
    )

    country = st.selectbox("#### Select Country",countries)
    edu = st.selectbox("#### Select Education",education)

    experience = st.slider("#### Years Of Experience",0,50,3)

    clicked = st.button("Calculate Salary")
    if clicked:
        X = np.array([[country, edu, experience ]])
        X[:, 0] = le_country.transform(X[:,0])
        X[:, 1] = le_education.transform(X[:,1])
        X = X.astype(float)
        salary = model_load.predict(X)
        
        st.subheader(f"The estimated salary is ${salary[0]:.2f}")
