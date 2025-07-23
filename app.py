import streamlit as st
import pandas as pd
import joblib

# Load model and columns
model = joblib.load("model.joblib")
model_columns = joblib.load("columns.joblib")

# Streamlit UI
st.title("Employee Salary Predictor")
st.write("Predict whether income >50K or <=50K based on user details.")

# Input form
with st.form("user_input"):
    age = st.number_input("Age", min_value=18, max_value=100, value=30)
    workclass = st.selectbox("Workclass", ["Private", "Self-emp-not-inc", "Self-emp-inc", "Federal-gov", "Local-gov", "State-gov", "Without-pay", "Never-worked"])
    education = st.selectbox("Education", ["Bachelors", "HS-grad", "11th", "Masters", "9th", "Some-college", "Assoc-acdm", "Assoc-voc", "7th-8th", "Doctorate", "Prof-school", "5th-6th", "10th", "1st-4th", "Preschool", "12th"])
    marital_status = st.selectbox("Marital Status", ["Married-civ-spouse", "Divorced", "Never-married", "Separated", "Widowed", "Married-spouse-absent", "Married-AF-spouse"])
    occupation = st.selectbox("Occupation", ["Tech-support", "Craft-repair", "Other-service", "Sales", "Exec-managerial", "Prof-specialty", "Handlers-cleaners", "Machine-op-inspct", "Adm-clerical", "Farming-fishing", "Transport-moving", "Priv-house-serv", "Protective-serv", "Armed-Forces"])
    relationship = st.selectbox("Relationship", ["Wife", "Own-child", "Husband", "Not-in-family", "Other-relative", "Unmarried"])
    race = st.selectbox("Race", ["White", "Asian-Pac-Islander", "Amer-Indian-Eskimo", "Other", "Black"])
    gender = st.selectbox("Gender", ["Male", "Female"])
    hours_per_week = st.slider("Hours per Week", min_value=1, max_value=99, value=40)
    native_country = st.selectbox("Native Country", ["United-States", "Mexico", "Philippines", "Germany", "Canada", "India", "England", "China", "Other"])

    submitted = st.form_submit_button("Predict")

if submitted:
    input_dict = {
        "age": [age],
        "workclass": [workclass],
        "education": [education],
        "marital-status": [marital_status],
        "occupation": [occupation],
        "relationship": [relationship],
        "race": [race],
        "gender": [gender],
        "hours-per-week": [hours_per_week],
        "native-country": [native_country]
    }

    input_df = pd.DataFrame(input_dict)
    input_encoded = pd.get_dummies(input_df)

    # Align columns with training
    for col in model_columns:
        if col not in input_encoded:
            input_encoded[col] = 0

    input_encoded = input_encoded[model_columns]

    prediction = model.predict(input_encoded)[0]
    result = ">50K" if prediction == 1 else "<=50K"
    st.success(f"Predicted Income: {result}")

# Footer credit
st.markdown("---")
st.markdown("*by Harshalee Malu as part of Edunet Foundation - IBM SkillsBuild AIML Internship*")
