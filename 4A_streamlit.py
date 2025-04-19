import streamlit as st
import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder

@st.cache_resource
def load_model(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

def encode(user_input, reference_df):
    df = pd.DataFrame(user_input)
    cat_cols = reference_df.select_dtypes(include='object').columns
    for col in cat_cols:
        le = LabelEncoder()
        values = list(reference_df[col].unique()) + list(df[col].unique())
        le.fit(values)
        df[col] = le.transform(df[col])
    return df

reference_df = pd.read_csv("Dataset_A_loan.csv").drop("loan_status", axis=1)
model = load_model("XGB_model_OOP.pkl")

st.title("Predict Loan Status")
st.markdown("This app uses **13 inputs** to predict the loan status.")

user_input = {
    'person_age': st.number_input("Person Age", min_value=20, max_value=100),
    'person_gender': st.selectbox("Gender", ['male', 'female']),
    'person_education': st.selectbox("Education", ['High School', 'Bachelor', 'Associate', 'Master', 'Doctorate']),
    'person_income': st.number_input("Monthly Income", value=0),
    'person_emp_exp': st.number_input("Employment Experience in Years", min_value=0, max_value=100),
    'person_home_ownership': st.selectbox("Home Ownership", ['RENT', 'OWN', 'MORTGAGE', 'OTHER']),
    'loan_amnt': st.number_input("Loan Amount", value=0),
    'loan_intent': st.selectbox("Loan Intent", ['EDUCATION', 'MEDICAL', 'VENTURE', 'PERSONAL', 'DEBTCONSOLIDATION', 'HOMEIMPROVEMENT']),
    'loan_int_rate': st.number_input("Interest Rate", value=12.5),
    'loan_percent_income': st.number_input("Loan % of Income", value=0.2),
    'cb_person_cred_hist_length': st.number_input("Credit History Length", value=3),
    'credit_score': st.number_input("Credit Score", value=0),
    'previous_loan_defaults_on_file': st.selectbox("Previous Loan Default?", ['Yes', 'No'])
}

if st.button("Predict"):
    input_df = encode([user_input], reference_df)
    prediction = model.predict(input_df)[0]
    result = "Request Accepted" if prediction == 1 else "Request Rejected"
    st.success(f"Prediction: {result}")

if st.checkbox("Run Test Cases"):
    st.markdown("## Test Case 1")
    st.write(
    'person_age': 23.0,
    'person_gender': 'female',
    'person_education': 'Bachelor',
    'person_income': 79753.0,
    'person_emp_exp': 0,
    'person_home_ownership': 'RENT',
    'loan_amnt': 35000.0,
    'loan_intent': 'MEDICAL',
    'loan_int_rate': 15.23,
    'loan_percent_income': 0.44,
    'cb_person_cred_hist_length': 2.0,
    'credit_score': 675,
    'previous_loan_defaults_on_file': 'No')
    test1 = [ {
    'person_age': 23.0,
    'person_gender': 'female',
    'person_education': 'Bachelor',
    'person_income': 79753.0,
    'person_emp_exp': 0,
    'person_home_ownership': 'RENT',
    'loan_amnt': 35000.0,
    'loan_intent': 'MEDICAL',
    'loan_int_rate': 15.23,
    'loan_percent_income': 0.44,
    'cb_person_cred_hist_length': 2.0,
    'credit_score': 675,
    'previous_loan_defaults_on_file': 'No'
    }]
    input_test1 = encode(test1, reference_df)
    pred1 = model.predict(input_test1)[0]
    st.write("Test Case 1 Prediction:", "Request Accepted" if pred1 == 1 else "Request Rejected")
    
    st.markdown("### Test Case 2")
    st.write(
    'person_gender': 'female',
    'person_education': 'High School',
    'person_income': 12282.0,
    'person_emp_exp': 0,
    'person_home_ownership': 'OWN',
    'loan_amnt': 1000.0,
    'loan_intent': 'EDUCATION',
    'loan_int_rate': 11.14,
    'loan_percent_income': 0.08,
    'cb_person_cred_hist_length': 2.0,
    'credit_score': 504,
    'previous_loan_defaults_on_file': 'Yes')
        
    test2 =[{'person_age': 21.0,
    'person_gender': 'female',
    'person_education': 'High School',
    'person_income': 12282.0,
    'person_emp_exp': 0,
    'person_home_ownership': 'OWN',
    'loan_amnt': 1000.0,
    'loan_intent': 'EDUCATION',
    'loan_int_rate': 11.14,
    'loan_percent_income': 0.08,
    'cb_person_cred_hist_length': 2.0,
    'credit_score': 504,
    'previous_loan_defaults_on_file': 'Yes'
    }]
    input_test2 = encode(test2, reference_df)
    pred2 = model.predict(input_test2)[0]
    st.write("Test Case 2 Prediction:", "Request Accepted" if pred2 == 1 else "Request Rejected")
