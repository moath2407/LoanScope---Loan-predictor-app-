import streamlit as st
import pandas as pd

st.set_page_config(page_title="LoanScope", page_icon=":moneybag:")

st.title(":heavy_dollar_sign: Loan Risk Scoring & Approval App:heavy_dollar_sign:")
st.write("\nWelcome to the **Loan Risk Scoring & Approval App**, a machine learning-powered tool designed to evaluate loan applications using a combination of predictive analytics and classification models.")



st.subheader("\n_**What This App Does**:_\n", divider="gray")
st.write("This app performs two key tasks:\n **Risk Scoring**: Predicts a numeric risk score (0–100) representing the likelihood of default based on 23 financial and personal criteria.- **Loan Approval Prediction**: Classifies the loan as either **Approved** or **Denied** using trained classifiers.")

st.subheader("\n_**How It Works:**_", divider="blue")

st.write("""- A machine learning model was trained on historical loan data using techniques such as:
  - **StandardScaler** for normalization
  - **One-hot encoding** for categorical variables
  - **Linear Regression** for risk score prediction
  - Multiple classification algorithms (e.g., GaussianNB, Random Forest, KNN, LinearSVC) for loan approval
- The model takes into account 23 features including income, credit score, debt ratios, loan purpose, and more.
- A FastAPI backend handles predictions, and this Streamlit frontend allows for easy data input and visualization.""")

st.subheader("\n_**Technologies Used:**_", divider="blue")
st.write("""
- **Python**, **scikit-learn**, **pandas**, **matplotlib**, **seaborn**
- **FastAPI** for the REST API backend
- **Streamlit** for this interactive web interface

Use the **Predict** page to get started!
""")