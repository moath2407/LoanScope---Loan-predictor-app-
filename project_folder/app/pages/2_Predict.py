import streamlit as st
import pandas as pd
import requests
st.set_page_config(layout="wide")
st.title("🔢 Loan Prediction")

st.markdown("""
Use this tool to evaluate a loan application by entering the applicant's details.  
The model will return:
- A **risk score** (indicating likelihood of default), and  
- A **loan approval decision** (Approved or Denied) based on predictive analysis.

Fill in the form below to get started.
""")

#User has to fill the following form to be able to predict
with st.form("loan_application"):
    AnnualIncome = st.number_input("Annual Income", min_value=0.0)
    CreditScore = st.number_input("Credit Score", min_value=0.0, max_value=850.0)
    EmploymentStatus = st.selectbox("Employment Status", ["Employed", "Unemployed", "Self-Employed", "Student"])
    EducationLevel = st.selectbox("Education Level", ["High School", "Bachelor", "Master", "PhD"])
    Experience = st.number_input("Years of Experience", min_value=0.0)

    LoanAmount = st.number_input("Loan Amount", min_value=0.0)
    LoanDuration = st.number_input("Loan Duration (months)", min_value=1.0)
    MaritalStatus = st.selectbox("Marital Status", ["Single", "Married", "Divorced", "Widowed"])
    NumberOfDependents = st.number_input("Number of Dependents", min_value=0, step=1)
    HomeOwnershipStatus = st.selectbox("Home Ownership Status", ["Rent", "Own", "Mortgage", "Other"])

    MonthlyDebtPayments = st.number_input("Monthly Debt Payments", min_value=0.0)
    CreditCardUtilizationRate = st.number_input("Credit Card Utilization Rate", min_value=0.0, max_value=1.0)
    NumberOfOpenCreditLines = st.number_input("Number of Open Credit Lines", min_value=0, step=1)
    NumberOfCreditInquiries = st.number_input("Number of Credit Inquiries", min_value=0, step=1)
    DebtToIncomeRatio = st.number_input("Debt-to-Income Ratio", min_value=0.0, max_value=1.0)

    BankruptcyHistory = st.selectbox("Bankruptcy History", [0, 1])
    LoanPurpose = st.selectbox("Loan Purpose", ["Education", "Home", "Business", "Personal", "Medical", "Other"])
    PreviousLoanDefaults = st.selectbox("Previous Loan Defaults", [0, 1])
    PaymentHistory = st.number_input("Payment History (months on-time)", min_value=0.0)
    NetWorth = st.number_input("Net Worth", min_value=0.0)

    MonthlyLoanPayment = st.number_input("Monthly Loan Payment", min_value=0.0)
    TotalDebtToIncomeRatio = st.number_input("Total Debt-to-Income Ratio", min_value=0.0, max_value=1.0)
    TotalLiabilities = st.number_input("Total Liabilities", min_value=0.0)

    submitted = st.form_submit_button("Submit & Predict!")

#Checks if submitted and creates a model_features variable to send to the API
    model_features = {
        "AnnualIncome": AnnualIncome,
        "CreditScore": CreditScore,
        "EmploymentStatus": EmploymentStatus,
        "EducationLevel": EducationLevel,
        "Experience": Experience,
        "LoanAmount": LoanAmount,
        "LoanDuration": LoanDuration,
        "MaritalStatus": MaritalStatus,
        "NumberOfDependents": NumberOfDependents,
        "HomeOwnershipStatus": HomeOwnershipStatus,
        "MonthlyDebtPayments": MonthlyDebtPayments,
        "CreditCardUtilizationRate": CreditCardUtilizationRate,
        "NumberOfOpenCreditLines": NumberOfOpenCreditLines,
        "NumberOfCreditInquiries": NumberOfCreditInquiries,
        "DebtToIncomeRatio": DebtToIncomeRatio,
        "BankruptcyHistory": BankruptcyHistory,
        "LoanPurpose": LoanPurpose,
        "PreviousLoanDefaults": PreviousLoanDefaults,
        "PaymentHistory": PaymentHistory,
        "NetWorth": NetWorth,
        "MonthlyLoanPayment": MonthlyLoanPayment,
        "TotalDebtToIncomeRatio": TotalDebtToIncomeRatio,
        "TotalLiabilities": TotalLiabilities
    }

#This should return the risk score and prediction
if submitted:
    r_apply_1 = requests.post(
        "http://127.0.0.1:8000/predict/apply", json = model_features
    )

    st.status("Prediction has been completed!")
    print(r_apply_1.json())
    st.write(r_apply_1.json())
else: 
    st.status("Waiting For Submission")
