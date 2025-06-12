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
    model_feature = {
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
numerical_cols = [
    'AnnualIncome','CreditScore','Experience','LoanAmount','LoanDuration',
    'NumberOfDependents','MonthlyDebtPayments','CreditCardUtilizationRate','NumberOfOpenCreditLines',
    'NumberOfCreditInquiries','DebtToIncomeRatio','NetWorth',
    'MonthlyLoanPayment', 'TotalDebtToIncomeRatio', 'TotalLiabilities' ]

categorical_var = ['EmploymentStatus', 'EducationLevel', 'MaritalStatus', 'HomeOwnershipStatus', 'LoanPurpose']


with open('model_features.pkl', 'rb') as f:
    model_features = pickle.load(f)

with open('model_Linearclassifier.pkl','rb') as f:
    LinearSVC = pickle.load(f)

with open('model_regressor.pkl','rb') as f:
    regressor = pickle.load(f)

with open('model_scaler.pkl','rb') as f:
    scaler = pickle.load(f)


if submitted:
    
    input_dict = model_feature.model_dump()
    input_df = pd.DataFrame([input_dict])
    
    # Encode categorical variables (same as during training)
    encoded_input = pd.get_dummies(input_df, columns=categorical_var, drop_first=True)
    
    # Ensure all expected columns are present (add missing ones with 0)
    for col in model_features:
        if col not in encoded_input.columns:
            encoded_input[col] = 0
    
    # Reorder columns to match training data
    encoded_input = encoded_input[model_features]
    
    # Standardize numerical columns (transform only, no fitting)
    encoded_input[numerical_cols] = scaler.transform(encoded_input[numerical_cols])
    
    # Predict the risk
    risk_pred = round(regressor.predict(encoded_input)[0], 3)
    
    # Predict the approval
    approval_pred = int(LinearSVC.predict(encoded_input)[0])
    
    if (approval_pred == 0):
        approval_pred = str("Deny!")
    else:
        approval_pred = str("Approve!")
    
        st.markdown("""The risk score is:":{risk_pred},
        "Loan Approval Status:":{approval_pred}""")
    
else:
    st.status("Prediction Failed.")

#This should return the risk score and prediction, ONLY USE IF YOU CAN DEPLOY AN API
#if submitted:
#    r_apply_1 = requests.post(
#        "http://127.0.0.1:8000/predict/apply", json = model_features
#    )

#    st.status("Prediction has been completed!")
#    print(r_apply_1.json())
#    st.write(r_apply_1.json())
#else: 
#    st.status("Waiting For Submission")
