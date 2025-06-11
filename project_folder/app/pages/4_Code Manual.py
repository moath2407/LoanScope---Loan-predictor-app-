import streamlit as st
import pandas as pd

code1 = '''import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import pickle





data = pd.read_csv('Loan.csv')
data = data.drop(columns=['ApplicationDate'])  

#print(data.head())
#print(data.describe())
#print(data.info())


#Heatmap of correlation
#data_numerical_columns = data.select_dtypes(include = ['number'])
#plt.figure(figsize = (50,50))
#sns.heatmap(data_numerical_columns.corr(), annot = True)
#plt.show()


input_features = ['AnnualIncome', 'CreditScore', 'EmploymentStatus', 'EducationLevel', 'Experience',
'LoanAmount', 'LoanDuration', 'MaritalStatus', 'NumberOfDependents', 'HomeOwnershipStatus',
'MonthlyDebtPayments', 'CreditCardUtilizationRate', 'NumberOfOpenCreditLines',
'NumberOfCreditInquiries', 'DebtToIncomeRatio', 'BankruptcyHistory', 'LoanPurpose',
'PreviousLoanDefaults', 'PaymentHistory', 'NetWorth', 'MonthlyLoanPayment',
'TotalDebtToIncomeRatio', 'TotalLiabilities']

LoanDF = data[input_features]
#print(LoanDF.describe)


#Handling categorical variables and encoding them
##only encode categorical variables (variables that are based on categories (male, married etc.))
###Ordinal encoding converts categories into binary/trueorfalse variables

categorical_var = ['EmploymentStatus', 'EducationLevel', 'MaritalStatus', 'HomeOwnershipStatus', 'LoanPurpose']
encoded_var = pd.get_dummies(LoanDF, columns=categorical_var, drop_first=True)
#print(encoded_var)

#Standardize using standardscaler()
scaler = StandardScaler()
#Seperate the numerical columns into a seperate list
numerical_cols= [
    'AnnualIncome','CreditScore','Experience','LoanAmount','LoanDuration',
    'NumberOfDependents','MonthlyDebtPayments','CreditCardUtilizationRate','NumberOfOpenCreditLines',
    'NumberOfCreditInquiries','DebtToIncomeRatio','NetWorth',
    'MonthlyLoanPayment', 'TotalDebtToIncomeRatio', 'TotalLiabilities' ]

#The encoded_var now has standardized numerical columns, and encoded categorical columns
encoded_var[numerical_cols] = scaler.fit_transform(encoded_var[numerical_cols])
#print(encoded_var)

#Training the model
X = encoded_var[:] #All features
y_risk = data['RiskScore']
y_loanapproval = data['LoanApproved']

#You should split the data into a risk set and approval set
X_trainrisk, X_testrisk, y_trainrisk, y_testrisk = train_test_split(X, y_risk, test_size = 0.2, random_state = 18)
X_trainapproval, X_testapproval, y_trainapproval, y_testapproval = train_test_split(X, y_loanapproval, test_size = 0.2, random_state = 18)

#Creating a linear model
regressor = LinearRegression()
regressor.fit(X_trainrisk, y_trainrisk)
y_predictrisk = regressor.predict(X_testrisk)

#Evaluate using MSE and R2
print("Mean Squared Error: ", round(mean_squared_error(y_testrisk, y_predictrisk), 2))
print("R2 score: ", round(r2_score(y_testrisk, y_predictrisk), 2))
#Scatter plot
x_axis=range(len(y_testrisk))
plt.scatter(x_axis, y_testrisk, label='Actual Risk Score', color='red')
plt.scatter(x_axis, y_predictrisk, label='Predicted Risk Score', color='blue')
plt.xlabel('Test Data Points')
plt.ylabel('Risk Score')
plt.legend()
plt.show()

#Classification - Gaussian is the simplest to use
from sklearn.naive_bayes import GaussianNB
clf = GaussianNB()
clf.fit(X_trainapproval, y_trainapproval)

from sklearn.metrics import accuracy_score
y_pred = clf.predict(X_testapproval)
print("Accuracy using Gaussian Classification:", accuracy_score(y_testapproval, y_pred))


#Another Classification method - TO COMPARE ACCURACIES - RFC
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators=100)
classifier.fit(X_trainapproval, y_trainapproval)
y_pred_approval = classifier.predict(X_testapproval)
print("Accuracy using RFC:", accuracy_score(y_testapproval, y_pred_approval))

#Another Classification method - TO COMPARE ACCURACIES - KNearest Neighbor
from sklearn.neighbors import KNeighborsClassifier
clf = KNeighborsClassifier()
clf.fit(X_trainapproval, y_trainapproval)
y_pred_2 = clf.predict(X_testapproval)
print("Accuracy using KNN: ", accuracy_score(y_testapproval, y_pred_2))

#Another Classification method - TO COMPARE ACCURACIES - LinearSVC
from sklearn.svm import LinearSVC
clf = LinearSVC()
clf.fit(X_trainapproval, y_trainapproval)
y_pred_3 = clf.predict(X_testapproval)
print("Accuracy usinng LinearSVC: ", accuracy_score(y_testapproval, y_pred_3))


#How to pickle:
with open ('model_regressor.pkl', 'wb') as f:
    pickle.dump(regressor, f)
with open('model_Linearclassifier.pkl', 'wb') as f:
    pickle.dump(clf, f)
with open('model_scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
with open('model_features.pkl', 'wb') as f:
    pickle.dump(list(input_features), f)
with open('model_features_encoded.pkl', 'wb') as f:
    pickle.dump(list(encoded_var), f)
'''

code2 = '''from fastapi import FastAPI
from pydantic import BaseModel
from enum import Enum
import pickle
import pandas as pd
import numpy as np

#Loading all the pickled assets
with open('model_features.pkl', 'rb') as f:
    model_features = pickle.load(f)

with open('model_Linearclassifier.pkl','rb') as f:
    LinearSVC = pickle.load(f)

with open('model_regressor.pkl','rb') as f:
    regressor = pickle.load(f)

with open('model_scaler.pkl','rb') as f:
    scaler = pickle.load(f)

with open('model_features_encoded.pkl','rb') as f:
    encoded = pickle.load(f)

app = FastAPI()

class Loan(BaseModel):
    AnnualIncome: float
    CreditScore: float
    EmploymentStatus: str
    EducationLevel: str
    Experience: float
    LoanAmount: float
    LoanDuration: float
    MaritalStatus: strv
    NumberOfDependents: int
    HomeOwnershipStatus: str
    MonthlyDebtPayments: float
    CreditCardUtilizationRate: float
    NumberOfOpenCreditLines: int
    NumberOfCreditInquiries: int
    DebtToIncomeRatio: float
    BankruptcyHistory: int
    LoanPurpose: str
    PreviousLoanDefaults: int
    PaymentHistory: float
    NetWorth: float
    MonthlyLoanPayment: float
    TotalDebtToIncomeRatio: float
    TotalLiabilities: float

numerical_cols = [
    'AnnualIncome','CreditScore','Experience','LoanAmount','LoanDuration',
    'NumberOfDependents','MonthlyDebtPayments','CreditCardUtilizationRate','NumberOfOpenCreditLines',
    'NumberOfCreditInquiries','DebtToIncomeRatio','NetWorth',
    'MonthlyLoanPayment', 'TotalDebtToIncomeRatio', 'TotalLiabilities' ]

categorical_var = ['EmploymentStatus', 'EducationLevel', 'MaritalStatus', 'HomeOwnershipStatus', 'LoanPurpose']
@app.get("/")
def check_if_it_works():
    return(list(numerical_cols))

@app.post("/predict/apply")
#application:Loan basically tells the API to expect an input in the form of the Loan class
def predict(application:Loan):

    #Model_dump is similar to application.dict()
    input_dict = application.model_dump()
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
    
    return {
        "The risk score is:":(risk_pred),
        "Loan Approval Status:":approval_pred
    }

'''
code3 = '''import requests

# Check if API root is responding
#r = requests.get("http://127.0.0.1:8000/")
#print(r.status_code)
#print(r.json())

# Send POST request with loan application JSON


r_apply = requests.post(
        "http://127.0.0.1:8000/predict/apply",
        json=
{
    "AnnualIncome": 52825,
    "CreditScore": 572,
    "EmploymentStatus": "Employed",
    "EducationLevel": "Bachelor",
    "Experience": 17,
    "LoanAmount": 14135,
    "LoanDuration": 24,
    "MaritalStatus": "Single",
    "NumberOfDependents": 0,
    "HomeOwnershipStatus": "Own",
    "MonthlyDebtPayments": 549,
    "CreditCardUtilizationRate": 0.17613715546323208,
    "NumberOfOpenCreditLines": 2,
    "NumberOfCreditInquiries": 1,
    "DebtToIncomeRatio": 0.2548345044296835,
    "BankruptcyHistory": 0,
    "LoanPurpose": "Education",
    "PreviousLoanDefaults": 0,
    "PaymentHistory": 26,
    "NetWorth": 52931,
    "MonthlyLoanPayment": 731.5003940899489,
    "TotalDebtToIncomeRatio": 0.2908850871572056,
    "TotalLiabilities": 33289
    
}
    )


print(r_apply.json()) '''

st.title("Code Used For The Model:")
st.subheader("\n**The Model**:\n", divider="red")

with st.expander("Press to expand"):
    code = code1
    st.code(code, language='python')

st.markdown("""
This machine learning pipeline processes loan application data to predict both risk scores (regression) and approval decisions (classification). After loading and cleaning the data, it:

    1. Encodes categorical variables (like employment status)

    2. Scales numerical features (like income and credit score)

    3. Splits data for separate risk/approval modeling

    4. Trains a Linear Regression for risk scoring (evaluated with MSE/R²)

    5. Implement classifier types (Naive Bayes, Random Forest, KNN, SVM) for approval (evaluated by accuracy)

    6. Saves all trained models and preprocessing objects for deployment

The system automates loan assessment while comparing algorithm performance, with visualizations (like scatter plots) to validate predictions. Key techniques include feature engineering, standardization, and model persistence via pickle files.
            """)


st.subheader("\n**FastAPI implementation**:\n", divider="blue")

with st.expander("Press to expand:"):
    code = code2
    st.code(code, language = "python")

st.markdown("""
This FastAPI application serves as a loan prediction API that:
1. Loads pre-trained models (LinearRegression for risk scores, LinearSVC for approvals) and preprocessing objects (scaler, feature lists) from pickle files

2. Defines endpoints:

    A. GET "/": Returns numerical features list (health check)

    B. POST "/predict/apply": Accepts loan application data (23 features via Pydantic model)

3. Processes requests by:

    A. Encoding categorical variables (one-hot)

    B. Aligning features with training structure

    C. Standardizing numerical values

    D. Returning both risk score (regression) and approval decision (classification)

Key features:

    A. Input validation via Loan model class

    B. Consistent preprocessing with training pipeline

    C. Returns denials as "Deny!" and approvals as "Approve!"

The purpose of this FastAPI application is to deploy the trained ML system as a microservice for real-time loan decisions.
""")

st.subheader("\n**Testing The API Endpoint**:\n", divider="green")

with st.expander("Press to expand:"):
    code = code3
    st.code(code, language = "python")

st.markdown(
    """
This code snippet is a manual test for your FastAPI loan prediction endpoint. It performs two main tasks:

1. It checks if the API root endpoint is available and responsive by sending a GET request. This helps verify that the server is up and running and can handle incoming requests.

2. It sends a POST request to the /predict/apply endpoint with a JSON payload that simulates a user's loan application. This payload contains all 23 required input fields for the model. When the request is received, the FastAPI backend processes the input by applying the same feature encoding and scaling steps used during training. Then, it uses the loaded machine learning models to predict the loan risk score and determine whether the loan should be approved or denied.

This test confirms several important aspects: _that your FastAPI application is running properly, the machine learning models are correctly loaded from their saved state, the input processing pipeline is consistent, and the prediction endpoint works correctly from end to end_ — from receiving data, processing it, to returning meaningful prediction results.
""")