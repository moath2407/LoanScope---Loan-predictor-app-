import streamlit as st
import pandas as pd
from PIL import Image
import os
st.set_page_config(layout="wide")
st.title("Model Evaluation & Data Overview section")
st.write("""Welcome to the Model Evaluation & Data Overview section of our loan application risk assessment tool.

Here, you can explore:
""")

st.subheader("\n**The Original Loan Dataset**:\n", divider="red")

import os
# Dynamically get the correct path
script_dir = os.path.dirname(__file__)  # Gets the folder where the script is
csv_path = os.path.join(script_dir, "Loan.csv")  # If CSV is in the same folder
LoanDF = pd.read_csv(csv_path)
st.write(LoanDF)

st.write("The dataset above contains 20,000 different loan applications that were used to train the ML model.")
st.markdown("""
_How It Works:_
1. **Data Preprocessing**:
   - Categorical inputs are encoded using one-hot encoding.
   - Numerical inputs are standardized using a trained scaler.
            
2. **Model Training**:
   - **Linear Regression** is used for continuous risk scoring.
   - **LinearSVC** is used for binary loan approval classification.
   - Other models (Naive Bayes, KNN, Random Forest) were evaluated, but LinearSVC performed best.

3. **Model Outputs**:
   - A **risk score** (e.g., 0.48)
   - A **loan decision**: “Approve” or “Deny”
            """)
st.subheader("\n_**The Performance Metrics**:_\n", divider="blue")


st.write("To measure the accuracy of the ML model, we randomly picked 31 loan applications from the original loan dataset, and ran it through the model.")
st.write("""
1. **Predicted Risk Score**: The risk score predicted by the model for each loan application. Values range from approximately 36.78 to 74.80.

2. **Predicted Status**: The loan decision based on the predicted risk score ("Deny!" or `"Approve!"`).

3. **Actual Risk Score**: The true risk score for the loan application, as determined by actual outcomes. Values range from approximately 36 to 73.

4. **Accuracy Rating (%)**: A calculated metric comparing the predicted and actual risk scores.

5. **Actual Status**: The true loan decision ("Denied" or "Approved"), based on the actual risk score (Provided in the Original loan dataset).""")
st.title("📄 Excel Sheet Preview")
script_dir1 = os.path.dirname(__file__)  # Gets the folder where the script is
xlsx_path = os.path.join(script_dir, "Miniloan.xlsx")  # If CSV is in the same folder
LoanminiDF = pd.read_excel(xlsx_path) 
st.write(LoanminiDF)


st.write("To visualize the variation between the **Predicted Risk Score** vs **Actual Risk Score**:")
image = Image.open("pages/LineGraph.png")
st.write(image)

st.subheader("\n_**Potential Issues With The Model**:_\n", divider="grey")

st.markdown("""
            1. Class Imbalance:
The dataset may have far more ~"Denied"~ or `"Approved"` cases, causing the model to favor the majority class.
            
2. Outcome Mismatches:
Row 31 shows a prediction errors (false denial), which may indicate model inaccuracies or feature-normalization misalignment.

3. Model Drift:
The model is static after training, which may lead to scaling/threshold issues, leading to potential poor model performance.
            """)
# Recommendations
st.markdown("---")
st.subheader("Recommended Fixes:")
st.markdown("""
1. **Review Mismatches**: Investigate rows where predictions failed (e.g., false denials).  
            
2. **Scheduled Retraining**: Periodically retrain the model with new, recent data.
            
3. **Resampling Techniques**: Undersample the majority class (_Denied_) if oversampling inflates training size too much.
""")
