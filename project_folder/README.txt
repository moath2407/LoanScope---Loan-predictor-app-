-------------------------------------------------------------
Welcome to LoanScope!

LoanScope is a machine learning-powered web application designed to evaluate loan applications by predicting both a risk score and an approval status. Built using a combination of FastAPI for the backend and Streamlit for the frontend, this tool simulates a realistic financial risk analysis environment. Users can input 23 variables related to their financial and personal background to receive an instant assessment.

The core aim of this project is to apply machine learning to support decision-making in the lending process. With real-time predictions, visual feedback, and performance insights, LoanScope not only helps users understand their loan eligibility but also provides transparency into how each decision is made.

The model behind LoanScope uses Linear Regression to calculate a numerical risk score, while a Linear Support Vector Classifier (LinearSVC) is employed to determine approval decisions. The model has been trained on a real-world dataset (20,000 real-world loan applications) and preprocessed with scaling and encoding to ensure consistent and accurate results.

This app also showcases metrics like model accuracy and visual comparisons between actual and predicted risk scores, while highlighting challenges such as class imbalance, model drift, and misclassification risk.

Whether you're testing an API endpoint, exploring machine learning deployment, or simulating a loan scenario, LoanScope serves as a complete end-to-end ML web solution.

-------------------------------------------------------------

Project Structure:

project_folder:
	api--------|--implem--|
			|---model_features.pkl---|
			|---model_Linearclassifier.pkl---|
			|---model_regressor.pkl---|
			|---model_scaler.pkl---|
	
	app--------|--pages--|
			|---1_Home---|
			|---2_Predict---|
			|---3_Model Insights---|
			|---4_Code Manual---|
		   |--LineGraph.png--|
		   |--Loan--|
		   |--Loan_Predictions_Report--|
		   |--Main.py--|

	README.txt
	requirements.txt

-------------------------------------------------------------

Prerequisites:

numpy
pandas
scikit-learn
seaborn
matplotlib
pickle
fastapi
uvicorn
enum
PIL
streamlit
requests
pydantic

--------------------------------------------------------------



