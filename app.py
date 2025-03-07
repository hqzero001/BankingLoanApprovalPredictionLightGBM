import streamlit as st
import pandas as pd
import os
import joblib

# Load trained model
MODEL_PATH = 'trained_pipeline.pkl'
loaded_pipeline = None

if os.path.exists(MODEL_PATH):
    loaded_pipeline = joblib.load(MODEL_PATH)
    if hasattr(loaded_pipeline, "predict"):
        st.success("Model Load Successfully !")
    else:
        st.error("Error: Something went wrong with the model, please check `trained_pipeline.pkl`.")
        loaded_pipeline = None
else:
    st.error("⚠️ Model file is missing, please upload `trained_pipeline.pkl`.")

# Streamlit App
st.title("Loan Approval Prediction App")

col1, col2, col3 = st.columns(3, gap = "medium")

# User Inputs
with col1:
    employment_status = st.selectbox("Employment Status", ["Employed", "Unemployed", "Self-Employed"])
    education_level = st.selectbox("Education Level", ["High School", "Bachelor", "Master", "Associate", "Doctorate"])
    marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced", "Widowed"])
    number_of_dependents = st.number_input("Number of Dependents", min_value=0, max_value=10, value=0)
    home_ownership_status = st.selectbox("Home Ownership Status", ["Own", "Rent", "Mortgage", "Other"])
    loan_duration = st.number_input("Loan Duration (months)", min_value=6, max_value=360, value=36)
    loan_purpose = st.selectbox("Loan Purpose", ["Home", "Debt Consolidation", "Education", "Auto", "Other"])
    number_of_open_credit_lines = st.number_input("Number of Open Credit Lines", min_value=0, value=5)
    number_of_credit_inquiries = st.number_input("Number of Credit Inquiries", min_value=0, value=2)
    bankruptcy_history = st.selectbox("Bankruptcy History (0 - NO, 1 - YES)", [0, 1])
    previous_loan_defaults = st.selectbox("Previous Loan Defaults (0 - NO, 1 - YES)", [0, 1])

with col2:
    net_worth = st.number_input("Net Worth", min_value=0, value=10000)
    loan_amount = st.number_input("Loan Amount", min_value=0, value=5000)
    total_assets = st.number_input("Total Assets", min_value=0, value=50000)
    job_tenure = st.number_input("Job Tenure (years)", min_value=0, value=5)
    debt_to_income_ratio = st.number_input("Debt to Income Ratio", min_value=0.0, value=0.2)
    total_debt_to_income_ratio = st.number_input("Total Debt to Income Ratio", min_value=0.0, max_value=1.0, value=0.3)
    credit_card_utilization_rate = st.number_input("Credit Card Utilization Rate", min_value=0.0, max_value=1.0, value=0.5)
    utility_bills_payment_history = st.number_input("Utility Bills Payment History", min_value=0.0, max_value=1.0, value=0.8)
    total_liabilities = st.number_input("Total Liabilities", min_value=0, value=20000)
    experience = st.number_input("Experience (years)", min_value=0, value=10)
    savings_account_balance = st.number_input("Savings Account Balance", min_value=0, value=10000)

with col3:
    monthly_income = st.number_input("Monthly Income", min_value=0.0, value=4000.0)
    monthly_loan_payment = st.number_input("Monthly Loan Payment", min_value=0.0, value=500.0)
    interest_rate = st.number_input("Interest Rate", min_value=0.0, max_value=1.0, value=0.05)
    credit_score = st.number_input("Credit Score", min_value=300, max_value=750, value=650)
    base_interest_rate = st.number_input("Base Interest Rate", min_value=0.0, max_value=1.0, value=0.03)
    payment_history = st.number_input("Payment History", min_value=0, value=5)
    age = st.number_input("Age", min_value=18, max_value=100, value=30)
    monthly_debt_payments = st.number_input("Monthly Debt Payments", min_value=0, value=1000)
    annual_income = st.number_input("Annual Income", min_value=0, value=50000)
    length_of_credit_history = st.number_input("Length of Credit History (years)", min_value=0, value=10)
    checking_account_balance = st.number_input("Checking Account Balance", min_value=0, value=5000)

# Predict button
if st.button("Predict!"):
    if loaded_pipeline is not None:
        input_data = pd.DataFrame({
            "EmploymentStatus": [employment_status],
            "EducationLevel": [education_level],
            "LoanDuration": [loan_duration],
            "MaritalStatus": [marital_status],
            "NumberOfDependents": [number_of_dependents],
            "HomeOwnershipStatus": [home_ownership_status],
            "NumberOfOpenCreditLines": [number_of_open_credit_lines],
            "NumberOfCreditInquiries": [number_of_credit_inquiries],
            "BankruptcyHistory": [bankruptcy_history],
            "LoanPurpose": [loan_purpose],
            "PreviousLoanDefaults": [previous_loan_defaults],
            "NetWorth": [net_worth],
            "LoanAmount": [loan_amount],
            "TotalAssets": [total_assets],
            "JobTenure": [job_tenure],
            "DebtToIncomeRatio": [debt_to_income_ratio],
            "TotalDebtToIncomeRatio": [total_debt_to_income_ratio],
            "CreditCardUtilizationRate": [credit_card_utilization_rate],
            "UtilityBillsPaymentHistory": [utility_bills_payment_history],
            "TotalLiabilities": [total_liabilities],
            "Experience": [experience],
            "SavingsAccountBalance": [savings_account_balance],
            "MonthlyIncome": [monthly_income],
            "MonthlyLoanPayment": [monthly_loan_payment],
            "InterestRate": [interest_rate],
            "CreditScore": [credit_score],
            "BaseInterestRate": [base_interest_rate],
            "PaymentHistory": [payment_history],
            "Age": [age],
            "MonthlyDebtPayments": [monthly_debt_payments],
            "AnnualIncome": [annual_income],
            "LengthOfCreditHistory": [length_of_credit_history],
            "CheckingAccountBalance": [checking_account_balance]
        })


        st.write("Input:", input_data)

        # Dự đoán kết quả
        try:
            prediction = loaded_pipeline.predict(input_data)
            if prediction[0] == 1:
                st.success("Loan Approved!")
            else:
                st.error("Loan Rejected.")
        except Exception as e:
            st.error(f"Something went wrong: {e}")
    else:
        st.error("Error: Something went wrong with the model, please check `trained_pipeline.pkl`.")
