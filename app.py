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
        st.success("✅ Mô hình đã được load thành công!")
    else:
        st.error("❌ LỖI: Model không hợp lệ, kiểm tra lại trained_pipeline.pkl!")
        loaded_pipeline = None
else:
    st.error("⚠️ Model file is missing. Please upload `trained_pipeline.pkl`.")

# Streamlit App
st.title("🏦 Loan Approval Prediction App")

# User Inputs
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

# Predict button
if st.button("📊 Predict"):
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
            "PreviousLoanDefaults": [previous_loan_defaults]
        })

        st.write("🔎 Dữ liệu đầu vào:", input_data)

        # Dự đoán kết quả
        try:
            prediction = loaded_pipeline.predict(input_data)
            if prediction[0] == 1:
                st.success("🎉 Loan Approved!")
            else:
                st.error("❌ Loan Rejected.")
        except Exception as e:
            st.error(f"⚠️ Lỗi khi dự đoán: {e}")
    else:
        st.error("⚠️ Model không khả dụng. Hãy kiểm tra `trained_pipeline.pkl`.")
