import streamlit as st
import joblib
import pandas as pd
import numpy as np

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Loan Default Risk Predictor",
    page_icon="💰",
    layout="wide"
)

# ---------------- LOAD ARTIFACTS ----------------
@st.cache_resource
def load_artifacts():
    model = joblib.load("xgb_default_model.pkl")
    scaler = joblib.load("scaler.pkl")
    label_encoders = joblib.load("label_encoders.pkl")
    model_columns = joblib.load("model_columns.pkl")
    threshold = joblib.load("threshold.pkl")
    return model, scaler, label_encoders, model_columns, threshold

try:
    model, scaler, label_encoders, model_columns, threshold = load_artifacts()
except Exception as e:
    st.error("❌ Required model files are missing or corrupted.")
    st.error(str(e))
    st.stop()

# ---------------- TITLE ----------------
st.title("🏦 Loan Default Risk Predictor")
st.markdown(
    """
    This application predicts **loan default risk** using a Machine Learning model.  
    ⚠️ Educational demo only.
    """
)

# ---------------- SIDEBAR ----------------
st.sidebar.header("📊 Model Info")
st.sidebar.metric("Model", "XGBoost")
st.sidebar.metric("Threshold", f"{threshold:.2f}")
st.sidebar.metric("Recall", "~88%")

# ---------------- INPUT UI ----------------
col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", 18, 100, 35)
    income = st.number_input("Annual Income", 0, 500000, 50000, step=1000)
    education = st.selectbox("Education", label_encoders["Education"].classes_)
    marital_status = st.selectbox("Marital Status", label_encoders["MaritalStatus"].classes_)
    has_dependents = st.selectbox("Has Dependents", label_encoders["HasDependents"].classes_)

with col2:
    loan_amount = st.number_input("Loan Amount", 0, 500000, 25000, step=1000)
    credit_score = st.number_input("Credit Score", 300, 850, 650)
    months_employed = st.number_input("Months Employed", 0, 600, 24)
    num_credit_lines = st.number_input("Number of Credit Lines", 0, 20, 3)

col3, col4 = st.columns(2)

with col3:
    employment_type = st.selectbox("Employment Type", label_encoders["EmploymentType"].classes_)
    has_mortgage = st.selectbox("Has Mortgage", label_encoders["HasMortgage"].classes_)
    loan_purpose = st.selectbox("Loan Purpose", label_encoders["LoanPurpose"].classes_)
    has_cosigner = st.selectbox("Has Co-Signer", label_encoders["HasCoSigner"].classes_)

with col4:
    interest_rate = st.number_input("Interest Rate (%)", 0.0, 30.0, 7.5, step=0.1)
    loan_term = st.selectbox("Loan Term (months)", [12, 24, 36, 48, 60])
    dti_ratio = st.number_input("Debt-to-Income Ratio", 0.0, 1.0, 0.35, step=0.01)

# ---------------- PREDICTION ----------------
if st.button("🔮 Predict Risk", use_container_width=True):
    with st.spinner("Evaluating loan risk..."):

        # 1️⃣ Create input dataframe
        input_df = pd.DataFrame({
            "Age": [age],
            "Income": [income],
            "LoanAmount": [loan_amount],
            "CreditScore": [credit_score],
            "MonthsEmployed": [months_employed],
            "NumCreditLines": [num_credit_lines],
            "InterestRate": [interest_rate],
            "LoanTerm": [loan_term],
            "DTIRatio": [dti_ratio],
            "Education": [education],
            "EmploymentType": [employment_type],
            "MaritalStatus": [marital_status],
            "HasMortgage": [has_mortgage],
            "HasDependents": [has_dependents],
            "LoanPurpose": [loan_purpose],
            "HasCoSigner": [has_cosigner],
        })

        # 2️⃣ Apply label encoders
        for col, le in label_encoders.items():
            if col in input_df.columns:
                input_df[col] = le.transform(input_df[col].astype(str))

        # 3️⃣ Align columns
        input_df = input_df.reindex(columns=model_columns, fill_value=0)

        # 4️⃣ Scale numeric features
        num_cols = scaler.feature_names_in_
        input_df[num_cols] = scaler.transform(input_df[num_cols])

        # 5️⃣ Predict
        probability = model.predict_proba(input_df)[0, 1]
        prediction = int(probability >= threshold)

        # ---------------- RESULTS ----------------
        st.divider()
        st.subheader("📊 Prediction Result")

        c1, c2, c3 = st.columns(3)

        with c1:
            st.metric("Default Probability", f"{probability*100:.1f}%")

        with c2:
            st.metric("Risk Level", "🔴 HIGH RISK" if prediction else "🟢 LOW RISK")

        with c3:
            st.metric("Model Decision", "Reject" if prediction else "Approve")

        # ---------------- RULE CHECKS ----------------
        st.subheader("🔍 Risk Indicators (Rule-based)")
        if credit_score < 600:
            st.warning("Low credit score")
        if dti_ratio > 0.43:
            st.warning("High debt-to-income ratio")
        if months_employed < 12:
            st.warning("Short employment history")
        if loan_amount > income * 0.5:
            st.warning("Loan amount high vs income")

        if prediction == 0:
            st.success("No major red flags detected.")

# ---------------- FOOTER ----------------
st.divider()
st.caption("Educational demo | ML predictions are probabilistic, not decisions.")
