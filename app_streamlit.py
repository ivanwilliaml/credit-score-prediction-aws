import json
import os

import boto3
import streamlit as st
from botocore.exceptions import ClientError, NoCredentialsError


ENDPOINT_NAME = os.environ.get("ENDPOINT_NAME", "credit-score-endpoint")
REGION = os.environ.get("AWS_REGION", "us-east-1")

CLASS_NAMES = ["Good", "Poor", "Standard"]

MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August"]
OCCUPATIONS = ["Scientist", "Teacher", "Engineer", "Entrepreneur", "Developer", "Lawyer",
               "Media_Manager", "Doctor", "Journalist", "Manager", "Accountant", "Musician",
               "Mechanic", "Writer", "Architect", "Unknown"]
LOAN_TYPES = ["Not Specified", "Auto Loan", "Credit-Builder Loan", "Personal Loan",
              "Home Equity Loan", "Mortgage Loan", "Student Loan",
              "Debt Consolidation Loan", "Payday Loan"]


@st.cache_resource
def get_runtime_client():
    return boto3.client("sagemaker-runtime", region_name=REGION)


# Kirim 1 baris fitur ke SageMaker endpoint (bukan load model langsung, beda dari local_deployment/app.py)
def invoke_endpoint(features: list) -> dict:
    runtime = get_runtime_client()
    payload = {"instances": [features]}
    response = runtime.invoke_endpoint(
        EndpointName=ENDPOINT_NAME,
        ContentType="application/json",
        Accept="application/json",
        Body=json.dumps(payload),
    )
    return json.loads(response["Body"].read().decode("utf-8"))


def show_result(label: str, probs: list):
    prob_map = {c: float(p) for c, p in zip(CLASS_NAMES, probs)}

    msg = f"Credit Score: {label}  |  Confidence: {prob_map[label]*100:.1f}%"
    if label == "Good":
        st.success(msg)
    elif label == "Poor":
        st.error(msg)
    else:
        st.warning(msg)

    st.write("**Probability Distribution**")
    for c in ["Poor", "Standard", "Good"]:
        st.caption(f"{c}  —  {prob_map[c]*100:.1f}%")
        st.progress(float(prob_map[c]))


st.set_page_config(page_title="Credit Score Classifier", page_icon="💳", layout="wide")
st.title("💳 Credit Score Classification")
st.markdown("Memprediksi performa kredit nasabah (**Poor / Standard / Good**) via AWS SageMaker Endpoint.")
st.markdown("---")

st.subheader("📝 Input Data Nasabah")
with st.form("manual_form"):

    # 1. Profil Nasabah
    st.markdown("#### 👤 Profil Nasabah")
    c1, c2, c3 = st.columns(3)
    with c1:
        month = st.selectbox("Month", MONTHS)
    with c2:
        age = st.number_input("Age", 18, 100, 35)
    with c3:
        occupation = st.selectbox("Occupation", OCCUPATIONS)

    st.divider()

    # 2. Pendapatan & Pengeluaran
    st.markdown("#### 💰 Pendapatan & Pengeluaran")
    c1, c2, c3 = st.columns(3)
    with c1:
        annual_income = st.number_input("Annual Income (USD)", 0.0, 300000.0, 40000.0)
        monthly_salary = st.number_input("Monthly Inhand Salary (USD)", 0.0, 30000.0, 3500.0)
    with c2:
        total_emi = st.number_input("Total EMI per Month (USD)", 0.0, 5000.0, 70.0)
        invested = st.number_input("Amount Invested Monthly (USD)", 0.0, 10000.0, 150.0)
    with c3:
        balance = st.number_input("Monthly Balance (USD)", 0.0, 2000.0, 350.0)

    st.divider()

    # 3. Akun & Riwayat Kredit
    st.markdown("#### 🏦 Akun & Riwayat Kredit")
    c1, c2, c3 = st.columns(3)
    with c1:
        num_bank = st.number_input("Num Bank Accounts", 0, 20, 5)
        num_card = st.number_input("Num Credit Card", 0, 20, 5)
        num_loan = st.number_input("Num of Loan", 0, 15, 3)
        loan_type = st.selectbox("Type of Loan (primary)", LOAN_TYPES)
    with c2:
        interest = st.number_input("Interest Rate (%)", 0, 50, 14)
        outstanding = st.number_input("Outstanding Debt (USD)", 0.0, 5000.0, 1200.0)
        util = st.number_input("Credit Utilization Ratio (%)", 0.0, 60.0, 32.0)
        hist_age = st.number_input("Credit History Age (months)", 0, 500, 220)
    with c3:
        changed_limit = st.number_input("Changed Credit Limit", -10.0, 50.0, 9.0)
        num_inq = st.number_input("Num Credit Inquiries", 0, 50, 6)
        credit_mix = st.radio("Credit Mix", ["Bad", "Standard", "Good", "Unknown"])

    st.divider()

    # 4. Perilaku Pembayaran
    st.markdown("#### 💳 Perilaku Pembayaran")
    c1, c2, c3 = st.columns(3)
    with c1:
        delay_due = st.number_input("Delay from Due Date (days)", 0, 100, 20)
        num_delayed = st.number_input("Num of Delayed Payment", 0, 50, 12)
    with c2:
        pay_min = st.radio("Payment of Min Amount", ["Yes", "No"], horizontal=True)
        spending = st.radio("Spending Level", ["Low", "High"], horizontal=True)
    with c3:
        pay_size = st.radio("Payment Size", ["Small", "Medium", "Large"], horizontal=True)

    st.divider()
    submitted = st.form_submit_button("🔮 Prediksi Credit Score", use_container_width=True)

if submitted:
    # Urutan list ini WAJIB sama persis dengan FEATURE_NAMES di inference.py
    features = [
        month, age, occupation, annual_income, monthly_salary, num_bank, num_card, interest,
        num_loan, loan_type, delay_due, num_delayed, changed_limit, num_inq, credit_mix,
        outstanding, util, hist_age, pay_min, total_emi, invested,
        f"{spending}_spent_{pay_size}_value_payments", balance,
    ]
    try:
        result = invoke_endpoint(features)
        label = result["labels"][0]
        probs = result["probabilities"][0]
        st.subheader("Hasil Prediksi")
        show_result(label, probs)
    except NoCredentialsError:
        st.error("No AWS credentials found. Pastikan EC2 instance memiliki IAM role LabInstanceProfile.")
    except ClientError as e:
        st.error(f"AWS error: {e.response['Error'].get('Message', str(e))}")
