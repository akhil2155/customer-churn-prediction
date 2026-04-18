import streamlit as st
import pandas as pd
import joblib

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Customer Churn Dashboard",
    page_icon="📊",
    layout="wide"
)

# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>
.main {
    background-color: transparent;
}

div.stButton > button {
    width: 100%;
    font-weight: bold;
    border-radius: 10px;
    height: 3em;
}

div.stButton > button:hover {
    opacity: 0.9;
}

/* Sidebar labels visible in both modes */
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] span,
section[data-testid="stSidebar"] div {
    color: inherit !important;
}
</style>
""", unsafe_allow_html=True)

# ---------------- TITLE ----------------
st.title("📊 Customer Churn Prediction Dashboard")
st.caption("AI-powered customer retention analytics")

# ---------------- KPI SECTION ----------------
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Model Accuracy", "79.8%")

with col2:
    st.metric("ROC-AUC", "0.836")

with col3:
    st.metric("Status", "Live ✅")

st.markdown("---")

# ---------------- LOAD MODEL ----------------
model = joblib.load("models/churn_model.pkl")

# ---------------- SIDEBAR ----------------
st.sidebar.header("📌 Customer Details")

gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
senior = st.sidebar.selectbox("Senior Citizen", [0, 1])
partner = st.sidebar.selectbox("Partner", [0, 1])
dependents = st.sidebar.selectbox("Dependents", [0, 1])

tenure = st.sidebar.slider("Tenure (Months)", 1, 72, 12)

# Contract Type
contract_option = st.sidebar.selectbox(
    "Contract Type",
    ["Month-to-Month", "One Year", "Two Year"]
)

contract_map = {
    "Month-to-Month": 0,
    "One Year": 1,
    "Two Year": 2
}

contract = contract_map[contract_option]

# Internet Service
internet_option = st.sidebar.selectbox(
    "Internet Service",
    ["DSL", "Fiber", "No"]
)

internet_map = {
    "DSL": 0,
    "Fiber": 1,
    "No": 2
}

internet = internet_map[internet_option]

tech = st.sidebar.selectbox("Tech Support", [0, 1])
paperless = st.sidebar.selectbox("Paperless Billing", [0, 1])

# Payment Method
payment_option = st.sidebar.selectbox(
    "Payment Method",
    ["Electronic Check", "Mailed Check", "Bank Transfer", "Credit Card"]
)

payment_map = {
    "Electronic Check": 0,
    "Mailed Check": 1,
    "Bank Transfer": 2,
    "Credit Card": 3
}

payment = payment_map[payment_option]

monthly = st.sidebar.slider("Monthly Charges", 10, 150, 70)

# ---------------- INPUT DATA ----------------
input_data = pd.DataFrame({
    "gender": [1 if gender == "Male" else 0],
    "SeniorCitizen": [senior],
    "Partner": [partner],
    "Dependents": [dependents],
    "tenure": [tenure],
    "PhoneService": [1],
    "MultipleLines": [0],
    "InternetService": [internet],
    "OnlineSecurity": [0],
    "OnlineBackup": [1],
    "DeviceProtection": [1],
    "TechSupport": [tech],
    "StreamingTV": [1],
    "StreamingMovies": [1],
    "Contract": [contract],
    "PaperlessBilling": [paperless],
    "PaymentMethod": [payment],
    "MonthlyCharges": [monthly],
    "TotalCharges": [monthly * tenure]
})

# ---------------- PREDICTION ----------------
st.subheader("🔮 Prediction")

if st.button("Predict Churn"):

    pred = model.predict(input_data)[0]
    prob = model.predict_proba(input_data)[0][1]

    # Risk Meter
    st.subheader("Risk Score")
    st.progress(int(prob * 100))
    st.write(f"Churn Probability: {prob:.2%}")

    # Prediction Result
    if pred == 1:
        st.error(f"⚠️ High Risk of Churn ({prob:.2%})")

        saved = monthly * 12
        st.info(f"💰 Potential Annual Revenue Saved: ₹{saved:,.0f}")

    else:
        st.success(f"✅ Likely to Stay ({1 - prob:.2%})")

    # Analytics Section
    st.markdown("---")
    st.subheader("📊 Quick Analytics")

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Monthly Revenue", f"₹{monthly}")

    with col2:
        st.metric("Customer Tenure", f"{tenure} Months")

    # Risk Chart
    chart_data = pd.DataFrame({
        "Category": ["Risk", "Safe"],
        "Value": [prob, 1 - prob]
    })

    st.bar_chart(chart_data.set_index("Category"))

# ---------------- FEATURE IMPORTANCE ----------------
st.markdown("---")
st.subheader("📈 Top Churn Drivers")

st.image(
    "models/shap_summary.png",
    caption="Feature Importance Analysis",
    use_container_width=True
)

# ---------------- FOOTER ----------------
st.markdown("---")
st.caption("Built with Python, Scikit-learn, Streamlit")