import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

# ── Page Config ───────────────────────────────────────────────
st.set_page_config(
    page_title="Credit Risk Engine",
    page_icon="🏦",
    layout="wide"
)

# ── Load Artifacts ────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    model         = joblib.load("model.pkl")
    model_columns = joblib.load("model_columns.pkl")
    metrics       = joblib.load("metrics.pkl")
    le            = joblib.load("label_encoder.pkl")
    return model, model_columns, metrics, le

model, model_columns, metrics, le = load_artifacts()

# ── Sidebar ───────────────────────────────────────────────────
st.sidebar.title("🏦 Credit Risk Engine")
st.sidebar.markdown("---")
st.sidebar.header("Applicant Details")

credit_policy  = st.sidebar.selectbox("Meets Credit Policy?", [1, 0],
                     format_func=lambda x: "Yes" if x == 1 else "No")
purpose        = st.sidebar.selectbox("Loan Purpose", [
                     "debt_consolidation", "credit_card", "home_improvement",
                     "other", "major_purchase", "small_business", "educational"])
int_rate       = st.sidebar.slider("Interest Rate", 0.05, 0.25, 0.12, step=0.01)
installment    = st.sidebar.slider("Monthly Installment (₹)", 50.0, 1500.0, 400.0)
log_annual_inc = st.sidebar.slider("Annual Income (log scale)", 8.0, 14.0, 11.0,
                     help="Natural log of income. 11 ≈ ₹60,000 | 12 ≈ ₹1,62,000")
dti            = st.sidebar.slider("Debt-to-Income Ratio (%)", 0.0, 35.0, 15.0)
fico           = st.sidebar.slider("FICO Score", 300, 850, 700)
days_cr_line   = st.sidebar.slider("Days with Credit Line", 200, 17000, 5000)
revol_bal      = st.sidebar.slider("Revolving Balance", 0, 120000, 15000)
revol_util     = st.sidebar.slider("Revolving Utilization (%)", 0.0, 120.0, 50.0)
inq_last_6mths = st.sidebar.slider("Inquiries Last 6 Months", 0, 10, 1)
delinq_2yrs    = st.sidebar.slider("Delinquencies (2 yrs)", 0, 10, 0)
pub_rec        = st.sidebar.slider("Public Records", 0, 5, 0)

# ── Encode & Predict ──────────────────────────────────────────
try:
    purpose_encoded = le.transform([purpose])[0]
except:
    purpose_encoded = 0

input_dict = {
    "credit.policy":     credit_policy,
    "int.rate":          int_rate,
    "installment":       installment,
    "log.annual.inc":    log_annual_inc,
    "dti":               dti,
    "fico":              fico,
    "days.with.cr.line": days_cr_line,
    "revol.bal":         revol_bal,
    "revol.util":        revol_util,
    "inq.last.6mths":    inq_last_6mths,
    "delinq.2yrs":       delinq_2yrs,
    "pub.rec":           pub_rec,
    "purpose":           purpose_encoded
}

input_df  = pd.DataFrame([input_dict])[model_columns]
prob      = model.predict_proba(input_df)[0][1]
threshold = 0.50

# ── Main UI ───────────────────────────────────────────────────
st.title("🏦 Credit Risk Evaluation Engine")
st.markdown("Machine learning system for loan default prediction · Random Forest · Lending Club Data")
st.markdown("---")

# Top metrics row
c1, c2, c3, c4 = st.columns(4)
c1.metric("Default Probability", f"{prob:.2%}")
c2.metric("Decision Threshold",  f"{threshold:.2%}")
c3.metric("FICO Score",          fico)
c4.metric("Interest Rate",       f"{int_rate:.2%}")

st.markdown("---")

# Decision
st.header("Loan Decision")
if prob > threshold:
    st.error(f"❌ **Loan Rejected** — Default probability {prob:.2%} exceeds threshold {threshold:.2%}")
else:
    st.success(f"✅ **Loan Approved** — Default probability {prob:.2%} is within acceptable range")

# Risk band
st.header("Risk Classification")
col1, col2 = st.columns([1, 2])
with col1:
    if prob < 0.20:
        st.success("🟢 Low Risk")
    elif prob < 0.35:
        st.info("🔵 Low-Medium Risk")
    elif prob < 0.50:
        st.warning("🟡 Medium Risk")
    elif prob < 0.65:
        st.warning("🟠 Medium-High Risk")
    else:
        st.error("🔴 High Risk")
with col2:
    st.markdown(f"**Risk Score: {prob:.2%}**")
    st.progress(float(prob))

st.markdown("---")

# Feature importance
st.header("Top Factors Driving This Decision")
importance = model.feature_importances_
feat_df = pd.DataFrame({
    "Feature":    model_columns,
    "Importance": importance
}).sort_values("Importance", ascending=True).tail(8)

fig, ax = plt.subplots(figsize=(8, 4))
ax.barh(feat_df["Feature"], feat_df["Importance"], color="#E63946")
ax.set_xlabel("Importance Score")
ax.set_title("Top 8 Features Used by the Model")
ax.spines[["top", "right"]].set_visible(False)
st.pyplot(fig)

st.markdown("---")

# Model performance
st.header("📊 Model Performance")
m1, m2, m3, m4 = st.columns(4)
m1.metric("AUC Score",     metrics["auc"])
m2.metric("Accuracy",      f"{metrics['accuracy']:.2%}")
m3.metric("Training Size", f"{metrics['train_size']:,}")
m4.metric("Default Rate",  f"{metrics['default_rate']:.2%}")
st.caption("Model: Random Forest | Dataset: Lending Club Style | 9,578 loan records")

# Applicant summary
st.markdown("---")
st.header("📋 Applicant Summary")
st.table(pd.DataFrame({
    "Feature": [
        "FICO Score", "Interest Rate", "DTI Ratio",
        "Loan Purpose", "Credit Policy", "Public Records",
        "Delinquencies", "Inquiries (6mo)"
    ],
    "Value": [
        fico, f"{int_rate:.2%}", f"{dti}%",
        purpose, "Yes" if credit_policy == 1 else "No",
        pub_rec, delinq_2yrs, inq_last_6mths
    ]
}))