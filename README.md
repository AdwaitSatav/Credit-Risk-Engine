# 🏦 Credit Risk Evaluation Engine

A machine learning web application that predicts loan default probability using a Random Forest classifier, deployed live on Streamlit Cloud.

🔗 **Live Demo:** [credit-risk-engine-adwait.streamlit.app](https://credit-risk-engine-adwait.streamlit.app)

---

## 📌 Project Overview

This project simulates a credit risk decision engine similar to systems used in fintech and banking. Given an applicant's financial profile, the model predicts the probability of loan default and classifies the applicant into a risk band.

Built as a personal project to explore end-to-end ML deployment in a fintech context. Not intended for production use.

---

## 🚀 Features

- **Real-time predictions** — adjustable inputs with instant model output
- **Risk classification** — 5-band system from Low to High Risk
- **Feature importance chart** — shows which factors drive each decision
- **Model performance metrics** — AUC, Accuracy, Training Size displayed in-app
- **Full ML pipeline** — data generation, training, evaluation, and deployment

---

## 🛠️ Tech Stack

| Layer | Tools |
|-------|-------|
| Model | Random Forest (scikit-learn) |
| App | Streamlit |
| Data | Lending Club-style synthetic dataset (15,000 records) |
| Deployment | Streamlit Cloud |
| Version Control | GitHub |

---

## 📊 Model Performance

| Metric | Score |
|--------|-------|
| AUC Score | ~0.7218 |
| Accuracy | ~68.77% |
| Training Size | 12,000 records |
| Features Used | 13 |

---

## 🔢 Input Features

- FICO Score
- Interest Rate
- Debt-to-Income Ratio (DTI)
- Annual Income (log scale)
- Monthly Installment
- Revolving Balance & Utilization
- Days with Credit Line
- Inquiries Last 6 Months
- Delinquencies (2 years)
- Public Records
- Loan Purpose
- Credit Policy Compliance

---

## 📁 Project Structure

```
Credit-Risk-Engine/
├── app.py                # Streamlit web app
├── train_model.py        # Model training script
├── model.pkl             # Trained Random Forest model
├── model_columns.pkl     # Feature column order
├── metrics.pkl           # Saved evaluation metrics
├── label_encoder.pkl     # Encoder for categorical features
└── requirements.txt      # Python dependencies
```

---

## ⚙️ Run Locally

```bash
# Install dependencies
pip install -r requirements.txt

# Train the model
python train_model.py

# Launch the app
streamlit run app.py
```

---

## 👤 Author

**Adwait Satav**  
[GitHub](https://github.com/AdwaitSatav)
