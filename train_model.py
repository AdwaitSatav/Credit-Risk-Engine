import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.preprocessing import LabelEncoder
import joblib

# ── 1. Generate High-Quality Realistic Loan Dataset ──────────
np.random.seed(42)
n = 15000

purposes = ["debt_consolidation", "credit_card", "home_improvement",
            "other", "major_purchase", "small_business", "educational"]

fico           = np.random.randint(612, 850, n)
int_rate       = np.round(np.random.uniform(0.06, 0.24, n), 4)
dti            = np.round(np.random.uniform(0, 35, n), 2)
credit_policy  = np.random.choice([0, 1], n, p=[0.2, 0.8])
installment    = np.round(np.random.uniform(50, 1400, n), 2)
log_annual_inc = np.round(np.random.normal(11.0, 0.6, n), 4)
days_cr_line   = np.round(np.random.uniform(200, 17000, n), 1)
revol_bal      = np.random.randint(0, 120000, n)
revol_util     = np.round(np.random.uniform(0, 119, n), 1)
inq_last_6mths = np.random.randint(0, 9, n)
delinq_2yrs    = np.random.randint(0, 5, n)
pub_rec        = np.random.randint(0, 3, n)
purpose        = np.random.choice(purposes, n)

df = pd.DataFrame({
    "credit.policy":     credit_policy,
    "purpose":           purpose,
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
})

# Strong, clear default logic — gives model clear patterns to learn
default_score = (
    - 0.008  * (df["fico"] - 650)          # higher FICO = lower risk
    + 8.0    *  df["int.rate"]              # higher rate = higher risk
    + 0.03   *  df["dti"]                   # higher DTI = higher risk
    - 0.8    *  df["credit.policy"]         # meets policy = lower risk
    + 0.08   *  df["inq.last.6mths"]        # more inquiries = higher risk
    + 0.15   *  df["pub.rec"]               # public records = higher risk
    + 0.12   *  df["delinq.2yrs"]           # delinquencies = higher risk
    - 0.3    * (df["log.annual.inc"] - 10)  # higher income = lower risk
    + 0.003  *  df["revol.util"]            # high utilization = higher risk
)

# Convert score to probability using sigmoid
default_prob = 1 / (1 + np.exp(-default_score))
default_prob = np.clip(default_prob, 0.02, 0.98)
df["not.fully.paid"] = np.random.binomial(1, default_prob)

print(f"✅ Dataset created: {df.shape}")
print(f"   Default rate: {df['not.fully.paid'].mean():.2%}")

# ── 2. Features & Target ──────────────────────────────────────
features = [
    "credit.policy", "int.rate", "installment",
    "log.annual.inc", "dti", "fico",
    "days.with.cr.line", "revol.bal", "revol.util",
    "inq.last.6mths", "delinq.2yrs", "pub.rec", "purpose"
]

le = LabelEncoder()
df["purpose"] = le.fit_transform(df["purpose"])
joblib.dump(le, "label_encoder.pkl")

X = df[features]
y = df["not.fully.paid"]

# ── 3. Train/Test Split ───────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ── 4. Train Model ────────────────────────────────────────────
model = RandomForestClassifier(
    n_estimators=300,
    max_depth=10,
    min_samples_split=20,
    min_samples_leaf=10,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)
model.fit(X_train, y_train)
print("✅ Model trained!")

# ── 5. Evaluate ───────────────────────────────────────────────
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]
auc    = roc_auc_score(y_test, y_prob)
acc    = (y_pred == y_test).mean()

print(f"\n✅ AUC Score:  {auc:.4f}")
print(f"✅ Accuracy:   {acc:.4f}")
print("\n", classification_report(y_test, y_pred))

# ── 6. Save Everything ────────────────────────────────────────
joblib.dump(model,           "model.pkl")
joblib.dump(list(X.columns), "model_columns.pkl")
joblib.dump({
    "auc":          round(auc, 4),
    "accuracy":     round(acc, 4),
    "train_size":   len(X_train),
    "test_size":    len(X_test),
    "default_rate": round(float(y.mean()), 4),
}, "metrics.pkl")

print("\n✅ All files saved: model.pkl, model_columns.pkl, metrics.pkl, label_encoder.pkl")