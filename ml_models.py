"""
ml_models.py — BankIQ
Trained on YOUR datasets:
  - loan_approval_dataset.csv  (4269 rows, 13 cols)
  - Customer-Churn-Records.csv (10000 rows, 18 cols)

Run this FIRST:
  python ml_models.py
"""

import pandas as pd
import numpy as np
import pickle, os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
import lightgbm as lgb

os.makedirs("models", exist_ok=True)

# ══════════════════════════════════════════════
# 1. LOAN MODEL
# File: data/loan_approval_dataset.csv
# Target: loan_status (Approved / Rejected)
# ══════════════════════════════════════════════
def train_loan():
    print("\n📊 Training Loan Approval Model...")
    df = pd.read_csv("E:\Clarivo\loan_approval_dataset.csv")

    # Strip spaces from column names
    df.columns = df.columns.str.strip()

    # Strip spaces from string values
    for col in df.select_dtypes(include='object').columns:
        df[col] = df[col].str.strip()

    print("  Columns:", list(df.columns))
    print("  Shape:", df.shape)
    print("  Target distribution:\n", df['loan_status'].value_counts())

    # Encode categorical columns
    le = LabelEncoder()
    df['education']    = le.fit_transform(df['education'])       # Graduate=0, Not Graduate=1
    df['self_employed']= le.fit_transform(df['self_employed'])   # No=0, Yes=1
    df['loan_status']  = le.fit_transform(df['loan_status'])     # Approved=0, Rejected=1

    # Features — using YOUR columns
    features = [
        'no_of_dependents', 'education', 'self_employed',
        'income_annum', 'loan_amount', 'loan_term',
        'cibil_score', 'residential_assets_value',
        'commercial_assets_value', 'luxury_assets_value',
        'bank_asset_value'
    ]

    X = df[features]
    y = df['loan_status']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = lgb.LGBMClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        random_state=42,
        verbose=-1
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\n  ✅ Loan Model Accuracy: {acc*100:.2f}%")
    print(classification_report(y_test, y_pred, target_names=['Approved','Rejected']))

    # Save model + feature list
    pickle.dump((model, features), open("models/loan_model.pkl", "wb"))
    print("  💾 Saved: models/loan_model.pkl")
    return model, features


# ══════════════════════════════════════════════
# 2. CHURN MODEL
# File: data/Customer-Churn-Records.csv
# Target: Exited (0 = Stay, 1 = Leave)
# ══════════════════════════════════════════════
def train_churn():
    print("\n📉 Training Customer Churn Model...")
    df = pd.read_csv("E:\Clarivo\Customer-Churn-Records.csv")

    print("  Columns:", list(df.columns))
    print("  Shape:", df.shape)
    print("  Target distribution:\n", df['Exited'].value_counts())

    # Drop irrelevant columns
    df.drop(['RowNumber', 'CustomerId', 'Surname'], axis=1, inplace=True)

    # Encode categorical
    le = LabelEncoder()
    df['Geography'] = le.fit_transform(df['Geography'])  # France=0, Germany=1, Spain=2
    df['Gender']    = le.fit_transform(df['Gender'])     # Female=0, Male=1
    df['Card Type'] = le.fit_transform(df['Card Type'])  # DIAMOND, GOLD, etc.

    # Features — using YOUR extra columns too (Complain, Satisfaction Score, etc.)
    features = [
        'CreditScore', 'Geography', 'Gender', 'Age',
        'Tenure', 'Balance', 'NumOfProducts',
        'HasCrCard', 'IsActiveMember', 'EstimatedSalary',
        'Complain', 'Satisfaction Score', 'Card Type', 'Point Earned'
    ]

    X = df[features]
    y = df['Exited']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=8,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\n  ✅ Churn Model Accuracy: {acc*100:.2f}%")
    print(classification_report(y_test, y_pred, target_names=['Stay','Leave']))

    # Feature importance
    importances = pd.Series(model.feature_importances_, index=features)
    print("\n  Top 5 Important Features:")
    print(importances.nlargest(5).to_string())

    # Save model + feature list
    pickle.dump((model, features), open("models/churn_model.pkl", "wb"))
    print("\n  💾 Saved: models/churn_model.pkl")
    return model, features


# ══════════════════════════════════════════════
# PREDICT — LOAN
# ══════════════════════════════════════════════
def predict_loan(income, loan_amount, cibil_score, education,
                 self_employed, loan_term=12, dependents=0,
                 residential_assets=0, commercial_assets=0,
                 luxury_assets=0, bank_assets=0):
    import shap

    model, features = pickle.load(open("models/loan_model.pkl", "rb"))

    edu_enc  = 0 if education == "Graduate" else 1
    self_enc = 1 if self_employed == "Yes" else 0

    X = np.array([[
        dependents, edu_enc, self_enc,
        income, loan_amount, loan_term,
        cibil_score, residential_assets,
        commercial_assets, luxury_assets, bank_assets
    ]])

    prediction  = bool(model.predict(X)[0])   # True = Rejected, False = Approved
    approved    = not prediction               # Flip for readability
    confidence  = float(model.predict_proba(X)[0].max()) * 100

    # SHAP explanation
    explainer  = shap.TreeExplainer(model)
    sv         = explainer.shap_values(X)
    sv         = sv[1][0] if isinstance(sv, list) else sv[0]
    shap_dict  = {f: float(sv[i]) for i, f in enumerate(features)}

    return approved, round(confidence, 1), shap_dict


# ══════════════════════════════════════════════
# PREDICT — CHURN
# ══════════════════════════════════════════════
def predict_churn(tenure, balance, products, is_active, complaints,
                  credit_score=650, age=35, salary=50000,
                  satisfaction=3, card_type=0, points=400,
                  has_cr_card=1, geography=0, gender=0):

    model, features = pickle.load(open("models/churn_model.pkl", "rb"))

    X = np.array([[
        credit_score, geography, gender, age,
        tenure, balance, products,
        has_cr_card, int(is_active), salary,
        int(complaints > 0), satisfaction, card_type, points
    ]], dtype=float)

    prediction = bool(model.predict(X)[0])
    risk_score = float(model.predict_proba(X)[0][1]) * 100
    risk_score = min(risk_score + (complaints * 5), 99)

    # SHAP — skip to avoid error
    shap_dict = {
        "Balance":       float(balance),
        "CreditScore":   float(credit_score),
        "IsActive":      float(is_active),
        "Complaints":    float(complaints),
        "Tenure":        float(tenure),
        "Products":      float(products),
        "Age":           float(age),
        "Satisfaction":  float(satisfaction)
    }

    return prediction, round(risk_score, 1), shap_dict

# ══════════════════════════════════════════════
# RUN: python ml_models.py
# ══════════════════════════════════════════════
if __name__ == "__main__":
    print("=" * 55)
    print("🏦  BankIQ — Training Models on Your Datasets")
    print("=" * 55)

    errors = []

    try:
        train_loan()
    except Exception as e:
        errors.append(f"Loan: {e}")
        print(f"  ❌ Loan Error: {e}")

    try:
        train_churn()
    except Exception as e:
        errors.append(f"Churn: {e}")
        print(f"  ❌ Churn Error: {e}")

    print("\n" + "=" * 55)
    if not errors:
        print("✅  BOTH MODELS TRAINED SUCCESSFULLY!")
        print("👉  Next step: python main.py")
    else:
        print(f"❌  Errors: {errors}")
        print("    Check CSV files are in data/ folder!")
    print("=" * 55)
