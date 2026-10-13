"""
train_model.py — Customer Churn Prediction Pipeline
=====================================================
Trains a Random Forest classifier on the Telco Customer Churn dataset and
persists the model + feature schema so a downstream service (API, batch
scorer, etc.) can reproduce identical preprocessing at inference time.

Usage:
    python train_model.py

Outputs:
    churn_model.pkl    – serialised Random Forest model
    model_features.pkl – ordered list of feature columns the model expects
"""

import warnings
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
import joblib

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# 1. LOAD DATA
# ---------------------------------------------------------------------------
# The dataset mirrors the IBM Telco Customer Churn public dataset.
# Each row represents one customer account; the target column `Churn`
# indicates whether the customer left the service within the last month.
# ---------------------------------------------------------------------------

DATA_PATH = "churn_data.csv"

print("=" * 60)
print("  CUSTOMER CHURN PREDICTION – Training Pipeline")
print("=" * 60)

df = pd.read_csv(DATA_PATH)
print(f"\n[OK] Loaded dataset: {df.shape[0]:,} rows x {df.shape[1]} columns")

# ---------------------------------------------------------------------------
# 2. DATA CLEANING
# ---------------------------------------------------------------------------
# Business context for each cleaning step:
#
# • customerID is a unique row identifier with zero predictive value; keeping
#   it would let the model memorise IDs instead of learning real patterns.
#
# • TotalCharges is stored as a string in the raw CSV.  Rows where tenure == 0
#   (brand-new customers) have TotalCharges = " " (whitespace), which Pandas
#   cannot coerce to float.  Rather than drop these customers — who are
#   *exactly* the high-risk cohort for early churn — we fill the missing
#   value with 0.0 (they haven't been charged anything yet).
#
# • SeniorCitizen is already encoded as 0/1 in the raw data, so we leave it
#   as-is (no string-to-number conversion needed).
# ---------------------------------------------------------------------------

# Drop the customer identifier — not a feature
df.drop("customerID", axis=1, inplace=True)
print("[OK] Dropped 'customerID' (non-predictive identifier)")

# Coerce TotalCharges to numeric; whitespace-only values become NaN
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

n_missing = df["TotalCharges"].isna().sum()
if n_missing > 0:
    # These are brand-new customers (tenure == 0) who haven't been billed yet
    df["TotalCharges"].fillna(0.0, inplace=True)
    print(f"[OK] Filled {n_missing} missing 'TotalCharges' values with 0.0")

# ---------------------------------------------------------------------------
# 3. FEATURE PREPROCESSING
# ---------------------------------------------------------------------------
# Encoding strategy rationale:
#
# LABEL ENCODING (0/1) for columns with exactly two distinct values.
#   → Preserves ordinality implicitly (e.g., "Yes" > "No") while keeping
#     the feature space compact.  Tree-based models handle this natively.
#
# ONE-HOT ENCODING for columns with 3+ distinct categories.
#   → Prevents the model from imposing a false ordinal relationship
#     (e.g., "DSL" < "Fiber optic" < "No" makes no business sense).
#     Each category becomes its own binary flag.
#
# We process the target column (Churn) with label encoding as well
# (Yes → 1, No → 0) so sklearn can consume it.
# ---------------------------------------------------------------------------

# Encode the target variable: Churn → 1 (left) / 0 (stayed)
le_target = LabelEncoder()
df["Churn"] = le_target.fit_transform(df["Churn"])

# Separate features from the target before encoding
target = df["Churn"]
features = df.drop("Churn", axis=1)

# Identify column types
numeric_cols = features.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols = features.select_dtypes(include=["object"]).columns.tolist()

# Split categoricals into binary (2 unique values) vs. multi-class (3+)
binary_cols = [c for c in categorical_cols if features[c].nunique() == 2]
multiclass_cols = [c for c in categorical_cols if features[c].nunique() > 2]

print(f"\n- Numeric features   ({len(numeric_cols)}): {numeric_cols}")
print(f"- Binary  features   ({len(binary_cols)}): {binary_cols}")
print(f"- Multi-class features ({len(multiclass_cols)}): {multiclass_cols}")

# 3a. LABEL ENCODE binary columns (Yes/No, Male/Female)
label_encoders = {}
for col in binary_cols:
    le = LabelEncoder()
    features[col] = le.fit_transform(features[col])
    label_encoders[col] = le

print(f"\n[OK] Label-encoded {len(binary_cols)} binary columns")

# 3b. ONE-HOT ENCODE multi-class columns (drop_first to avoid
#     multicollinearity — e.g., if InternetService has 3 categories,
#     2 dummy columns are sufficient to represent all 3 states).
features = pd.get_dummies(features, columns=multiclass_cols, drop_first=True)

print(f"[OK] One-hot-encoded {len(multiclass_cols)} multi-class columns")
print(f"[OK] Final feature matrix: {features.shape[1]} columns")

# Save the ordered list of feature column names.
# This is CRITICAL for production: at inference time the serving layer must
# build a DataFrame with *exactly* these columns in *exactly* this order.
feature_columns = features.columns.tolist()

# ---------------------------------------------------------------------------
# 4. TRAIN / TEST SPLIT
# ---------------------------------------------------------------------------
# 80/20 split with stratification on the target to preserve the natural
# churn-rate imbalance (~26 % churn) in both partitions.
# A fixed random_state guarantees reproducibility across runs.
# ---------------------------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    features,
    target,
    test_size=0.20,
    random_state=42,
    stratify=target,
)

print(f"\n[OK] Train set: {X_train.shape[0]:,} samples")
print(f"[OK] Test  set: {X_test.shape[0]:,} samples")

# ---------------------------------------------------------------------------
# 5. MODEL TRAINING
# ---------------------------------------------------------------------------
# Random Forest is chosen because:
#   • It handles mixed numeric / encoded-categorical features well.
#   • It is robust to outliers and doesn't require feature scaling.
#   • Built-in feature importance helps stakeholders understand drivers.
#   • n_estimators=100 gives a good bias-variance trade-off for ~7 k rows.
#   • class_weight="balanced" upweights the minority class (churners) so
#     the model doesn't simply predict "No Churn" for everyone.
# ---------------------------------------------------------------------------

print("\n>> Training Random Forest Classifier ...")

model = RandomForestClassifier(
    n_estimators=100,
    max_depth=12,
    min_samples_split=5,
    min_samples_leaf=2,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1,
)
model.fit(X_train, y_train)

print("[OK] Training complete")

# ---------------------------------------------------------------------------
# 6. MODEL EVALUATION
# ---------------------------------------------------------------------------
# Accuracy alone can be misleading with imbalanced classes; the
# classification report adds precision, recall, and F1 per class.
#   • Precision for "Churn = 1": Of all customers we flagged as churners,
#     how many actually churned?  (→ controls wasted retention offers)
#   • Recall for "Churn = 1": Of all real churners, how many did we catch?
#     (→ controls missed revenue loss)
# ---------------------------------------------------------------------------

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print("\n" + "=" * 60)
print("  EVALUATION RESULTS")
print("=" * 60)
print(f"\n  Accuracy: {accuracy:.4f}  ({accuracy * 100:.2f} %)\n")
print(classification_report(y_test, y_pred, target_names=["Stayed", "Churned"]))

# ---------------------------------------------------------------------------
# 7. FEATURE IMPORTANCE (Top 10)
# ---------------------------------------------------------------------------
# Surfacing the most influential features helps the business team
# prioritise retention strategies (e.g., if "Contract_Month-to-month"
# ranks #1, offering annual plans is a clear lever).
# ---------------------------------------------------------------------------

importances = pd.Series(model.feature_importances_, index=feature_columns)
top10 = importances.sort_values(ascending=False).head(10)

print("-" * 60)
print("  TOP 10 FEATURE IMPORTANCES")
print("-" * 60)
for rank, (feat, imp) in enumerate(top10.items(), start=1):
    bar = "#" * int(imp * 100)
    print(f"  {rank:>2}. {feat:<40s} {imp:.4f}  {bar}")

# ---------------------------------------------------------------------------
# 8. SAVE ARTIFACTS
# ---------------------------------------------------------------------------
# Two files are persisted:
#   • churn_model.pkl    — the trained Random Forest (used at inference)
#   • model_features.pkl — the exact feature column order (used to align
#                          new input DataFrames before calling model.predict)
# ---------------------------------------------------------------------------

MODEL_PATH = "churn_model.pkl"
FEATURES_PATH = "model_features.pkl"

joblib.dump(model, MODEL_PATH)
joblib.dump(feature_columns, FEATURES_PATH)

print(f"\n[OK] Model saved    -> {MODEL_PATH}")
print(f"[OK] Features saved -> {FEATURES_PATH}")
print(f"\n{'=' * 60}")
print("  Pipeline finished successfully.")
print(f"{'=' * 60}\n")
