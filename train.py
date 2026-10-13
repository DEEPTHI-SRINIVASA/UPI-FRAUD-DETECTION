# train.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import joblib
import os

# Load dataset
df = pd.read_csv("sampledata.csv")

# Clean column names and values
df.columns = df.columns.str.strip()
for col in df.select_dtypes(include=['object']).columns:
    df[col] = df[col].astype(str).str.strip()

# ✅ Create target column from Detection_Flagged
df["Is_Fraudulent"] = df["Detection_Flagged"].apply(lambda x: 1 if x.lower() == "flagged" else 0)

# ✅ Keep only useful features
use_cols = ["Transaction_amt", "location_state", "Trans_type", "channel", "Is_Fraudulent"]
df = df[use_cols].copy()

# ✅ Encode categorical columns
encoders = {}
for col in ["location_state", "Trans_type", "channel"]:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    encoders[col] = le

# ✅ Features and target
X = df.drop(columns=["Is_Fraudulent"])
y = df["Is_Fraudulent"]

# ✅ Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

# ✅ Train model
model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

# ✅ Evaluate
print("Fraud counts:", y.value_counts().to_dict())
print("Train accuracy:", model.score(X_train, y_train))
print("Test accuracy:", model.score(X_test, y_test))

# ✅ Save model and encoders
joblib.dump(model, "fraud_model.pkl")
joblib.dump(encoders, "encoders.pkl")
print("✅ Model and encoders saved successfully.")
