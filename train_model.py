import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import RandomOverSampler
import joblib

# --- Load dataset ---
df = pd.read_csv("sampledata.csv")

# 1) Clean column names
df.columns = df.columns.str.strip()

# 2) Clean string cells
for col in df.columns:
    if df[col].dtype == "object":
        df[col] = df[col].astype(str).str.strip()

# 3) Clean target column robustly
df["Is_Fraudulent"] = (
    df["Is_Fraudulent"]
    .astype(str)
    .str.extract(r"([01])")
    .fillna("0")
    .astype(int)
)

print("Fraud counts before resampling:\n", df["Is_Fraudulent"].value_counts())

# 4) Select features (Risk Score removed)
X = df[["Transaction_Amount", "Location_State", "Transaction_Type", "Channel"]]
y = df["Is_Fraudulent"]

# 5) Ensure Transaction_Amount numeric
X.loc[:, "Transaction_Amount"] = pd.to_numeric(X["Transaction_Amount"], errors="coerce").fillna(X["Transaction_Amount"].median())

# 6) Oversample to balance fraud vs safe
ros = RandomOverSampler(random_state=42)
X_res, y_res = ros.fit_resample(X, y)
print("Resampled counts:\n", y_res.value_counts())

# 7) Preprocessing: numeric passthrough, categorical one-hot
numeric_cols = ["Transaction_Amount"]
categorical_cols = ["Location_State", "Transaction_Type", "Channel"]

preprocessor = ColumnTransformer(
    transformers=[
        ("num", "passthrough", numeric_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_cols),
    ]
)

# 8) Model pipeline
model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        class_weight="balanced_subsample"
    ))
])

# 9) Train/test split AFTER resampling
X_train, X_test, y_train, y_test = train_test_split(
    X_res, y_res, test_size=0.2, random_state=42
)

# 10) Train
model.fit(X_train, y_train)

# 11) Debug scores
print("Train accuracy:", model.score(X_train, y_train))
print("Test accuracy:", model.score(X_test, y_test))

# 12) Save model
joblib.dump(model, "fraud_model.pkl")
print("✅ Model trained and saved as fraud_model.pkl")
