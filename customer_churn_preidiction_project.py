import numpy as np
import pandas as pd
import joblib

from catboost import CatBoostClassifier
import lightgbm as lgb
from xgboost import XGBClassifier

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, VotingClassifier, StackingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from imblearn.pipeline import Pipeline as ImbPipeline

pd.set_option("display.max_row", None)
pd.set_option("display.max_columns", None)

# 1. Load Data
df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")
for i in df.select_dtypes(include=['object']).columns:
    print(f"----{i}----")
    print(df[i].unique())
for i in df.select_dtypes(include=['float64', 'int64']).columns:
    print(f"----{i}----")
    print(df[i].min())
    print(df[i].max())
print(df.head(5))
print(df.info())
for i in df.select_dtypes(include=['object']).columns:
    print(f"----{i}----")
    print(df[i].unique())

def engineer_features(df):
    df = df.copy()
    print("--- Starting Feature Engineering ---")
    
    # Drop ID column
    if 'customerID' in df.columns:
        df = df.drop("customerID", axis=1)
    
    # Gender mapping
    df["gender"] = df["gender"].map({"Female": 0, "Male": 1})
    
    # Clean TotalCharges
    if 'TotalCharges' in df.columns:
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
        df['TotalCharges'] = df['TotalCharges'].fillna(df['TotalCharges'].median())
    
    # Standardize categorical values
    if 'MultipleLines' in df.columns:
        df["MultipleLines"] = df["MultipleLines"].replace({"No phone service": "No"})
        df["MultipleLines"] = df["MultipleLines"].map({"No": 0, "Yes": 1})
    
    # Churn mapping
    if 'Churn' in df.columns:
        df["Churn"] = df["Churn"].map({"No": 0, "Yes": 1})
    
    # Replace "No internet service" with "No"
    col_replace = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 
                   'TechSupport', 'StreamingTV', 'StreamingMovies']
    for col in col_replace:
        if col in df.columns:
            df[col] = df[col].replace({"No internet service": "No"})
            df[col] = df[col].map({"No": 0, "Yes": 1})
    
    # Binary mapping
    col_replace2 = ["Partner", "Dependents", "PhoneService", "PaperlessBilling"]
    for col in col_replace2:
        if col in df.columns:
            df[col] = df[col].map({"No": 0, "Yes": 1})
    
    # -----------------------------
    # Custom Domain-Specific Features
    # -----------------------------
    service_cols = ["PhoneService", "MultipleLines", "OnlineSecurity", "OnlineBackup",
                    "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"]
    
    # Total services count
    df["TotalServicesCount"] = df[service_cols].sum(axis=1)
    
    # Spend ratio
    df["SpendRatio"] = (df["MonthlyCharges"] / (df["TotalCharges"] + 1)).round(3)
    
    # High friction payment (Mailed check / Electronic check)
    df["HighFrictionPayment"] = df["PaymentMethod"].str.contains("Mailed check", case=False).astype(int)
    df["IsElectronicPayment"] = df["PaymentMethod"].str.contains("electronic check", case=False).astype(int)
    
    # Feature A: High Risk New Customer
    if 'Contract' in df.columns and 'tenure' in df.columns:
        df["Contract"] = df["Contract"].map({"Month-to-month": 0.083, "One year": 1, "Two year": 2})
        df['HighRiskNewCustomer'] = ((df['tenure'] <= 40) & (df['Contract'] == 0.083)).astype(int)
    
    # Feature B: Fiber Optic without Tech Support
    if 'InternetService' in df.columns and 'TechSupport' in df.columns:
        df['Fiber_NoTech'] = ((df['InternetService'] == 'Fiber optic') & (df['TechSupport'] == 0)).astype(int)
    
    # Feature C: Is AutoPay
    if 'PaymentMethod' in df.columns:
        df['IsAutoPay'] = df['PaymentMethod'].str.contains('automatic', case=False, na=False).astype(int)
    
    # -----------------------------
    # Extra Advanced Features
    # -----------------------------
    # Tenure in years
    df["TenureYears"] = (df["tenure"] / 12).round(2)
    
    # Average monthly spend
    df["AvgMonthlySpend"] = (df["TotalCharges"] / (df["tenure"] + 1)).round(2)
    
    # Senior alone risk
    df["SeniorAlone"] = ((df["SeniorCitizen"] == 1) & (df["Partner"] == 0) & (df["Dependents"] == 0)).astype(int)
    
    # Service richness
    df["ServiceRich"] = (df["TotalServicesCount"] >= 4).astype(int)
    
    # Charges per service
    df["ChargesPerService"] = (df["MonthlyCharges"] / (df["TotalServicesCount"] + 1)).round(2)
    
    # Contract-tenure interaction
    df["Contract_Tenure"] = df["Contract"] * df["TenureYears"]
    
    print("--- Feature Engineering Completed Successfully! ---")
    return df

# Run the function
df = engineer_features(df)
print(df.head(5))
print(df.info())
X = df.drop("Churn", axis=1)
y = df["Churn"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

one_hot_columns = ['InternetService', 'PaymentMethod']
numerical_columns = X.select_dtypes(include=["int64", "float64"]).columns

num_pipe = Pipeline([("imputer", SimpleImputer(strategy="median"))])
ohe_pipe = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                     ("ohe", OneHotEncoder(handle_unknown='ignore', sparse_output=False))])
preprocessor = ColumnTransformer([
    ("num", num_pipe, numerical_columns),
    ("ohe", ohe_pipe, one_hot_columns)
])

# -----------------------------
# 3. Tuned Models
# -----------------------------
gbr_model = GradientBoostingClassifier(
    n_estimators=400, learning_rate=0.02, max_depth=5,
    max_features="sqrt", subsample=0.8, random_state=42, verbose=1
)

rfc_model = RandomForestClassifier(
    n_estimators=1200, max_depth=15, min_samples_split=6,
    min_samples_leaf=2, max_features="sqrt", criterion="gini",
    n_jobs=-1, class_weight="balanced", random_state=42
)

xgb_model = XGBClassifier(
    n_estimators=400, learning_rate=0.02, max_depth=6,
    subsample=0.8, reg_alpha=1.5, reg_lambda=1.5,
    colsample_bytree=0.8, eval_metric="logloss",
    random_state=42, scale_pos_weight=1.5, n_jobs=-1
)
lgb_model = lgb.LGBMClassifier(n_estimators=400, learning_rate=0.01, max_depth=5,
                               scale_pos_weight=1.5, random_state=42, verbose=-1)

cat_model = CatBoostClassifier(iterations=400, learning_rate=0.01, depth=5,
                               auto_class_weights="Balanced", random_seed=42, verbose=0)

gbr_pipe = ImbPipeline([
    ("preprocessor", preprocessor),
    ("model", gbr_model)
])

rfc_pipe = ImbPipeline([
    ("preprocessor", preprocessor),
    ("model", rfc_model)
])

xgb_pipe = ImbPipeline([
    ("preprocessor", preprocessor),
    ("model", xgb_model)
])
lgb_pipe = ImbPipeline([
    ("preprocessor", preprocessor),
    ("model", lgb_model)
])

cat_pipe = ImbPipeline([
    ("preprocessor", preprocessor),
    ("model", cat_model)
])

# -----------------------------
# 7. Voting Ensemble (all models)
# -----------------------------
from sklearn.linear_model import LogisticRegression

estimators = [
    ("gbr", gbr_pipe),
    ("rf", rfc_pipe),
    ("xgb", xgb_pipe),
    ("lgb", lgb_pipe),
    ("cat", cat_pipe)
]

meta = LogisticRegression(C=0.1, penalty="l2", max_iter=1000)

stacking = StackingClassifier(
    estimators=estimators,
    final_estimator=meta,
    cv=5,
    passthrough=False
)

models = {
    "CatBoost": cat_pipe,
    "LightGBM": lgb_pipe,
    "stacking_Enseble": stacking
}

lgb_pipe.fit(X_train, y_train)
y_probs = lgb_pipe.predict_proba(X_test)[:, 1]
custom_threshold1 = 0.45
y_pred_custom = (y_probs >= custom_threshold1).astype(int)
print("-------- light gbm Boost Train and Test score-------")
print("Train Score:", lgb_pipe.score(X_train, y_train) * 100)
print("Test Score:", lgb_pipe.score(X_test, y_test) * 100)
print(f"--- LightGBM Model Evaluation (Threshold = {custom_threshold1}) ---")
print("Precision:", round(precision_score(y_test, y_pred_custom) * 100, 2), "%")
print("Recall:", round(recall_score(y_test, y_pred_custom) * 100, 2), "%")
print("F1-Score:", round(f1_score(y_test, y_pred_custom) * 100, 2), "%")
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred_custom))

cat_pipe.fit(X_train, y_train)
y_probs = cat_pipe.predict_proba(X_test)[:, 1]
custom_threshold2 = 0.65
y_pred_custom = (y_probs >= custom_threshold2).astype(int)
print("-------- Cat boost Train and Test score-------")
print("Train Score:", cat_pipe.score(X_train, y_train) * 100)
print("Test Score:", cat_pipe.score(X_test, y_test) * 100)
print(f"--- CatBoost Model Evaluation (Threshold = {custom_threshold2}) ---")
print("Precision:", round(precision_score(y_test, y_pred_custom) * 100, 2), "%")
print("Recall:", round(recall_score(y_test, y_pred_custom) * 100, 2), "%")
print("F1-Score:", round(f1_score(y_test, y_pred_custom) * 100, 2), "%")
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred_custom))

voting = VotingClassifier(
    estimators=[
        ("gbr", gbr_pipe),
        ("rf", rfc_pipe),
        ("xgb", xgb_pipe),
        ("lgb", lgb_pipe),
        ("cat", cat_pipe)
    ],
    voting="soft",
    n_jobs=-1
)

voting.fit(X_train, y_train)
y_probs = voting.predict_proba(X_test)[:, 1]
custom_threshold3 = 0.55
y_pred_custom = (y_probs >= custom_threshold3).astype(int)

print("-------- Voting Ensemble Train and Test score -------")
print("Train Score:", voting.score(X_train, y_train) * 100)
print("Test Score:", voting.score(X_test, y_test) * 100)
print(f"--- Voting Ensemble Evaluation (Threshold = {custom_threshold3}) ---")
print("Precision:", round(precision_score(y_test, y_pred_custom) * 100, 2), "%")
print("Recall:", round(recall_score(y_test, y_pred_custom) * 100, 2), "%")
print("F1-Score:", round(f1_score(y_test, y_pred_custom) * 100, 2), "%")
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred_custom))

joblib.dump(voting, "voting_model_pipeline.pkl")