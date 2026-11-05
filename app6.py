import streamlit as st 
import joblib 
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix

st.set_page_config(
    page_title="Customer Churn Prediction Engine",
    page_icon="📉",
    layout="wide"
)

# Load Trained Voting Pipeline Model (.pkl)
@st.cache_resource
def load_model():
    return joblib.load("voting_model_pipeline.pkl")
    
model = load_model()

st.title("📉 Customer Churn Prediction System")
st.write("Enterprise Retention Engine: Analyze individual customer churn risks or batch process customer portfolios using your trained Voting Pipeline.")

def engineer_features(df):
    df = df.copy()
    print("--- Starting Feature Engineering ---")
    
    # Drop ID column
    if 'customerID' in df.columns:
        df = df.drop("customerID", axis=1)
    
    # Gender mapping
    if 'gender' in df.columns:
        df["gender"] = df["gender"].map({"Female":0, "Male":1})
    
    # Clean TotalCharges
    if 'TotalCharges' in df.columns:
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
        df['TotalCharges'] = df['TotalCharges'].fillna(df['TotalCharges'].median())
    
    # Standardize categorical values
    if 'MultipleLines' in df.columns:
        df["MultipleLines"] = df["MultipleLines"].replace({"No phone service": "No"})
        df["MultipleLines"] = df["MultipleLines"].map({"No":0, "Yes":1})
    
    # Churn mapping
    if 'Churn' in df.columns:
        df["Churn"] = df["Churn"].map({"No":0 , "Yes":1})
    
    # Replace "No internet service" with "No"
    col_replace = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 
                   'TechSupport', 'StreamingTV', 'StreamingMovies']
    for col in col_replace:
        if col in df.columns:
            df[col] = df[col].replace({"No internet service": "No"})
            df[col] = df[col].map({"No":0, "Yes":1})
    
    # Binary mapping
    col_replace2 = ["Partner","Dependents","PhoneService","PaperlessBilling"]
    for col in col_replace2:
        if col in df.columns:
            df[col] = df[col].map({"No":0, "Yes":1})
    
    # -----------------------------
    # Custom Domain-Specific Features
    # -----------------------------
    service_cols = ["PhoneService","MultipleLines","OnlineSecurity","OnlineBackup",
                    "DeviceProtection","TechSupport","StreamingTV","StreamingMovies"]
    
    existing_service_cols = [c for c in service_cols if c in df.columns]
    if existing_service_cols:
        df["TotalServicesCount"] = df[existing_service_cols].sum(axis=1)
    else:
        df["TotalServicesCount"] = 0
    
    # Spend ratio
    if 'MonthlyCharges' in df.columns and 'TotalCharges' in df.columns:
        df["SpendRatio"] = (df["MonthlyCharges"] / (df["TotalCharges"]+1)).round(3)
    
    # High friction payment
    if 'PaymentMethod' in df.columns:
        df["HighFrictionPayment"] = df["PaymentMethod"].str.contains("Mailed check", case=False, na=False).astype(int)
        df["IsElectronicPayment"] = df["PaymentMethod"].str.contains("electronic check", case=False, na=False).astype(int)
        
    # Feature A: High Risk New Customer
    if 'Contract' in df.columns and 'tenure' in df.columns:
        df["Contract"] = df["Contract"].map({"Month-to-month":0.083, "One year":1, "Two year":2})
        df['HighRiskNewCustomer'] = ((df['tenure'] <= 6) & (df['Contract'] == 0.083)).astype(int)
    
    # Feature B: Fiber Optic without Tech Support
    if 'InternetService' in df.columns and 'TechSupport' in df.columns:
        df['Fiber_NoTech'] = ((df['InternetService'] == 'Fiber optic') & (df['TechSupport'] == 0)).astype(int)
    
    # Feature C: Is AutoPay
    if 'PaymentMethod' in df.columns:
        df['IsAutoPay'] = df['PaymentMethod'].str.contains('automatic', case=False, na=False).astype(int)
    
    # -----------------------------
    # Extra Advanced Features
    # -----------------------------
    if 'tenure' in df.columns:
        df["TenureYears"] = (df["tenure"] / 12).round(2)
        
    if 'TotalCharges' in df.columns and 'tenure' in df.columns:
        df["AvgMonthlySpend"] = (df["TotalCharges"] / (df["tenure"]+1)).round(2)
        
    if all(c in df.columns for c in ["SeniorCitizen", "Partner", "Dependents"]):
        df["SeniorAlone"] = ((df["SeniorCitizen"]==1) & (df["Partner"]==0) & (df["Dependents"]==0)).astype(int)
        
    if "TotalServicesCount" in df.columns:
        df["ServiceRich"] = (df["TotalServicesCount"] >= 4).astype(int)
        
    if 'MonthlyCharges' in df.columns and "TotalServicesCount" in df.columns:
        df["ChargesPerService"] = (df["MonthlyCharges"] / (df["TotalServicesCount"]+1)).round(2)
        
    if 'Contract' in df.columns and 'TenureYears' in df.columns:
        df["Contract_Tenure"] = df["Contract"] * df["TenureYears"]
    
    print("--- Feature Engineering Completed Successfully! ---")
    return df

tab1, tab2 = st.tabs(["👤 Single Customer Audit", "📁 Bulk CSV Batch Audit"])

# ==================== TAB 1: SINGLE CUSTOMER AUDIT ====================
with tab1:
    st.sidebar.header("Customer Profile Parameters")
    st.sidebar.subheader("📊 Customer Details")
    
    gender = st.sidebar.selectbox("Gender", ['Female', 'Male'], index=0)
    senior_citizen = st.sidebar.selectbox("Senior Citizen", [0, 1], index=0, help="0 = No, 1 = Yes")
    partner = st.sidebar.selectbox("Partner", ['Yes', 'No'], index=1)
    dependents = st.sidebar.selectbox("Dependents", ['No', 'Yes'], index=0)
    
    tenure = st.sidebar.slider("Tenure (Months)", 0, 72, 12)
    phone_service = st.sidebar.selectbox("Phone Service", ['No', 'Yes'], index=1)
    multiple_lines = st.sidebar.selectbox("Multiple Lines", ['No', 'Yes', 'No phone service'], index=0)
    
    internet_service = st.sidebar.selectbox("Internet Service", ['DSL', 'Fiber optic', 'No'], index=1)
    online_security = st.sidebar.selectbox("Online Security", ['No', 'Yes', 'No internet service'], index=0)
    online_backup = st.sidebar.selectbox("Online Backup", ['Yes', 'No', 'No internet service'], index=0)
    device_protection = st.sidebar.selectbox("Device Protection", ['No', 'Yes', 'No internet service'], index=0)
    tech_support = st.sidebar.selectbox("Tech Support", ['No', 'Yes', 'No internet service'], index=0)
    streaming_tv = st.sidebar.selectbox("Streaming TV", ['No', 'Yes', 'No internet service'], index=0)
    streaming_movies = st.sidebar.selectbox("Streaming Movies", ['No', 'Yes', 'No internet service'], index=0)
    
    contract = st.sidebar.selectbox("Contract Type", ['Month-to-month', 'One year', 'Two year'], index=0)
    paperless_billing = st.sidebar.selectbox("Paperless Billing", ['Yes', 'No'], index=0)
    payment_method = st.sidebar.selectbox(
        "Payment Method", 
        ['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'], 
        index=0
    )
    
    monthly_charges = st.sidebar.slider("Monthly Charges ($)", 18.25, 118.75, 70.0)
    total_charges = st.sidebar.slider("Total Charges ($)", 1.0, 10000.0, 1500.0)

    if st.button("🚀 Predict Customer Churn Risk", type="primary"):
        input_data = {
            'gender': [gender],
            'SeniorCitizen': [senior_citizen],
            'Partner': [partner],
            'Dependents': [dependents],
            'tenure': [tenure],
            'PhoneService': [phone_service],
            'MultipleLines': [multiple_lines],
            'InternetService': [internet_service],
            'OnlineSecurity': [online_security],
            'OnlineBackup': [online_backup],
            'DeviceProtection': [device_protection],
            'TechSupport': [tech_support],
            'StreamingTV': [streaming_tv],
            'StreamingMovies': [streaming_movies],
            'Contract': [contract],
            'PaperlessBilling': [paperless_billing],
            'PaymentMethod': [payment_method],
            'MonthlyCharges': [monthly_charges],
            'TotalCharges': [total_charges]
        }

        df_input = pd.DataFrame(input_data)
        processed_input = engineer_features(df_input)

        # Predict Probabilities
        probabilities = model.predict_proba(processed_input)[:, 1]
        threshold = 0.40  
        prediction = 1 if probabilities[0] >= threshold else 0

        st.subheader("📊 Assessment Result")
        if prediction == 1:
            st.error(f"⚠️ High Risk of Customer Churn Detected! (Churn Probability: {probabilities[0]*100:.2f}%)")
        else:
            st.success(f"✅ Low Risk! Customer is Likely to Stay (Retained). (Churn Probability: {probabilities[0]*100:.2f}%)")

# ==================== TAB 2: BULK CSV BATCH ====================
with tab2:
    st.header("📁 Bulk Customer Portfolio Audit & Evaluation")

    uploaded_file = st.file_uploader("Upload Customer Dataset CSV File for Batch Prediction", type=["csv"], key="file_uploader_tab2")

    if uploaded_file is not None:
        batch_df = pd.read_csv(uploaded_file)

        if "Unnamed: 0" in batch_df.columns:
            batch_df = batch_df.drop(columns=["Unnamed: 0"])

        st.info(f"Loaded customer portfolio dataset with **{len(batch_df)}** records.")

        threshold = st.slider(
            "Select Churn Probability Threshold",
            min_value=0.01,
            max_value=0.95,
            value=0.40,
            step=0.01,
            help="Threshold adjust karke model ki sensitivity control kar sakte ho.",
            key="threshold_slider_tab2",
        )

        target_col = None
        for col in batch_df.columns:
            cleaned_name = col.strip().lower().replace(" ", "_")
            if cleaned_name in ["churn", "target", "churn_value"]:
                target_col = col
                break

        if target_col:
            batch_df_features = batch_df.drop(columns=[target_col])
        else:
            batch_df_features = batch_df

        # 🌟 Corrected function name here:
        processed_batch = engineer_features(batch_df_features)
        batch_probs = model.predict_proba(processed_batch)[:, 1]

        batch_df["Churn_Probability_%"] = (batch_probs * 100).round(2)
        batch_df["Predicted_Churn_Status"] = (batch_probs >= threshold).astype(int)

        total_customers = len(batch_df)
        high_risk_customers = (batch_df["Predicted_Churn_Status"] == 1).sum()
        safe_customers = (batch_df["Predicted_Churn_Status"] == 0).sum()

        st.markdown("---")
        st.subheader("📊 Portfolio Churn Summary")

        col1, col2, col3 = st.columns(3)
        col1.metric("Total Customers Audited", total_customers)
        col2.metric("⚠️ High Risk (Likely to Churn)", int(high_risk_customers), delta_color="inverse")
        col3.metric("✅ Safe (Retained)", int(safe_customers), delta_color="normal")

        if target_col:
            raw_target = batch_df[target_col]
            if raw_target.dtype == object or raw_target.isin(['Yes', 'No', 'yes', 'no']).any():
                actuals = raw_target.map({"Yes": 1, "No": 0, "yes": 1, "no": 0}).fillna(0).values
            else:
                actuals = raw_target.values
                
            preds = batch_df["Predicted_Churn_Status"].values

            tn, fp, fn, tp = confusion_matrix(actuals, preds).ravel()
            acc = accuracy_score(actuals, preds) * 100
            correct_count = (preds == actuals).sum()
            wrong_count = total_customers - correct_count

            st.markdown("---")
            st.subheader("🎯 Detailed Error Analysis (FP & FN Breakdown)")

            m1, m2, m3, m4, m5 = st.columns(5)
            m1.metric("✅ Correct", correct_count)
            m2.metric("❌ Wrong", wrong_count, delta_color="inverse")
            m3.metric("⚠️ False Positives (FP)", int(fp), help="Model ne loyal customer ko churn bol diya")
            m4.metric("🚨 False Negatives (FN)", int(fn), help="Model ne churn hone wale customer ko safe bol diya")
            m5.metric("Model Accuracy", f"{acc:.2f}%")

        st.markdown("---")
        st.success("Batch portfolio processing complete!")

        st.subheader("🔎 Filter & Search Audited Customers")
        risk_filter = st.selectbox(
            "Filter Customers by Status:",
            ["All Customers", "🚨 High Risk Churners Only", "✅ Safe Customers Only"],
            key="risk_filter_tab2",
        )

        if risk_filter == "🚨 High Risk Churners Only":
            filtered_df = batch_df[batch_df["Predicted_Churn_Status"] == 1]
        elif risk_filter == "✅ Safe Customers Only":
            filtered_df = batch_df[batch_df["Predicted_Churn_Status"] == 0]
        else:
            filtered_df = batch_df

        st.dataframe(filtered_df, use_container_width=True)

        csv_output = batch_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "📥 Download Batch Audit Report (CSV)",
            csv_output,
            "Customer_Churn_Batch_Results.csv",
            "text/csv",
            key="download_btn_tab2",
        )