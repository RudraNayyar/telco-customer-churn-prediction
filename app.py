import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

st.set_page_config(page_title="Telco Customer Churn Prediction", layout="centered")

# --- Sidebar ---
st.sidebar.title("About This Project")
st.sidebar.info(
    """
    **Telco Customer Churn Prediction**
    
    Enter customer details to predict the likelihood of churn. This project uses a machine learning model trained on the Kaggle Telco Churn dataset.
    """
)

# --- Load Model ---
def load_model():
    model_path = os.path.join('models', 'churn_lightgbm_tuned.pkl')
    if not os.path.exists(model_path):
        st.error("Trained model not found! Please train the model first.")
        st.stop()
    model = joblib.load(model_path)
    return model

model = load_model()

# --- Feature List (from preprocessing) ---
# These should match the columns after preprocessing and one-hot encoding
feature_names = joblib.load('models/feature_names.pkl')

# --- User Input Form ---
st.title("Telco Customer Churn Prediction")
st.write("Fill in the customer details below:")

with st.form("churn_form"):
    col1, col2 = st.columns(2)
    with col1:
        gender = st.selectbox("Gender", ["Female", "Male"])
        senior = st.selectbox("Senior Citizen", ["No", "Yes"])
        partner = st.selectbox("Has Partner", ["No", "Yes"])
        dependents = st.selectbox("Has Dependents", ["No", "Yes"])
        tenure = st.slider("Tenure (months)", 0, 72, 12)
        phone = st.selectbox("Phone Service", ["Yes", "No"])
        multiple = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
        internet = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
        online_sec = st.selectbox("Online Security", ["No", "Yes"])
        online_bkp = st.selectbox("Online Backup", ["No", "Yes"])
        device = st.selectbox("Device Protection", ["No", "Yes"])
    with col2:
        tech = st.selectbox("Tech Support", ["No", "Yes"])
        stream_tv = st.selectbox("Streaming TV", ["No", "Yes"])
        stream_mov = st.selectbox("Streaming Movies", ["No", "Yes"])
        contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
        paperless = st.selectbox("Paperless Billing", ["Yes", "No"])
        payment = st.selectbox("Payment Method", [
            "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
        ])
        monthly = st.number_input("Monthly Charges", min_value=0.0, max_value=200.0, value=50.0)
        total = st.number_input("Total Charges", min_value=0.0, max_value=10000.0, value=500.0)
    submitted = st.form_submit_button("Predict Churn")

# --- Prepare Input for Model ---
def preprocess_input():
    row = dict.fromkeys(feature_names, 0)
    row['tenure'] = tenure
    row['MonthlyCharges'] = monthly
    row['TotalCharges'] = total
    row['gender_Male'] = 1 if gender == 'Male' else 0
    row['SeniorCitizen'] = 1 if senior == 'Yes' else 0
    row['Partner_Yes'] = 1 if partner == 'Yes' else 0
    row['Dependents_Yes'] = 1 if dependents == 'Yes' else 0
    row['PhoneService_Yes'] = 1 if phone == 'Yes' else 0
    row['MultipleLines_No phone service'] = 1 if multiple == 'No phone service' else 0
    row['MultipleLines_Yes'] = 1 if multiple == 'Yes' else 0
    row['InternetService_Fiber optic'] = 1 if internet == 'Fiber optic' else 0
    row['InternetService_No'] = 1 if internet == 'No' else 0
    row['OnlineSecurity_Yes'] = 1 if online_sec == 'Yes' else 0
    row['OnlineBackup_Yes'] = 1 if online_bkp == 'Yes' else 0
    row['DeviceProtection_Yes'] = 1 if device == 'Yes' else 0
    row['TechSupport_Yes'] = 1 if tech == 'Yes' else 0
    row['StreamingTV_Yes'] = 1 if stream_tv == 'Yes' else 0
    row['StreamingMovies_Yes'] = 1 if stream_mov == 'Yes' else 0
    row['Contract_One year'] = 1 if contract == 'One year' else 0
    row['Contract_Two year'] = 1 if contract == 'Two year' else 0
    row['PaperlessBilling_Yes'] = 1 if paperless == 'Yes' else 0
    row['PaymentMethod_Credit card (automatic)'] = 1 if payment == 'Credit card (automatic)' else 0
    row['PaymentMethod_Electronic check'] = 1 if payment == 'Electronic check' else 0
    row['PaymentMethod_Mailed check'] = 1 if payment == 'Mailed check' else 0
    input_df = pd.DataFrame([row])
    # Ensure all columns are present and in the right order
    input_df = input_df.reindex(columns=feature_names, fill_value=0)
    return input_df

# --- Prediction and Output ---
if submitted:
    input_df = preprocess_input()
    # Predict probability
    prob = model.predict_proba(input_df)[0][1]
    st.subheader("Prediction Result")
    st.write(f"**Churn Probability:** {prob*100:.2f}%")
    if prob > 0.7:
        st.error("High risk of churn! Consider retention strategies.")
    elif prob > 0.4:
        st.warning("Medium risk of churn.")
    else:
        st.success("Low risk of churn.")

    # Feature importance chart
    if hasattr(model, 'feature_importances_'):
        st.subheader("Feature Importances")
        importances = model.feature_importances_
        feat_imp = pd.Series(importances, index=feature_names).sort_values(ascending=False)
        st.bar_chart(feat_imp.head(10))
    else:
        st.info("Feature importances not available for this model.")

st.markdown("---")
# st.caption("Made with Streamlit for a college project. ") 