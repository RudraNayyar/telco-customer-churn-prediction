import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import time

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
feature_names = joblib.load('models/feature_names.pkl')

st.title("Telco Customer Churn Prediction")

# --- Tabs for Single/Batch Prediction (future-proof) ---
tabs = st.tabs(["Single Prediction", "About"])
with tabs[0]:
    with st.form("churn_form"):
        # --- Personal Info ---
        with st.expander("Personal Info", expanded=True):
            col1, col2 = st.columns(2)
            with col1:
                gender = st.selectbox("Gender", ["Female", "Male"], index=0, help="Customer's gender")
                senior = st.selectbox("Senior Citizen", ["No", "Yes"], index=0, help="Is the customer a senior citizen?")
                partner = st.selectbox("Has Partner", ["No", "Yes"], index=0, help="Does the customer have a partner?")
                dependents = st.selectbox("Has Dependents", ["No", "Yes"], index=0, help="Does the customer have dependents?")
            with col2:
                tenure = st.slider("Tenure (months)", 0, 72, 24, help="Number of months the customer has stayed")
        # --- Services ---
        with st.expander("Services", expanded=True):
            col1, col2 = st.columns(2)
            with col1:
                phone = st.selectbox("Phone Service", ["Yes", "No"], index=0, help="Does the customer have phone service?")
                multiple = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"], index=0, help="Does the customer have multiple lines?")
                internet = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"], index=0, help="Type of internet service")
                online_sec = st.selectbox("Online Security", ["No", "Yes"], index=0, help="Does the customer have online security?")
                online_bkp = st.selectbox("Online Backup", ["No", "Yes"], index=0, help="Does the customer have online backup?")
            with col2:
                device = st.selectbox("Device Protection", ["No", "Yes"], index=0, help="Does the customer have device protection?")
                tech = st.selectbox("Tech Support", ["No", "Yes"], index=0, help="Does the customer have tech support?")
                stream_tv = st.selectbox("Streaming TV", ["No", "Yes"], index=0, help="Does the customer have streaming TV?")
                stream_mov = st.selectbox("Streaming Movies", ["No", "Yes"], index=0, help="Does the customer have streaming movies?")
        # --- Billing ---
        with st.expander("Billing", expanded=True):
            col1, col2 = st.columns(2)
            with col1:
                contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"], index=0, help="Type of contract")
                paperless = st.selectbox("Paperless Billing", ["Yes", "No"], index=0, help="Is billing paperless?")
            with col2:
                payment = st.selectbox("Payment Method", [
                    "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
                ], index=0, help="Customer's payment method")
                monthly = st.number_input("Monthly Charges", min_value=0.0, max_value=200.0, value=70.0, help="Current monthly charges")
                total = st.number_input("Total Charges", min_value=0.0, max_value=10000.0, value=1500.0, help="Total charges to date")
        submitted = st.form_submit_button("Predict Churn")

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
        input_df = input_df.reindex(columns=feature_names, fill_value=0)
        return input_df

    if submitted:
        input_df = preprocess_input()
        with st.spinner('Predicting...'):
            progress = st.progress(0)
            for percent in range(1, 101, 10):
                time.sleep(0.05)
                progress.progress(percent)
            prob = model.predict_proba(input_df)[0][1]
            progress.empty()
        st.subheader("Prediction Result")
        if prob > 0.7:
            st.markdown(f"<h2 style='color:red;'>🔴 High risk of churn! ({prob*100:.2f}%)</h2>", unsafe_allow_html=True)
            st.info("Suggested action: Offer a discount or personalized retention call.")
            st.toast("⚠️ High risk detected! Take action.")
        elif prob > 0.4:
            st.markdown(f"<h2 style='color:orange;'>🟡 Medium risk of churn. ({prob*100:.2f}%)</h2>", unsafe_allow_html=True)
            st.info("Suggested action: Send a satisfaction survey or check-in email.")
            st.snow()
        else:
            st.markdown(f"<h2 style='color:green;'>🟢 Low risk of churn. ({prob*100:.2f}%)</h2>", unsafe_allow_html=True)
            st.success("Suggested action: Thank the customer for loyalty!")
            st.balloons()
        # Feature importance chart
        if hasattr(model, 'feature_importances_'):
            st.subheader("Feature Importances")
            importances = model.feature_importances_
            feat_imp = pd.Series(importances, index=feature_names).sort_values(ascending=False)
            st.bar_chart(feat_imp.head(10))
        else:
            st.info("Feature importances not available for this model.")

with tabs[1]:
    st.markdown("""
    ### About
    This app predicts customer churn for a telecom company using a machine learning model trained on the Kaggle Telco Churn dataset. 
    Enter customer details in the form to get a churn risk prediction and suggested action.
    """)

st.markdown("---")
