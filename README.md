# Telco Customer Churn Prediction

Predict whether a customer will churn (leave) a telecom service using machine learning. This project uses the Kaggle Telco Customer Churn dataset and provides an interactive web app built with Streamlit.

## 🚀 Live Demo

   [Try the app on Streamlit Cloud!](https://telco-customer-churn-prediction-mgqwyx2znqu8tmjaear77c.streamlit.app/)
   
## 🚀 Features
- Cleaned and preprocessed data
- Multiple machine learning models (Logistic Regression, Random Forest, XGBoost, LightGBM)
- Hyperparameter tuning for best performance
- Feature importance insights
- Beautiful Streamlit app for interactive predictions

## 📊 Demo
Run the app locally or deploy to Streamlit Cloud to try it out!

---

## 🛠️ How to Run Locally
1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/telco-customer-churn-prediction.git
   cd telco-customer-churn-prediction
   ```
2. **Set up a virtual environment (optional but recommended):**
   ```bash
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   # On Mac/Linux:
   source venv/bin/activate
   ```
3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
4. **Add the dataset:**
   - Download `WA_Fn-UseC_-Telco-Customer-Churn.csv` from [Kaggle](https://www.kaggle.com/blastchar/telco-customer-churn)
   - Place it in the `data/` folder
5. **Preprocess and train the model:**
   ```bash
   python src/preprocessing.py
   python src/train_model.py
   ```
6. **Run the Streamlit app:**
   ```bash
   streamlit run app.py
   ```

---

## 🌐 Deploy on Streamlit Cloud
1. Push your code to GitHub (including `requirements.txt` and `app.py`)
2. Go to [Streamlit Cloud](https://streamlit.io/cloud) and connect your repo
3. Set the main file as `app.py` and deploy!

---

## 📁 Project Structure
```
telco-customer-churn-prediction/
│
├── data/                  # Raw and processed data
├── models/                # Saved models and feature names
├── src/                   # Preprocessing and training scripts
├── app.py                 # Streamlit app
├── requirements.txt       # Python dependencies
└── README.md              # Project info (this file)
```

---

## 📚 Credits
- Dataset: [Kaggle Telco Customer Churn](https://www.kaggle.com/blastchar/telco-customer-churn)
- Built with: Python, pandas, scikit-learn, xgboost, lightgbm, streamlit

---

Feel free to use, modify, and share this project! 
