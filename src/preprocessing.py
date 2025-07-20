import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os
import joblib


file_path = os.path.join('data', 'WA_Fn-UseC_-Telco-Customer-Churn.csv')
data = pd.read_csv(file_path)


data['TotalCharges'] = pd.to_numeric(data['TotalCharges'], errors='coerce')
data['TotalCharges'] = data['TotalCharges'].fillna(0)


data = data.drop('customerID', axis=1)


data['Churn'] = data['Churn'].map({'Yes': 1, 'No': 0})


cat_cols = data.select_dtypes(include=['object']).columns.tolist()
data = pd.get_dummies(data, columns=cat_cols, drop_first=True)


X = data.drop('Churn', axis=1)
y = data['Churn']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


scaler = StandardScaler()
num_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
X_test[num_cols] = scaler.transform(X_test[num_cols])


X_train.to_csv('data/X_train.csv', index=False)
X_test.to_csv('data/X_test.csv', index=False)
y_train.to_csv('data/y_train.csv', index=False)
y_test.to_csv('data/y_test.csv', index=False)

joblib.dump(X_train.columns.tolist(), 'models/feature_names.pkl')

print('Done preprocessing!') 