import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import RandomizedSearchCV
import joblib
import os
import xgboost as xgb
import lightgbm as lgb


X_train = pd.read_csv('data/X_train.csv')
X_test = pd.read_csv('data/X_test.csv')
y_train = pd.read_csv('data/y_train.csv').values.ravel()
y_test = pd.read_csv('data/y_test.csv').values.ravel()

results = {}
models = {}


lr_model = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')
lr_model.fit(X_train, y_train)
lr_preds = lr_model.predict(X_test)
lr_probs = lr_model.predict_proba(X_test)[:, 1]
results['Logistic Regression'] = {
    'accuracy': accuracy_score(y_test, lr_preds),
    'precision': precision_score(y_test, lr_preds),
    'recall': recall_score(y_test, lr_preds),
    'f1': f1_score(y_test, lr_preds),
    'roc_auc': roc_auc_score(y_test, lr_probs)
}
models['Logistic Regression'] = lr_model


rf_model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
rf_model.fit(X_train, y_train)
rf_preds = rf_model.predict(X_test)
rf_probs = rf_model.predict_proba(X_test)[:, 1]
results['Random Forest'] = {
    'accuracy': accuracy_score(y_test, rf_preds),
    'precision': precision_score(y_test, rf_preds),
    'recall': recall_score(y_test, rf_preds),
    'f1': f1_score(y_test, rf_preds),
    'roc_auc': roc_auc_score(y_test, rf_probs)
}
models['Random Forest'] = rf_model


xgb_model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42, scale_pos_weight=(sum(y_train==0)/sum(y_train==1)))
xgb_model.fit(X_train, y_train)
xgb_preds = xgb_model.predict(X_test)
xgb_probs = xgb_model.predict_proba(X_test)[:, 1]
results['XGBoost'] = {
    'accuracy': accuracy_score(y_test, xgb_preds),
    'precision': precision_score(y_test, xgb_preds),
    'recall': recall_score(y_test, xgb_preds),
    'f1': f1_score(y_test, xgb_preds),
    'roc_auc': roc_auc_score(y_test, xgb_probs)
}
models['XGBoost'] = xgb_model


lgb_model = lgb.LGBMClassifier(random_state=42, class_weight='balanced')
lgb_model.fit(X_train, y_train)
lgb_preds = lgb_model.predict(X_test)
lgb_probs = lgb_model.predict_proba(X_test)[:, 1]
results['LightGBM'] = {
    'accuracy': accuracy_score(y_test, lgb_preds),
    'precision': precision_score(y_test, lgb_preds),
    'recall': recall_score(y_test, lgb_preds),
    'f1': f1_score(y_test, lgb_preds),
    'roc_auc': roc_auc_score(y_test, lgb_probs)
}
models['LightGBM'] = lgb_model


print('\nModel Results:')
for name, metrics in results.items():
    print(f'\n{name}:')
    for metric, value in metrics.items():
        print(f'  {metric}: {value:.4f}')


top_models = sorted(results.items(), key=lambda x: x[1]['roc_auc'], reverse=True)[:2]
tuned_models = {}
for name, _ in top_models:
    print(f'\nTuning {name}...')
    if name == 'XGBoost':
        params = {
            'n_estimators': [100, 200, 300],
            'max_depth': [3, 5, 7],
            'learning_rate': [0.01, 0.05, 0.1],
            'subsample': [0.8, 1.0],
            'colsample_bytree': [0.8, 1.0]
        }
        search = RandomizedSearchCV(xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42, scale_pos_weight=(sum(y_train==0)/sum(y_train==1))),
                                   param_distributions=params, n_iter=10, scoring='roc_auc', cv=3, random_state=42, n_jobs=-1)
        search.fit(X_train, y_train)
        tuned_models[name] = search.best_estimator_
    elif name == 'LightGBM':
        params = {
            'n_estimators': [100, 200, 300],
            'max_depth': [3, 5, 7, -1],
            'learning_rate': [0.01, 0.05, 0.1],
            'num_leaves': [31, 50, 100],
            'subsample': [0.8, 1.0]
        }
        search = RandomizedSearchCV(lgb.LGBMClassifier(random_state=42, class_weight='balanced'),
                                   param_distributions=params, n_iter=10, scoring='roc_auc', cv=3, random_state=42, n_jobs=-1)
        search.fit(X_train, y_train)
        tuned_models[name] = search.best_estimator_
    elif name == 'Random Forest':
        params = {
            'n_estimators': [100, 200, 300],
            'max_depth': [5, 10, 20, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
        search = RandomizedSearchCV(RandomForestClassifier(random_state=42, class_weight='balanced'),
                                   param_distributions=params, n_iter=10, scoring='roc_auc', cv=3, random_state=42, n_jobs=-1)
        search.fit(X_train, y_train)
        tuned_models[name] = search.best_estimator_
    elif name == 'Logistic Regression':
        params = {
            'C': [0.01, 0.1, 1, 10, 100],
            'solver': ['liblinear', 'lbfgs']
        }
        search = RandomizedSearchCV(LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced'),
                                   param_distributions=params, n_iter=10, scoring='roc_auc', cv=3, random_state=42, n_jobs=-1)
        search.fit(X_train, y_train)
        tuned_models[name] = search.best_estimator_
    else:
        continue
    # Evaluate tuned model
    preds = tuned_models[name].predict(X_test)
    probs = tuned_models[name].predict_proba(X_test)[:, 1]
    print(f"  Tuned ROC-AUC: {roc_auc_score(y_test, probs):.4f}")


best_name = None
best_model = None
best_auc = 0
for name, model in tuned_models.items():
    probs = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, probs)
    if auc > best_auc:
        best_auc = auc
        best_name = name
        best_model = model
if best_model is not None:
    os.makedirs('models', exist_ok=True)
    model_path = f"models/churn_{best_name.replace(' ', '_').lower()}_tuned.pkl"
    joblib.dump(best_model, model_path)
    print(f'\nBest tuned model ({best_name}) saved to {model_path}')
    if hasattr(best_model, 'feature_importances_'):
        importances = best_model.feature_importances_
        feat_names = X_train.columns
        feat_imp = pd.Series(importances, index=feat_names).sort_values(ascending=False)
        print('\nTop 10 Feature Importances:')
        print(feat_imp.head(10))
else:
    print('\nNo tuned model was found.') 
