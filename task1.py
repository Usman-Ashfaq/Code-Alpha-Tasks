import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report

# Generate sample credit data
np.random.seed(42)
n_samples = 5000

data = {
    'income': np.random.normal(50000, 20000, n_samples),
    'age': np.random.randint(18, 70, n_samples),
    'employment_years': np.random.randint(0, 40, n_samples),
    'debt_to_income': np.random.uniform(0, 0.5, n_samples),
    'credit_utilization': np.random.uniform(0, 1, n_samples),
    'late_payments': np.random.poisson(2, n_samples),
    'num_credit_accounts': np.random.randint(1, 10, n_samples),
    'loan_amount': np.random.normal(15000, 10000, n_samples)
}

df = pd.DataFrame(data)

# Create target variable (creditworthy or not)
df['creditworthy'] = ((df['income'] > 30000) & 
                      (df['debt_to_income'] < 0.4) & 
                      (df['late_payments'] < 3) & 
                      (df['credit_utilization'] < 0.7)).astype(int)

# Feature engineering
df['income_per_account'] = df['income'] / df['num_credit_accounts']
df['debt_ratio'] = df['loan_amount'] / df['income']

# Prepare features and target
X = df.drop('creditworthy', axis=1)
y = df['creditworthy']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Logistic Regression Model
log_reg = LogisticRegression(random_state=42)
log_reg.fit(X_train_scaled, y_train)
y_pred_log = log_reg.predict(X_test_scaled)
y_pred_proba_log = log_reg.predict_proba(X_test_scaled)[:, 1]

print("Logistic Regression Results:")
print(f"Accuracy: {accuracy_score(y_test, y_pred_log):.4f}")
print(f"Precision: {precision_score(y_test, y_pred_log):.4f}")
print(f"Recall: {recall_score(y_test, y_pred_log):.4f}")
print(f"F1-Score: {f1_score(y_test, y_pred_log):.4f}")
print(f"ROC-AUC: {roc_auc_score(y_test, y_pred_proba_log):.4f}")
print()

# Random Forest Model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train_scaled, y_train)
y_pred_rf = rf_model.predict(X_test_scaled)
y_pred_proba_rf = rf_model.predict_proba(X_test_scaled)[:, 1]

print("Random Forest Results:")
print(f"Accuracy: {accuracy_score(y_test, y_pred_rf):.4f}")
print(f"Precision: {precision_score(y_test, y_pred_rf):.4f}")
print(f"Recall: {recall_score(y_test, y_pred_rf):.4f}")
print(f"F1-Score: {f1_score(y_test, y_pred_rf):.4f}")
print(f"ROC-AUC: {roc_auc_score(y_test, y_pred_proba_rf):.4f}")

# Feature importance
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': rf_model.feature_importances_
}).sort_values('importance', ascending=False)
print("\nTop 5 Important Features:")
print(feature_importance.head(5))