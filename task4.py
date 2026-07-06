import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Generate sample diabetes dataset
np.random.seed(42)
n_samples = 1000

diabetes_data = {
    'age': np.random.randint(20, 80, n_samples),
    'bmi': np.random.uniform(15, 45, n_samples),
    'blood_pressure': np.random.uniform(80, 180, n_samples),
    'glucose': np.random.uniform(70, 200, n_samples),
    'insulin': np.random.uniform(0, 300, n_samples),
    'pregnancies': np.random.randint(0, 15, n_samples),
    'skin_thickness': np.random.uniform(10, 50, n_samples),
    'diabetes_pedigree': np.random.uniform(0, 2.5, n_samples)
}

df = pd.DataFrame(diabetes_data)

# Create target based on glucose and bmi
df['has_diabetes'] = ((df['glucose'] > 126) & (df['bmi'] > 30) & (df['blood_pressure'] > 130)).astype(int)

# Or use UCI dataset loading
def load_uci_diabetes():
    try:
        # Pima Indians Diabetes Database
        url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
        column_names = ['pregnancies', 'glucose', 'blood_pressure', 'skin_thickness', 
                       'insulin', 'bmi', 'diabetes_pedigree', 'age', 'outcome']
        df_uci = pd.read_csv(url, names=column_names)
        return df_uci
    except:
        print("Could not load UCI dataset, using generated dataset")
        return None

# Try to load real dataset
df_uci = load_uci_diabetes()

if df_uci is not None:
    df = df_uci
    X = df.drop('outcome', axis=1)
    y = df['outcome']
else:
    X = df.drop('has_diabetes', axis=1)
    y = df['has_diabetes']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define models
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'SVM': SVC(probability=True, random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'XGBoost': GradientBoostingClassifier(random_state=42)
}

# Train and evaluate each model
results = []
for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    # Cross validation
    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5)
    
    results.append({
        'Model': name,
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1,
        'CV Mean': cv_scores.mean(),
        'CV Std': cv_scores.std()
    })
    
    print(f"{name}:")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall: {recall:.4f}")
    print(f"  F1-Score: {f1:.4f}")
    print(f"  CV Mean: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
    print()

# Convert results to DataFrame
results_df = pd.DataFrame(results)
print("\nOverall Results:")
print(results_df.to_string(index=False))

# Feature importance for Random Forest
rf_model = models['Random Forest']
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': rf_model.feature_importances_
}).sort_values('importance', ascending=False)

print("\nTop 5 Important Features:")
print(feature_importance.head(5))

# Confusion Matrix for best model
best_model_name = results_df.loc[results_df['Accuracy'].idxmax(), 'Model']
best_model = models[best_model_name]
y_pred_best = best_model.predict(X_test_scaled)

plt.figure(figsize=(8, 6))
cm = confusion_matrix(y_test, y_pred_best)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title(f'Confusion Matrix - {best_model_name}')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.savefig('confusion_matrix.png')
plt.show()

# Save model
import joblib
joblib.dump(best_model, 'disease_prediction_model.pkl')
print(f"\nBest model ({best_model_name}) saved as 'disease_prediction_model.pkl'")

# Function to predict new patient data
def predict_disease(new_patient_data):
    loaded_model = joblib.load('disease_prediction_model.pkl')
    loaded_scaler = scaler  # Use the fitted scaler
    
    if isinstance(new_patient_data, dict):
        new_df = pd.DataFrame([new_patient_data])
    else:
        new_df = pd.DataFrame(new_patient_data)
    
    new_scaled = loaded_scaler.transform(new_df)
    prediction = loaded_model.predict(new_scaled)
    probability = loaded_model.predict_proba(new_scaled)
    
    return prediction, probability

# Example prediction
sample_patient = {
    'pregnancies': 5,
    'glucose': 140,
    'blood_pressure': 80,
    'skin_thickness': 25,
    'insulin': 100,
    'bmi': 32,
    'diabetes_pedigree': 0.8,
    'age': 45
}

pred, prob = predict_disease(sample_patient)
print(f"\nSample prediction for patient:")
print(f"Predicted: {'Has Diabetes' if pred[0] == 1 else 'No Diabetes'}")
print(f"Probability of having diabetes: {prob[0][1]:.2%}")