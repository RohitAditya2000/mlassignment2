"""
Bank Marketing Classification - Model Training Script
This script trains 6 different ML models on the Bank Marketing dataset
and saves them for use in the Streamlit application.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (accuracy_score, roc_auc_score, precision_score, 
                             recall_score, f1_score, matthews_corrcoef, 
                             confusion_matrix, classification_report)
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

# Create directory for saving models
os.makedirs('model/trained_models', exist_ok=True)

print("="*70)
print("BANK MARKETING CLASSIFICATION - MODEL TRAINING")
print("="*70)

# Load the dataset
print("\n[1/6] Loading dataset...")
df = pd.read_csv('bank/bank-full.csv', sep=';')
print(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")

# Display basic info
print("\nDataset Info:")
print(f"Target distribution:\n{df['y'].value_counts()}")
print(f"\nFeatures: {list(df.columns)}")

# Preprocessing
print("\n[2/6] Preprocessing data...")

# Encode categorical variables
label_encoders = {}
categorical_columns = df.select_dtypes(include=['object']).columns.tolist()
categorical_columns.remove('y')  # Don't encode target yet

for col in categorical_columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Encode target variable
df['y'] = df['y'].map({'no': 0, 'yes': 1})

# Save label encoders for use in the app
joblib.dump(label_encoders, 'model/trained_models/label_encoders.pkl')
print(f"Encoded {len(categorical_columns)} categorical features")

# Split features and target
X = df.drop('y', axis=1)
y = df['y']

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Training set: {X_train.shape[0]} samples")
print(f"Test set: {X_test.shape[0]} samples")

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
joblib.dump(scaler, 'model/trained_models/scaler.pkl')
print("Features scaled and scaler saved")

# Save feature names for the app
feature_names = X.columns.tolist()
joblib.dump(feature_names, 'model/trained_models/feature_names.pkl')

# Save test data for the app demo
test_df = X_test.copy()
test_df['y'] = y_test.values
test_df.to_csv('model/trained_models/test_data.csv', index=False)
print("Test data saved for app demo")

# Define models
print("\n[3/6] Initializing models...")
models = {
    'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
    'Decision Tree': DecisionTreeClassifier(random_state=42, max_depth=10),
    'kNN': KNeighborsClassifier(n_neighbors=5),
    'Naive Bayes': GaussianNB(),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, max_depth=15),
    'XGBoost': XGBClassifier(n_estimators=100, random_state=42, max_depth=7, 
                             learning_rate=0.1, eval_metric='logloss')
}

# Train and evaluate models
print("\n[4/6] Training models...")
results = []

for name, model in models.items():
    print(f"\n{'='*70}")
    print(f"Training: {name}")
    print('='*70)
    
    # Train
    if name in ['Logistic Regression', 'kNN', 'Naive Bayes']:
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
    else:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred_proba)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    mcc = matthews_corrcoef(y_test, y_pred)
    
    print(f"✓ Accuracy:  {accuracy:.3f}")
    print(f"✓ AUC:       {auc:.3f}")
    print(f"✓ Precision: {precision:.3f}")
    print(f"✓ Recall:    {recall:.3f}")
    print(f"✓ F1 Score:  {f1:.3f}")
    print(f"✓ MCC:       {mcc:.3f}")
    
    # Save results
    results.append({
        'Model': name,
        'Accuracy': round(accuracy, 3),
        'AUC': round(auc, 3),
        'Precision': round(precision, 3),
        'Recall': round(recall, 3),
        'F1': round(f1, 3),
        'MCC': round(mcc, 3)
    })
    
    # Save model
    model_filename = f"model/trained_models/{name.replace(' ', '_').lower()}.pkl"
    joblib.dump(model, model_filename)
    print(f"✓ Model saved: {model_filename}")

print("\n[5/6] Saving results summary...")
results_df = pd.DataFrame(results)
results_df.to_csv('model/trained_models/model_results.csv', index=False)
print("\nFinal Results Summary:")
print("="*70)
print(results_df.to_string(index=False))
print("="*70)

# Find best model
best_model_mcc = results_df.loc[results_df['MCC'].idxmax(), 'Model']
best_model_f1 = results_df.loc[results_df['F1'].idxmax(), 'Model']
print(f"\n✨ Best Model (MCC): {best_model_mcc}")
print(f"✨ Best Model (F1):  {best_model_f1}")

print("\n[6/6] Training complete!")
print(f"\n✓ All models saved in: model/trained_models/")
print(f"✓ Results saved in: model/trained_models/model_results.csv")
print(f"✓ Ready to run: streamlit run app.py")
print("\n" + "="*70)
