# ============================================
# CREDIT SCORING MODEL (END-TO-END)
# ============================================

import pandas as pd
import numpy as np

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Sklearn
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, classification_report, confusion_matrix
)

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# Handle imbalance
from imblearn.over_sampling import SMOTE

# Save model
import joblib

import warnings
warnings.filterwarnings('ignore')


# ============================================
# 1. LOAD DATASET
# ============================================

df = pd.read_csv("credit_risk_dataset.csv")

print("Dataset Shape:", df.shape)
print(df.head())


# ============================================
# 2. DATA CLEANING
# ============================================

# Rename target column (if needed)
df.rename(columns={'loan_status': 'target'}, inplace=True)

# Drop missing values
df.dropna(inplace=True)

print("\nAfter cleaning:", df.shape)


# ============================================
# 3. FEATURE ENGINEERING
# ============================================

# Debt-to-Income ratio
df['debt_to_income'] = df['loan_amnt'] / (df['person_income'] + 1)

# Credit history ratio
df['credit_history_ratio'] = df['cb_person_cred_hist_length'] / (df['person_age'] + 1)

# Encode categorical features
df = pd.get_dummies(df, drop_first=True)


# ============================================
# 4. SPLIT DATA
# ============================================

X = df.drop('target', axis=1)
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("\nTrain size:", X_train.shape)
print("Test size:", X_test.shape)


# ============================================
# 5. HANDLE CLASS IMBALANCE
# ============================================

smote = SMOTE(random_state=42)
X_train, y_train = smote.fit_resample(X_train, y_train)

print("\nAfter SMOTE:", X_train.shape)


# ============================================
# 6. SCALING (for Logistic Regression)
# ============================================

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)


# ============================================
# 7. TRAIN MODELS
# ============================================

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(max_depth=6),
    "Random Forest": RandomForestClassifier(n_estimators=200)
}

results = {}

for name, model in models.items():
    
    X_tr = X_train_scaled if name == "Logistic Regression" else X_train
    X_te = X_test_scaled if name == "Logistic Regression" else X_test

    model.fit(X_tr, y_train)

    y_pred = model.predict(X_te)
    y_prob = model.predict_proba(X_te)[:, 1]

    results[name] = {
        "model": model,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1": f1_score(y_test, y_pred),
        "ROC-AUC": roc_auc_score(y_test, y_prob),
        "y_pred": y_pred
    }

    print(f"\n{'='*50}")
    print(name)
    print('='*50)
    print(classification_report(y_test, y_pred))


# ============================================
# 8. MODEL COMPARISON
# ============================================

metrics_df = pd.DataFrame({
    name: {k: v for k, v in vals.items() if k not in ('model', 'y_pred')}
    for name, vals in results.items()
}).T

print("\nModel Comparison:\n", metrics_df.round(4))


# ============================================
# 9. CROSS VALIDATION (Random Forest)
# ============================================

rf_model = results["Random Forest"]["model"]

cv_scores = cross_val_score(rf_model, X, y, cv=5, scoring='roc_auc')

print("\nCross-Validation ROC-AUC:", cv_scores.mean())


# ============================================
# 10. VISUALIZATION
# ============================================

plt.figure(figsize=(10,5))

metrics_df[['Accuracy','Precision','Recall','F1','ROC-AUC']].plot(kind='bar')
plt.title("Model Performance Comparison")
plt.xticks(rotation=0)
plt.ylabel("Score")
plt.tight_layout()
plt.savefig("model_comparison.png", dpi=150, bbox_inches='tight')
plt.show()


# Confusion Matrix
best_model_name = metrics_df['ROC-AUC'].idxmax()
cm = confusion_matrix(y_test, results[best_model_name]['y_pred'])

sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title(f"Confusion Matrix ({best_model_name})")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=150, bbox_inches='tight')
plt.show()


# Feature Importance
importances = pd.Series(
    rf_model.feature_importances_,
    index=X.columns
).sort_values(ascending=False)

importances.head(10).plot(kind='barh')
plt.title("Top Features (Random Forest)")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig("feature_importance.png", dpi=150, bbox_inches='tight')
plt.show()


# ============================================
# 11. SAVE MODEL
# ============================================

joblib.dump(rf_model, "credit_model.pkl")
joblib.dump(scaler, "scaler.pkl")

print("\nModel saved successfully!")


# ============================================
# 12. PREDICT NEW CUSTOMER
# ============================================

sample = X_test.iloc[0:1]

prediction = rf_model.predict(sample)[0]
prob = rf_model.predict_proba(sample)[0][1]

print("\nNew Prediction:")
print("Creditworthy:", "Yes" if prediction == 1 else "No")
print("Confidence:", round(prob * 100, 2), "%")