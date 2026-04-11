# Credit Scoring Model
# Objective: Predict an individual's creditworthiness using past financial data.
# Approach: Classification algorithms - Logistic Regression, Decision Trees, Random Forest

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, classification_report, confusion_matrix
)
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


# ─────────────────────────────────────────────
# 1. Generate / Load Dataset
# ─────────────────────────────────────────────
# Using synthetic data that mimics real credit data.
# Replace this section with: df = pd.read_csv('your_dataset.csv')

np.random.seed(42)
n_samples = 1000

df = pd.DataFrame({
    'age':             np.random.randint(21, 65, n_samples),
    'income':          np.random.randint(20000, 120000, n_samples),
    'loan_amount':     np.random.randint(1000, 50000, n_samples),
    'loan_tenure':     np.random.randint(6, 60, n_samples),
    'existing_debts':  np.random.randint(0, 30000, n_samples),
    'missed_payments': np.random.randint(0, 10, n_samples),
    'credit_history':  np.random.randint(0, 15, n_samples),
    'employment_type': np.random.choice(['Salaried', 'Self-employed', 'Unemployed'], n_samples),
    'creditworthy':    np.random.randint(0, 2, n_samples)   # 0 = Not creditworthy, 1 = Creditworthy
})

print("Dataset shape:", df.shape)
print("\nFirst 5 rows:")
print(df.head())
print("\nClass distribution:")
print(df['creditworthy'].value_counts())


# ─────────────────────────────────────────────
# 2. Feature Engineering
# ─────────────────────────────────────────────

# Debt-to-Income Ratio
df['debt_to_income'] = df['existing_debts'] / (df['income'] + 1)

# Loan-to-Income Ratio
df['loan_to_income'] = df['loan_amount'] / (df['income'] + 1)

# Payment reliability score (inverse of missed payments)
df['payment_reliability'] = 1 / (df['missed_payments'] + 1)

# Encode categorical column
le = LabelEncoder()
df['employment_type_encoded'] = le.fit_transform(df['employment_type'])


# ─────────────────────────────────────────────
# 3. Prepare Features and Target
# ─────────────────────────────────────────────

feature_columns = [
    'age', 'income', 'loan_amount', 'loan_tenure',
    'existing_debts', 'missed_payments', 'credit_history',
    'employment_type_encoded', 'debt_to_income',
    'loan_to_income', 'payment_reliability'
]

X = df[feature_columns]
y = df['creditworthy']

# Train-test split (80/20)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Feature scaling (important for Logistic Regression)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

print(f"\nTraining samples: {X_train.shape[0]}")
print(f"Testing samples:  {X_test.shape[0]}")


# ─────────────────────────────────────────────
# 4. Train Models
# ─────────────────────────────────────────────

models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Decision Tree':       DecisionTreeClassifier(max_depth=5, random_state=42),
    'Random Forest':       RandomForestClassifier(n_estimators=100, random_state=42)
}

results = {}

for name, model in models.items():
    # Logistic Regression uses scaled data; tree-based models use raw
    X_tr = X_train_scaled if name == 'Logistic Regression' else X_train
    X_te = X_test_scaled  if name == 'Logistic Regression' else X_test

    model.fit(X_tr, y_train)
    y_pred      = model.predict(X_te)
    y_pred_prob = model.predict_proba(X_te)[:, 1]

    results[name] = {
        'Accuracy':  accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred),
        'Recall':    recall_score(y_test, y_pred),
        'F1-Score':  f1_score(y_test, y_pred),
        'ROC-AUC':   roc_auc_score(y_test, y_pred_prob),
        'model':     model,
        'y_pred':    y_pred
    }

    print(f"\n{'='*45}")
    print(f"  {name}")
    print(f"{'='*45}")
    print(classification_report(y_test, y_pred, target_names=['Not Creditworthy', 'Creditworthy']))


# ─────────────────────────────────────────────
# 5. Compare Model Performance
# ─────────────────────────────────────────────

metrics_df = pd.DataFrame({
    name: {k: v for k, v in vals.items() if k not in ('model', 'y_pred')}
    for name, vals in results.items()
}).T

print("\nModel Comparison:")
print(metrics_df.round(4).to_string())


# ─────────────────────────────────────────────
# 6. Visualizations
# ─────────────────────────────────────────────

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle('Credit Scoring Model — Performance Comparison', fontsize=14, fontweight='bold')

# (a) Bar chart — metric comparison
metrics_to_plot = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
x = np.arange(len(metrics_to_plot))
width = 0.25
colors = ['#378ADD', '#1D9E75', '#D85A30']

for i, (name, vals) in enumerate(results.items()):
    scores = [vals[m] for m in metrics_to_plot]
    axes[0].bar(x + i * width, scores, width, label=name, color=colors[i], alpha=0.85)

axes[0].set_xticks(x + width)
axes[0].set_xticklabels(metrics_to_plot, rotation=15)
axes[0].set_ylim(0, 1.1)
axes[0].set_title('Metric Comparison')
axes[0].legend()
axes[0].set_ylabel('Score')

# (b) Confusion matrix — best model (Random Forest)
best_model_name = max(results, key=lambda n: results[n]['ROC-AUC'])
cm = confusion_matrix(y_test, results[best_model_name]['y_pred'])
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[1],
            xticklabels=['Not Creditworthy', 'Creditworthy'],
            yticklabels=['Not Creditworthy', 'Creditworthy'])
axes[1].set_title(f'Confusion Matrix\n({best_model_name})')
axes[1].set_xlabel('Predicted')
axes[1].set_ylabel('Actual')

# (c) Feature importance — Random Forest
rf_model = results['Random Forest']['model']
importances = pd.Series(rf_model.feature_importances_, index=feature_columns).sort_values(ascending=True)
importances.plot(kind='barh', ax=axes[2], color='#1D9E75', alpha=0.85)
axes[2].set_title('Feature Importance\n(Random Forest)')
axes[2].set_xlabel('Importance Score')

plt.tight_layout()
plt.savefig('credit_scoring_results.png', dpi=150, bbox_inches='tight')
plt.show()
print("\nChart saved as 'credit_scoring_results.png'")


# ─────────────────────────────────────────────
# 7. Predict on a New Applicant
# ─────────────────────────────────────────────

new_applicant = pd.DataFrame([{
    'age': 35,
    'income': 55000,
    'loan_amount': 15000,
    'loan_tenure': 24,
    'existing_debts': 5000,
    'missed_payments': 1,
    'credit_history': 7,
    'employment_type_encoded': 0,          # 0 = Salaried
    'debt_to_income': 5000 / (55000 + 1),
    'loan_to_income': 15000 / (55000 + 1),
    'payment_reliability': 1 / (1 + 1)
}])

rf = results['Random Forest']['model']
prediction  = rf.predict(new_applicant)[0]
probability = rf.predict_proba(new_applicant)[0][1]

print("\n" + "="*45)
print("  New Applicant Prediction (Random Forest)")
print("="*45)
print(f"  Creditworthy:  {'Yes' if prediction == 1 else 'No'}")
print(f"  Confidence:    {probability:.1%}")
print("="*45)