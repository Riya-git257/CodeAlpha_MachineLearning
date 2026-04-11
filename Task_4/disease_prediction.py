# Disease Prediction from Medical Data
# Objective: Predict the possibility of diseases based on patient data.
# Approach: Classification techniques on structured medical datasets.
# Algorithms: SVM, Logistic Regression, Random Forest, XGBoost
# Datasets: Heart Disease, Diabetes, Breast Cancer (UCI ML Repository)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, classification_report,
    confusion_matrix, roc_curve
)
from sklearn.datasets import load_breast_cancer
import xgboost as xgb


# ─────────────────────────────────────────────
# Configuration — choose your disease dataset
# ─────────────────────────────────────────────

# Options: 'diabetes' | 'heart' | 'breast_cancer'
DISEASE_TARGET = 'breast_cancer'


# ─────────────────────────────────────────────
# 1. Load Dataset
# ─────────────────────────────────────────────

def load_dataset(target=DISEASE_TARGET):
    """
    Load one of three medical datasets.
    - breast_cancer : built into sklearn (no download needed)
    - diabetes      : download from https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database
    - heart         : download from https://www.kaggle.com/datasets/cherngs/heart-disease-cleveland-uci
    """

    if target == 'breast_cancer':
        print("Loading Breast Cancer dataset (sklearn built-in)...")
        data = load_breast_cancer()
        df = pd.DataFrame(data.data, columns=data.feature_names)
        df['target'] = data.target
        # 0 = malignant, 1 = benign
        class_names = ['Malignant', 'Benign']

    elif target == 'diabetes':
        print("Loading Pima Indians Diabetes dataset...")
        # Download CSV from Kaggle and place it as 'diabetes.csv'
        try:
            df = pd.read_csv('diabetes.csv')
            df.rename(columns={'Outcome': 'target'}, inplace=True)
        except FileNotFoundError:
            print("diabetes.csv not found. Generating synthetic data for demo.")
            df = generate_synthetic_data(n_samples=768, n_features=8, target='diabetes')
        class_names = ['No Diabetes', 'Diabetes']

    elif target == 'heart':
        print("Loading Heart Disease dataset...")
        try:
            df = pd.read_csv('heart.csv')
            # Cleveland dataset uses 'target' or 'condition'
            if 'condition' in df.columns:
                df.rename(columns={'condition': 'target'}, inplace=True)
            df['target'] = (df['target'] > 0).astype(int)
        except FileNotFoundError:
            print("heart.csv not found. Generating synthetic data for demo.")
            df = generate_synthetic_data(n_samples=303, n_features=13, target='heart')
        class_names = ['No Disease', 'Heart Disease']

    else:
        raise ValueError(f"Unknown target: {target}")

    print(f"Dataset shape  : {df.shape}")
    print(f"Target classes : {class_names}")
    print(f"Class balance  :\n{df['target'].value_counts()}\n")
    return df, class_names


def generate_synthetic_data(n_samples, n_features, target):
    """Synthetic fallback data when CSV is not available."""
    np.random.seed(42)
    feature_names = [f'feature_{i+1}' for i in range(n_features)]
    X = np.random.randn(n_samples, n_features)
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    df = pd.DataFrame(X, columns=feature_names)
    df['target'] = y
    return df


# ─────────────────────────────────────────────
# 2. Exploratory Data Analysis (EDA)
# ─────────────────────────────────────────────

def perform_eda(df, class_names):
    print("=" * 50)
    print("  Exploratory Data Analysis")
    print("=" * 50)
    print(df.describe().round(2))
    print(f"\nMissing values:\n{df.isnull().sum()}")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Class distribution
    counts = df['target'].value_counts()
    axes[0].bar(class_names, counts.values, color=['#378ADD', '#1D9E75'], alpha=0.85, width=0.5)
    axes[0].set_title('Class Distribution', fontsize=13)
    axes[0].set_ylabel('Count')
    for i, v in enumerate(counts.values):
        axes[0].text(i, v + 2, str(v), ha='center', fontsize=12)

    # Correlation heatmap (top 10 features)
    numeric_df = df.select_dtypes(include=[np.number])
    top_features = numeric_df.corr()['target'].abs().sort_values(ascending=False).head(11).index
    corr = numeric_df[top_features].corr()
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm',
                ax=axes[1], linewidths=0.5, square=True)
    axes[1].set_title('Correlation Heatmap (Top 10 Features)', fontsize=13)

    plt.suptitle('Medical Data — EDA', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('eda_plots.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("EDA plots saved as 'eda_plots.png'")


# ─────────────────────────────────────────────
# 3. Preprocessing
# ─────────────────────────────────────────────

def preprocess(df):
    # Separate features and target
    X = df.drop('target', axis=1)
    y = df['target']

    # Handle missing values with column median
    X = X.fillna(X.median())

    # Encode any remaining categorical columns
    for col in X.select_dtypes(include='object').columns:
        X[col] = LabelEncoder().fit_transform(X[col])

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Feature scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)

    print(f"Training samples : {X_train.shape[0]}")
    print(f"Testing samples  : {X_test.shape[0]}")
    print(f"Number of features: {X_train.shape[1]}\n")

    return X_train_scaled, X_test_scaled, y_train, y_test, X.columns.tolist(), scaler


# ─────────────────────────────────────────────
# 4. Train Models
# ─────────────────────────────────────────────

def train_models(X_train, X_test, y_train, y_test):
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'SVM':                 SVC(kernel='rbf', probability=True, random_state=42),
        'Random Forest':       RandomForestClassifier(n_estimators=100, random_state=42),
        'XGBoost':             xgb.XGBClassifier(
                                    n_estimators=100, learning_rate=0.1,
                                    use_label_encoder=False, eval_metric='logloss',
                                    random_state=42, verbosity=0
                               )
    }

    results = {}

    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train, y_train)

        y_pred      = model.predict(X_test)
        y_pred_prob = model.predict_proba(X_test)[:, 1]

        # Cross-validation
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')

        results[name] = {
            'model':      model,
            'y_pred':     y_pred,
            'y_pred_prob': y_pred_prob,
            'Accuracy':   accuracy_score(y_test, y_pred),
            'Precision':  precision_score(y_test, y_pred, zero_division=0),
            'Recall':     recall_score(y_test, y_pred, zero_division=0),
            'F1-Score':   f1_score(y_test, y_pred, zero_division=0),
            'ROC-AUC':    roc_auc_score(y_test, y_pred_prob),
            'CV Mean':    cv_scores.mean(),
            'CV Std':     cv_scores.std(),
        }

        print(f"  Accuracy: {results[name]['Accuracy']:.4f} | "
              f"ROC-AUC: {results[name]['ROC-AUC']:.4f} | "
              f"CV: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    return results


# ─────────────────────────────────────────────
# 5. Compare & Visualize Results
# ─────────────────────────────────────────────

def compare_models(results, y_test, class_names):
    # Summary table
    summary = pd.DataFrame({
        name: {k: v for k, v in vals.items()
               if k not in ('model', 'y_pred', 'y_pred_prob')}
        for name, vals in results.items()
    }).T

    print("\n" + "=" * 60)
    print("  Model Comparison Summary")
    print("=" * 60)
    print(summary.round(4).to_string())

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Disease Prediction — Model Comparison', fontsize=14, fontweight='bold')

    model_names = list(results.keys())
    colors = ['#378ADD', '#D85A30', '#1D9E75', '#BA7517']

    # (a) Metric bar chart
    metrics   = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
    x         = np.arange(len(metrics))
    bar_width = 0.2

    for i, (name, vals) in enumerate(results.items()):
        scores = [vals[m] for m in metrics]
        axes[0, 0].bar(x + i * bar_width, scores, bar_width,
                       label=name, color=colors[i], alpha=0.85)

    axes[0, 0].set_xticks(x + bar_width * 1.5)
    axes[0, 0].set_xticklabels(metrics, rotation=10)
    axes[0, 0].set_ylim(0, 1.15)
    axes[0, 0].set_title('Metric Comparison')
    axes[0, 0].set_ylabel('Score')
    axes[0, 0].legend(fontsize=9)

    # (b) ROC Curves
    for i, (name, vals) in enumerate(results.items()):
        fpr, tpr, _ = roc_curve(y_test, vals['y_pred_prob'])
        axes[0, 1].plot(fpr, tpr, color=colors[i],
                        label=f"{name} (AUC={vals['ROC-AUC']:.3f})", linewidth=2)
    axes[0, 1].plot([0, 1], [0, 1], 'k--', linewidth=1)
    axes[0, 1].set_title('ROC Curves')
    axes[0, 1].set_xlabel('False Positive Rate')
    axes[0, 1].set_ylabel('True Positive Rate')
    axes[0, 1].legend(fontsize=9)
    axes[0, 1].grid(alpha=0.3)

    # (c) Confusion matrix — best model
    best_name = max(results, key=lambda n: results[n]['ROC-AUC'])
    cm = confusion_matrix(y_test, results[best_name]['y_pred'])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[1, 0],
                xticklabels=class_names, yticklabels=class_names)
    axes[1, 0].set_title(f'Confusion Matrix — {best_name}')
    axes[1, 0].set_xlabel('Predicted')
    axes[1, 0].set_ylabel('Actual')

    # (d) Cross-validation scores
    cv_means = [results[n]['CV Mean'] for n in model_names]
    cv_stds  = [results[n]['CV Std']  for n in model_names]
    axes[1, 1].barh(model_names, cv_means, xerr=cv_stds,
                    color=colors, alpha=0.85, capsize=5)
    axes[1, 1].set_xlim(0, 1.1)
    axes[1, 1].set_title('Cross-Validation Accuracy (5-Fold)')
    axes[1, 1].set_xlabel('Accuracy')
    axes[1, 1].axvline(x=max(cv_means), color='gray', linestyle='--', linewidth=1)
    for i, (mean, std) in enumerate(zip(cv_means, cv_stds)):
        axes[1, 1].text(mean + 0.01, i, f'{mean:.3f}', va='center', fontsize=10)

    plt.tight_layout()
    plt.savefig('model_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("\nComparison plots saved as 'model_comparison.png'")

    # Classification report for best model
    print(f"\nClassification Report — {best_name}")
    print(classification_report(y_test, results[best_name]['y_pred'],
                                 target_names=class_names))
    return best_name


# ─────────────────────────────────────────────
# 6. Feature Importance
# ─────────────────────────────────────────────

def plot_feature_importance(results, feature_names, top_n=15):
    rf_model = results['Random Forest']['model']
    importances = pd.Series(
        rf_model.feature_importances_, index=feature_names
    ).sort_values(ascending=True).tail(top_n)

    plt.figure(figsize=(10, 6))
    importances.plot(kind='barh', color='#1D9E75', alpha=0.85)
    plt.title(f'Top {top_n} Feature Importances (Random Forest)', fontsize=13)
    plt.xlabel('Importance Score')
    plt.tight_layout()
    plt.savefig('feature_importance.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("Feature importance plot saved as 'feature_importance.png'")


# ─────────────────────────────────────────────
# 7. Hyperparameter Tuning (best model)
# ─────────────────────────────────────────────

def tune_best_model(X_train, y_train, best_name):
    print(f"\nTuning hyperparameters for: {best_name}")

    param_grids = {
        'Random Forest': {
            'n_estimators': [100, 200],
            'max_depth':    [None, 10, 20],
            'min_samples_split': [2, 5]
        },
        'XGBoost': {
            'n_estimators':  [100, 200],
            'learning_rate': [0.05, 0.1],
            'max_depth':     [3, 5]
        },
        'SVM': {
            'C':      [0.1, 1, 10],
            'kernel': ['rbf', 'linear']
        },
        'Logistic Regression': {
            'C':       [0.01, 0.1, 1, 10],
            'penalty': ['l2']
        }
    }

    if best_name not in param_grids:
        print("No tuning grid defined for this model.")
        return None

    base_models = {
        'Random Forest':       RandomForestClassifier(random_state=42),
        'XGBoost':             xgb.XGBClassifier(use_label_encoder=False,
                                                  eval_metric='logloss',
                                                  random_state=42, verbosity=0),
        'SVM':                 SVC(probability=True, random_state=42),
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42)
    }

    grid_search = GridSearchCV(
        base_models[best_name],
        param_grids[best_name],
        cv=5, scoring='roc_auc', n_jobs=-1, verbose=0
    )
    grid_search.fit(X_train, y_train)

    print(f"Best params : {grid_search.best_params_}")
    print(f"Best ROC-AUC: {grid_search.best_score_:.4f}")
    return grid_search.best_estimator_


# ─────────────────────────────────────────────
# 8. Predict on New Patient
# ─────────────────────────────────────────────

def predict_patient(model, scaler, feature_names, patient_data: dict, class_names):
    """
    Predict disease risk for a new patient.
    patient_data: dict with feature names as keys.
    """
    patient_df = pd.DataFrame([patient_data])[feature_names]
    patient_scaled = scaler.transform(patient_df)

    prediction  = model.predict(patient_scaled)[0]
    probability = model.predict_proba(patient_scaled)[0]

    print("\n" + "=" * 45)
    print("  New Patient Prediction")
    print("=" * 45)
    for feature, value in patient_data.items():
        print(f"  {feature:<25} : {value}")
    print("-" * 45)
    print(f"  Prediction   : {class_names[prediction]}")
    print(f"  Confidence   : {probability[prediction]:.1%}")
    print(f"  Risk Score   : {probability[1]:.1%} chance of disease")
    print("=" * 45)

    return prediction, probability


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

if __name__ == '__main__':

    print("=" * 55)
    print("  Disease Prediction from Medical Data")
    print("=" * 55)

    # 1. Load
    df, class_names = load_dataset(DISEASE_TARGET)

    # 2. EDA
    perform_eda(df, class_names)

    # 3. Preprocess
    X_train, X_test, y_train, y_test, feature_names, scaler = preprocess(df)

    # 4. Train all models
    print("=" * 50)
    print("  Training Models")
    print("=" * 50)
    results = train_models(X_train, X_test, y_train, y_test)

    # 5. Compare
    best_name = compare_models(results, y_test, class_names)

    # 6. Feature importance
    plot_feature_importance(results, feature_names)

    # 7. Tune best model
    tuned_model = tune_best_model(X_train, y_train, best_name)

    # 8. Predict on a sample patient (Breast Cancer example)
    if DISEASE_TARGET == 'breast_cancer':
        sample_patient = {feat: float(np.random.rand()) for feat in feature_names}
    else:
        sample_patient = {feat: float(np.random.rand()) for feat in feature_names}

    best_model = tuned_model if tuned_model else results[best_name]['model']
    predict_patient(best_model, scaler, feature_names, sample_patient, class_names)

    print("\nAll done! Check the saved plots in your working directory.")