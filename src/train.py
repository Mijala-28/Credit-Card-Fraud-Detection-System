import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    average_precision_score, confusion_matrix
)
from imblearn.over_sampling import SMOTE

def load_data(data_dir="data"):
    X_train = pd.read_csv(f"{data_dir}/X_train.csv")
    X_test = pd.read_csv(f"{data_dir}/X_test.csv")
    y_train = pd.read_csv(f"{data_dir}/y_train.csv").squeeze()
    y_test = pd.read_csv(f"{data_dir}/y_test.csv").squeeze()
    return X_train, X_test, y_train, y_test

def evaluate(model, X_test, y_test, name):
    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1]

    result = {
        "model": name,
        "precision": precision_score(y_test, preds),
        "recall": recall_score(y_test, preds),
        "f1": f1_score(y_test, preds),
        "pr_auc": average_precision_score(y_test, probs),
    }
    print(f"\n--- {name} ---")
    for k, v in result.items():
        if k != "model":
            print(f"{k}: {v:.4f}")
    print("Confusion matrix:\n", confusion_matrix(y_test, preds))
    return result

def train_class_weight_models(X_train, y_train, X_test, y_test):
    models = {
        "LogisticRegression (class_weight)": LogisticRegression(
            class_weight="balanced", max_iter=1000, random_state=42
        ),
        "RandomForest (class_weight)": RandomForestClassifier(
            class_weight="balanced", n_estimators=200, random_state=42, n_jobs=-1
        ),
        "XGBoost (scale_pos_weight)": XGBClassifier(
            scale_pos_weight=(y_train == 0).sum() / (y_train == 1).sum(),
            random_state=42, eval_metric="logloss"
        ),
    }

    results = []
    for name, model in models.items():
        model.fit(X_train, y_train)
        results.append(evaluate(model, X_test, y_test, name))
    return results, models

def train_smote_models(X_train, y_train, X_test, y_test):
    # SMOTE fits ONLY on training data - test stays untouched, 100% real
    smote = SMOTE(random_state=42)
    X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)

    print(f"\nBefore SMOTE - train fraud count: {y_train.sum()}")
    print(f"After SMOTE  - train fraud count: {y_train_sm.sum()}")
    print(f"After SMOTE  - train total rows: {len(y_train_sm)}")

    models = {
        "LogisticRegression (SMOTE)": LogisticRegression(max_iter=1000, random_state=42),
        "RandomForest (SMOTE)": RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1),
        "XGBoost (SMOTE)": XGBClassifier(random_state=42, eval_metric="logloss"),
    }

    results = []
    for name, model in models.items():
        model.fit(X_train_sm, y_train_sm)
        results.append(evaluate(model, X_test, y_test, name))  # evaluate on ORIGINAL test set
    return results, models

if __name__ == "__main__":
    import os
    os.makedirs("results", exist_ok=True)

    X_train, X_test, y_train, y_test = load_data()

    print("\n########## CLASS WEIGHT APPROACH ##########")
    cw_results, cw_models = train_class_weight_models(X_train, y_train, X_test, y_test)

    print("\n########## SMOTE APPROACH ##########")
    smote_results, smote_models = train_smote_models(X_train, y_train, X_test, y_test)

    all_results = pd.DataFrame(cw_results + smote_results)
    print("\n\n=== Full Comparison ===")
    print(all_results.to_string(index=False))
    all_results.to_csv("results/model_comparison_full.csv", index=False)
