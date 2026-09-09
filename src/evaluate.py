import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns 
from sklearn.metrics import precision_recall_curve, confusion_matrix, average_precision_score
import os

def load_test_data(data_dir="data"):
    X_test = pd.read_csv(f"{data_dir}/X_test.csv")
    y_test = pd.read_csv(f"{data_dir}/y_test.csv").squeeze()
    return X_test, y_test

def plot_pr_curve(model, X_test, y_test, out_path="results/pr_curve.png"):
    probs = model.predict_proba(X_test)[:, 1]
    precision, recall, _ = precision_recall_curve(y_test, probs)
    pr_auc = average_precision_score(y_test, probs)

    plt.figure(figsize=(7, 6))
    plt.plot(recall, precision, label=f"XGBoost (PR-AUC ={pr_auc:.4f})")
    plt.xlabel("Recall")
    plt.ylabel("precision")
    plt.title("Precision-Recall Curve - Fraud Detection")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved {out_path}")

def plot_confusion_matrix(model, X_test, y_test, out_path="results/confusion_matrix.png"):
    preds = model.predict(X_test)
    cm = confusion_matrix(y_test, preds)

    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=["Predicted: Not Fraud", "Predicted: Fraud"],
        yticklabels=["Actual: Not Fraud", "Actual: Fraud"],   
    )
    plt.title("Confusion Matrix - Final Model (XGBoost, class_weight)")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved {out_path}")

if __name__ == "__main__":
    os.makedirs("results", exist_ok=True)
    X_test, y_test = load_test_data()
    model = joblib.load("models/final_model.pkl")

    plot_pr_curve(model, X_test, y_test)
    plot_confusion_matrix(model, X_test, y_test)

