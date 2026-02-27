import numpy as np
import pandas as pd
from pathlib import Path
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    precision_recall_curve,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    average_precision_score,
)

DATA_PATH = Path("data/creditcard.csv")
MODEL_PATH = Path("artifacts/fraud_model.joblib")

print("Loading data and model...")
df = pd.read_csv(DATA_PATH)
X = df.drop("Class", axis=1)
y = df["Class"]

# same split settings as train.py
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

model = joblib.load(MODEL_PATH)
probs = model.predict_proba(X_test)[:, 1]

roc = roc_auc_score(y_test, probs)
pr = average_precision_score(y_test, probs)
print(f"ROC-AUC: {roc:.4f}")
print(f"PR-AUC: {pr:.4f}")

# ---- Threshold selection (cost-based grid search) ----
# Business costs (edit anytime)
cost_fn = 500   # missed fraud
cost_fp = 5     # false alarm

# Search thresholds on a grid (avoid weird edge cases)
grid = np.linspace(0.001, 0.999, 999)

best = {
    "t": None,
    "cost": float("inf"),
    "tn": None, "fp": None, "fn": None, "tp": None,
    "precision": None, "recall": None
}

for t in grid:
    preds = (probs >= t).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, preds).ravel()

    # avoid division by zero
    precision = tp / (tp + fp + 1e-12)
    recall = tp / (tp + fn + 1e-12)

    total_cost = fn * cost_fn + fp * cost_fp

    if total_cost < best["cost"]:
        best.update({
            "t": t,
            "cost": total_cost,
            "tn": tn, "fp": fp, "fn": fn, "tp": tp,
            "precision": precision,
            "recall": recall
        })

print(f"\nBest threshold by cost: {best['t']:.3f}")
print(f"Precision: {best['precision']:.4f} | Recall: {best['recall']:.4f}")
print(f"Confusion (tn, fp, fn, tp): {(best['tn'], best['fp'], best['fn'], best['tp'])}")
print(f"Estimated cost: ${best['cost']:,.0f} (FN=${cost_fn}, FP=${cost_fp})")

# Use best threshold for final report
best_t = float(best["t"])
preds = (probs >= best_t).astype(int)

print("\nClassification report:")
print(classification_report(y_test, preds, digits=4))