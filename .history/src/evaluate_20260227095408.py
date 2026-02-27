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

# Find threshold that achieves at least a target precision while maximizing recall
target_precision = 0.90

precisions, recalls, thresholds = precision_recall_curve(y_test, probs)
# precision_recall_curve returns thresholds of length n-1
thresholds = np.append(thresholds, 1.0)

best_t = None
best_recall = -1

for p, r, t in zip(precisions, recalls, thresholds):
    if p >= target_precision and r > best_recall:
        best_recall = r
        best_t = t

if best_t is None:
    # fallback: pick threshold maximizing F1 (simple)
    f1 = (2 * precisions * recalls) / (precisions + recalls + 1e-12)
    best_idx = int(np.nanargmax(f1))
    best_t = thresholds[best_idx]
    print("Could not reach target precision. Using best F1 threshold instead.")

print(f"Selected threshold: {best_t:.4f}")

preds = (probs >= best_t).astype(int)

cm = confusion_matrix(y_test, preds)
tn, fp, fn, tp = cm.ravel()

print("\nConfusion Matrix (tn, fp, fn, tp):")
print((tn, fp, fn, tp))

print("\nClassification report:")
print(classification_report(y_test, preds, digits=4))

# Simple business cost example:
# assume a false negative (missed fraud) costs $500, false positive costs $5
cost_fn = 500
cost_fp = 5
total_cost = fn * cost_fn + fp * cost_fp
print(f"\nEstimated cost at this threshold: ${total_cost:,.0f} (FN=${cost_fn}, FP=${cost_fp})")