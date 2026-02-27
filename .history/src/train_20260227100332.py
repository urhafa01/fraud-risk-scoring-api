import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score
import joblib

# Paths
DATA_PATH = Path("data/creditcard.csv")
ARTIFACTS_DIR = Path("artifacts")
ARTIFACTS_DIR.mkdir(exist_ok=True)

print("Loading dataset...")
df = pd.read_csv(DATA_PATH)

# Features and target
X = df.drop("Class", axis=1)
y = df["Class"]

# Stratified split (important for fraud data)
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

print("Training model...")

model = LogisticRegression(
    max_iter=1000,
    class_weight="balanced"
)

model.fit(X_train, y_train)

# Probabilities
probs = model.predict_proba(X_test)[:, 1]

# Metrics
roc = roc_auc_score(y_test, probs)
pr = average_precision_score(y_test, probs)

print(f"ROC-AUC: {roc:.4f}")
print(f"PR-AUC: {pr:.4f}")

# Save model
joblib.dump(model, ARTIFACTS_DIR / "fraud_model.joblib")

print("Model saved to artifacts/")

# Save default threshold (can be updated after evaluation)
threshold = 0.978  # placeholder
joblib.dump(threshold, ARTIFACTS_DIR / "threshold.joblib")