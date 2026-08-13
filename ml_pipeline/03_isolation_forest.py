from pathlib import Path
import pandas as pd
import numpy as np

from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = PROJECT_ROOT / "datasets" / "public_rotating_machine" / "processed" / "vibration_window_features.csv"
OUTPUT_DIR = PROJECT_ROOT / "ml_pipeline" / "outputs"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("\n=== Isolation Forest Baseline ===")
print(f"Loading: {DATA_FILE}")

df = pd.read_csv(DATA_FILE)

print(f"Dataset shape: {df.shape}")
print("\nLabel counts:")
print(df["label"].value_counts())

# Remove non-ML columns
drop_columns = [
    "window_id",
    "label",
    "condition",
    "source_file",
    "sample_index_mean",
    "sample_index_std",
    "sample_index_min",
    "sample_index_max",
]

X = df.drop(columns=drop_columns, errors="ignore")
y = df["label"]

# Train Isolation Forest only on normal data
X_normal = X[y == 0]

scaler = StandardScaler()
X_normal_scaled = scaler.fit_transform(X_normal)
X_all_scaled = scaler.transform(X)

model = IsolationForest(
    n_estimators=100,
    contamination=0.05,
    random_state=42
)

print("\nTraining model on normal data only...")
model.fit(X_normal_scaled)

# Isolation Forest output:
#  1 = normal
# -1 = anomaly
raw_predictions = model.predict(X_all_scaled)

# Convert to project labels:
# 0 = normal
# 1 = anomaly
predictions = np.where(raw_predictions == -1, 1, 0)

df["anomaly_score"] = model.decision_function(X_all_scaled)
df["prediction"] = predictions

print("\nConfusion Matrix:")
print(confusion_matrix(y, predictions))

print("\nClassification Report:")
print(classification_report(y, predictions, target_names=["normal", "ball_fault"]))

output_file = OUTPUT_DIR / "isolation_forest_results.csv"
df.to_csv(output_file, index=False)

print(f"\nResults saved to: {output_file}")
print("\nDone.")