from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_FILE = (
    PROJECT_ROOT
    / "datasets"
    / "public_rotating_machine"
    / "processed"
    / "vibration_window_features.csv"
)

OUTPUT_DIR = PROJECT_ROOT / "ml_pipeline" / "outputs"
MODEL_DIR = PROJECT_ROOT / "ml_pipeline" / "models"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)


SENSOR_COLUMNS = [
    "bearingA_x",
    "bearingA_y",
    "bearingB_x",
    "bearingB_y",
]

STATISTICS = [
    "mean",
    "std",
    "min",
    "max",
    "rms",
]

FEATURE_COLUMNS = [
    f"{sensor}_{statistic}"
    for sensor in SENSOR_COLUMNS
    for statistic in STATISTICS
]

METADATA_COLUMNS = [
    "window_id",
    "sample_start",
    "sample_end",
    "window_size",
    "label",
    "condition",
    "source_file",
]

TARGET_FALSE_ALARM_RATE = 0.05


print("\n=== Isolation Forest Vibration Baseline ===")
print(f"Loading: {DATA_FILE}")


if not DATA_FILE.exists():
    raise FileNotFoundError(
        f"Feature dataset not found: {DATA_FILE}"
    )


data = pd.read_csv(DATA_FILE)

print(f"Dataset shape: {data.shape}")

print("\nClass distribution:")
print(
    data
    .groupby(["label", "condition"])
    .size()
)


# Confirm that all expected columns exist.
required_columns = FEATURE_COLUMNS + METADATA_COLUMNS

missing_columns = [
    column
    for column in required_columns
    if column not in data.columns
]

if missing_columns:
    raise ValueError(
        f"Dataset is missing columns: {missing_columns}"
    )


# Confirm labels use the expected format.
unexpected_labels = (
    set(data["label"].unique()) - {0, 1}
)

if unexpected_labels:
    raise ValueError(
        f"Unexpected labels found: {unexpected_labels}"
    )


# Select only the 20 physical vibration features.
X_all = (
    data[FEATURE_COLUMNS]
    .apply(pd.to_numeric, errors="coerce")
)


if X_all.isna().any().any():
    raise ValueError(
        "Missing or non-numerical feature values found."
    )


if not np.isfinite(X_all.to_numpy()).all():
    raise ValueError(
        "Infinite feature values found."
    )


print(f"\nModel feature count: {len(FEATURE_COLUMNS)}")

print("\nModel features:")
for feature in FEATURE_COLUMNS:
    print(feature)


# Separate the known conditions.
normal_data = (
    data[data["label"] == 0]
    .sort_values("window_id")
    .reset_index(drop=True)
)

fault_data = (
    data[data["label"] == 1]
    .sort_values("window_id")
    .reset_index(drop=True)
)


# Chronological normal-data split:
# 60% training, 20% validation, 20% testing.
normal_train_end = int(len(normal_data) * 0.60)
normal_validation_end = int(len(normal_data) * 0.80)

normal_train = normal_data.iloc[
    :normal_train_end
].copy()

normal_validation = normal_data.iloc[
    normal_train_end:normal_validation_end
].copy()

normal_test = normal_data.iloc[
    normal_validation_end:
].copy()


# Use the final 20% of fault windows for testing.
fault_test_start = int(len(fault_data) * 0.80)

fault_test = fault_data.iloc[
    fault_test_start:
].copy()


X_train = normal_train[FEATURE_COLUMNS]
X_validation = normal_validation[FEATURE_COLUMNS]

test_data = pd.concat(
    [normal_test, fault_test],
    ignore_index=True,
)

X_test = test_data[FEATURE_COLUMNS]
y_test = test_data["label"].astype(int)


print("\nData split:")
print(f"Normal training windows: {len(normal_train)}")
print(
    f"Normal validation windows: "
    f"{len(normal_validation)}"
)
print(f"Normal test windows: {len(normal_test)}")
print(f"Ball-fault test windows: {len(fault_test)}")


model = IsolationForest(
    n_estimators=200,
    max_samples="auto",
    max_features=1.0,
    contamination="auto",
    random_state=42,
    n_jobs=-1,
)


print("\nTraining Isolation Forest on normal windows...")
model.fit(X_train)


# Lower score_samples values mean more abnormal.
validation_scores = model.score_samples(
    X_validation
)

score_threshold = float(
    np.quantile(
        validation_scores,
        TARGET_FALSE_ALARM_RATE,
    )
)


print(
    "\nSelected normality-score threshold: "
    f"{score_threshold:.6f}"
)

print(
    "Target validation false-alarm rate: "
    f"{TARGET_FALSE_ALARM_RATE:.1%}"
)


test_normality_scores = model.score_samples(
    X_test
)

# A score below the threshold is classified as anomalous.
predictions = (
    test_normality_scores < score_threshold
).astype(int)

# Higher anomaly_score means more abnormal.
test_anomaly_scores = -test_normality_scores


confusion = confusion_matrix(
    y_test,
    predictions,
    labels=[0, 1],
)

tn, fp, fn, tp = confusion.ravel()

accuracy = accuracy_score(
    y_test,
    predictions,
)

precision = precision_score(
    y_test,
    predictions,
    zero_division=0,
)

recall = recall_score(
    y_test,
    predictions,
    zero_division=0,
)

f1 = f1_score(
    y_test,
    predictions,
    zero_division=0,
)

roc_auc = roc_auc_score(
    y_test,
    test_anomaly_scores,
)

false_alarm_rate = (
    fp / (fp + tn)
    if (fp + tn) > 0
    else 0.0
)

specificity = (
    tn / (tn + fp)
    if (tn + fp) > 0
    else 0.0
)


print("\nConfusion matrix:")
print(confusion)

print("\nConfusion-matrix meaning:")
print(f"True normal: {tn}")
print(f"False alarms: {fp}")
print(f"Missed faults: {fn}")
print(f"Detected faults: {tp}")

print("\nClassification report:")
print(
    classification_report(
        y_test,
        predictions,
        labels=[0, 1],
        target_names=[
            "normal",
            "ball_fault",
        ],
        zero_division=0,
    )
)

print("\nSummary metrics:")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Fault recall: {recall:.4f}")
print(f"F1-score: {f1:.4f}")
print(f"Specificity: {specificity:.4f}")
print(
    f"False-alarm rate: "
    f"{false_alarm_rate:.4f}"
)
print(f"ROC AUC: {roc_auc:.4f}")


# Save test-window results.
results = test_data[METADATA_COLUMNS].copy()

results["normality_score"] = (
    test_normality_scores
)

results["anomaly_score"] = (
    test_anomaly_scores
)

results["score_threshold"] = score_threshold
results["prediction"] = predictions

results["predicted_condition"] = np.where(
    predictions == 1,
    "anomaly",
    "normal",
)

results["correct_prediction"] = (
    results["label"]
    == results["prediction"]
)


results_file = (
    OUTPUT_DIR
    / "isolation_forest_results.csv"
)

results.to_csv(
    results_file,
    index=False,
)


# Save evaluation metrics.
metrics = {
    "training_normal_windows": len(normal_train),
    "validation_normal_windows": len(normal_validation),
    "test_normal_windows": len(normal_test),
    "test_ball_fault_windows": len(fault_test),
    "feature_count": len(FEATURE_COLUMNS),
    "threshold": score_threshold,
    "target_false_alarm_rate": (
        TARGET_FALSE_ALARM_RATE
    ),
    "true_normal": int(tn),
    "false_alarms": int(fp),
    "missed_faults": int(fn),
    "detected_faults": int(tp),
    "accuracy": float(accuracy),
    "precision": float(precision),
    "fault_recall": float(recall),
    "f1_score": float(f1),
    "specificity": float(specificity),
    "false_alarm_rate": float(false_alarm_rate),
    "roc_auc": float(roc_auc),
}

metrics_file = (
    OUTPUT_DIR
    / "isolation_forest_metrics.json"
)

with metrics_file.open(
    "w",
    encoding="utf-8",
) as file:
    json.dump(
        metrics,
        file,
        indent=4,
    )


# Save the trained model and everything required
# to use it again.
model_bundle = {
    "model": model,
    "feature_columns": FEATURE_COLUMNS,
    "score_threshold": score_threshold,
    "target_false_alarm_rate": (
        TARGET_FALSE_ALARM_RATE
    ),
    "window_size": 1_000,
    "label_mapping": {
        0: "normal",
        1: "anomaly",
    },
}

model_file = (
    MODEL_DIR
    / "isolation_forest_vibration.joblib"
)

joblib.dump(
    model_bundle,
    model_file,
)


print(f"\nResults saved to: {results_file}")
print(f"Metrics saved to: {metrics_file}")
print(f"Model saved to: {model_file}")
print("\nDone.")