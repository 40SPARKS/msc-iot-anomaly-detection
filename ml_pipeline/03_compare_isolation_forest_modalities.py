from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
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
    / "multisensor_window_features.csv"
)

OUTPUT_DIR = PROJECT_ROOT / "ml_pipeline" / "outputs"
MODEL_DIR = PROJECT_ROOT / "ml_pipeline" / "models"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)


CURRENT_SENSORS = [
    "current_R",
    "current_S",
    "current_T",
]

VIBRATION_SENSORS = [
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

CURRENT_FEATURES = [
    f"{sensor}_{statistic}"
    for sensor in CURRENT_SENSORS
    for statistic in STATISTICS
]

VIBRATION_FEATURES = [
    f"{sensor}_{statistic}"
    for sensor in VIBRATION_SENSORS
    for statistic in STATISTICS
]

RPM_FEATURES = [
    "rpm_interpolated",
    "rpm_change_per_second",
]

FEATURE_GROUPS = {
    "current_only": CURRENT_FEATURES,
    "vibration_only": VIBRATION_FEATURES,
    "rpm_only": RPM_FEATURES,
    "all_sensors": (
        CURRENT_FEATURES
        + VIBRATION_FEATURES
        + RPM_FEATURES
    ),
}

RESULT_METADATA_COLUMNS = [
    "window_id",
    "time_start_seconds",
    "time_end_seconds",
    "window_duration_seconds",
    "label",
    "condition",
]

TARGET_FALSE_ALARM_RATE = 0.05


def validate_features(data):
    required_columns = set(
        RESULT_METADATA_COLUMNS
    )

    for features in FEATURE_GROUPS.values():
        required_columns.update(features)

    missing_columns = sorted(
        required_columns - set(data.columns)
    )

    if missing_columns:
        raise ValueError(
            f"Dataset is missing columns: {missing_columns}"
        )

    all_features = FEATURE_GROUPS["all_sensors"]

    numeric_features = data[all_features].apply(
        pd.to_numeric,
        errors="coerce",
    )

    if numeric_features.isna().any().any():
        raise ValueError(
            "Missing or non-numerical feature values found."
        )

    if not np.isfinite(
        numeric_features.to_numpy()
    ).all():
        raise ValueError(
            "Infinite feature values found."
        )

    unexpected_labels = (
        set(data["label"].unique()) - {0, 1}
    )

    if unexpected_labels:
        raise ValueError(
            f"Unexpected labels found: {unexpected_labels}"
        )


def prepare_splits(data):
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

    normal_train_end = int(
        len(normal_data) * 0.60
    )

    normal_validation_end = int(
        len(normal_data) * 0.80
    )

    fault_test_start = int(
        len(fault_data) * 0.80
    )

    normal_train = normal_data.iloc[
        :normal_train_end
    ].copy()

    normal_validation = normal_data.iloc[
        normal_train_end:normal_validation_end
    ].copy()

    normal_test = normal_data.iloc[
        normal_validation_end:
    ].copy()

    fault_test = fault_data.iloc[
        fault_test_start:
    ].copy()

    test_data = pd.concat(
        [normal_test, fault_test],
        ignore_index=True,
    )

    return (
        normal_train,
        normal_validation,
        test_data,
        normal_test,
        fault_test,
    )


def train_and_evaluate(
    model_name,
    feature_columns,
    normal_train,
    normal_validation,
    test_data,
):
    X_train = normal_train[feature_columns]
    X_validation = normal_validation[feature_columns]
    X_test = test_data[feature_columns]
    y_test = test_data["label"].astype(int)

    model = IsolationForest(
        n_estimators=200,
        max_samples="auto",
        max_features=1.0,
        contamination="auto",
        random_state=42,
        n_jobs=-1,
    )

    model.fit(X_train)

    validation_scores = model.score_samples(
        X_validation
    )

    threshold = float(
        np.quantile(
            validation_scores,
            TARGET_FALSE_ALARM_RATE,
        )
    )

    normality_scores = model.score_samples(X_test)
    anomaly_scores = -normality_scores

    predictions = (
        normality_scores < threshold
    ).astype(int)

    confusion = confusion_matrix(
        y_test,
        predictions,
        labels=[0, 1],
    )

    tn, fp, fn, tp = confusion.ravel()

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

    metrics = {
        "model": model_name,
        "feature_count": len(feature_columns),
        "threshold": threshold,
        "true_normal": int(tn),
        "false_alarms": int(fp),
        "missed_faults": int(fn),
        "detected_faults": int(tp),
        "accuracy": accuracy_score(
            y_test,
            predictions,
        ),
        "precision": precision_score(
            y_test,
            predictions,
            zero_division=0,
        ),
        "fault_recall": recall_score(
            y_test,
            predictions,
            zero_division=0,
        ),
        "f1_score": f1_score(
            y_test,
            predictions,
            zero_division=0,
        ),
        "specificity": specificity,
        "false_alarm_rate": false_alarm_rate,
        "roc_auc": roc_auc_score(
            y_test,
            anomaly_scores,
        ),
        "average_precision": average_precision_score(
            y_test,
            anomaly_scores,
        ),
    }

    model_bundle = {
        "model_name": model_name,
        "model": model,
        "feature_columns": feature_columns,
        "score_threshold": threshold,
        "target_false_alarm_rate": (
            TARGET_FALSE_ALARM_RATE
        ),
        "window_duration_seconds": 0.1,
        "label_mapping": {
            0: "normal",
            1: "anomaly",
        },
    }

    model_file = (
        MODEL_DIR
        / f"isolation_forest_{model_name}.joblib"
    )

    joblib.dump(model_bundle, model_file)

    result_columns = {
        f"{model_name}_normality_score": (
            normality_scores
        ),
        f"{model_name}_anomaly_score": (
            anomaly_scores
        ),
        f"{model_name}_threshold": np.full(
            len(test_data),
            threshold,
        ),
        f"{model_name}_prediction": predictions,
    }

    return metrics, result_columns


def main():
    print(
        "\n=== Isolation Forest Modality Comparison ==="
    )
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

    validate_features(data)

    (
        normal_train,
        normal_validation,
        test_data,
        normal_test,
        fault_test,
    ) = prepare_splits(data)

    print("\nData split:")
    print(
        f"Normal training windows: "
        f"{len(normal_train)}"
    )
    print(
        f"Normal validation windows: "
        f"{len(normal_validation)}"
    )
    print(
        f"Normal test windows: {len(normal_test)}"
    )
    print(
        f"Ball-fault test windows: {len(fault_test)}"
    )

    metrics_rows = []

    results = test_data[
        RESULT_METADATA_COLUMNS
    ].copy()

    for model_name, feature_columns in (
        FEATURE_GROUPS.items()
    ):
        print(
            f"\nTraining: {model_name} "
            f"({len(feature_columns)} features)"
        )

        metrics, result_columns = train_and_evaluate(
            model_name=model_name,
            feature_columns=feature_columns,
            normal_train=normal_train,
            normal_validation=normal_validation,
            test_data=test_data,
        )

        metrics_rows.append(metrics)

        for column, values in result_columns.items():
            results[column] = values

    metrics_table = pd.DataFrame(metrics_rows)

    metrics_table = metrics_table.sort_values(
        by="f1_score",
        ascending=False,
    ).reset_index(drop=True)

    metrics_file = (
        OUTPUT_DIR
        / "isolation_forest_modality_comparison.csv"
    )

    results_file = (
        OUTPUT_DIR
        / "isolation_forest_modality_results.csv"
    )

    metrics_table.to_csv(
        metrics_file,
        index=False,
    )

    results.to_csv(
        results_file,
        index=False,
    )

    display_columns = [
        "model",
        "feature_count",
        "true_normal",
        "false_alarms",
        "missed_faults",
        "detected_faults",
        "accuracy",
        "precision",
        "fault_recall",
        "f1_score",
        "false_alarm_rate",
        "roc_auc",
        "average_precision",
    ]

    print("\n=== MODEL COMPARISON ===")
    print(
        metrics_table[display_columns]
        .round(4)
        .to_string(index=False)
    )

    print(f"\nMetrics saved to: {metrics_file}")
    print(f"Results saved to: {results_file}")
    print(f"Models saved in: {MODEL_DIR}")
    print("\nDone.")


if __name__ == "__main__":
    main()
