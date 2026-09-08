from pathlib import Path
import json
import random
import time

import joblib
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay,
    roc_curve,
    precision_recall_curve,
)


# ============================================================
# REPRODUCIBILITY
# ============================================================

RANDOM_SEED = 42

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = (
    PROJECT_ROOT
    / "datasets"
    / "synthetic_conveyor"
)

MODEL_DIR = (
    PROJECT_ROOT
    / "ml_pipeline"
    / "models"
    / "synthetic"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "ml_pipeline"
    / "outputs"
    / "synthetic"
    / "isolation_forest"
)

PLOT_DIR = (
    OUTPUT_DIR
    / "plots"
)

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

PLOT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# DATA FILES
# ============================================================

TRAIN_NORMAL_FILE = (
    DATA_DIR
    / "synthetic_normal_0.csv"
)

VALIDATION_NORMAL_FILE = (
    DATA_DIR
    / "synthetic_normal_1.csv"
)

TEST_NORMAL_FILE = (
    DATA_DIR
    / "synthetic_normal_2.csv"
)

TEST_FAULT_FILE = (
    DATA_DIR
    / "synthetic_fault_2.csv"
)


# ============================================================
# MODEL FEATURES
# ============================================================

FEATURE_COLUMNS = [
    "current_A",
    "voltage_V",
    "rpm",
    "vibration_g",
    "piezo_value",
    "ir_entry",
    "ir_exit",
]


# ============================================================
# EXPERIMENT SETTINGS
# ============================================================

TARGET_FALSE_ALARM_RATE = 0.05

N_ESTIMATORS = 300

MAX_SAMPLES = "auto"


# ============================================================
# OUTPUT FILES
# ============================================================

MODEL_FILE = (
    MODEL_DIR
    / "synthetic_isolation_forest.joblib"
)

METRICS_FILE = (
    OUTPUT_DIR
    / "synthetic_isolation_forest_metrics.json"
)

PREDICTIONS_FILE = (
    OUTPUT_DIR
    / "synthetic_isolation_forest_predictions.csv"
)

SCORE_DISTRIBUTION_FILE = (
    PLOT_DIR
    / "synthetic_if_score_distribution.png"
)

CONFUSION_MATRIX_FILE = (
    PLOT_DIR
    / "synthetic_if_confusion_matrix.png"
)

ROC_CURVE_FILE = (
    PLOT_DIR
    / "synthetic_if_roc_curve.png"
)

PR_CURVE_FILE = (
    PLOT_DIR
    / "synthetic_if_precision_recall_curve.png"
)


# ============================================================
# LOAD ONE RECORDING
# ============================================================

def load_recording(
    file_path,
    expected_label,
):
    """
    Load and validate one synthetic conveyor recording.
    """

    if not file_path.exists():
        raise FileNotFoundError(
            f"Synthetic dataset file not found:\n"
            f"{file_path}"
        )

    data = pd.read_csv(
        file_path
    )

    required_columns = (
        ["timestamp"]
        + FEATURE_COLUMNS
        + ["label"]
    )

    missing_columns = [
        column
        for column in required_columns
        if column not in data.columns
    ]

    if missing_columns:
        raise ValueError(
            f"{file_path.name} is missing columns:\n"
            f"{missing_columns}"
        )

    # --------------------------------------------------------
    # Convert required columns to numeric
    # --------------------------------------------------------

    numeric_columns = (
        ["timestamp"]
        + FEATURE_COLUMNS
        + ["label"]
    )

    data[numeric_columns] = (
        data[numeric_columns]
        .apply(
            pd.to_numeric,
            errors="coerce",
        )
    )

    # --------------------------------------------------------
    # Check missing / infinite values
    # --------------------------------------------------------

    invalid_values = (
        data[numeric_columns]
        .replace(
            [np.inf, -np.inf],
            np.nan,
        )
        .isna()
        .sum()
        .sum()
    )

    if invalid_values:
        raise ValueError(
            f"{file_path.name} contains "
            f"{int(invalid_values)} invalid values."
        )

    # --------------------------------------------------------
    # Validate expected label
    # --------------------------------------------------------

    labels = set(
        data["label"]
        .astype(int)
        .unique()
        .tolist()
    )

    if labels != {expected_label}:
        raise ValueError(
            f"{file_path.name}: expected only label "
            f"{expected_label}, found {sorted(labels)}."
        )

    # --------------------------------------------------------
    # Check timestamp ordering
    # --------------------------------------------------------

    if not data["timestamp"].is_monotonic_increasing:
        raise ValueError(
            f"{file_path.name}: timestamps are not "
            "monotonically increasing."
        )

    return data


# ============================================================
# CREATE FEATURE MATRIX
# ============================================================

def feature_matrix(
    data,
):
    """
    Return model input matrix.

    Timestamp and label are deliberately excluded.
    """

    return (
        data[FEATURE_COLUMNS]
        .to_numpy(
            dtype=np.float64
        )
    )


# ============================================================
# CALCULATE ISOLATION FOREST SCORES
# ============================================================

def get_scores(
    model,
    X,
):
    """
    IsolationForest score_samples():

    Higher value = more normal
    Lower value  = more anomalous
    """

    return model.score_samples(
        X
    )


# ============================================================
# CREATE PREDICTIONS USING CUSTOM THRESHOLD
# ============================================================

def classify_scores(
    scores,
    threshold,
):
    """
    Binary labels:

    0 = normal
    1 = fault/anomaly

    Lower Isolation Forest scores represent more anomalous
    behaviour.
    """

    return (
        scores < threshold
    ).astype(int)


# ============================================================
# SAVE SCORE DISTRIBUTION
# ============================================================

def save_score_distribution(
    normal_scores,
    fault_scores,
    threshold,
):
    """
    Plot final unseen Run 2 Isolation Forest score
    distributions.
    """

    plt.figure(
        figsize=(9, 5)
    )

    plt.hist(
        normal_scores,
        bins=50,
        alpha=0.6,
        label="Normal Run 2",
    )

    plt.hist(
        fault_scores,
        bins=50,
        alpha=0.6,
        label="Fault Run 2",
    )

    plt.axvline(
        threshold,
        linestyle="--",
        label="Anomaly threshold",
    )

    plt.xlabel(
        "Isolation Forest score"
    )

    plt.ylabel(
        "Number of samples"
    )

    plt.title(
        "Synthetic Conveyor Isolation Forest Score Distribution"
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        SCORE_DISTRIBUTION_FILE,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()


# ============================================================
# SAVE CONFUSION MATRIX
# ============================================================

def save_confusion_matrix(
    y_true,
    y_pred,
):
    """
    Save final Run 2 confusion matrix.
    """

    matrix = confusion_matrix(
        y_true,
        y_pred,
        labels=[
            0,
            1,
        ],
    )

    display = ConfusionMatrixDisplay(
        confusion_matrix=matrix,
        display_labels=[
            "Normal",
            "Fault",
        ],
    )

    display.plot(
        values_format="d"
    )

    plt.title(
        "Synthetic Isolation Forest Test Confusion Matrix"
    )

    plt.tight_layout()

    plt.savefig(
        CONFUSION_MATRIX_FILE,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    return matrix


# ============================================================
# SAVE ROC CURVE
# ============================================================

def save_roc_curve(
    y_true,
    anomaly_scores,
):
    """
    Higher anomaly score means more abnormal.
    """

    false_positive_rate, true_positive_rate, _ = (
        roc_curve(
            y_true,
            anomaly_scores,
        )
    )

    auc_value = roc_auc_score(
        y_true,
        anomaly_scores,
    )

    plt.figure(
        figsize=(7, 5)
    )

    plt.plot(
        false_positive_rate,
        true_positive_rate,
        label=f"ROC-AUC = {auc_value:.4f}",
    )

    plt.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        label="Random classifier",
    )

    plt.xlabel(
        "False Positive Rate"
    )

    plt.ylabel(
        "True Positive Rate"
    )

    plt.title(
        "Synthetic Isolation Forest ROC Curve"
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        ROC_CURVE_FILE,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()


# ============================================================
# SAVE PRECISION-RECALL CURVE
# ============================================================

def save_precision_recall_curve(
    y_true,
    anomaly_scores,
):
    """
    Save final precision-recall curve.
    """

    precision_values, recall_values, _ = (
        precision_recall_curve(
            y_true,
            anomaly_scores,
        )
    )

    average_precision = (
        average_precision_score(
            y_true,
            anomaly_scores,
        )
    )

    plt.figure(
        figsize=(7, 5)
    )

    plt.plot(
        recall_values,
        precision_values,
        label=(
            f"Average precision = "
            f"{average_precision:.4f}"
        ),
    )

    plt.xlabel(
        "Recall"
    )

    plt.ylabel(
        "Precision"
    )

    plt.title(
        "Synthetic Isolation Forest Precision-Recall Curve"
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        PR_CURVE_FILE,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()


# ============================================================
# MAIN EXPERIMENT
# ============================================================

def main():

    print("=" * 70)
    print(
        "SYNTHETIC CONVEYOR ISOLATION FOREST"
    )
    print("=" * 70)

    print(
        "\nExperimental design:"
    )

    print(
        "Run 0 NORMAL -> training"
    )

    print(
        "Run 1 NORMAL -> threshold calibration"
    )

    print(
        "Run 2 NORMAL + FAULT -> final unseen test"
    )

    print(
        "\nImportant:"
    )

    print(
        "Fault data is NOT used for model training."
    )

    print(
        "Fault data is NOT used to choose the threshold."
    )

    # ========================================================
    # LOAD DATA
    # ========================================================

    print("\nLoading datasets...")

    train_normal = load_recording(
        TRAIN_NORMAL_FILE,
        expected_label=0,
    )

    validation_normal = load_recording(
        VALIDATION_NORMAL_FILE,
        expected_label=0,
    )

    test_normal = load_recording(
        TEST_NORMAL_FILE,
        expected_label=0,
    )

    test_fault = load_recording(
        TEST_FAULT_FILE,
        expected_label=1,
    )

    print(
        f"\nTraining normal samples:   "
        f"{len(train_normal)}"
    )

    print(
        f"Validation normal samples: "
        f"{len(validation_normal)}"
    )

    print(
        f"Test normal samples:       "
        f"{len(test_normal)}"
    )

    print(
        f"Test fault samples:        "
        f"{len(test_fault)}"
    )

    print(
        f"Feature count:             "
        f"{len(FEATURE_COLUMNS)}"
    )

    print(
        "\nModel features:"
    )

    for feature in FEATURE_COLUMNS:
        print(
            f"  {feature}"
        )

    # ========================================================
    # FEATURE MATRICES
    # ========================================================

    X_train = feature_matrix(
        train_normal
    )

    X_validation = feature_matrix(
        validation_normal
    )

    X_test_normal = feature_matrix(
        test_normal
    )

    X_test_fault = feature_matrix(
        test_fault
    )

    # ========================================================
    # TRAIN ISOLATION FOREST
    # ========================================================

    print("\n" + "=" * 70)
    print(
        "TRAINING ISOLATION FOREST"
    )
    print("=" * 70)

    model = IsolationForest(
        n_estimators=N_ESTIMATORS,
        max_samples=MAX_SAMPLES,
        contamination="auto",
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )

    training_start = (
        time.perf_counter()
    )

    model.fit(
        X_train
    )

    training_end = (
        time.perf_counter()
    )

    training_time_seconds = (
        training_end
        - training_start
    )

    print(
        f"\nTraining completed in "
        f"{training_time_seconds:.3f} seconds."
    )

    # ========================================================
    # SAVE MODEL
    # ========================================================

    joblib.dump(
        model,
        MODEL_FILE,
    )

    print(
        f"\nModel saved to:"
    )

    print(
        MODEL_FILE
    )

    # ========================================================
    # VALIDATION THRESHOLD
    # ========================================================

    validation_scores = get_scores(
        model,
        X_validation,
    )

    # Lower scores are more anomalous.
    #
    # A 5% target false alarm rate means the threshold is
    # chosen at the 5th percentile of healthy validation
    # scores.

    threshold = float(
        np.quantile(
            validation_scores,
            TARGET_FALSE_ALARM_RATE,
        )
    )

    validation_predictions = (
        classify_scores(
            validation_scores,
            threshold,
        )
    )

    validation_false_alarms = int(
        validation_predictions.sum()
    )

    validation_false_alarm_rate = (
        validation_false_alarms
        / len(validation_predictions)
    )

    print("\n" + "=" * 70)
    print(
        "THRESHOLD CALIBRATION"
    )
    print("=" * 70)

    print(
        f"Target false alarm rate: "
        f"{TARGET_FALSE_ALARM_RATE:.2%}"
    )

    print(
        f"Threshold: "
        f"{threshold:.6f}"
    )

    print(
        f"Validation false alarms: "
        f"{validation_false_alarms} / "
        f"{len(validation_predictions)}"
    )

    print(
        f"Validation false alarm rate: "
        f"{validation_false_alarm_rate:.2%}"
    )

    # ========================================================
    # FINAL TEST SCORES
    # ========================================================

    normal_scores = get_scores(
        model,
        X_test_normal,
    )

    fault_scores = get_scores(
        model,
        X_test_fault,
    )

    normal_predictions = classify_scores(
        normal_scores,
        threshold,
    )

    fault_predictions = classify_scores(
        fault_scores,
        threshold,
    )

    # ========================================================
    # BUILD FINAL TEST SET
    # ========================================================

    y_true = np.concatenate(
        [
            np.zeros(
                len(test_normal),
                dtype=int,
            ),
            np.ones(
                len(test_fault),
                dtype=int,
            ),
        ]
    )

    y_pred = np.concatenate(
        [
            normal_predictions,
            fault_predictions,
        ]
    )

    raw_scores = np.concatenate(
        [
            normal_scores,
            fault_scores,
        ]
    )

    # Convert to intuitive anomaly score:
    #
    # higher value = more anomalous

    anomaly_scores = (
        -raw_scores
    )

    # ========================================================
    # METRICS
    # ========================================================

    accuracy = accuracy_score(
        y_true,
        y_pred,
    )

    precision = precision_score(
        y_true,
        y_pred,
        zero_division=0,
    )

    fault_recall = recall_score(
        y_true,
        y_pred,
        zero_division=0,
    )

    f1 = f1_score(
        y_true,
        y_pred,
        zero_division=0,
    )

    roc_auc = roc_auc_score(
        y_true,
        anomaly_scores,
    )

    average_precision = (
        average_precision_score(
            y_true,
            anomaly_scores,
        )
    )

    matrix = save_confusion_matrix(
        y_true,
        y_pred,
    )

    true_normal = int(
        matrix[0, 0]
    )

    false_alarms = int(
        matrix[0, 1]
    )

    missed_faults = int(
        matrix[1, 0]
    )

    detected_faults = int(
        matrix[1, 1]
    )

    specificity = (
        true_normal
        / (
            true_normal
            + false_alarms
        )
        if (
            true_normal
            + false_alarms
        ) > 0
        else 0.0
    )

    false_alarm_rate = (
        false_alarms
        / (
            true_normal
            + false_alarms
        )
        if (
            true_normal
            + false_alarms
        ) > 0
        else 0.0
    )

    # ========================================================
    # CLASSIFICATION REPORT
    # ========================================================

    report = classification_report(
        y_true,
        y_pred,
        target_names=[
            "normal",
            "fault",
        ],
        output_dict=True,
        zero_division=0,
    )

    # ========================================================
    # SAVE TEST PREDICTIONS
    # ========================================================

    normal_results = pd.DataFrame(
        {
            "source_file": (
                TEST_NORMAL_FILE.name
            ),
            "timestamp": (
                test_normal[
                    "timestamp"
                ].to_numpy()
            ),
            "true_label": 0,
            "true_condition": "normal",
            "isolation_forest_score": (
                normal_scores
            ),
            "anomaly_score": (
                -normal_scores
            ),
            "threshold": (
                threshold
            ),
            "predicted_label": (
                normal_predictions
            ),
        }
    )

    fault_results = pd.DataFrame(
        {
            "source_file": (
                TEST_FAULT_FILE.name
            ),
            "timestamp": (
                test_fault[
                    "timestamp"
                ].to_numpy()
            ),
            "true_label": 1,
            "true_condition": "fault",
            "isolation_forest_score": (
                fault_scores
            ),
            "anomaly_score": (
                -fault_scores
            ),
            "threshold": (
                threshold
            ),
            "predicted_label": (
                fault_predictions
            ),
        }
    )

    predictions = pd.concat(
        [
            normal_results,
            fault_results,
        ],
        ignore_index=True,
    )

    predictions[
        "predicted_condition"
    ] = np.where(
        predictions[
            "predicted_label"
        ] == 0,
        "normal",
        "fault",
    )

    predictions[
        "correct_prediction"
    ] = (
        predictions[
            "true_label"
        ]
        == predictions[
            "predicted_label"
        ]
    )

    predictions.to_csv(
        PREDICTIONS_FILE,
        index=False,
    )

    # ========================================================
    # SAVE REPORT PLOTS
    # ========================================================

    save_score_distribution(
        normal_scores,
        fault_scores,
        threshold,
    )

    save_roc_curve(
        y_true,
        anomaly_scores,
    )

    save_precision_recall_curve(
        y_true,
        anomaly_scores,
    )

    # ========================================================
    # SCORE SUMMARIES
    # ========================================================

    normal_score_summary = {
        "minimum": float(
            np.min(normal_scores)
        ),
        "mean": float(
            np.mean(normal_scores)
        ),
        "standard_deviation": float(
            np.std(normal_scores)
        ),
        "maximum": float(
            np.max(normal_scores)
        ),
    }

    fault_score_summary = {
        "minimum": float(
            np.min(fault_scores)
        ),
        "mean": float(
            np.mean(fault_scores)
        ),
        "standard_deviation": float(
            np.std(fault_scores)
        ),
        "maximum": float(
            np.max(fault_scores)
        ),
    }

    # ========================================================
    # SAVE METRICS
    # ========================================================

    metrics = {

        "experiment": (
            "synthetic_conveyor_isolation_forest"
        ),

        "dataset": (
            "synthetic_conveyor"
        ),

        "classification_task": (
            "normal_vs_fault"
        ),

        "random_seed": (
            RANDOM_SEED
        ),

        "training_run": 0,

        "threshold_validation_run": 1,

        "test_run": 2,

        "training_condition": (
            "normal_only"
        ),

        "threshold_condition": (
            "normal_only"
        ),

        "feature_count": (
            len(FEATURE_COLUMNS)
        ),

        "features": (
            FEATURE_COLUMNS
        ),

        "training_normal_samples": int(
            len(train_normal)
        ),

        "validation_normal_samples": int(
            len(validation_normal)
        ),

        "test_normal_samples": int(
            len(test_normal)
        ),

        "test_fault_samples": int(
            len(test_fault)
        ),

        "n_estimators": (
            N_ESTIMATORS
        ),

        "max_samples": (
            MAX_SAMPLES
        ),

        "target_false_alarm_rate": (
            TARGET_FALSE_ALARM_RATE
        ),

        "threshold": (
            threshold
        ),

        "validation_false_alarms": (
            validation_false_alarms
        ),

        "validation_false_alarm_rate": float(
            validation_false_alarm_rate
        ),

        "training_time_seconds": float(
            training_time_seconds
        ),

        "true_normal": (
            true_normal
        ),

        "false_alarms": (
            false_alarms
        ),

        "missed_faults": (
            missed_faults
        ),

        "detected_faults": (
            detected_faults
        ),

        "accuracy": float(
            accuracy
        ),

        "precision": float(
            precision
        ),

        "fault_recall": float(
            fault_recall
        ),

        "f1_score": float(
            f1
        ),

        "specificity": float(
            specificity
        ),

        "false_alarm_rate": float(
            false_alarm_rate
        ),

        "roc_auc": float(
            roc_auc
        ),

        "average_precision": float(
            average_precision
        ),

        "confusion_matrix": (
            matrix.tolist()
        ),

        "classification_report": (
            report
        ),

        "normal_test_score_summary": (
            normal_score_summary
        ),

        "fault_test_score_summary": (
            fault_score_summary
        ),

        "model_file": (
            str(
                MODEL_FILE
            )
        ),

        "predictions_file": (
            str(
                PREDICTIONS_FILE
            )
        ),
    }

    with open(
        METRICS_FILE,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            metrics,
            file,
            indent=4,
        )

    # ========================================================
    # TERMINAL RESULTS
    # ========================================================

    print("\n" + "=" * 70)
    print(
        "FINAL UNSEEN RUN 2 RESULTS"
    )
    print("=" * 70)

    print(
        f"\nAccuracy:          "
        f"{accuracy:.4f} "
        f"({accuracy * 100:.2f}%)"
    )

    print(
        f"Precision:         "
        f"{precision:.4f} "
        f"({precision * 100:.2f}%)"
    )

    print(
        f"Fault recall:      "
        f"{fault_recall:.4f} "
        f"({fault_recall * 100:.2f}%)"
    )

    print(
        f"F1 score:          "
        f"{f1:.4f}"
    )

    print(
        f"Specificity:       "
        f"{specificity:.4f}"
    )

    print(
        f"False alarm rate:  "
        f"{false_alarm_rate:.4f}"
    )

    print(
        f"ROC-AUC:           "
        f"{roc_auc:.4f}"
    )

    print(
        f"Average precision: "
        f"{average_precision:.4f}"
    )

    print(
        "\nConfusion matrix:"
    )

    print(
        matrix
    )

    print(
        f"\nTrue normal:      "
        f"{true_normal}"
    )

    print(
        f"False alarms:     "
        f"{false_alarms}"
    )

    print(
        f"Missed faults:    "
        f"{missed_faults}"
    )

    print(
        f"Detected faults:  "
        f"{detected_faults}"
    )

    print(
        "\nClassification report:"
    )

    print(
        classification_report(
            y_true,
            y_pred,
            target_names=[
                "normal",
                "fault",
            ],
            zero_division=0,
        )
    )

    # ========================================================
    # FINAL OUTPUT SUMMARY
    # ========================================================

    print("\n" + "=" * 70)
    print(
        "SYNTHETIC ISOLATION FOREST COMPLETE"
    )
    print("=" * 70)

    print(
        "\nModel:"
    )
    print(
        MODEL_FILE
    )

    print(
        "\nMetrics:"
    )
    print(
        METRICS_FILE
    )

    print(
        "\nPredictions:"
    )
    print(
        PREDICTIONS_FILE
    )

    print(
        "\nPlots:"
    )
    print(
        SCORE_DISTRIBUTION_FILE
    )
    print(
        CONFUSION_MATRIX_FILE
    )
    print(
        ROC_CURVE_FILE
    )
    print(
        PR_CURVE_FILE
    )


# ============================================================
# SCRIPT ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()