from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = (
    PROJECT_ROOT
    / "datasets"
    / "synthetic_conveyor"
)

PROCESSED_DIR = (
    DATA_DIR
    / "processed"
    / "gru"
)

MODEL_DIR = (
    PROJECT_ROOT
    / "ml_pipeline"
    / "models"
    / "synthetic"
    / "gru"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "ml_pipeline"
    / "outputs"
    / "synthetic"
    / "gru"
)

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


FEATURE_COLUMNS = [
    "current_A",
    "voltage_V",
    "rpm",
    "vibration_g",
    "piezo_value",
    "ir_entry",
    "ir_exit",
]

SEQUENCE_LENGTH = 20
SEQUENCE_STRIDE = 5
SAMPLE_INTERVAL_SECONDS = 0.1


def load_recording(run_id, condition):
    """Load and validate one synthetic conveyor recording."""

    if condition not in {"normal", "fault"}:
        raise ValueError(
            "Condition must be 'normal' or 'fault'."
        )

    file_path = (
        DATA_DIR
        / f"synthetic_{condition}_{run_id}.csv"
    )

    if not file_path.exists():
        raise FileNotFoundError(
            f"Dataset file not found: {file_path}"
        )

    data = pd.read_csv(file_path)

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
            f"{file_path.name} is missing columns: "
            f"{missing_columns}"
        )

    data[required_columns] = (
        data[required_columns]
        .apply(pd.to_numeric, errors="coerce")
    )

    invalid_values = (
        data[required_columns]
        .replace([np.inf, -np.inf], np.nan)
        .isna()
        .sum()
        .sum()
    )

    if invalid_values:
        raise ValueError(
            f"{file_path.name} contains "
            f"{int(invalid_values)} invalid values."
        )

    expected_label = 0 if condition == "normal" else 1

    observed_labels = set(
        data["label"]
        .astype(int)
        .unique()
        .tolist()
    )

    if observed_labels != {expected_label}:
        raise ValueError(
            f"{file_path.name} should contain only label "
            f"{expected_label}, but found "
            f"{sorted(observed_labels)}."
        )

    if not data["timestamp"].is_monotonic_increasing:
        raise ValueError(
            f"{file_path.name} timestamps are not "
            "in increasing order."
        )

    return data


def load_run(run_id):
    """Load normal and fault recordings for one run."""

    normal_data = load_recording(
        run_id=run_id,
        condition="normal",
    )

    fault_data = load_recording(
        run_id=run_id,
        condition="fault",
    )

    return normal_data, fault_data


def create_sequences(data, scaler):
    """
    Convert one recording into overlapping two-second
    sequences.

    Each sequence contains:
        20 time steps x 7 sensor features
    """

    scaled_features = scaler.transform(
        data[FEATURE_COLUMNS]
    )

    label = int(
        data["label"].iloc[0]
    )

    sequences = []
    labels = []

    for start in range(
        0,
        len(data) - SEQUENCE_LENGTH + 1,
        SEQUENCE_STRIDE,
    ):
        end = start + SEQUENCE_LENGTH

        sequences.append(
            scaled_features[start:end]
        )

        labels.append(label)

    X = np.asarray(
        sequences,
        dtype=np.float32,
    )

    y = np.asarray(
        labels,
        dtype=np.int64,
    )

    return X, y


def create_run_sequences(
    normal_data,
    fault_data,
    scaler,
):
    """
    Create sequences separately for normal and fault
    recordings so no sequence crosses a condition boundary.
    """

    X_normal, y_normal = create_sequences(
        normal_data,
        scaler,
    )

    X_fault, y_fault = create_sequences(
        fault_data,
        scaler,
    )

    X = np.concatenate(
        [
            X_normal,
            X_fault,
        ],
        axis=0,
    )

    y = np.concatenate(
        [
            y_normal,
            y_fault,
        ],
        axis=0,
    )

    return X, y


def save_split(name, X, y):
    """Save one prepared GRU dataset."""

    output_file = (
        PROCESSED_DIR
        / f"synthetic_gru_{name}.npz"
    )

    np.savez_compressed(
        output_file,
        X=X,
        y=y,
    )

    unique_labels, counts = np.unique(
        y,
        return_counts=True,
    )

    class_counts = {
        int(label): int(count)
        for label, count in zip(
            unique_labels,
            counts,
        )
    }

    print(f"\n{name.capitalize()} data")
    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")
    print(f"Class counts: {class_counts}")
    print(f"Saved to: {output_file}")


def get_class_counts(labels):
    unique_labels, counts = np.unique(
        labels,
        return_counts=True,
    )

    return {
        str(int(label)): int(count)
        for label, count in zip(
            unique_labels,
            counts,
        )
    }


def main():
    print("Synthetic GRU data preparation")
    print()

    print("Experimental split:")
    print("Run 0 -> training")
    print("Run 1 -> validation")
    print("Run 2 -> final test")

    print()
    print(
        f"Sequence length: {SEQUENCE_LENGTH} samples "
        f"({SEQUENCE_LENGTH * SAMPLE_INTERVAL_SECONDS:.1f} seconds)"
    )

    print(
        f"Sequence stride: {SEQUENCE_STRIDE} samples "
        f"({SEQUENCE_STRIDE * SAMPLE_INTERVAL_SECONDS:.1f} seconds)"
    )

    print(
        f"Sensor features: {len(FEATURE_COLUMNS)}"
    )

    # Load the three independent experimental runs
    train_normal, train_fault = load_run(0)
    validation_normal, validation_fault = load_run(1)
    test_normal, test_fault = load_run(2)

    print()
    print("Raw recording sizes:")
    print(
        f"Run 0: normal={len(train_normal)}, "
        f"fault={len(train_fault)}"
    )
    print(
        f"Run 1: normal={len(validation_normal)}, "
        f"fault={len(validation_fault)}"
    )
    print(
        f"Run 2: normal={len(test_normal)}, "
        f"fault={len(test_fault)}"
    )

    # Fit the scaler only on Run 0.
    #
    # Both Run 0 normal and fault data are legitimate training
    # data for the supervised GRU classifier.
    training_features = pd.concat(
        [
            train_normal[FEATURE_COLUMNS],
            train_fault[FEATURE_COLUMNS],
        ],
        ignore_index=True,
    )

    scaler = StandardScaler()

    scaler.fit(
        training_features
    )

    scaler_file = (
        MODEL_DIR
        / "synthetic_gru_scaler.joblib"
    )

    joblib.dump(
        scaler,
        scaler_file,
    )

    print()
    print(
        "StandardScaler fitted using Run 0 training data only."
    )
    print(
        f"Scaler saved to: {scaler_file}"
    )

    # Build two-second temporal sequences
    X_train, y_train = create_run_sequences(
        train_normal,
        train_fault,
        scaler,
    )

    X_validation, y_validation = create_run_sequences(
        validation_normal,
        validation_fault,
        scaler,
    )

    X_test, y_test = create_run_sequences(
        test_normal,
        test_fault,
        scaler,
    )

    # Confirm all splits have the same input structure
    expected_shape = (
        SEQUENCE_LENGTH,
        len(FEATURE_COLUMNS),
    )

    for name, X in {
        "train": X_train,
        "validation": X_validation,
        "test": X_test,
    }.items():

        if X.shape[1:] != expected_shape:
            raise ValueError(
                f"{name} has unexpected sequence shape "
                f"{X.shape[1:]}. Expected {expected_shape}."
            )

        if not np.isfinite(X).all():
            raise ValueError(
                f"{name} contains NaN or infinite values."
            )

    save_split(
        "train",
        X_train,
        y_train,
    )

    save_split(
        "validation",
        X_validation,
        y_validation,
    )

    save_split(
        "test",
        X_test,
        y_test,
    )

    metadata = {
        "dataset": "synthetic_conveyor",
        "classification_task": "normal_vs_fault",
        "labels": {
            "0": "normal",
            "1": "fault",
        },
        "sample_interval_seconds": (
            SAMPLE_INTERVAL_SECONDS
        ),
        "sampling_rate_hz": (
            1 / SAMPLE_INTERVAL_SECONDS
        ),
        "sequence_length_samples": (
            SEQUENCE_LENGTH
        ),
        "sequence_duration_seconds": (
            SEQUENCE_LENGTH
            * SAMPLE_INTERVAL_SECONDS
        ),
        "sequence_stride_samples": (
            SEQUENCE_STRIDE
        ),
        "sequence_stride_seconds": (
            SEQUENCE_STRIDE
            * SAMPLE_INTERVAL_SECONDS
        ),
        "feature_count": (
            len(FEATURE_COLUMNS)
        ),
        "features": (
            FEATURE_COLUMNS
        ),
        "training_run": 0,
        "validation_run": 1,
        "test_run": 2,
        "scaling_method": "StandardScaler",
        "scaler_fit_dataset": (
            "Run 0 normal and fault training recordings only"
        ),
        "train_shape": list(
            X_train.shape
        ),
        "validation_shape": list(
            X_validation.shape
        ),
        "test_shape": list(
            X_test.shape
        ),
        "train_class_counts": (
            get_class_counts(
                y_train
            )
        ),
        "validation_class_counts": (
            get_class_counts(
                y_validation
            )
        ),
        "test_class_counts": (
            get_class_counts(
                y_test
            )
        ),
        "data_leakage_control": (
            "Independent synthetic runs are used for training, "
            "validation and testing. The feature scaler is "
            "fitted using Run 0 only."
        ),
        "sequence_boundary_policy": (
            "Normal and fault recordings are converted to "
            "sequences independently so sequences never cross "
            "between operating conditions."
        ),
    }

    metadata_file = (
        OUTPUT_DIR
        / "synthetic_gru_preparation_metadata.json"
    )

    with open(
        metadata_file,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            metadata,
            file,
            indent=4,
        )

    print()
    print("Synthetic GRU preparation complete.")
    print(
        f"Metadata saved to: {metadata_file}"
    )


if __name__ == "__main__":
    main()