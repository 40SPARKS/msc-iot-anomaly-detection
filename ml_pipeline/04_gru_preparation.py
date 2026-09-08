from pathlib import Path
import json
import runpy

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

PIPELINE_DIR = PROJECT_ROOT / "ml_pipeline"

PROCESSED_DIR = (
    PROJECT_ROOT
    / "datasets"
    / "public_rotating_machine"
    / "processed"
)

GRU_DATA_DIR = PROCESSED_DIR / "gru"

GRU_MODEL_DIR = (
    PIPELINE_DIR
    / "models"
    / "gru"
)

GRU_OUTPUT_DIR = (
    PIPELINE_DIR
    / "outputs"
    / "gru"
)

GRU_DATA_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

GRU_MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

GRU_OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# REUSE EXISTING FEATURE EXTRACTION CODE
# ============================================================

EXTRACTOR_FILE = (
    PIPELINE_DIR
    / "02_extract_multisensor_features.py"
)

if not EXTRACTOR_FILE.exists():
    raise FileNotFoundError(
        f"Existing feature extraction script not found:\n"
        f"{EXTRACTOR_FILE}"
    )


print(
    "Loading existing multi-sensor feature extraction "
    "functions..."
)

extractor_namespace = runpy.run_path(
    str(EXTRACTOR_FILE)
)


extract_stream_features = (
    extractor_namespace[
        "extract_stream_features"
    ]
)

create_rpm_features = (
    extractor_namespace[
        "create_rpm_features"
    ]
)

DATA_DIR = (
    extractor_namespace[
        "DATA_DIR"
    ]
)

CURRENT_COLUMNS = (
    extractor_namespace[
        "CURRENT_COLUMNS"
    ]
)

VIBRATION_COLUMNS = (
    extractor_namespace[
        "VIBRATION_COLUMNS"
    ]
)

CURRENT_SAMPLES_PER_WINDOW = (
    extractor_namespace[
        "CURRENT_SAMPLES_PER_WINDOW"
    ]
)

VIBRATION_SAMPLES_PER_WINDOW = (
    extractor_namespace[
        "VIBRATION_SAMPLES_PER_WINDOW"
    ]
)

WINDOW_DURATION_SECONDS = float(
    extractor_namespace[
        "WINDOW_DURATION_SECONDS"
    ]
)


# ============================================================
# GRU SEQUENCE SETTINGS
# ============================================================

# Existing feature windows are 0.1 seconds long.
#
# 20 consecutive windows:
# 20 x 0.1 seconds = 2 seconds
#
# Each GRU sequence therefore represents 2 seconds
# of machine behaviour.

SEQUENCE_LENGTH = 20


# Start a new sequence every 5 feature windows.
#
# 5 x 0.1 seconds = 0.5 seconds
#
# This gives some overlap between sequences while avoiding
# creation of an unnecessarily large number of almost
# identical samples.

SEQUENCE_STRIDE = 5


# ============================================================
# FEATURE COLUMN SELECTION
# ============================================================

def get_feature_columns(data):
    """
    Select the same numerical features used by the existing
    multi-sensor Isolation Forest experiment.

    Expected features:

    Current:
        3 channels x
        mean/std/min/max/RMS
        = 15

    Vibration:
        4 channels x
        mean/std/min/max/RMS
        = 20

    RPM:
        rpm_interpolated
        rpm_change_per_second
        = 2

    Total = 37 features
    """

    feature_columns = [
        column
        for column in data.columns
        if (
            column.endswith(
                (
                    "_mean",
                    "_std",
                    "_min",
                    "_max",
                    "_rms",
                )
            )
            or column in {
                "rpm_interpolated",
                "rpm_change_per_second",
            }
        )
    ]

    return feature_columns


# ============================================================
# DETERMINE HOW MANY WINDOWS RPM SUPPORTS
# ============================================================

def get_rpm_supported_windows(
    file_path,
):
    """
    Determine how many 0.1-second feature windows are fully
    supported by the RPM timestamps.

    Window centres occur at:

        0.05 s
        0.15 s
        0.25 s
        ...
        299.95 s

    We do not extrapolate RPM beyond the final available RPM
    timestamp.
    """

    if not file_path.exists():
        raise FileNotFoundError(
            f"RPM file not found:\n{file_path}"
        )

    rpm_time_data = pd.read_csv(
        file_path,
        usecols=["time"],
    )

    rpm_time = pd.to_numeric(
        rpm_time_data["time"],
        errors="coerce",
    ).dropna()

    if rpm_time.empty:
        raise ValueError(
            f"No valid RPM timestamps found in "
            f"{file_path.name}"
        )

    rpm_end_time = float(
        rpm_time.max()
    )

    supported_windows = int(
        np.floor(
            (
                rpm_end_time
                / WINDOW_DURATION_SECONDS
            )
            + 0.5
        )
    )

    if supported_windows <= 0:
        raise ValueError(
            f"{file_path.name} does not contain enough "
            "RPM data for one complete feature window."
        )

    return (
        supported_windows,
        rpm_end_time,
    )


# ============================================================
# BUILD ONE CONDITION FOR GRU
# ============================================================

def build_gru_condition_dataset(
    run_id,
    condition,
    label,
    current_file_name,
    vibration_file_name,
    rpm_file_name,
):
    """
    Build one normal or ball-fault condition for a particular
    recording run.

    The function reuses the same feature extraction logic as
    02_extract_multisensor_features.py.

    A cache file is created after successful processing so
    multi-gigabyte raw files do not need to be repeatedly
    processed if the GRU preparation script is run again.
    """

    cache_file = (
        GRU_DATA_DIR
        / (
            f"gru_run{run_id}_"
            f"{condition}_window_features.csv"
        )
    )

    # --------------------------------------------------------
    # Reuse cache if this condition has already been processed
    # --------------------------------------------------------

    if cache_file.exists():

        print("\n" + "=" * 60)
        print(
            f"Reusing cached GRU feature data:"
        )
        print(cache_file)

        cached_data = pd.read_csv(
            cache_file
        )

        print(
            f"Cached windows: "
            f"{len(cached_data)}"
        )

        return cached_data

    # --------------------------------------------------------
    # Start fresh processing
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print(
        f"Run {run_id}"
    )
    print(
        f"Condition: {condition}"
    )
    print(
        f"Assigned label: {label}"
    )
    print("=" * 60)

    # --------------------------------------------------------
    # Current features
    # --------------------------------------------------------

    current_features = (
        extract_stream_features(
            file_path=(
                DATA_DIR
                / current_file_name
            ),
            sensor_columns=(
                CURRENT_COLUMNS
            ),
            samples_per_window=(
                CURRENT_SAMPLES_PER_WINDOW
            ),
            chunk_size=(
                CURRENT_SAMPLES_PER_WINDOW
                * 100
            ),
        )
    )

    # --------------------------------------------------------
    # Vibration features
    # --------------------------------------------------------

    vibration_features = (
        extract_stream_features(
            file_path=(
                DATA_DIR
                / vibration_file_name
            ),
            sensor_columns=(
                VIBRATION_COLUMNS
            ),
            samples_per_window=(
                VIBRATION_SAMPLES_PER_WINDOW
            ),
            chunk_size=(
                VIBRATION_SAMPLES_PER_WINDOW
                * 100
            ),
        )
    )

    # --------------------------------------------------------
    # Determine current/vibration common duration
    # --------------------------------------------------------

    sensor_windows = min(
        len(current_features),
        len(vibration_features),
    )

    if (
        len(current_features)
        != len(vibration_features)
    ):

        print(
            "\nWarning:"
        )

        print(
            "Current and vibration window counts differ."
        )

        print(
            f"Current windows: "
            f"{len(current_features)}"
        )

        print(
            f"Vibration windows: "
            f"{len(vibration_features)}"
        )

        print(
            f"Using first "
            f"{sensor_windows} "
            "common sensor windows."
        )

    # --------------------------------------------------------
    # Determine RPM-supported duration
    # --------------------------------------------------------

    rpm_file_path = (
        DATA_DIR
        / rpm_file_name
    )

    (
        rpm_supported_windows,
        rpm_end_time,
    ) = get_rpm_supported_windows(
        rpm_file_path
    )

    # --------------------------------------------------------
    # Final common number of valid windows
    # --------------------------------------------------------

    common_windows = min(
        sensor_windows,
        rpm_supported_windows,
    )

    print("\nAlignment information:")
    print(
        f"Current/vibration windows available: "
        f"{sensor_windows}"
    )

    print(
        f"RPM final timestamp: "
        f"{rpm_end_time:.6f} seconds"
    )

    print(
        f"RPM-supported 0.1 s windows: "
        f"{rpm_supported_windows}"
    )

    print(
        f"Final common windows retained: "
        f"{common_windows}"
    )

    # --------------------------------------------------------
    # Report trimming if required
    # --------------------------------------------------------

    trimmed_windows = (
        sensor_windows
        - common_windows
    )

    if trimmed_windows > 0:

        print(
            f"Trimming {trimmed_windows} "
            "final window(s)."
        )

        print(
            "Reason: RPM recording ends slightly before "
            "the final current/vibration window."
        )

        print(
            "No RPM values are extrapolated."
        )

    # --------------------------------------------------------
    # Trim current and vibration to common duration
    # --------------------------------------------------------

    current_features = (
        current_features
        .iloc[:common_windows]
        .copy()
        .reset_index(drop=True)
    )

    vibration_features = (
        vibration_features
        .iloc[:common_windows]
        .copy()
        .reset_index(drop=True)
    )

    # --------------------------------------------------------
    # Create RPM features over exactly same duration
    # --------------------------------------------------------

    rpm_features = (
        create_rpm_features(
            file_path=rpm_file_path,
            number_of_windows=(
                common_windows
            ),
        )
    )

    # --------------------------------------------------------
    # Merge current + vibration
    # --------------------------------------------------------

    combined = (
        current_features
        .merge(
            vibration_features,
            on="window_id",
            how="inner",
            validate="one_to_one",
        )
    )

    # --------------------------------------------------------
    # Merge RPM
    # --------------------------------------------------------

    combined = (
        combined
        .merge(
            rpm_features,
            on="window_id",
            how="inner",
            validate="one_to_one",
        )
    )

    # --------------------------------------------------------
    # Add timing information
    # --------------------------------------------------------

    combined[
        "time_start_seconds"
    ] = (
        combined["window_id"]
        * WINDOW_DURATION_SECONDS
    )

    combined[
        "time_end_seconds"
    ] = (
        combined[
            "time_start_seconds"
        ]
        + WINDOW_DURATION_SECONDS
    )

    combined[
        "window_duration_seconds"
    ] = WINDOW_DURATION_SECONDS

    combined[
        "current_samples_per_window"
    ] = CURRENT_SAMPLES_PER_WINDOW

    combined[
        "vibration_samples_per_window"
    ] = VIBRATION_SAMPLES_PER_WINDOW

    # --------------------------------------------------------
    # Add labels / metadata
    # --------------------------------------------------------

    combined["label"] = label
    combined["condition"] = condition
    combined["run_id"] = run_id

    combined[
        "source_current"
    ] = current_file_name

    combined[
        "source_vibration"
    ] = vibration_file_name

    combined[
        "source_rpm"
    ] = rpm_file_name

    combined[
        "rpm_end_time_seconds"
    ] = rpm_end_time

    combined[
        "trimmed_final_windows"
    ] = trimmed_windows

    # --------------------------------------------------------
    # Save condition cache
    # --------------------------------------------------------

    combined.to_csv(
        cache_file,
        index=False,
    )

    print(
        f"\nSaved condition cache:"
    )

    print(
        cache_file
    )

    print(
        f"Aligned windows retained: "
        f"{len(combined)}"
    )

    return combined


# ============================================================
# BUILD / LOAD ONE COMPLETE RUN
# ============================================================

def build_run(
    run_id,
):
    """
    Build one complete recording containing:

        normal
        ball_fault

    Run 0 reuses the feature table already produced by the
    completed Isolation Forest pipeline.

    Runs 1 and 2 are prepared specifically for GRU use.
    """

    print("\n" + "=" * 70)
    print(
        f"PREPARING RUN {run_id}"
    )
    print("=" * 70)

    # --------------------------------------------------------
    # RUN 0
    # --------------------------------------------------------
    #
    # This already exists because it was created by the
    # existing multi-sensor Isolation Forest preprocessing.
    #
    # We reuse it and do NOT alter the original file.
    # --------------------------------------------------------

    if run_id == 0:

        existing_file = (
            PROCESSED_DIR
            / "multisensor_window_features.csv"
        )

        if not existing_file.exists():

            raise FileNotFoundError(
                "Existing Run 0 multi-sensor feature "
                "table not found:\n"
                f"{existing_file}"
            )

        print(
            "Reusing existing Run 0 processed "
            "feature table:"
        )

        print(
            existing_file
        )

        data = pd.read_csv(
            existing_file
        )

        data["run_id"] = 0

        print(
            f"Run 0 rows loaded: "
            f"{len(data)}"
        )

        return data

    # --------------------------------------------------------
    # RUN 1 / RUN 2
    # --------------------------------------------------------

    combined_cache_file = (
        GRU_DATA_DIR
        / f"gru_run{run_id}_window_features.csv"
    )

    if combined_cache_file.exists():

        print(
            "Reusing cached complete run:"
        )

        print(
            combined_cache_file
        )

        data = pd.read_csv(
            combined_cache_file
        )

        print(
            f"Cached rows: "
            f"{len(data)}"
        )

        return data

    # --------------------------------------------------------
    # Normal condition
    # --------------------------------------------------------

    normal_data = (
        build_gru_condition_dataset(
            run_id=run_id,
            condition="normal",
            label=0,
            current_file_name=(
                f"current_normal_{run_id}.csv"
            ),
            vibration_file_name=(
                f"vibration_normal_{run_id}.csv"
            ),
            rpm_file_name=(
                f"rpm_normal_{run_id}.csv"
            ),
        )
    )

    # --------------------------------------------------------
    # Ball fault condition
    # --------------------------------------------------------

    ball_data = (
        build_gru_condition_dataset(
            run_id=run_id,
            condition="ball_fault",
            label=1,
            current_file_name=(
                f"current_ball_{run_id}.csv"
            ),
            vibration_file_name=(
                f"vibration_ball_{run_id}.csv"
            ),
            rpm_file_name=(
                f"rpm_ball_{run_id}.csv"
            ),
        )
    )

    # --------------------------------------------------------
    # Combine conditions
    # --------------------------------------------------------

    data = pd.concat(
        [
            normal_data,
            ball_data,
        ],
        ignore_index=True,
    )

    data["run_id"] = run_id

    # --------------------------------------------------------
    # Save complete run cache
    # --------------------------------------------------------

    data.to_csv(
        combined_cache_file,
        index=False,
    )

    print(
        f"\nSaved complete Run {run_id} "
        "feature table:"
    )

    print(
        combined_cache_file
    )

    print(
        f"Run {run_id} total rows: "
        f"{len(data)}"
    )

    return data


# ============================================================
# VALIDATE ONE RUN
# ============================================================

def validate_run(
    data,
    run_id,
    expected_features=None,
):
    """
    Validate feature structure and labels before sequence
    creation.
    """

    required_columns = {
        "window_id",
        "label",
        "condition",
    }

    missing_columns = (
        required_columns
        - set(data.columns)
    )

    if missing_columns:

        raise ValueError(
            f"Run {run_id} is missing required "
            f"columns:\n"
            f"{sorted(missing_columns)}"
        )

    feature_columns = (
        get_feature_columns(
            data
        )
    )

    # --------------------------------------------------------
    # We expect exactly 37 model features
    # --------------------------------------------------------

    if len(feature_columns) != 37:

        raise ValueError(
            f"Run {run_id}: expected 37 model "
            f"features but found "
            f"{len(feature_columns)}."
        )

    # --------------------------------------------------------
    # Ensure Run 1 and Run 2 match Run 0 feature structure
    # --------------------------------------------------------

    if expected_features is not None:

        if (
            feature_columns
            != expected_features
        ):

            raise ValueError(
                f"Run {run_id}: feature columns or "
                "feature ordering do not match "
                "training Run 0."
            )

    # --------------------------------------------------------
    # Check numerical validity
    # --------------------------------------------------------

    invalid_values = int(
        data[
            feature_columns
        ]
        .replace(
            [
                np.inf,
                -np.inf,
            ],
            np.nan,
        )
        .isna()
        .sum()
        .sum()
    )

    if invalid_values > 0:

        raise ValueError(
            f"Run {run_id} contains "
            f"{invalid_values} invalid "
            "numerical feature values."
        )

    # --------------------------------------------------------
    # Check classes
    # --------------------------------------------------------

    observed_labels = set(
        data["label"]
        .astype(int)
        .unique()
        .tolist()
    )

    if observed_labels != {0, 1}:

        raise ValueError(
            f"Run {run_id}: expected labels "
            f"{{0, 1}}, found "
            f"{sorted(observed_labels)}"
        )

    # --------------------------------------------------------
    # Ensure each condition has only one label
    # --------------------------------------------------------

    condition_label_counts = (
        data
        .groupby("condition")["label"]
        .nunique()
    )

    if (
        condition_label_counts > 1
    ).any():

        raise ValueError(
            f"Run {run_id}: at least one condition "
            "contains more than one label."
        )

    # --------------------------------------------------------
    # Display validation summary
    # --------------------------------------------------------

    print("\n" + "-" * 60)
    print(
        f"RUN {run_id} VALIDATION"
    )
    print("-" * 60)

    print(
        f"Rows: "
        f"{len(data)}"
    )

    print(
        f"Model features: "
        f"{len(feature_columns)}"
    )

    print(
        "\nClass distribution:"
    )

    print(
        data
        .groupby(
            [
                "label",
                "condition",
            ]
        )
        .size()
    )

    return feature_columns


# ============================================================
# CREATE TEMPORAL GRU SEQUENCES
# ============================================================

def create_sequences(
    data,
    feature_columns,
    scaler,
    sequence_length,
    stride,
):
    """
    Convert consecutive 0.1-second feature windows into
    temporal GRU sequences.

    Each condition is processed independently.

    This prevents a sequence from accidentally crossing from
    the final normal window into the first ball-fault window.
    """

    sequence_features = []
    sequence_labels = []

    for (
        condition,
        condition_data,
    ) in data.groupby(
        "condition",
        sort=False,
    ):

        condition_data = (
            condition_data
            .sort_values(
                "window_id"
            )
            .reset_index(
                drop=True
            )
        )

        labels = (
            condition_data[
                "label"
            ]
            .to_numpy(
                dtype=np.int64
            )
        )

        unique_labels = np.unique(
            labels
        )

        if len(unique_labels) != 1:

            raise ValueError(
                f"Condition "
                f"'{condition}' contains "
                "multiple labels."
            )

        # ----------------------------------------------------
        # Apply scaler learned from training Run 0
        # ----------------------------------------------------

        scaled_values = (
            scaler.transform(
                condition_data[
                    feature_columns
                ]
            )
        )

        # ----------------------------------------------------
        # Build consecutive temporal sequences
        # ----------------------------------------------------

        for start_index in range(
            0,
            (
                len(condition_data)
                - sequence_length
                + 1
            ),
            stride,
        ):

            end_index = (
                start_index
                + sequence_length
            )

            sequence = (
                scaled_values[
                    start_index:end_index
                ]
            )

            sequence_features.append(
                sequence
            )

            sequence_labels.append(
                int(
                    unique_labels[0]
                )
            )

    X = np.asarray(
        sequence_features,
        dtype=np.float32,
    )

    y = np.asarray(
        sequence_labels,
        dtype=np.int64,
    )

    if len(X) == 0:

        raise ValueError(
            "No GRU sequences were created."
        )

    return X, y


# ============================================================
# SAVE ONE DATA SPLIT
# ============================================================

def save_split(
    split_name,
    X,
    y,
):
    """
    Save prepared GRU input arrays.
    """

    output_file = (
        GRU_DATA_DIR
        / f"gru_{split_name}.npz"
    )

    np.savez_compressed(
        output_file,
        X=X,
        y=y,
    )

    unique_labels, counts = (
        np.unique(
            y,
            return_counts=True,
        )
    )

    class_counts = dict(
        zip(
            unique_labels.tolist(),
            counts.tolist(),
        )
    )

    print("\n" + "-" * 60)
    print(
        f"{split_name.upper()} DATA"
    )
    print("-" * 60)

    print(
        f"X shape: "
        f"{X.shape}"
    )

    print(
        f"y shape: "
        f"{y.shape}"
    )

    print(
        f"Class counts: "
        f"{class_counts}"
    )

    print(
        f"Saved to:"
    )

    print(
        output_file
    )


# ============================================================
# CLASS COUNT HELPER
# ============================================================

def get_class_counts(
    labels,
):
    """
    Convert NumPy class counts to JSON-safe dictionary.
    """

    unique_labels, counts = (
        np.unique(
            labels,
            return_counts=True,
        )
    )

    return {
        str(int(label)): int(count)
        for label, count in zip(
            unique_labels,
            counts,
        )
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print(
        "GRU DATA PREPARATION"
    )
    print("=" * 70)

    print(
        "\nExperimental design:"
    )

    print(
        "Run 0 -> Training"
    )

    print(
        "Run 1 -> Validation"
    )

    print(
        "Run 2 -> Final Test"
    )

    print(
        "\nClassification task:"
    )

    print(
        "0 = normal"
    )

    print(
        "1 = ball_fault"
    )

    print(
        "\nFeature window duration:"
    )

    print(
        f"{WINDOW_DURATION_SECONDS} seconds"
    )

    print(
        "\nGRU sequence length:"
    )

    print(
        f"{SEQUENCE_LENGTH} windows"
    )

    print(
        f"= "
        f"{SEQUENCE_LENGTH * WINDOW_DURATION_SECONDS:.1f} "
        "seconds"
    )

    print(
        "\nGRU sequence stride:"
    )

    print(
        f"{SEQUENCE_STRIDE} windows"
    )

    print(
        f"= "
        f"{SEQUENCE_STRIDE * WINDOW_DURATION_SECONDS:.1f} "
        "seconds"
    )

    # ========================================================
    # BUILD / LOAD RUNS
    # ========================================================

    train_data = build_run(
        run_id=0
    )

    validation_data = build_run(
        run_id=1
    )

    test_data = build_run(
        run_id=2
    )

    # ========================================================
    # VALIDATE FEATURE CONSISTENCY
    # ========================================================

    feature_columns = (
        validate_run(
            train_data,
            run_id=0,
        )
    )

    validate_run(
        validation_data,
        run_id=1,
        expected_features=(
            feature_columns
        ),
    )

    validate_run(
        test_data,
        run_id=2,
        expected_features=(
            feature_columns
        ),
    )

    # ========================================================
    # SCALE FEATURES
    # ========================================================
    #
    # IMPORTANT:
    #
    # The scaler is fitted ONLY using Run 0.
    #
    # Validation and test recordings are transformed using the
    # Run 0 scaler.
    #
    # This prevents data leakage.
    # ========================================================

    scaler = StandardScaler()

    scaler.fit(
        train_data[
            feature_columns
        ]
    )

    scaler_file = (
        GRU_MODEL_DIR
        / "gru_feature_scaler.joblib"
    )

    joblib.dump(
        scaler,
        scaler_file,
    )

    print("\n" + "=" * 70)
    print(
        "FEATURE SCALING"
    )
    print("=" * 70)

    print(
        "Scaler fitted using "
        "TRAINING RUN 0 ONLY."
    )

    print(
        "Validation and test data were not used "
        "to fit the scaler."
    )

    print(
        "\nScaler saved to:"
    )

    print(
        scaler_file
    )

    # ========================================================
    # CREATE TRAINING SEQUENCES
    # ========================================================

    X_train, y_train = (
        create_sequences(
            data=train_data,
            feature_columns=(
                feature_columns
            ),
            scaler=scaler,
            sequence_length=(
                SEQUENCE_LENGTH
            ),
            stride=(
                SEQUENCE_STRIDE
            ),
        )
    )

    # ========================================================
    # CREATE VALIDATION SEQUENCES
    # ========================================================

    (
        X_validation,
        y_validation,
    ) = create_sequences(
        data=validation_data,
        feature_columns=(
            feature_columns
        ),
        scaler=scaler,
        sequence_length=(
            SEQUENCE_LENGTH
        ),
        stride=(
            SEQUENCE_STRIDE
        ),
    )

    # ========================================================
    # CREATE TEST SEQUENCES
    # ========================================================

    X_test, y_test = (
        create_sequences(
            data=test_data,
            feature_columns=(
                feature_columns
            ),
            scaler=scaler,
            sequence_length=(
                SEQUENCE_LENGTH
            ),
            stride=(
                SEQUENCE_STRIDE
            ),
        )
    )

    # ========================================================
    # SAVE DATA ARRAYS
    # ========================================================

    save_split(
        split_name="train",
        X=X_train,
        y=y_train,
    )

    save_split(
        split_name="validation",
        X=X_validation,
        y=y_validation,
    )

    save_split(
        split_name="test",
        X=X_test,
        y=y_test,
    )

    # ========================================================
    # EXPERIMENT METADATA
    # ========================================================

    metadata = {

        "dataset": (
            "public_rotating_machine"
        ),

        "classification_task": (
            "normal_vs_ball_fault"
        ),

        "labels": {
            "0": "normal",
            "1": "ball_fault",
        },

        "feature_window_duration_seconds": (
            WINDOW_DURATION_SECONDS
        ),

        "sequence_length_windows": (
            SEQUENCE_LENGTH
        ),

        "sequence_duration_seconds": (
            SEQUENCE_LENGTH
            * WINDOW_DURATION_SECONDS
        ),

        "sequence_stride_windows": (
            SEQUENCE_STRIDE
        ),

        "sequence_stride_seconds": (
            SEQUENCE_STRIDE
            * WINDOW_DURATION_SECONDS
        ),

        "feature_count": (
            len(feature_columns)
        ),

        "features": (
            feature_columns
        ),

        "training_run": 0,

        "validation_run": 1,

        "test_run": 2,

        "training_windows_before_sequence": (
            len(train_data)
        ),

        "validation_windows_before_sequence": (
            len(validation_data)
        ),

        "test_windows_before_sequence": (
            len(test_data)
        ),

        "train_shape": (
            list(
                X_train.shape
            )
        ),

        "validation_shape": (
            list(
                X_validation.shape
            )
        ),

        "test_shape": (
            list(
                X_test.shape
            )
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

        "scaling_method": (
            "StandardScaler"
        ),

        "scaler_fit_dataset": (
            "Run 0 training data only"
        ),

        "data_leakage_control": (
            "Independent recordings used for training, "
            "validation and testing. Feature scaler fitted "
            "only on Run 0."
        ),

        "rpm_alignment_policy": (
            "Final current/vibration windows are trimmed "
            "when the corresponding RPM recording ends "
            "before the final feature-window centre. "
            "RPM extrapolation is not performed."
        ),
    }

    metadata_file = (
        GRU_OUTPUT_DIR
        / "gru_preparation_metadata.json"
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

    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    print("\n" + "=" * 70)
    print(
        "GRU PREPARATION COMPLETE"
    )
    print("=" * 70)

    print(
        f"\nFeature count: "
        f"{len(feature_columns)}"
    )

    print(
        f"Sequence length: "
        f"{SEQUENCE_LENGTH} windows"
    )

    print(
        f"Sequence duration: "
        f"{SEQUENCE_LENGTH * WINDOW_DURATION_SECONDS:.1f} "
        "seconds"
    )

    print(
        f"Sequence stride: "
        f"{SEQUENCE_STRIDE} windows"
    )

    print(
        f"Sequence stride duration: "
        f"{SEQUENCE_STRIDE * WINDOW_DURATION_SECONDS:.1f} "
        "seconds"
    )

    print(
        "\nTrain:"
    )

    print(
        f"X = {X_train.shape}"
    )

    print(
        f"y = {y_train.shape}"
    )

    print(
        "\nValidation:"
    )

    print(
        f"X = {X_validation.shape}"
    )

    print(
        f"y = {y_validation.shape}"
    )

    print(
        "\nTest:"
    )

    print(
        f"X = {X_test.shape}"
    )

    print(
        f"y = {y_test.shape}"
    )

    print(
        "\nPreparation metadata saved to:"
    )

    print(
        metadata_file
    )

    print(
        "\nGRU data preparation is complete."
    )

    print(
        "Next stage: 05_gru_train.py"
    )


# ============================================================
# SCRIPT ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()