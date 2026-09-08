from pathlib import Path
import json
import random
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import tensorflow as tf

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
)

from tensorflow.keras import Sequential
from tensorflow.keras.layers import (
    Input,
    GRU,
    Dropout,
    Dense,
)
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau,
)


# ============================================================
# REPRODUCIBILITY
# ============================================================

RANDOM_SEED = 42

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

GRU_DATA_DIR = (
    PROJECT_ROOT
    / "datasets"
    / "public_rotating_machine"
    / "processed"
    / "gru"
)

GRU_MODEL_DIR = (
    PROJECT_ROOT
    / "ml_pipeline"
    / "models"
    / "gru"
)

GRU_OUTPUT_DIR = (
    PROJECT_ROOT
    / "ml_pipeline"
    / "outputs"
    / "gru"
)

GRU_PLOT_DIR = (
    GRU_OUTPUT_DIR
    / "plots"
)

GRU_MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

GRU_OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

GRU_PLOT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# INPUT FILES
# ============================================================

TRAIN_FILE = (
    GRU_DATA_DIR
    / "gru_train.npz"
)

VALIDATION_FILE = (
    GRU_DATA_DIR
    / "gru_validation.npz"
)

TEST_FILE = (
    GRU_DATA_DIR
    / "gru_test.npz"
)

PREPARATION_METADATA_FILE = (
    GRU_OUTPUT_DIR
    / "gru_preparation_metadata.json"
)


# ============================================================
# OUTPUT FILES
# ============================================================

BEST_MODEL_FILE = (
    GRU_MODEL_DIR
    / "downloaded_gru_best.keras"
)

FINAL_MODEL_FILE = (
    GRU_MODEL_DIR
    / "downloaded_gru_final.keras"
)

HISTORY_FILE = (
    GRU_OUTPUT_DIR
    / "downloaded_gru_training_history.csv"
)

METRICS_FILE = (
    GRU_OUTPUT_DIR
    / "downloaded_gru_metrics.json"
)

PREDICTIONS_FILE = (
    GRU_OUTPUT_DIR
    / "downloaded_gru_test_predictions.csv"
)

LOSS_PLOT_FILE = (
    GRU_PLOT_DIR
    / "downloaded_gru_loss.png"
)

ACCURACY_PLOT_FILE = (
    GRU_PLOT_DIR
    / "downloaded_gru_accuracy.png"
)

CONFUSION_MATRIX_FILE = (
    GRU_PLOT_DIR
    / "downloaded_gru_confusion_matrix.png"
)


# ============================================================
# TRAINING SETTINGS
# ============================================================

MAX_EPOCHS = 60

BATCH_SIZE = 32

LEARNING_RATE = 0.001

CLASSIFICATION_THRESHOLD = 0.5


# ============================================================
# LOAD NPZ DATA
# ============================================================

def load_dataset(file_path):
    """
    Load one prepared GRU NPZ dataset.
    """

    if not file_path.exists():
        raise FileNotFoundError(
            f"GRU dataset not found:\n"
            f"{file_path}"
        )

    data = np.load(
        file_path
    )

    if (
        "X" not in data
        or "y" not in data
    ):
        raise ValueError(
            f"{file_path.name} must contain "
            "X and y arrays."
        )

    X = data["X"].astype(
        np.float32
    )

    y = data["y"].astype(
        np.float32
    )

    return X, y


# ============================================================
# VALIDATE DATASETS
# ============================================================

def validate_datasets(
    X_train,
    y_train,
    X_validation,
    y_validation,
    X_test,
    y_test,
):
    """
    Verify that train, validation and test arrays are
    compatible before model training begins.
    """

    datasets = {
        "train": (
            X_train,
            y_train,
        ),
        "validation": (
            X_validation,
            y_validation,
        ),
        "test": (
            X_test,
            y_test,
        ),
    }

    reference_shape = (
        X_train.shape[1:]
    )

    for name, (X, y) in datasets.items():

        if X.ndim != 3:
            raise ValueError(
                f"{name}: expected X to be "
                f"3-dimensional, found "
                f"{X.ndim} dimensions."
            )

        if y.ndim != 1:
            raise ValueError(
                f"{name}: expected y to be "
                "1-dimensional."
            )

        if len(X) != len(y):
            raise ValueError(
                f"{name}: X and y contain "
                "different numbers of samples."
            )

        if X.shape[1:] != reference_shape:
            raise ValueError(
                f"{name}: sequence shape "
                f"{X.shape[1:]} does not match "
                f"training shape "
                f"{reference_shape}."
            )

        if not np.isfinite(X).all():
            raise ValueError(
                f"{name}: X contains NaN "
                "or infinite values."
            )

        if not np.isfinite(y).all():
            raise ValueError(
                f"{name}: y contains invalid values."
            )

        unique_labels = set(
            np.unique(y)
            .astype(int)
            .tolist()
        )

        if unique_labels != {0, 1}:
            raise ValueError(
                f"{name}: expected binary labels "
                f"0 and 1, found "
                f"{sorted(unique_labels)}."
            )

        print(
            f"{name.upper():<12} "
            f"X={X.shape} "
            f"y={y.shape}"
        )

    print(
        "\nDataset validation passed."
    )


# ============================================================
# BUILD GRU MODEL
# ============================================================

def build_model(
    sequence_length,
    feature_count,
):
    """
    Build a compact binary GRU classifier.

    Input:
        20 time steps x 37 features

    Output:
        probability of ball fault
    """

    model = Sequential(
        [
            Input(
                shape=(
                    sequence_length,
                    feature_count,
                )
            ),

            GRU(
                units=32,
                return_sequences=False,
            ),

            Dropout(
                rate=0.20
            ),

            Dense(
                units=16,
                activation="relu",
            ),

            Dense(
                units=1,
                activation="sigmoid",
            ),
        ]
    )

    optimiser = tf.keras.optimizers.Adam(
        learning_rate=LEARNING_RATE
    )

    model.compile(
        optimizer=optimiser,
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.AUC(
                name="auc"
            ),
        ],
    )

    return model


# ============================================================
# SAVE TRAINING CURVES
# ============================================================

def save_training_plots(
    history,
):
    """
    Save training/validation loss and accuracy plots.
    """

    history_data = history.history

    epochs = range(
        1,
        len(history_data["loss"]) + 1,
    )

    # --------------------------------------------------------
    # Loss
    # --------------------------------------------------------

    plt.figure(
        figsize=(8, 5)
    )

    plt.plot(
        epochs,
        history_data["loss"],
        label="Training loss",
    )

    plt.plot(
        epochs,
        history_data["val_loss"],
        label="Validation loss",
    )

    plt.xlabel(
        "Epoch"
    )

    plt.ylabel(
        "Binary cross-entropy loss"
    )

    plt.title(
        "GRU Training and Validation Loss"
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        LOSS_PLOT_FILE,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    # --------------------------------------------------------
    # Accuracy
    # --------------------------------------------------------

    plt.figure(
        figsize=(8, 5)
    )

    plt.plot(
        epochs,
        history_data["accuracy"],
        label="Training accuracy",
    )

    plt.plot(
        epochs,
        history_data["val_accuracy"],
        label="Validation accuracy",
    )

    plt.xlabel(
        "Epoch"
    )

    plt.ylabel(
        "Accuracy"
    )

    plt.title(
        "GRU Training and Validation Accuracy"
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        ACCURACY_PLOT_FILE,
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
    Save final test-set confusion matrix.
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
            "Ball fault",
        ],
    )

    display.plot(
        values_format="d"
    )

    plt.title(
        "GRU Test Confusion Matrix"
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
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print(
        "DOWNLOADED DATASET GRU TRAINING"
    )
    print("=" * 70)

    print(
        f"\nTensorFlow version: "
        f"{tf.__version__}"
    )

    devices = (
        tf.config.list_physical_devices()
    )

    print(
        f"Available TensorFlow devices:"
    )

    for device in devices:
        print(
            f"  {device}"
        )

    # ========================================================
    # LOAD PREPARED DATA
    # ========================================================

    print(
        "\nLoading GRU datasets..."
    )

    X_train, y_train = (
        load_dataset(
            TRAIN_FILE
        )
    )

    (
        X_validation,
        y_validation,
    ) = load_dataset(
        VALIDATION_FILE
    )

    X_test, y_test = (
        load_dataset(
            TEST_FILE
        )
    )

    print(
        "\nPrepared dataset shapes:"
    )

    validate_datasets(
        X_train,
        y_train,
        X_validation,
        y_validation,
        X_test,
        y_test,
    )

    sequence_length = (
        X_train.shape[1]
    )

    feature_count = (
        X_train.shape[2]
    )

    # ========================================================
    # LOAD PREPARATION METADATA
    # ========================================================

    preparation_metadata = {}

    if PREPARATION_METADATA_FILE.exists():

        with open(
            PREPARATION_METADATA_FILE,
            "r",
            encoding="utf-8",
        ) as file:

            preparation_metadata = (
                json.load(file)
            )

    # ========================================================
    # BUILD MODEL
    # ========================================================

    print(
        "\nBuilding GRU model..."
    )

    model = build_model(
        sequence_length=(
            sequence_length
        ),
        feature_count=(
            feature_count
        ),
    )

    print()

    model.summary()

    # ========================================================
    # CALLBACKS
    # ========================================================

    callbacks = [

        # Stop training once validation loss stops improving.
        EarlyStopping(
            monitor="val_loss",
            patience=8,
            restore_best_weights=True,
            verbose=1,
        ),

        # Preserve best model according to validation loss.
        ModelCheckpoint(
            filepath=str(
                BEST_MODEL_FILE
            ),
            monitor="val_loss",
            save_best_only=True,
            verbose=1,
        ),

        # Reduce learning rate if validation loss plateaus.
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=4,
            min_lr=1e-6,
            verbose=1,
        ),
    ]

    # ========================================================
    # TRAIN
    # ========================================================

    print("\n" + "=" * 70)
    print(
        "TRAINING"
    )
    print("=" * 70)

    print(
        f"Maximum epochs: "
        f"{MAX_EPOCHS}"
    )

    print(
        f"Batch size: "
        f"{BATCH_SIZE}"
    )

    print(
        f"Initial learning rate: "
        f"{LEARNING_RATE}"
    )

    training_start = (
        time.perf_counter()
    )

    history = model.fit(
        X_train,
        y_train,
        validation_data=(
            X_validation,
            y_validation,
        ),
        epochs=MAX_EPOCHS,
        batch_size=BATCH_SIZE,
        shuffle=True,
        callbacks=callbacks,
        verbose=1,
    )

    training_end = (
        time.perf_counter()
    )

    training_time_seconds = (
        training_end
        - training_start
    )

    # ========================================================
    # SAVE FINAL MODEL
    # ========================================================

    model.save(
        FINAL_MODEL_FILE
    )

    # ========================================================
    # SAVE TRAINING HISTORY
    # ========================================================

    history_dataframe = pd.DataFrame(
        history.history
    )

    history_dataframe[
        "epoch"
    ] = np.arange(
        1,
        len(history_dataframe) + 1,
    )

    history_dataframe.to_csv(
        HISTORY_FILE,
        index=False,
    )

    # ========================================================
    # PLOTS
    # ========================================================

    save_training_plots(
        history
    )

    # ========================================================
    # TEST PREDICTIONS
    # ========================================================

    print("\n" + "=" * 70)
    print(
        "FINAL TEST EVALUATION"
    )
    print("=" * 70)

    test_start = (
        time.perf_counter()
    )

    test_probabilities = (
        model.predict(
            X_test,
            batch_size=BATCH_SIZE,
            verbose=0,
        )
        .reshape(-1)
    )

    test_end = (
        time.perf_counter()
    )

    inference_time_seconds = (
        test_end
        - test_start
    )

    test_predictions = (
        test_probabilities
        >= CLASSIFICATION_THRESHOLD
    ).astype(int)

    y_test_integer = (
        y_test.astype(int)
    )

    # ========================================================
    # METRICS
    # ========================================================

    accuracy = accuracy_score(
        y_test_integer,
        test_predictions,
    )

    precision = precision_score(
        y_test_integer,
        test_predictions,
        zero_division=0,
    )

    recall = recall_score(
        y_test_integer,
        test_predictions,
        zero_division=0,
    )

    f1 = f1_score(
        y_test_integer,
        test_predictions,
        zero_division=0,
    )

    roc_auc = roc_auc_score(
        y_test_integer,
        test_probabilities,
    )

    average_precision = (
        average_precision_score(
            y_test_integer,
            test_probabilities,
        )
    )

    matrix = save_confusion_matrix(
        y_test_integer,
        test_predictions,
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
    # SAVE PREDICTIONS
    # ========================================================

    predictions_dataframe = pd.DataFrame(
        {
            "sequence_index": (
                np.arange(
                    len(y_test_integer)
                )
            ),
            "true_label": (
                y_test_integer
            ),
            "true_condition": (
                np.where(
                    y_test_integer == 0,
                    "normal",
                    "ball_fault",
                )
            ),
            "fault_probability": (
                test_probabilities
            ),
            "predicted_label": (
                test_predictions
            ),
            "predicted_condition": (
                np.where(
                    test_predictions == 0,
                    "normal",
                    "ball_fault",
                )
            ),
            "correct_prediction": (
                y_test_integer
                == test_predictions
            ),
        }
    )

    predictions_dataframe.to_csv(
        PREDICTIONS_FILE,
        index=False,
    )

    # ========================================================
    # DETERMINE BEST EPOCH
    # ========================================================

    validation_losses = np.asarray(
        history.history[
            "val_loss"
        ]
    )

    best_epoch = int(
        np.argmin(
            validation_losses
        )
        + 1
    )

    best_validation_loss = float(
        np.min(
            validation_losses
        )
    )

    epochs_completed = int(
        len(
            history.history[
                "loss"
            ]
        )
    )

    # ========================================================
    # CLASSIFICATION REPORT
    # ========================================================

    report = classification_report(
        y_test_integer,
        test_predictions,
        target_names=[
            "normal",
            "ball_fault",
        ],
        output_dict=True,
        zero_division=0,
    )

    # ========================================================
    # SAVE METRICS
    # ========================================================

    results = {

        "experiment": (
            "downloaded_dataset_gru"
        ),

        "dataset": (
            "public_rotating_machine"
        ),

        "classification_task": (
            "normal_vs_ball_fault"
        ),

        "tensorflow_version": (
            tf.__version__
        ),

        "random_seed": (
            RANDOM_SEED
        ),

        "training_run": (
            preparation_metadata.get(
                "training_run",
                0,
            )
        ),

        "validation_run": (
            preparation_metadata.get(
                "validation_run",
                1,
            )
        ),

        "test_run": (
            preparation_metadata.get(
                "test_run",
                2,
            )
        ),

        "sequence_length": (
            sequence_length
        ),

        "feature_count": (
            feature_count
        ),

        "sequence_duration_seconds": (
            preparation_metadata.get(
                "sequence_duration_seconds",
                None,
            )
        ),

        "train_sequences": int(
            len(X_train)
        ),

        "validation_sequences": int(
            len(X_validation)
        ),

        "test_sequences": int(
            len(X_test)
        ),

        "gru_units": 32,

        "dense_units": 16,

        "dropout_rate": 0.20,

        "batch_size": (
            BATCH_SIZE
        ),

        "maximum_epochs": (
            MAX_EPOCHS
        ),

        "epochs_completed": (
            epochs_completed
        ),

        "best_epoch": (
            best_epoch
        ),

        "best_validation_loss": (
            best_validation_loss
        ),

        "learning_rate": (
            LEARNING_RATE
        ),

        "classification_threshold": (
            CLASSIFICATION_THRESHOLD
        ),

        "training_time_seconds": float(
            training_time_seconds
        ),

        "test_inference_time_seconds": float(
            inference_time_seconds
        ),

        "accuracy": float(
            accuracy
        ),

        "precision": float(
            precision
        ),

        "fault_recall": float(
            recall
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

        "confusion_matrix": (
            matrix.tolist()
        ),

        "classification_report": (
            report
        ),

        "best_model_file": (
            str(
                BEST_MODEL_FILE
            )
        ),

        "final_model_file": (
            str(
                FINAL_MODEL_FILE
            )
        ),
    }

    with open(
        METRICS_FILE,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            results,
            file,
            indent=4,
        )

    # ========================================================
    # TERMINAL RESULTS
    # ========================================================

    print(
        f"\nTraining completed in "
        f"{training_time_seconds:.2f} seconds."
    )

    print(
        f"Epochs completed: "
        f"{epochs_completed}"
    )

    print(
        f"Best epoch: "
        f"{best_epoch}"
    )

    print(
        f"Best validation loss: "
        f"{best_validation_loss:.6f}"
    )

    print("\nFinal unseen Run 2 results:")

    print(
        f"Accuracy:          "
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
        f"{recall:.4f} "
        f"({recall * 100:.2f}%)"
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

    print("\nConfusion matrix:")

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
            y_test_integer,
            test_predictions,
            target_names=[
                "normal",
                "ball_fault",
            ],
            zero_division=0,
        )
    )

    # ========================================================
    # OUTPUT SUMMARY
    # ========================================================

    print("\n" + "=" * 70)
    print(
        "GRU TRAINING AND EVALUATION COMPLETE"
    )
    print("=" * 70)

    print(
        "\nBest model:"
    )
    print(
        BEST_MODEL_FILE
    )

    print(
        "\nFinal model:"
    )
    print(
        FINAL_MODEL_FILE
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
        "\nTraining history:"
    )
    print(
        HISTORY_FILE
    )

    print(
        "\nPlots:"
    )
    print(
        LOSS_PLOT_FILE
    )
    print(
        ACCURACY_PLOT_FILE
    )
    print(
        CONFUSION_MATRIX_FILE
    )


# ============================================================
# SCRIPT ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()