from pathlib import Path
import json
import random
import time

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from tensorflow.keras import Sequential
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau,
)
from tensorflow.keras.layers import (
    Dense,
    Dropout,
    GRU,
    Input,
)


RANDOM_SEED = 42

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = (
    PROJECT_ROOT
    / "datasets"
    / "synthetic_conveyor"
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

PLOT_DIR = OUTPUT_DIR / "plots"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
PLOT_DIR.mkdir(parents=True, exist_ok=True)


TRAIN_FILE = DATA_DIR / "synthetic_gru_train.npz"

VALIDATION_FILE = (
    DATA_DIR
    / "synthetic_gru_validation.npz"
)

TEST_FILE = DATA_DIR / "synthetic_gru_test.npz"

PREPARATION_METADATA_FILE = (
    OUTPUT_DIR
    / "synthetic_gru_preparation_metadata.json"
)


BEST_MODEL_FILE = (
    MODEL_DIR
    / "synthetic_gru_best.keras"
)

FINAL_MODEL_FILE = (
    MODEL_DIR
    / "synthetic_gru_final.keras"
)

METRICS_FILE = (
    OUTPUT_DIR
    / "synthetic_gru_metrics.json"
)

HISTORY_FILE = (
    OUTPUT_DIR
    / "synthetic_gru_training_history.csv"
)

PREDICTIONS_FILE = (
    OUTPUT_DIR
    / "synthetic_gru_test_predictions.csv"
)

LOSS_PLOT_FILE = (
    PLOT_DIR
    / "synthetic_gru_loss.png"
)

ACCURACY_PLOT_FILE = (
    PLOT_DIR
    / "synthetic_gru_accuracy.png"
)

CONFUSION_MATRIX_FILE = (
    PLOT_DIR
    / "synthetic_gru_confusion_matrix.png"
)


MAX_EPOCHS = 60
BATCH_SIZE = 32
LEARNING_RATE = 0.001
CLASSIFICATION_THRESHOLD = 0.5


def load_dataset(file_path):
    """Load a prepared GRU dataset."""

    if not file_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {file_path}"
        )

    data = np.load(file_path)

    if "X" not in data or "y" not in data:
        raise ValueError(
            f"{file_path.name} must contain X and y arrays."
        )

    X = data["X"].astype(np.float32)
    y = data["y"].astype(np.float32)

    return X, y


def validate_dataset(name, X, y, expected_shape=None):
    """Check one dataset before training."""

    if X.ndim != 3:
        raise ValueError(
            f"{name}: X should have three dimensions."
        )

    if y.ndim != 1:
        raise ValueError(
            f"{name}: y should have one dimension."
        )

    if len(X) != len(y):
        raise ValueError(
            f"{name}: X and y have different sample counts."
        )

    if expected_shape is not None:
        if X.shape[1:] != expected_shape:
            raise ValueError(
                f"{name}: expected sequence shape "
                f"{expected_shape}, found {X.shape[1:]}."
            )

    if not np.isfinite(X).all():
        raise ValueError(
            f"{name}: X contains invalid values."
        )

    if not np.isfinite(y).all():
        raise ValueError(
            f"{name}: y contains invalid values."
        )

    labels = set(
        np.unique(y)
        .astype(int)
        .tolist()
    )

    if labels != {0, 1}:
        raise ValueError(
            f"{name}: expected labels 0 and 1, "
            f"found {sorted(labels)}."
        )

    print(
        f"{name:<12} X={X.shape} y={y.shape}"
    )


def build_model(sequence_length, feature_count):
    """Build the synthetic conveyor GRU classifier."""

    model = Sequential(
        [
            Input(
                shape=(
                    sequence_length,
                    feature_count,
                )
            ),

            GRU(
                32,
                return_sequences=False,
            ),

            Dropout(0.20),

            Dense(
                16,
                activation="relu",
            ),

            Dense(
                1,
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


def save_training_plots(history):
    """Save loss and accuracy curves."""

    history_data = history.history

    epochs = range(
        1,
        len(history_data["loss"]) + 1,
    )

    plt.figure(figsize=(8, 5))

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

    plt.xlabel("Epoch")
    plt.ylabel("Binary cross-entropy loss")
    plt.title(
        "Synthetic GRU Training and Validation Loss"
    )
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        LOSS_PLOT_FILE,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()


    plt.figure(figsize=(8, 5))

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

    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title(
        "Synthetic GRU Training and Validation Accuracy"
    )
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        ACCURACY_PLOT_FILE,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()


def save_confusion_matrix(y_true, y_pred):
    """Save the final unseen test confusion matrix."""

    matrix = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1],
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
        "Synthetic GRU Test Confusion Matrix"
    )

    plt.tight_layout()

    plt.savefig(
        CONFUSION_MATRIX_FILE,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    return matrix


def main():
    print("Synthetic conveyor GRU training")
    print()

    print(
        f"TensorFlow version: {tf.__version__}"
    )

    print("TensorFlow devices:")

    for device in tf.config.list_physical_devices():
        print(f"  {device}")

    print()
    print("Loading prepared datasets...")

    X_train, y_train = load_dataset(
        TRAIN_FILE
    )

    X_validation, y_validation = load_dataset(
        VALIDATION_FILE
    )

    X_test, y_test = load_dataset(
        TEST_FILE
    )

    sequence_shape = X_train.shape[1:]

    validate_dataset(
        "Train",
        X_train,
        y_train,
    )

    validate_dataset(
        "Validation",
        X_validation,
        y_validation,
        expected_shape=sequence_shape,
    )

    validate_dataset(
        "Test",
        X_test,
        y_test,
        expected_shape=sequence_shape,
    )

    sequence_length = X_train.shape[1]
    feature_count = X_train.shape[2]

    preparation_metadata = {}

    if PREPARATION_METADATA_FILE.exists():

        with open(
            PREPARATION_METADATA_FILE,
            "r",
            encoding="utf-8",
        ) as file:
            preparation_metadata = json.load(
                file
            )

    print()
    print("Building model...")

    model = build_model(
        sequence_length,
        feature_count,
    )

    model.summary()

    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            patience=8,
            restore_best_weights=True,
            verbose=1,
        ),

        ModelCheckpoint(
            filepath=str(
                BEST_MODEL_FILE
            ),
            monitor="val_loss",
            save_best_only=True,
            verbose=1,
        ),

        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=4,
            min_lr=1e-6,
            verbose=1,
        ),
    ]

    print()
    print("Training model...")
    print(
        f"Maximum epochs: {MAX_EPOCHS}"
    )
    print(
        f"Batch size: {BATCH_SIZE}"
    )
    print(
        f"Learning rate: {LEARNING_RATE}"
    )

    start_time = time.perf_counter()

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

    training_time_seconds = (
        time.perf_counter()
        - start_time
    )

    model.save(
        FINAL_MODEL_FILE
    )

    history_dataframe = pd.DataFrame(
        history.history
    )

    history_dataframe["epoch"] = np.arange(
        1,
        len(history_dataframe) + 1,
    )

    history_dataframe.to_csv(
        HISTORY_FILE,
        index=False,
    )

    save_training_plots(
        history
    )

    print()
    print("Evaluating final unseen Run 2...")

    inference_start = time.perf_counter()

    probabilities = (
        model.predict(
            X_test,
            batch_size=BATCH_SIZE,
            verbose=0,
        )
        .reshape(-1)
    )

    inference_time_seconds = (
        time.perf_counter()
        - inference_start
    )

    predictions = (
        probabilities
        >= CLASSIFICATION_THRESHOLD
    ).astype(int)

    y_true = y_test.astype(int)

    accuracy = accuracy_score(
        y_true,
        predictions,
    )

    precision = precision_score(
        y_true,
        predictions,
        zero_division=0,
    )

    recall = recall_score(
        y_true,
        predictions,
        zero_division=0,
    )

    f1 = f1_score(
        y_true,
        predictions,
        zero_division=0,
    )

    roc_auc = roc_auc_score(
        y_true,
        probabilities,
    )

    average_precision = (
        average_precision_score(
            y_true,
            probabilities,
        )
    )

    matrix = save_confusion_matrix(
        y_true,
        predictions,
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

    validation_losses = np.asarray(
        history.history["val_loss"]
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

    epochs_completed = len(
        history.history["loss"]
    )

    test_results = pd.DataFrame(
        {
            "sequence_index": np.arange(
                len(y_true)
            ),
            "true_label": y_true,
            "true_condition": np.where(
                y_true == 0,
                "normal",
                "fault",
            ),
            "fault_probability": probabilities,
            "predicted_label": predictions,
            "predicted_condition": np.where(
                predictions == 0,
                "normal",
                "fault",
            ),
            "correct_prediction": (
                y_true == predictions
            ),
        }
    )

    test_results.to_csv(
        PREDICTIONS_FILE,
        index=False,
    )

    report = classification_report(
        y_true,
        predictions,
        target_names=[
            "normal",
            "fault",
        ],
        output_dict=True,
        zero_division=0,
    )

    results = {
        "experiment": (
            "synthetic_conveyor_gru"
        ),
        "dataset": (
            "synthetic_conveyor"
        ),
        "classification_task": (
            "normal_vs_fault"
        ),
        "tensorflow_version": (
            tf.__version__
        ),
        "random_seed": (
            RANDOM_SEED
        ),
        "training_run": 0,
        "validation_run": 1,
        "test_run": 2,
        "sequence_length": (
            sequence_length
        ),
        "feature_count": (
            feature_count
        ),
        "sequence_duration_seconds": (
            preparation_metadata.get(
                "sequence_duration_seconds"
            )
        ),
        "sequence_stride_seconds": (
            preparation_metadata.get(
                "sequence_stride_seconds"
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
        "epochs_completed": int(
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
        "best_model_file": str(
            BEST_MODEL_FILE
        ),
        "final_model_file": str(
            FINAL_MODEL_FILE
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

    print()
    print(
        f"Training completed in "
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

    print()
    print("Final unseen Run 2 results:")

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

    print()
    print("Confusion matrix:")
    print(matrix)

    print()
    print(
        f"True normal:      "
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

    print()
    print("Classification report:")

    print(
        classification_report(
            y_true,
            predictions,
            target_names=[
                "normal",
                "fault",
            ],
            zero_division=0,
        )
    )

    print()
    print("Synthetic GRU training complete.")

    print(
        f"Best model: {BEST_MODEL_FILE}"
    )

    print(
        f"Final model: {FINAL_MODEL_FILE}"
    )

    print(
        f"Metrics: {METRICS_FILE}"
    )

    print(
        f"Predictions: {PREDICTIONS_FILE}"
    )

    print(
        f"Training history: {HISTORY_FILE}"
    )

    print(
        f"Loss plot: {LOSS_PLOT_FILE}"
    )

    print(
        f"Accuracy plot: {ACCURACY_PLOT_FILE}"
    )

    print(
        f"Confusion matrix plot: "
        f"{CONFUSION_MATRIX_FILE}"
    )


if __name__ == "__main__":
    main()