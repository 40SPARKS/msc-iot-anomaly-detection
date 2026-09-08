from pathlib import Path
import json

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

OUTPUT_DIR = (
    PROJECT_ROOT
    / "ml_pipeline"
    / "outputs"
)

COMPARISON_DIR = OUTPUT_DIR / "comparison"
PLOT_DIR = COMPARISON_DIR / "plots"

COMPARISON_DIR.mkdir(parents=True, exist_ok=True)
PLOT_DIR.mkdir(parents=True, exist_ok=True)


DOWNLOADED_IF_FILE = (
    OUTPUT_DIR
    / "isolation_forest_modality_comparison.csv"
)

DOWNLOADED_GRU_FILE = (
    OUTPUT_DIR
    / "gru"
    / "downloaded_gru_metrics.json"
)

SYNTHETIC_IF_FILE = (
    OUTPUT_DIR
    / "synthetic"
    / "isolation_forest"
    / "synthetic_isolation_forest_metrics.json"
)

SYNTHETIC_GRU_FILE = (
    OUTPUT_DIR
    / "synthetic"
    / "gru"
    / "synthetic_gru_metrics.json"
)


def load_json(file_path):
    if not file_path.exists():
        raise FileNotFoundError(
            f"Result file not found: {file_path}"
        )

    with open(
        file_path,
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


def load_downloaded_isolation_forest():
    if not DOWNLOADED_IF_FILE.exists():
        raise FileNotFoundError(
            f"Result file not found: {DOWNLOADED_IF_FILE}"
        )

    data = pd.read_csv(
        DOWNLOADED_IF_FILE
    )

    all_sensor_rows = data[
        data["model"] == "all_sensors"
    ]

    if len(all_sensor_rows) != 1:
        raise ValueError(
            "Could not uniquely identify the "
            "all_sensors Isolation Forest result."
        )

    return all_sensor_rows.iloc[0]


def create_result_row(
    dataset,
    algorithm,
    feature_count,
    accuracy,
    precision,
    fault_recall,
    f1_score,
    specificity,
    false_alarm_rate,
    roc_auc,
    average_precision,
    true_normal,
    false_alarms,
    missed_faults,
    detected_faults,
):
    return {
        "dataset": dataset,
        "algorithm": algorithm,
        "feature_count": int(feature_count),
        "accuracy": float(accuracy),
        "precision": float(precision),
        "fault_recall": float(fault_recall),
        "f1_score": float(f1_score),
        "specificity": float(specificity),
        "false_alarm_rate": float(false_alarm_rate),
        "roc_auc": float(roc_auc),
        "average_precision": float(average_precision),
        "true_normal": int(true_normal),
        "false_alarms": int(false_alarms),
        "missed_faults": int(missed_faults),
        "detected_faults": int(detected_faults),
    }


def save_metric_plot(results, metric, title, ylabel, filename):
    labels = (
        results["dataset"]
        + "\n"
        + results["algorithm"]
    )

    values = results[metric].to_numpy()

    plt.figure(figsize=(9, 5))

    bars = plt.bar(
        labels,
        values,
    )

    plt.ylabel(ylabel)
    plt.title(title)

    if metric != "false_alarm_rate":
        plt.ylim(0, 1.05)

    for bar, value in zip(bars, values):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.015,
            f"{value * 100:.1f}%",
            ha="center",
            va="bottom",
        )

    plt.tight_layout()

    plt.savefig(
        PLOT_DIR / filename,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()


def main():
    print("Loading completed model results...")

    downloaded_if = (
        load_downloaded_isolation_forest()
    )

    downloaded_gru = load_json(
        DOWNLOADED_GRU_FILE
    )

    synthetic_if = load_json(
        SYNTHETIC_IF_FILE
    )

    synthetic_gru = load_json(
        SYNTHETIC_GRU_FILE
    )

    rows = []

    rows.append(
        create_result_row(
            dataset="Downloaded",
            algorithm="Isolation Forest",
            feature_count=downloaded_if["feature_count"],
            accuracy=downloaded_if["accuracy"],
            precision=downloaded_if["precision"],
            fault_recall=downloaded_if["fault_recall"],
            f1_score=downloaded_if["f1_score"],
            specificity=downloaded_if["specificity"],
            false_alarm_rate=downloaded_if[
                "false_alarm_rate"
            ],
            roc_auc=downloaded_if["roc_auc"],
            average_precision=downloaded_if[
                "average_precision"
            ],
            true_normal=downloaded_if["true_normal"],
            false_alarms=downloaded_if["false_alarms"],
            missed_faults=downloaded_if["missed_faults"],
            detected_faults=downloaded_if[
                "detected_faults"
            ],
        )
    )

    rows.append(
        create_result_row(
            dataset="Downloaded",
            algorithm="GRU",
            feature_count=downloaded_gru["feature_count"],
            accuracy=downloaded_gru["accuracy"],
            precision=downloaded_gru["precision"],
            fault_recall=downloaded_gru["fault_recall"],
            f1_score=downloaded_gru["f1_score"],
            specificity=downloaded_gru["specificity"],
            false_alarm_rate=downloaded_gru[
                "false_alarm_rate"
            ],
            roc_auc=downloaded_gru["roc_auc"],
            average_precision=downloaded_gru[
                "average_precision"
            ],
            true_normal=downloaded_gru["true_normal"],
            false_alarms=downloaded_gru["false_alarms"],
            missed_faults=downloaded_gru["missed_faults"],
            detected_faults=downloaded_gru[
                "detected_faults"
            ],
        )
    )

    rows.append(
        create_result_row(
            dataset="Synthetic",
            algorithm="Isolation Forest",
            feature_count=synthetic_if["feature_count"],
            accuracy=synthetic_if["accuracy"],
            precision=synthetic_if["precision"],
            fault_recall=synthetic_if["fault_recall"],
            f1_score=synthetic_if["f1_score"],
            specificity=synthetic_if["specificity"],
            false_alarm_rate=synthetic_if[
                "false_alarm_rate"
            ],
            roc_auc=synthetic_if["roc_auc"],
            average_precision=synthetic_if[
                "average_precision"
            ],
            true_normal=synthetic_if["true_normal"],
            false_alarms=synthetic_if["false_alarms"],
            missed_faults=synthetic_if["missed_faults"],
            detected_faults=synthetic_if[
                "detected_faults"
            ],
        )
    )

    rows.append(
        create_result_row(
            dataset="Synthetic",
            algorithm="GRU",
            feature_count=synthetic_gru["feature_count"],
            accuracy=synthetic_gru["accuracy"],
            precision=synthetic_gru["precision"],
            fault_recall=synthetic_gru["fault_recall"],
            f1_score=synthetic_gru["f1_score"],
            specificity=synthetic_gru["specificity"],
            false_alarm_rate=synthetic_gru[
                "false_alarm_rate"
            ],
            roc_auc=synthetic_gru["roc_auc"],
            average_precision=synthetic_gru[
                "average_precision"
            ],
            true_normal=synthetic_gru["true_normal"],
            false_alarms=synthetic_gru["false_alarms"],
            missed_faults=synthetic_gru["missed_faults"],
            detected_faults=synthetic_gru[
                "detected_faults"
            ],
        )
    )

    results = pd.DataFrame(rows)

    raw_output_file = (
        COMPARISON_DIR
        / "model_comparison.csv"
    )

    results.to_csv(
        raw_output_file,
        index=False,
    )

    report_table = results[
        [
            "dataset",
            "algorithm",
            "feature_count",
            "accuracy",
            "precision",
            "fault_recall",
            "f1_score",
            "specificity",
            "false_alarm_rate",
            "roc_auc",
        ]
    ].copy()

    percentage_columns = [
        "accuracy",
        "precision",
        "fault_recall",
        "f1_score",
        "specificity",
        "false_alarm_rate",
        "roc_auc",
    ]

    for column in percentage_columns:
        report_table[column] = (
            report_table[column] * 100
        )

    report_output_file = (
        COMPARISON_DIR
        / "model_comparison_percentages.csv"
    )

    report_table.to_csv(
        report_output_file,
        index=False,
        float_format="%.2f",
    )

    save_metric_plot(
        results,
        metric="accuracy",
        title="Model Accuracy Comparison",
        ylabel="Accuracy",
        filename="accuracy_comparison.png",
    )

    save_metric_plot(
        results,
        metric="fault_recall",
        title="Fault Recall Comparison",
        ylabel="Fault Recall",
        filename="fault_recall_comparison.png",
    )

    save_metric_plot(
        results,
        metric="f1_score",
        title="F1 Score Comparison",
        ylabel="F1 Score",
        filename="f1_comparison.png",
    )

    save_metric_plot(
        results,
        metric="roc_auc",
        title="ROC-AUC Comparison",
        ylabel="ROC-AUC",
        filename="roc_auc_comparison.png",
    )

    save_metric_plot(
        results,
        metric="false_alarm_rate",
        title="False Alarm Rate Comparison",
        ylabel="False Alarm Rate",
        filename="false_alarm_rate_comparison.png",
    )

    print()
    print("Final model comparison")
    print()

    display_table = report_table.copy()

    print(
        display_table.to_string(
            index=False
        )
    )

    print()
    print(
        f"Raw results saved to: "
        f"{raw_output_file}"
    )

    print(
        f"Report-ready table saved to: "
        f"{report_output_file}"
    )

    print(
        f"Comparison plots saved to: "
        f"{PLOT_DIR}"
    )


if __name__ == "__main__":
    main()