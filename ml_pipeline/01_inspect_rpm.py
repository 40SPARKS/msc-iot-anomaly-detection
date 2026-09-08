from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = (
    PROJECT_ROOT
    / "datasets"
    / "public_rotating_machine"
    / "selected_sample"
)

RPM_FILES = {
    "normal": "rpm_normal_0.csv",
    "ball_fault": "rpm_ball_0.csv",
}


rpm_datasets = {}


for condition, file_name in RPM_FILES.items():
    file_path = DATA_DIR / file_name

    if not file_path.exists():
        raise FileNotFoundError(
            f"File not found: {file_path}"
        )

    data = pd.read_csv(file_path)

    data["time"] = pd.to_numeric(
        data["time"],
        errors="coerce",
    )

    data["rpm"] = pd.to_numeric(
        data["rpm"],
        errors="coerce",
    )

    data = (
        data
        .dropna(subset=["time", "rpm"])
        .sort_values("time")
        .reset_index(drop=True)
    )

    rpm_datasets[condition] = data

    time_differences = np.diff(
        data["time"].to_numpy()
    )

    positive_differences = time_differences[
        time_differences > 0
    ]

    median_interval = float(
        np.median(positive_differences)
    )

    estimated_sampling_rate = (
        1.0 / median_interval
    )

    print(f"\n=== {condition} ===")
    print(f"Source file: {file_name}")
    print(f"Rows: {len(data)}")

    print(
        f"Duration: "
        f"{data['time'].max() - data['time'].min():.3f} seconds"
    )

    print(
        f"Median sampling interval: "
        f"{median_interval:.6f} seconds"
    )

    print(
        f"Estimated sampling rate: "
        f"{estimated_sampling_rate:.3f} Hz"
    )

    print("\nRPM summary:")
    print(
        data["rpm"].describe(
            percentiles=[
                0.05,
                0.25,
                0.50,
                0.75,
                0.95,
            ]
        )
    )


normal_rpm = rpm_datasets["normal"]["rpm"]
fault_rpm = rpm_datasets["ball_fault"]["rpm"]

overlap_minimum = max(
    normal_rpm.min(),
    fault_rpm.min(),
)

overlap_maximum = min(
    normal_rpm.max(),
    fault_rpm.max(),
)


print("\n=== RPM RANGE OVERLAP ===")

if overlap_minimum <= overlap_maximum:
    print(
        f"Common RPM range: "
        f"{overlap_minimum:.2f} to "
        f"{overlap_maximum:.2f} RPM"
    )

    for condition, data in rpm_datasets.items():
        inside_overlap = data["rpm"].between(
            overlap_minimum,
            overlap_maximum,
        )

        percentage = (
            inside_overlap.mean() * 100
        )

        print(
            f"{condition} readings inside "
            f"common range: {percentage:.2f}%"
        )
else:
    print(
        "The normal and ball-fault recordings "
        "have no common RPM range."
    )