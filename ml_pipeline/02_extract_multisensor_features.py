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

OUTPUT_DIR = (
    PROJECT_ROOT
    / "datasets"
    / "public_rotating_machine"
    / "processed"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# Sampling rates documented for the downloaded dataset.
CURRENT_SAMPLING_RATE = 100_000
VIBRATION_SAMPLING_RATE = 25_600

# A 0.1-second window gives approximately 3,000 windows
# during each 300-second recording.
WINDOW_DURATION_SECONDS = 0.1

CURRENT_SAMPLES_PER_WINDOW = int(
    CURRENT_SAMPLING_RATE * WINDOW_DURATION_SECONDS
)

VIBRATION_SAMPLES_PER_WINDOW = int(
    VIBRATION_SAMPLING_RATE * WINDOW_DURATION_SECONDS
)

CURRENT_COLUMNS = [
    "current_R",
    "current_S",
    "current_T",
]

VIBRATION_COLUMNS = [
    "bearingA_x",
    "bearingA_y",
    "bearingB_x",
    "bearingB_y",
]


def extract_stream_features(
    file_path,
    sensor_columns,
    samples_per_window,
    chunk_size,
):
    """Extract time-domain features without loading a full CSV."""

    if not file_path.exists():
        raise FileNotFoundError(
            f"Input file not found: {file_path}"
        )

    print(f"\nProcessing: {file_path.name}")
    print(f"Samples per window: {samples_per_window}")

    feature_parts = []
    carry_over = pd.DataFrame(columns=sensor_columns)

    total_rows = 0
    next_window_id = 0

    reader = pd.read_csv(
        file_path,
        usecols=sensor_columns,
        chunksize=chunk_size,
    )

    for chunk_number, chunk in enumerate(reader, start=1):
        total_rows += len(chunk)

        chunk = chunk.apply(
            pd.to_numeric,
            errors="coerce",
        )

        invalid_count = int(
            chunk.isna().any(axis=1).sum()
        )

        if invalid_count:
            raise ValueError(
                f"{file_path.name}, chunk {chunk_number}: "
                f"found {invalid_count} invalid rows."
            )

        if not carry_over.empty:
            chunk = pd.concat(
                [carry_over, chunk],
                ignore_index=True,
            )
        else:
            chunk = chunk.reset_index(drop=True)

        usable_rows = (
            len(chunk) // samples_per_window
        ) * samples_per_window

        if usable_rows == 0:
            carry_over = chunk.copy()
            continue

        complete_data = chunk.iloc[:usable_rows]

        carry_over = (
            chunk
            .iloc[usable_rows:]
            .copy()
            .reset_index(drop=True)
        )

        number_of_windows = (
            usable_rows // samples_per_window
        )

        values = complete_data.to_numpy(
            dtype=np.float64,
        )

        windows = values.reshape(
            number_of_windows,
            samples_per_window,
            len(sensor_columns),
        )

        means = windows.mean(axis=1)
        standard_deviations = windows.std(
            axis=1,
            ddof=1,
        )
        minimums = windows.min(axis=1)
        maximums = windows.max(axis=1)
        rms_values = np.sqrt(
            np.mean(np.square(windows), axis=1)
        )

        feature_data = {
            "window_id": np.arange(
                next_window_id,
                next_window_id + number_of_windows,
            )
        }

        for column_index, column in enumerate(sensor_columns):
            feature_data[f"{column}_mean"] = (
                means[:, column_index]
            )
            feature_data[f"{column}_std"] = (
                standard_deviations[:, column_index]
            )
            feature_data[f"{column}_min"] = (
                minimums[:, column_index]
            )
            feature_data[f"{column}_max"] = (
                maximums[:, column_index]
            )
            feature_data[f"{column}_rms"] = (
                rms_values[:, column_index]
            )

        feature_parts.append(
            pd.DataFrame(feature_data)
        )

        next_window_id += number_of_windows

        if chunk_number == 1 or chunk_number % 10 == 0:
            print(
                f"Chunk {chunk_number}: "
                f"total windows {next_window_id}"
            )

    if not feature_parts:
        raise ValueError(
            f"No complete windows extracted from {file_path.name}"
        )

    result = pd.concat(
        feature_parts,
        ignore_index=True,
    )

    print(f"Raw rows read: {total_rows}")
    print(f"Complete windows: {len(result)}")
    print(f"Unused final samples: {len(carry_over)}")

    return result


def create_rpm_features(
    file_path,
    number_of_windows,
):
    """Interpolate RPM at each common window centre."""

    if not file_path.exists():
        raise FileNotFoundError(
            f"Input file not found: {file_path}"
        )

    rpm_data = pd.read_csv(file_path)

    required_columns = ["time", "rpm"]

    missing_columns = [
        column
        for column in required_columns
        if column not in rpm_data.columns
    ]

    if missing_columns:
        raise ValueError(
            f"{file_path.name} is missing columns: "
            f"{missing_columns}"
        )

    rpm_data[required_columns] = (
        rpm_data[required_columns]
        .apply(pd.to_numeric, errors="coerce")
    )

    invalid_count = int(
        rpm_data[required_columns]
        .isna()
        .any(axis=1)
        .sum()
    )

    if invalid_count:
        raise ValueError(
            f"{file_path.name} contains "
            f"{invalid_count} invalid rows."
        )

    rpm_data = (
        rpm_data
        .sort_values("time")
        .drop_duplicates(subset="time")
        .reset_index(drop=True)
    )

    window_ids = np.arange(number_of_windows)

    window_centres = (
        window_ids + 0.5
    ) * WINDOW_DURATION_SECONDS

    if window_centres[-1] > rpm_data["time"].max():
        raise ValueError(
            f"{file_path.name} ends before the final "
            "current/vibration window."
        )

    interpolated_rpm = np.interp(
        window_centres,
        rpm_data["time"].to_numpy(dtype=float),
        rpm_data["rpm"].to_numpy(dtype=float),
    )

    rpm_change_per_second = np.diff(
        interpolated_rpm,
        prepend=interpolated_rpm[0],
    ) / WINDOW_DURATION_SECONDS

    return pd.DataFrame(
        {
            "window_id": window_ids,
            "rpm_interpolated": interpolated_rpm,
            "rpm_change_per_second": rpm_change_per_second,
        }
    )


def build_condition_dataset(
    condition,
    label,
    current_file_name,
    vibration_file_name,
    rpm_file_name,
):
    """Create one aligned current, vibration and RPM table."""

    print(f"\n{'=' * 60}")
    print(f"Condition: {condition}")
    print(f"Assigned label: {label}")

    current_features = extract_stream_features(
        file_path=DATA_DIR / current_file_name,
        sensor_columns=CURRENT_COLUMNS,
        samples_per_window=CURRENT_SAMPLES_PER_WINDOW,
        chunk_size=CURRENT_SAMPLES_PER_WINDOW * 100,
    )

    vibration_features = extract_stream_features(
        file_path=DATA_DIR / vibration_file_name,
        sensor_columns=VIBRATION_COLUMNS,
        samples_per_window=VIBRATION_SAMPLES_PER_WINDOW,
        chunk_size=VIBRATION_SAMPLES_PER_WINDOW * 100,
    )

    common_windows = min(
        len(current_features),
        len(vibration_features),
    )

    if len(current_features) != len(vibration_features):
        print(
            "Warning: current and vibration window counts differ. "
            f"Using the first {common_windows} aligned windows."
        )

    current_features = current_features.iloc[
        :common_windows
    ].copy()

    vibration_features = vibration_features.iloc[
        :common_windows
    ].copy()

    rpm_features = create_rpm_features(
        file_path=DATA_DIR / rpm_file_name,
        number_of_windows=common_windows,
    )

    combined = current_features.merge(
        vibration_features,
        on="window_id",
        how="inner",
        validate="one_to_one",
    )

    combined = combined.merge(
        rpm_features,
        on="window_id",
        how="inner",
        validate="one_to_one",
    )

    combined["time_start_seconds"] = (
        combined["window_id"]
        * WINDOW_DURATION_SECONDS
    )

    combined["time_end_seconds"] = (
        combined["time_start_seconds"]
        + WINDOW_DURATION_SECONDS
    )

    combined["window_duration_seconds"] = (
        WINDOW_DURATION_SECONDS
    )

    combined["current_samples_per_window"] = (
        CURRENT_SAMPLES_PER_WINDOW
    )

    combined["vibration_samples_per_window"] = (
        VIBRATION_SAMPLES_PER_WINDOW
    )

    combined["label"] = label
    combined["condition"] = condition
    combined["source_current"] = current_file_name
    combined["source_vibration"] = vibration_file_name
    combined["source_rpm"] = rpm_file_name

    print(f"Aligned multi-sensor windows: {len(combined)}")

    return combined


def main():
    normal_data = build_condition_dataset(
        condition="normal",
        label=0,
        current_file_name="current_normal_0.csv",
        vibration_file_name="vibration_normal_0.csv",
        rpm_file_name="rpm_normal_0.csv",
    )

    ball_fault_data = build_condition_dataset(
        condition="ball_fault",
        label=1,
        current_file_name="current_ball_0.csv",
        vibration_file_name="vibration_ball_0.csv",
        rpm_file_name="rpm_ball_0.csv",
    )

    multisensor_data = pd.concat(
        [normal_data, ball_fault_data],
        ignore_index=True,
    )

    output_file = (
        OUTPUT_DIR
        / "multisensor_window_features.csv"
    )

    multisensor_data.to_csv(
        output_file,
        index=False,
    )

    feature_columns = [
        column
        for column in multisensor_data.columns
        if column.endswith(
            ("_mean", "_std", "_min", "_max", "_rms")
        )
        or column in {
            "rpm_interpolated",
            "rpm_change_per_second",
        }
    ]

    print("\nMulti-sensor feature extraction complete.")
    print(f"Saved to: {output_file}")
    print(f"Final shape: {multisensor_data.shape}")
    print(f"Model feature count: {len(feature_columns)}")

    print("\nClass distribution:")
    print(
        multisensor_data
        .groupby(["label", "condition"])
        .size()
    )

    print("\nModel features:")
    for column in feature_columns:
        print(column)


if __name__ == "__main__":
    main()
