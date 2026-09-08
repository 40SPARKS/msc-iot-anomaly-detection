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


CHUNK_SIZE = 100_000
WINDOW_SIZE = 1_000

SENSOR_COLUMNS = [
    "bearingA_x",
    "bearingA_y",
    "bearingB_x",
    "bearingB_y",
]


def calculate_rms(series):
    """Calculate root mean square for one sensor window."""

    values = series.to_numpy(dtype=float)

    return float(
        np.sqrt(
            np.mean(
                np.square(values)
            )
        )
    )


def extract_features_from_file(
    file_name,
    label,
    condition,
):
    """Extract window features from one vibration CSV file."""

    file_path = DATA_DIR / file_name

    if not file_path.exists():
        raise FileNotFoundError(
            f"Input file not found: {file_path}"
        )

    print(f"\nProcessing: {file_name}")
    print(f"Condition: {condition}")
    print(f"Assigned label: {label}")

    feature_parts = []

    # Stores samples that do not complete a full window
    # at the end of a chunk.
    carry_over = pd.DataFrame(
        columns=SENSOR_COLUMNS
    )

    next_window_id = 0
    total_raw_rows = 0

    csv_reader = pd.read_csv(
        file_path,
        chunksize=CHUNK_SIZE,
    )

    for chunk_number, raw_chunk in enumerate(
        csv_reader,
        start=1,
    ):
        total_raw_rows += len(raw_chunk)

        missing_columns = [
            column
            for column in SENSOR_COLUMNS
            if column not in raw_chunk.columns
        ]

        if missing_columns:
            raise ValueError(
                f"{file_name} is missing columns: "
                f"{missing_columns}"
            )

        chunk = raw_chunk[SENSOR_COLUMNS].copy()

        chunk = chunk.apply(
            pd.to_numeric,
            errors="coerce",
        )

        invalid_rows = chunk.isna().any(axis=1)
        invalid_count = int(invalid_rows.sum())

        if invalid_count > 0:
            raise ValueError(
                f"{file_name}, chunk {chunk_number}: "
                f"found {invalid_count} rows containing "
                f"missing or non-numerical values."
            )

        chunk = chunk.reset_index(drop=True)

        # Add samples left over from the preceding chunk.
        if not carry_over.empty:
            chunk = pd.concat(
                [carry_over, chunk],
                ignore_index=True,
            )

        usable_rows = (
            len(chunk) // WINDOW_SIZE
        ) * WINDOW_SIZE

        # Not enough samples to complete one window.
        if usable_rows == 0:
            carry_over = chunk.copy()
            continue

        window_data = (
            chunk
            .iloc[:usable_rows]
            .copy()
        )

        carry_over = (
            chunk
            .iloc[usable_rows:]
            .copy()
            .reset_index(drop=True)
        )

        number_of_windows = (
            usable_rows // WINDOW_SIZE
        )

        window_data["window_id"] = np.repeat(
            np.arange(
                next_window_id,
                next_window_id + number_of_windows,
            ),
            WINDOW_SIZE,
        )

        grouped = window_data.groupby(
            "window_id",
            sort=True,
        )

        feature_df = grouped[SENSOR_COLUMNS].agg(
            ["mean", "std", "min", "max"]
        )

        feature_df.columns = [
            f"{sensor}_{statistic}"
            for sensor, statistic
            in feature_df.columns
        ]

        for sensor in SENSOR_COLUMNS:
            feature_df[f"{sensor}_rms"] = (
                grouped[sensor]
                .apply(calculate_rms)
            )

        feature_df = feature_df.reset_index()

        # These are metadata for tracing windows.
        # They must not be model features.
        feature_df["sample_start"] = (
            feature_df["window_id"] * WINDOW_SIZE
        )

        feature_df["sample_end"] = (
            feature_df["sample_start"]
            + WINDOW_SIZE
            - 1
        )

        feature_df["window_size"] = WINDOW_SIZE
        feature_df["label"] = label
        feature_df["condition"] = condition
        feature_df["source_file"] = file_name

        feature_parts.append(feature_df)

        next_window_id += number_of_windows

        print(
            f"Chunk {chunk_number}: "
            f"created {number_of_windows} windows"
        )

    if not feature_parts:
        raise ValueError(
            f"No complete windows extracted from {file_name}"
        )

    result = pd.concat(
        feature_parts,
        ignore_index=True,
    )

    print(f"Raw rows read: {total_raw_rows}")
    print(f"Complete windows: {len(result)}")
    print(
        f"Unused final samples: {len(carry_over)}"
    )

    return result


normal_features = extract_features_from_file(
    file_name="vibration_normal_0.csv",
    label=0,
    condition="normal",
)

ball_features = extract_features_from_file(
    file_name="vibration_ball_0.csv",
    label=1,
    condition="ball_fault",
)


combined = pd.concat(
    [
        normal_features,
        ball_features,
    ],
    ignore_index=True,
)


output_file = (
    OUTPUT_DIR
    / "vibration_window_features.csv"
)

combined.to_csv(
    output_file,
    index=False,
)


print("\nFeature extraction complete.")
print(f"Saved to: {output_file}")
print(f"Final shape: {combined.shape}")

print("\nClass distribution:")
print(
    combined
    .groupby(["label", "condition"])
    .size()
)

print("\nGenerated columns:")
for column in combined.columns:
    print(column)