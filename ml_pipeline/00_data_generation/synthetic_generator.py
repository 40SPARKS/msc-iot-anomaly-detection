from pathlib import Path
import json

import numpy as np
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

SAMPLE_RATE = 10
DURATION_SECONDS = 300

TOTAL_SAMPLES = (
    SAMPLE_RATE
    * DURATION_SECONDS
)

RUN_IDS = [0, 1, 2]

BASE_RANDOM_SEED = 42


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[2]
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "datasets"
    / "synthetic_conveyor"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# TIME VECTOR
# ============================================================

TIME = (
    np.arange(
        TOTAL_SAMPLES
    )
    / SAMPLE_RATE
)


# ============================================================
# CREATE OBJECT-DETECTION SIGNALS
# ============================================================

def create_ir_signals(
    rpm,
    rng,
):
    """
    Create entry and exit object-detection pulses.

    Objects enter approximately every 5 seconds.

    Exit timing depends slightly on conveyor speed, meaning
    slower operation produces a longer travel time.
    """

    ir_entry = np.zeros(
        TOTAL_SAMPLES,
        dtype=np.int8,
    )

    ir_exit = np.zeros(
        TOTAL_SAMPLES,
        dtype=np.int8,
    )

    # One object approximately every 5 seconds.
    object_interval_samples = (
        5 * SAMPLE_RATE
    )

    # Sensor detection pulse lasts approximately 0.2 seconds.
    pulse_width = 2

    for entry_index in range(
        20,
        TOTAL_SAMPLES,
        object_interval_samples,
    ):

        ir_entry[
            entry_index:
            min(
                entry_index + pulse_width,
                TOTAL_SAMPLES,
            )
        ] = 1

        local_rpm = max(
            float(rpm[entry_index]),
            150.0,
        )

        # Around 3 seconds at approximately 320 RPM.
        travel_time_seconds = (
            3.0
            * (320.0 / local_rpm)
        )

        # Small realistic measurement variation.
        travel_time_seconds += (
            rng.normal(
                0.0,
                0.08,
            )
        )

        travel_time_seconds = max(
            travel_time_seconds,
            2.5,
        )

        delay_samples = int(
            round(
                travel_time_seconds
                * SAMPLE_RATE
            )
        )

        exit_index = (
            entry_index
            + delay_samples
        )

        if exit_index < TOTAL_SAMPLES:

            ir_exit[
                exit_index:
                min(
                    exit_index + pulse_width,
                    TOTAL_SAMPLES,
                )
            ] = 1

    return (
        ir_entry,
        ir_exit,
    )


# ============================================================
# ADD PIEZO EVENTS
# ============================================================

def add_piezo_events(
    piezo,
    ir_entry,
    ir_exit,
    rng,
    fault=False,
):
    """
    Add short impact responses to the piezo signal.

    Normal operation contains small object-related impacts.

    Fault operation also contains occasional stronger
    mechanical shock events.
    """

    result = piezo.copy()

    object_events = np.where(
        (ir_entry == 1)
        | (ir_exit == 1)
    )[0]

    for index in object_events:

        if rng.random() < 0.30:

            result[index] += (
                rng.uniform(
                    10,
                    35,
                )
            )

    if fault:

        # Additional intermittent mechanical impacts.
        number_of_fault_impacts = 35

        impact_indices = rng.choice(
            TOTAL_SAMPLES,
            size=number_of_fault_impacts,
            replace=False,
        )

        for index in impact_indices:

            result[index] += (
                rng.uniform(
                    25,
                    90,
                )
            )

    return result


# ============================================================
# GENERATE ONE RECORDING
# ============================================================

def generate_recording(
    run_id,
    condition,
):
    """
    Generate one independent 300-second conveyor recording.

    condition:
        "normal"
        "fault"
    """

    if condition not in {
        "normal",
        "fault",
    }:
        raise ValueError(
            "condition must be "
            "'normal' or 'fault'"
        )

    # --------------------------------------------------------
    # Independent but reproducible random generator
    # --------------------------------------------------------

    condition_offset = (
        0
        if condition == "normal"
        else 100
    )

    random_seed = (
        BASE_RANDOM_SEED
        + run_id
        + condition_offset
    )

    rng = np.random.default_rng(
        random_seed
    )

    # --------------------------------------------------------
    # Slightly different baseline for each run
    # --------------------------------------------------------

    current_baselines = [
        0.450,
        0.465,
        0.440,
    ]

    voltage_baselines = [
        5.000,
        5.005,
        4.995,
    ]

    rpm_baselines = [
        320.0,
        323.0,
        317.0,
    ]

    vibration_baselines = [
        0.030,
        0.032,
        0.029,
    ]

    current_base = (
        current_baselines[run_id]
    )

    voltage_base = (
        voltage_baselines[run_id]
    )

    rpm_base = (
        rpm_baselines[run_id]
    )

    vibration_base = (
        vibration_baselines[run_id]
    )

    phase = (
        run_id
        * 0.7
    )

    # --------------------------------------------------------
    # NORMAL MACHINE VARIATION
    # --------------------------------------------------------

    current = (
        current_base
        + 0.012
        * np.sin(
            2
            * np.pi
            * TIME
            / 45.0
            + phase
        )
        + rng.normal(
            0.0,
            0.025,
            TOTAL_SAMPLES,
        )
    )

    voltage = (
        voltage_base
        + rng.normal(
            0.0,
            0.020,
            TOTAL_SAMPLES,
        )
    )

    rpm = (
        rpm_base
        + 2.5
        * np.sin(
            2
            * np.pi
            * TIME
            / 35.0
            + phase
        )
        + rng.normal(
            0.0,
            2.5,
            TOTAL_SAMPLES,
        )
    )

    vibration = (
        vibration_base
        + 0.003
        * np.sin(
            2
            * np.pi
            * TIME
            / 18.0
            + phase
        )
        + rng.normal(
            0.0,
            0.006,
            TOTAL_SAMPLES,
        )
    )

    piezo = rng.normal(
        5.0,
        1.2,
        TOTAL_SAMPLES,
    )

    # --------------------------------------------------------
    # FAULT CONDITION
    # --------------------------------------------------------

    if condition == "fault":

        # Fault severity deliberately varies with time rather
        # than remaining at one fixed abnormal level.
        #
        # This creates developing/reducing mechanical stress
        # while the recording remains labelled as a faulty
        # machine condition.

        severity = (
            0.15
            + 0.55
            * (
                0.5
                + 0.5
                * np.sin(
                    2
                    * np.pi
                    * TIME
                    / 55.0
                    + phase
                )
            )
            + 0.15
            * (
                TIME
                / DURATION_SECONDS
            )
        )

        severity += (
            0.08
            * np.sin(
                2
                * np.pi
                * TIME
                / 8.0
            )
        )

        severity = np.clip(
            severity,
            0.10,
            0.90,
        )

        # ----------------------------------------------------
        # Electrical effort increases
        # ----------------------------------------------------

        current += (
            0.035
            + 0.20
            * severity
            + rng.normal(
                0.0,
                0.020,
                TOTAL_SAMPLES,
            )
        )

        # ----------------------------------------------------
        # Mechanical speed decreases
        # ----------------------------------------------------

        rpm -= (
            7.0
            + 45.0
            * severity
            + rng.normal(
                0.0,
                2.0,
                TOTAL_SAMPLES,
            )
        )

        # ----------------------------------------------------
        # Vibration increases
        # ----------------------------------------------------

        vibration += (
            0.010
            + 0.070
            * severity
            + rng.normal(
                0.0,
                0.006,
                TOTAL_SAMPLES,
            )
        )

        # ----------------------------------------------------
        # Small voltage loading effect
        # ----------------------------------------------------

        voltage -= (
            0.012
            * severity
        )

        label = np.ones(
            TOTAL_SAMPLES,
            dtype=np.int8,
        )

    else:

        label = np.zeros(
            TOTAL_SAMPLES,
            dtype=np.int8,
        )

    # --------------------------------------------------------
    # Prevent physically meaningless negative values
    # --------------------------------------------------------

    current = np.clip(
        current,
        0.0,
        None,
    )

    voltage = np.clip(
        voltage,
        0.0,
        None,
    )

    rpm = np.clip(
        rpm,
        0.0,
        None,
    )

    vibration = np.clip(
        vibration,
        0.0,
        None,
    )

    piezo = np.clip(
        piezo,
        0.0,
        None,
    )

    # --------------------------------------------------------
    # IR sensors
    # --------------------------------------------------------

    (
        ir_entry,
        ir_exit,
    ) = create_ir_signals(
        rpm=rpm,
        rng=rng,
    )

    # --------------------------------------------------------
    # Piezo responses
    # --------------------------------------------------------

    piezo = add_piezo_events(
        piezo=piezo,
        ir_entry=ir_entry,
        ir_exit=ir_exit,
        rng=rng,
        fault=(
            condition == "fault"
        ),
    )

    # --------------------------------------------------------
    # Final dataframe
    # --------------------------------------------------------

    data = pd.DataFrame(
        {
            "timestamp": TIME,
            "current_A": current,
            "voltage_V": voltage,
            "rpm": rpm,
            "vibration_g": vibration,
            "piezo_value": piezo,
            "ir_entry": ir_entry,
            "ir_exit": ir_exit,
            "label": label,
        }
    )

    return data


# ============================================================
# SAVE ALL SIX RECORDINGS
# ============================================================

def main():

    print("=" * 70)
    print(
        "SYNTHETIC CONVEYOR DATA GENERATION"
    )
    print("=" * 70)

    generated_files = []

    for run_id in RUN_IDS:

        for condition in [
            "normal",
            "fault",
        ]:

            print(
                f"\nGenerating "
                f"{condition} Run {run_id}..."
            )

            data = generate_recording(
                run_id=run_id,
                condition=condition,
            )

            output_file = (
                OUTPUT_DIR
                / (
                    f"synthetic_"
                    f"{condition}_"
                    f"{run_id}.csv"
                )
            )

            data.to_csv(
                output_file,
                index=False,
            )

            generated_files.append(
                {
                    "run_id": run_id,
                    "condition": condition,
                    "label": (
                        0
                        if condition == "normal"
                        else 1
                    ),
                    "file": output_file.name,
                    "rows": len(data),
                    "duration_seconds": (
                        DURATION_SECONDS
                    ),
                }
            )

            print(
                f"Saved: "
                f"{output_file}"
            )

            print(
                f"Shape: "
                f"{data.shape}"
            )

            print(
                "Current mean: "
                f"{data['current_A'].mean():.4f} A"
            )

            print(
                "RPM mean: "
                f"{data['rpm'].mean():.2f}"
            )

            print(
                "Vibration mean: "
                f"{data['vibration_g'].mean():.4f} g"
            )

    # ========================================================
    # GENERATION METADATA
    # ========================================================

    metadata = {

        "dataset": (
            "synthetic_conveyor"
        ),

        "sample_rate_hz": (
            SAMPLE_RATE
        ),

        "sample_interval_seconds": (
            1 / SAMPLE_RATE
        ),

        "recording_duration_seconds": (
            DURATION_SECONDS
        ),

        "samples_per_recording": (
            TOTAL_SAMPLES
        ),

        "number_of_runs": (
            len(RUN_IDS)
        ),

        "experimental_split": {
            "training_run": 0,
            "validation_run": 1,
            "test_run": 2,
        },

        "labels": {
            "0": "normal",
            "1": "fault",
        },

        "sensor_columns": [
            "current_A",
            "voltage_V",
            "rpm",
            "vibration_g",
            "piezo_value",
            "ir_entry",
            "ir_exit",
        ],

        "base_random_seed": (
            BASE_RANDOM_SEED
        ),

        "generation_design": (
            "Independent normal and fault recordings are "
            "generated for Runs 0, 1 and 2. Fault recordings "
            "contain varying mechanical resistance severity, "
            "represented by increased current and vibration "
            "and reduced RPM. Normal operating variation and "
            "run-to-run baseline differences are retained."
        ),

        "files": (
            generated_files
        ),
    }

    metadata_file = (
        OUTPUT_DIR
        / "synthetic_generation_metadata.json"
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
        "SYNTHETIC DATA GENERATION COMPLETE"
    )
    print("=" * 70)

    print(
        f"\nFiles generated: "
        f"{len(generated_files)}"
    )

    print(
        "\nExperimental split:"
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
        "\nEach file:"
    )

    print(
        f"{TOTAL_SAMPLES} rows"
    )

    print(
        f"{DURATION_SECONDS} seconds"
    )

    print(
        f"{SAMPLE_RATE} Hz"
    )

    print(
        "\nMetadata saved to:"
    )

    print(
        metadata_file
    )


if __name__ == "__main__":
    main()