import pandas as pd
import numpy as np
import os


# -----------------------------
# Configuration
# -----------------------------

SAMPLE_RATE = 10        # samples per second
DURATION_SECONDS = 3600 # 1 hour

TOTAL_SAMPLES = SAMPLE_RATE * DURATION_SECONDS


np.random.seed(42)


# -----------------------------
# Generate timestamps
# -----------------------------

time = np.arange(
    0,
    TOTAL_SAMPLES / SAMPLE_RATE,
    1 / SAMPLE_RATE
)


# -----------------------------
# Normal conveyor behaviour
# -----------------------------

current = np.random.normal(
    0.45,
    0.03,
    TOTAL_SAMPLES
)

voltage = np.random.normal(
    5.0,
    0.02,
    TOTAL_SAMPLES
)

rpm = np.random.normal(
    320,
    3,
    TOTAL_SAMPLES
)

vibration = np.random.normal(
    0.03,
    0.005,
    TOTAL_SAMPLES
)

piezo = np.random.normal(
    5,
    1,
    TOTAL_SAMPLES
)

ir_entry = np.zeros(TOTAL_SAMPLES)

ir_exit = np.zeros(TOTAL_SAMPLES)

label = np.zeros(TOTAL_SAMPLES)



# -----------------------------
# Inject mechanical resistance
# -----------------------------

fault_start = 12000
fault_end = 13000


current[fault_start:fault_end] += np.linspace(
    0.2,
    0.8,
    fault_end-fault_start
)


rpm[fault_start:fault_end] -= np.linspace(
    0,
    100,
    fault_end-fault_start
)


vibration[fault_start:fault_end] += np.linspace(
    0.05,
    0.25,
    fault_end-fault_start
)


label[fault_start:fault_end] = 1



# -----------------------------
# Inject motor stall
# -----------------------------

stall_start = 20000
stall_end = 20500


current[stall_start:stall_end] = 1.8

rpm[stall_start:stall_end] = 0

vibration[stall_start:stall_end] = 0.8

label[stall_start:stall_end] = 1



# -----------------------------
# Object detection events
# -----------------------------

for i in range(0, TOTAL_SAMPLES, 500):

    ir_entry[i:i+5] = 1

    if i + 30 < TOTAL_SAMPLES:
        ir_exit[i+30:i+35] = 1



# -----------------------------
# Create dataframe
# -----------------------------

data = pd.DataFrame({

    "timestamp": time,

    "current_A": current,

    "voltage_V": voltage,

    "rpm": rpm,

    "vibration_g": vibration,

    "piezo_value": piezo,

    "ir_entry": ir_entry,

    "ir_exit": ir_exit,

    "label": label

})


# -----------------------------
# Save dataset
# -----------------------------

output_folder = (
    "datasets/synthetic_conveyor"
)


os.makedirs(
    output_folder,
    exist_ok=True
)


output_file = (
    output_folder +
    "/synthetic_conveyor_data.csv"
)


data.to_csv(
    output_file,
    index=False
)


print("Synthetic dataset created")
print(data.head())

print("\nDataset shape:")
print(data.shape)

print("\nClass distribution:")
print(data["label"].value_counts())