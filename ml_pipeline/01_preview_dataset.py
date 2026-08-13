from pathlib import Path
import pandas as pd

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "datasets" / "public_rotating_machine" / "selected_sample"

files = [
    "current_normal_0.csv",
    "rpm_normal_0.csv",
    "vibration_normal_0.csv",
    "current_ball_0.csv",
    "rpm_ball_0.csv",
    "vibration_ball_0.csv",
]

print("\n=== Dataset Preview Script ===")
print(f"Project root: {PROJECT_ROOT}")
print(f"Data folder:  {DATA_DIR}")

for file_name in files:
    file_path = DATA_DIR / file_name

    print("\n" + "=" * 80)
    print(f"File: {file_name}")
    print(f"Path: {file_path}")

    if not file_path.exists():
        print("ERROR: File not found")
        continue

    # Read only first 5 rows so we do not load the huge file into memory
    df = pd.read_csv(file_path, nrows=5)

    print("\nColumns:")
    print(list(df.columns))

    print("\nFirst 5 rows:")
    print(df)

print("\nPreview complete.")