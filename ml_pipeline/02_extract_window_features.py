from pathlib import Path
import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "datasets" / "public_rotating_machine" / "selected_sample"
OUTPUT_DIR = PROJECT_ROOT / "datasets" / "public_rotating_machine" / "processed"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CHUNK_SIZE = 100_000
WINDOW_SIZE = 1000


def rms(series):
    return np.sqrt(np.mean(np.square(series)))


def extract_features_from_file(file_name, label, condition):
    file_path = DATA_DIR / file_name

    print(f"\nProcessing: {file_name}")

    all_features = []
    global_row_start = 0

    for chunk in pd.read_csv(file_path, chunksize=CHUNK_SIZE):
        chunk = chunk.apply(pd.to_numeric, errors="coerce")
        chunk = chunk.dropna()

        if chunk.empty:
            continue

        chunk["sample_index"] = np.arange(global_row_start, global_row_start + len(chunk))
        chunk["window_id"] = chunk["sample_index"] // WINDOW_SIZE

        feature_df = chunk.groupby("window_id").agg(["mean", "std", "min", "max"])

        feature_df.columns = [
            f"{col[0]}_{col[1]}" for col in feature_df.columns
        ]

        for col in chunk.columns:
            if col not in ["sample_index", "window_id"]:
                feature_df[f"{col}_rms"] = chunk.groupby("window_id")[col].apply(rms)

        feature_df = feature_df.reset_index()
        feature_df["label"] = label
        feature_df["condition"] = condition
        feature_df["source_file"] = file_name

        all_features.append(feature_df)

        global_row_start += len(chunk)

    if not all_features:
        print(f"No features extracted from {file_name}")
        return None

    result = pd.concat(all_features, ignore_index=True)
    print(f"Extracted windows: {len(result)}")

    return result


normal_features = extract_features_from_file(
    file_name="vibration_normal_0.csv",
    label=0,
    condition="normal"
)

ball_features = extract_features_from_file(
    file_name="vibration_ball_0.csv",
    label=1,
    condition="ball_fault"
)

combined = pd.concat([normal_features, ball_features], ignore_index=True)

output_file = OUTPUT_DIR / "vibration_window_features.csv"
combined.to_csv(output_file, index=False)

print("\nFeature extraction complete.")
print(f"Saved to: {output_file}")
print(f"Final shape: {combined.shape}")
print("\nColumns:")
print(combined.columns.tolist())