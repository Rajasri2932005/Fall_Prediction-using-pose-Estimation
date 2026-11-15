import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "Output"

csv_files = [
    "FALL_01_pose_labeled.csv",
    "FALL_05_pose_labeled.csv",
    "FALL_08_pose_labeled.csv",
    "ADL_05_pose_labeled.csv",
    "ADL_08_pose_labeled.csv",
    "ADL_16_pose_labeled.csv"
]

all_data = []

for filename in csv_files:

    file_path = OUTPUT_DIR / filename

    print("Reading:", filename)

    df = pd.read_csv(file_path)

    # Add video identifier
    df["video"] = filename.replace("_pose_labeled.csv", "")

    # Add category
    if filename.startswith("FALL"):
        df["category"] = "FALL"
    else:
        df["category"] = "ADL"

    all_data.append(df)

# Combine all files
combined_df = pd.concat(
    all_data,
    ignore_index=True
)

# Save combined dataset
output_path = OUTPUT_DIR / "combined_pose_dataset.csv"
combined_df.to_csv(
    output_path,
    index=False
)

print()
print("=" * 70)
print("COMBINED DATASET CREATED")
print("=" * 70)

print("Total frames :", len(combined_df))
print("Total columns:", len(combined_df.columns))

print()
print("Label distribution:")
print(combined_df["label"].value_counts().sort_index())

print()
print("Video distribution:")
print(combined_df["video"].value_counts())

print()
print("Saved at:")
print(output_path)
