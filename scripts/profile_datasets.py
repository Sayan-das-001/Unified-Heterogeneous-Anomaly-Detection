import pandas as pd
import os

files = {
    "UNSW": "Phase2_clean/UNSW_clean.csv",
    "BoT": "Phase2_clean/BoT_clean.csv",
    "CIC": "Phase2_clean/CIC_clean.csv",
    "5G": "Phase2_clean/5G_clean.csv"
}

os.makedirs("reports/feature_inventory", exist_ok=True)

for name, file in files.items():

    print(f"\nProcessing {name} ...")

    df = pd.read_csv(file, nrows=50000, low_memory=False)

    output_file = f"reports/feature_inventory/{name}_profile.txt"

    with open(output_file, "w") as f:

        f.write(f"DATASET: {name}\n")
        f.write("="*60 + "\n\n")

        f.write(f"ROWS SAMPLED: {len(df)}\n")
        f.write(f"COLUMNS: {len(df.columns)}\n\n")

        f.write("FEATURE NAMES\n")
        f.write("-"*60 + "\n")

        for col in df.columns:
            f.write(f"{col}\n")

        f.write("\n\nDATATYPES\n")
        f.write("-"*60 + "\n")

        f.write(df.dtypes.to_string())

        f.write("\n\n\nMISSING VALUES\n")
        f.write("-"*60 + "\n")

        missing = df.isnull().sum()

        for col, count in missing.items():
            if count > 0:
                f.write(f"{col}: {count}\n")

        f.write("\n\nCONSTANT COLUMNS\n")
        f.write("-"*60 + "\n")

        for col in df.columns:
            if df[col].nunique() == 1:
                f.write(f"{col}\n")

    print(f"Saved: {output_file}")

print("\nDone.")
