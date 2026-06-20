import pandas as pd
import json
import os

feature_file = "raw_data/UNSW_FULL/NUSW-NB15_features.csv"

cols = pd.read_csv(
    feature_file,
    encoding="latin1"
)

column_names = cols["Name"].tolist()

files = [
    "UNSW-NB15_1.csv",
    "UNSW-NB15_2.csv",
    "UNSW-NB15_3.csv",
    "UNSW-NB15_4.csv"
]

report = {}

for file in files:

    path = f"raw_data/UNSW_FULL/{file}"

    df = pd.read_csv(
        path,
        header=None,
        names=column_names,
        low_memory=False
    )

    report[file] = {
        "rows": len(df),
        "columns": len(df.columns),
        "duplicates": int(df.duplicated().sum())
    }

with open(
    "Phase1_audit/UNSW_FULL_audit.json",
    "w"
) as f:
    json.dump(
        report,
        f,
        indent=4
    )

print("Audit Saved")
