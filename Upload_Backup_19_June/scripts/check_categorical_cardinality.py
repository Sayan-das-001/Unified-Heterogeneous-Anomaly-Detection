import pandas as pd

df = pd.read_csv(
    "Phase6_final_dataset/preprocessed_dataset.csv",
    low_memory=False
)

categorical_cols = [
    "proto",
    "state",
    "service",
    "ct_ftp_cmd",
    "domain",
    "attack_category"
]

for col in categorical_cols:

    print("\n" + "="*50)
    print(col)
    print("="*50)

    print("Unique Values:",
          df[col].nunique())

    print("\nTop 20 Values:")

    print(
        df[col]
        .value_counts()
        .head(20)
    )
