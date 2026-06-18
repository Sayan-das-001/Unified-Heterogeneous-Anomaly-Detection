import pandas as pd

print("Loading Dataset...")

df = pd.read_csv(
    "Phase5_heterogeneous/heterogeneous_dataset_1M.csv",
    low_memory=False
)

print("Shape:", df.shape)

# =====================================
# CATEGORICAL COLUMNS
# =====================================

categorical_cols = [
    "proto",
    "state",
    "service",
    "ct_ftp_cmd",
    "domain",
    "attack_category"
]

# =====================================
# FILL CATEGORICAL NaN
# =====================================

for col in categorical_cols:

    if col in df.columns:

        df[col] = df[col].fillna(
            "UNKNOWN"
        )

# =====================================
# NUMERICAL NaN → 0
# =====================================

numeric_cols = df.columns.difference(
    categorical_cols
)

df[numeric_cols] = df[
    numeric_cols
].fillna(0)

print("Saving...")

df.to_csv(
    "Phase6_final_dataset/preprocessed_dataset.csv",
    index=False
)

print("Done")
print(df.shape)
