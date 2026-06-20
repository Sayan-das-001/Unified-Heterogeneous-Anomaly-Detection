import pandas as pd

df = pd.read_csv(
    "Phase5_heterogeneous/heterogeneous_dataset_1M.csv",
    low_memory=False
)

for domain in df["domain"].unique():

    subset = df[
        df["domain"] == domain
    ]

    dup = subset.duplicated().sum()

    print("\n", domain)
    print("Rows:", len(subset))
    print("Duplicates:", dup)
