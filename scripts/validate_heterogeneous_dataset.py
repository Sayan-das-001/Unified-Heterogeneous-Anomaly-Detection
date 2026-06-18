import pandas as pd

print("Loading Dataset...")

df = pd.read_csv(
    "Phase5_heterogeneous/heterogeneous_dataset_1M.csv",
    low_memory=False
)

print("\n" + "="*60)
print("DATASET OVERVIEW")
print("="*60)

print("Shape:", df.shape)

print("\n" + "="*60)
print("DUPLICATES")
print("="*60)

duplicates = df.duplicated().sum()
print("Duplicate Rows:", duplicates)

print("\n" + "="*60)
print("DATA TYPES")
print("="*60)

print(df.dtypes.value_counts())

print("\n" + "="*60)
print("DOMAIN DISTRIBUTION")
print("="*60)

print(df["domain"].value_counts())

print("\n" + "="*60)
print("BINARY LABEL DISTRIBUTION")
print("="*60)

print(df["binary_label"].value_counts())

print("\n" + "="*60)
print("TOP NULL FEATURES")
print("="*60)

null_pct = (
    df.isnull().sum() /
    len(df)
) * 100

print(
    null_pct.sort_values(
        ascending=False
    ).head(25)
)
