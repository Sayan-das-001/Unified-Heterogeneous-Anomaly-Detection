import pandas as pd

cols = pd.read_csv(
    "raw_data/UNSW_FULL/NUSW-NB15_features.csv",
    encoding="latin1"
)

names = cols["Name"].tolist()

files = [
    "raw_data/UNSW_FULL/UNSW-NB15_1.csv",
    "raw_data/UNSW_FULL/UNSW-NB15_2.csv",
    "raw_data/UNSW_FULL/UNSW-NB15_3.csv",
    "raw_data/UNSW_FULL/UNSW-NB15_4.csv"
]

dfs = []

for file in files:

    print("Loading",file)

    df = pd.read_csv(
        file,
        header=None,
        names=names,
        low_memory=False
    )

    dfs.append(df)

full = pd.concat(
    dfs,
    ignore_index=True
)

print("\nTotal Rows:",len(full))
print("Global Duplicates:",full.duplicated().sum())
print("Unique Rows:",len(full.drop_duplicates()))
