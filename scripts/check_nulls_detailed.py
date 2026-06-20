import pandas as pd

df = pd.read_csv(
    "Phase5_heterogeneous/heterogeneous_dataset_1M.csv",
    low_memory=False
)

nulls = (
    df.isnull().sum()
    .sort_values(ascending=False)
)

for col in nulls.index:

    if nulls[col] > 0:

        print(
            f"{col} : "
            f"{nulls[col]} "
            f"({round(nulls[col]/len(df)*100,2)}%)"
        )
