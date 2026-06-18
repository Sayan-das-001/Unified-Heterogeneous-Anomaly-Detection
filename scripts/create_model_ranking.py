import pandas as pd

df = pd.read_csv(
    "paper_assets/tables/publication_table.csv"
)

df = df.sort_values(
    by="F1",
    ascending=False
)

df.to_csv(
    "paper_assets/tables/model_ranking.csv",
    index=False
)

print(df)
