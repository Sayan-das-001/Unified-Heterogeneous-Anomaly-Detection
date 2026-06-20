import pandas as pd

df = pd.read_csv(
    "paper_assets/tables/publication_table.csv"
)

df.to_csv(
    "paper_assets/results_discussion/experimental_results.csv",
    index=False
)

print(df)
