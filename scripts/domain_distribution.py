import pandas as pd

df = pd.read_csv(
    "Phase6_final_dataset/scaled_dataset_v2.csv",
    usecols=["domain"]
)

dist = df["domain"].value_counts()

dist.to_csv(
    "paper_assets/dataset_analysis/domain_distribution.csv"
)

print(dist)
