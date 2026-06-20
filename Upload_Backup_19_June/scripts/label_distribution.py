import pandas as pd

df = pd.read_csv(
    "Phase6_final_dataset/scaled_dataset_v2.csv",
    usecols=["binary_label"]
)

dist = df["binary_label"].value_counts()

dist.to_csv(
    "paper_assets/dataset_analysis/binary_distribution.csv"
)

print(dist)
