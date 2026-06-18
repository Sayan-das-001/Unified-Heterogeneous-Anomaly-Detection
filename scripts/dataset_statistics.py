import pandas as pd

df = pd.read_csv(
    "Phase6_final_dataset/scaled_dataset_v2.csv"
)

summary = pd.DataFrame({
    "Mean":df.mean(),
    "Std":df.std(),
    "Min":df.min(),
    "Max":df.max()
})

summary.to_csv(
    "paper_assets/dataset_analysis/feature_statistics.csv"
)

print(summary.head())

