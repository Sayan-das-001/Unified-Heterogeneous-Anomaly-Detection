import pandas as pd

df = pd.read_csv(
    "paper_assets/final_paper/tables/domain_wise_results.csv"
)

mapping = {
    0:"5G",
    1:"BoTIoT",
    2:"CICIoT2023",
    3:"UNSW"
}

df["Domain"] = df["Domain"].map(mapping)

df = df.sort_values(
    "F1",
    ascending=False
)

df.to_csv(
    "paper_assets/final_paper/tables/domain_results_named.csv",
    index=False
)

print(df)
