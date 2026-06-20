import pandas as pd

domain_df = pd.DataFrame({
    "Domain":[
        "5G",
        "BoTIoT",
        "CICIoT2023",
        "UNSW"
    ],
    "Samples":[
        300000,
        200000,
        200000,
        300000
    ]
})

domain_df.to_csv(
    "paper_assets/dataset_analysis/domain_table.csv",
    index=False
)

print(domain_df)
