import pandas as pd

df = pd.DataFrame({

    "Domain":[
        "5G",
        "UNSW",
        "CICIoT2023",
        "BoTIoT"
    ],

    "Samples":[
        300000,
        300000,
        200000,
        200000
    ],

    "Percentage":[
        30,
        30,
        20,
        20
    ]
})

df.to_csv(
    "paper_assets/feature_analysis/domain_contribution_table.csv",
    index=False
)

print(df)
