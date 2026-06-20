import pandas as pd

dataset_info = pd.DataFrame({

    "Attribute":[
        "Total Samples",
        "Total Features",
        "Domains",
        "Normal Samples",
        "Attack Samples",
        "Normal Ratio (%)",
        "Attack Ratio (%)"
    ],

    "Value":[
        1000000,
        39,
        4,
        592127,
        407873,
        59.21,
        40.79
    ]
})

dataset_info.to_csv(
    "paper_assets/dataset_analysis/dataset_characteristics.csv",
    index=False
)

print(dataset_info)
