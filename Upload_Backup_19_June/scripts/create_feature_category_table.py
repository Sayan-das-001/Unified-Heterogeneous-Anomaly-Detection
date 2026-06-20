import pandas as pd

table = pd.DataFrame({

    "Category":[
        "Traffic Features",
        "Protocol Features",
        "Flow Features",
        "Statistical Features",
        "Domain Identifier"
    ],

    "Examples":[
        "duration, rate",
        "TCP, UDP, ICMP",
        "Load, Loss",
        "mean_value, variance",
        "domain"
    ]
})

table.to_csv(
    "paper_assets/dataset_analysis/feature_categories.csv",
    index=False
)

print(table)
