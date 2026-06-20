import pandas as pd

table = pd.DataFrame({

    "Metric":[
        "Total Samples",
        "Total Features",
        "Domains",
        "Normal Samples",
        "Attack Samples"
    ],

    "Value":[
        "1,000,000",
        "39",
        "4",
        "592,127",
        "407,873"
    ]
})

table.to_csv(
    "paper_assets/tables/dataset_summary_table.csv",
    index=False
)

print(table)
