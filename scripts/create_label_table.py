import pandas as pd

df = pd.DataFrame({

    "Class":[
        "Normal",
        "Attack"
    ],

    "Samples":[
        592127,
        407873
    ],

    "Percentage":[
        59.21,
        40.79
    ]
})

df.to_csv(
    "paper_assets/dataset_analysis/label_table.csv",
    index=False
)

print(df)
