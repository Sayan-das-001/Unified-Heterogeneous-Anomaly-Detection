import pandas as pd

table = pd.DataFrame({

    "Comparison":[
        "AutoEncoder -> LSTM",
        "LSTM -> LSTM V2",
        "Transformer -> Transformer V2"
    ],

    "F1 Gain (%)":[
        round(90.88 - 77.77,2),
        round(90.94 - 90.88,2),
        round(90.92 - 90.36,2)
    ],

    "ROC Gain":[
        round(0.9799 - 0.7618,4),
        round(0.9808 - 0.9799,4),
        round(0.9795 - 0.9774,4)
    ]
})

table.to_csv(
    "paper_assets/tables/performance_gain_table.csv",
    index=False
)

print(table)
