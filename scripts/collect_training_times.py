import pandas as pd

times = pd.DataFrame({

    "Model":[
        "LSTM",
        "Transformer",
        "Global Transformer",
        "XGBoost",
        "Proposed Ensemble-4"
    ],

    "Training_Time_Minutes":[
        0,
        0,
        0,
        0,
        0
    ]
})

times.to_csv(
    "paper_assets/final_paper/training_time/training_time_table.csv",
    index=False
)

print(times)
