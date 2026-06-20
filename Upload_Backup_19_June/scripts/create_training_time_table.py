import pandas as pd

df = pd.DataFrame({

    "Model":[
        "LSTM",
        "Transformer",
        "Global Transformer",
        "XGBoost",
        "Proposed Ensemble-4"
    ],

    "Train_Time_Min":[
        0,
        0,
        0,
        0,
        0
    ]
})

df.to_csv(
    "paper_assets/final_paper/tables/training_time_comparison.csv",
    index=False
)

print(df)import pandas as pd

df = pd.DataFrame({

    "Model":[
        "LSTM",
        "Transformer",
        "Global Transformer",
        "XGBoost",
        "Proposed Ensemble-4"
    ],

    "Train_Time_Min":[
        0,
        0,
        0,
        0,
        0
    ]
})

df.to_csv(
    "paper_assets/final_paper/tables/training_time.csv",
    index=False
)

print(df)
