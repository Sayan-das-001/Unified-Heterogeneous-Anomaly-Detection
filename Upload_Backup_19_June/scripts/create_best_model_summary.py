import pandas as pd

table = pd.DataFrame({

    "Metric":[
        "Best Accuracy",
        "Best F1",
        "Best ROC-AUC",
        "Best Recall"
    ],

    "Model":[
        "Transformer V2",
        "LSTM V2",
        "LSTM V2",
        "LSTM V2"
    ],

    "Value":[
        91.98,
        90.94,
        0.9808,
        99.41
    ]
})

table.to_csv(
    "paper_assets/tables/best_model_summary.csv",
    index=False
)

print(table)
