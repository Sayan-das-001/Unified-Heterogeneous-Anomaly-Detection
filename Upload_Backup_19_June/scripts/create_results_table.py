import pandas as pd

results = [
    ["AutoEncoder",84.26,91.75,67.48,77.77,0.7618],
    ["LSTM",91.92,84.20,98.71,90.88,0.9799],
    ["LSTM V2",91.92,83.80,99.41,90.94,0.9808],
    ["Transformer",91.43,83.52,98.42,90.36,0.9774],
    ["Transformer V2",91.98,84.42,98.52,90.92,0.9795],
    ["Global Transformer",88.21,78.14,98.72,87.23,0.9682]
]

df = pd.DataFrame(
    results,
    columns=[
        "Model",
        "Accuracy",
        "Precision",
        "Recall",
        "F1",
        "ROC_AUC"
    ]
)

df.to_csv(
    "paper_assets/tables/model_comparison.csv",
    index=False
)

print(df)
