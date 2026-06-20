import pandas as pd

ablation = [
    ["Transformer",91.43,90.36,0.9774],
    ["Transformer V2",91.98,90.92,0.9795],
    ["Global Transformer",88.21,87.23,0.9682]
]

df = pd.DataFrame(
    ablation,
    columns=[
        "Variant",
        "Accuracy",
        "F1",
        "ROC_AUC"
    ]
)

df.to_csv(
    "paper_assets/ablation/transformer_ablation.csv",
    index=False
)

print(df)
