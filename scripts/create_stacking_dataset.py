import pandas as pd

lstm = pd.read_csv(
    "paper_assets/roc_curves/data/lstm_v2_probs.csv"
)

transformer = pd.read_csv(
    "paper_assets/roc_curves/data/transformer_v2_probs.csv"
)

xgb = pd.read_csv(
    "ensemble/predictions/xgb_probs.csv"
)

y = pd.read_csv(
    "paper_assets/roc_curves/data/y_true.csv"
)

stack = pd.DataFrame()

stack["lstm_prob"] = lstm.iloc[:,0]
stack["transformer_prob"] = transformer.iloc[:,0]
stack["xgb_prob"] = xgb.iloc[:,0]

stack["label"] = y.iloc[:,0]

stack.to_csv(
    "ensemble/stacking_dataset.csv",
    index=False
)

print(stack.head())
print()
print("Shape:", stack.shape)
