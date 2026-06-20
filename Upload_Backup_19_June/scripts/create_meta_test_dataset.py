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

meta_test = pd.DataFrame()

meta_test["lstm_prob"] = lstm.iloc[:,0]
meta_test["transformer_prob"] = transformer.iloc[:,0]
meta_test["xgb_prob"] = xgb.iloc[:,0]
meta_test["label"] = y.iloc[:,0]

meta_test.to_csv(
    "ensemble/proper_stacking/meta_test.csv",
    index=False
)

print(meta_test.head())
print()
print("Shape:", meta_test.shape)
