import pandas as pd

meta = pd.DataFrame()

meta["lstm_prob"] = pd.read_csv(
    "ensemble/predictions/lstm_probs.csv"
).iloc[:,0]

meta["transformer_prob"] = pd.read_csv(
    "ensemble/predictions/transformer_probs.csv"
).iloc[:,0]

meta["global_prob"] = pd.read_csv(
    "ensemble/predictions/global_transformer_probs.csv"
).iloc[:,0]

meta["xgb_prob"] = pd.read_csv(
    "ensemble/predictions/xgb_probs.csv"
).iloc[:,0]

meta["label"] = pd.read_csv(
    "ensemble/predictions/y_true.csv"
).iloc[:,0]

meta.to_csv(
    "ensemble/proper_stacking/meta_test_4models.csv",
    index=False
)

print(meta.head())
print(meta.shape)
