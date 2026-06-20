import pandas as pd

meta = pd.DataFrame()

meta["lstm_prob"] = pd.read_csv(
    "ensemble/validation_predictions/lstm_val_probs.csv"
).iloc[:,0]

meta["transformer_prob"] = pd.read_csv(
    "ensemble/validation_predictions/transformer_val_probs.csv"
).iloc[:,0]

meta["global_prob"] = pd.read_csv(
    "ensemble/validation_predictions/global_transformer_val_probs.csv"
).iloc[:,0]

meta["xgb_prob"] = pd.read_csv(
    "ensemble/validation_predictions/xgb_val_probs.csv"
).iloc[:,0]

meta["label"] = pd.read_csv(
    "Phase7_splits/y_val_lstm.csv"
).iloc[:,0]

meta.to_csv(
    "ensemble/proper_stacking/meta_train_4models.csv",
    index=False
)

print(meta.head())
print(meta.shape)
