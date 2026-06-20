import pandas as pd

lstm = pd.read_csv(
    "ensemble/validation_predictions/lstm_val_probs.csv"
)

transformer = pd.read_csv(
    "ensemble/validation_predictions/transformer_val_probs.csv"
)

xgb = pd.read_csv(
    "ensemble/validation_predictions/xgb_val_probs.csv"
)

y = pd.read_csv(
    "Phase7_splits/y_val_lstm.csv"
)

stack_df = pd.DataFrame()

stack_df["lstm_prob"] = lstm.iloc[:,0]
stack_df["transformer_prob"] = transformer.iloc[:,0]
stack_df["xgb_prob"] = xgb.iloc[:,0]
stack_df["label"] = y.iloc[:,0]

stack_df.to_csv(
    "ensemble/proper_stacking/meta_train.csv",
    index=False
)

print(stack_df.head())

print()
print("Shape:", stack_df.shape)
