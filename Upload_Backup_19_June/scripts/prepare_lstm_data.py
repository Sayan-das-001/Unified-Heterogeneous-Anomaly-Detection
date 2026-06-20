import pandas as pd

print("Loading datasets...")

train = pd.read_csv(
    "Phase7_splits/train.csv"
)

val = pd.read_csv(
    "Phase7_splits/val.csv"
)

test = pd.read_csv(
    "Phase7_splits/test.csv"
)

# -------------------------
# Features
# -------------------------

drop_cols = [
    "binary_label",
    "attack_category"
]

X_train = train.drop(
    columns=drop_cols
)

X_val = val.drop(
    columns=drop_cols
)

X_test = test.drop(
    columns=drop_cols
)

# -------------------------
# Labels
# -------------------------

y_train = train["binary_label"]

y_val = val["binary_label"]

y_test = test["binary_label"]

# -------------------------
# Save
# -------------------------

X_train.to_csv(
    "Phase7_splits/X_train_lstm.csv",
    index=False
)

X_val.to_csv(
    "Phase7_splits/X_val_lstm.csv",
    index=False
)

X_test.to_csv(
    "Phase7_splits/X_test_lstm.csv",
    index=False
)

y_train.to_csv(
    "Phase7_splits/y_train_lstm.csv",
    index=False
)

y_val.to_csv(
    "Phase7_splits/y_val_lstm.csv",
    index=False
)

y_test.to_csv(
    "Phase7_splits/y_test_lstm.csv",
    index=False
)

print("\nSaved Successfully")

print(
    "\nTrain:",
    X_train.shape
)

print(
    "Val:",
    X_val.shape
)

print(
    "Test:",
    X_test.shape
)

