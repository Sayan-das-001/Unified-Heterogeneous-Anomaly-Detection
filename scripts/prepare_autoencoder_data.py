import pandas as pd

print("Loading datasets...")

train = pd.read_csv("Phase7_splits/train.csv")
val = pd.read_csv("Phase7_splits/val.csv")
test = pd.read_csv("Phase7_splits/test.csv")

# ----------------------------------
# Train on NORMAL only
# ----------------------------------

train_normal = train[
    train["binary_label"] == 0
].copy()

# ----------------------------------
# Remove labels from input features
# ----------------------------------

drop_cols = [
    "binary_label",
    "attack_category"
]

X_train = train_normal.drop(
    columns=drop_cols
)

X_val = val.drop(
    columns=drop_cols
)

X_test = test.drop(
    columns=drop_cols
)

# Labels for evaluation

y_val = val["binary_label"]
y_test = test["binary_label"]

print("\nTrain Shape:", X_train.shape)
print("Validation Shape:", X_val.shape)
print("Test Shape:", X_test.shape)

print("\nInput Features:", X_train.shape[1])

# Save

X_train.to_csv(
    "Phase7_splits/X_train_autoencoder.csv",
    index=False
)

X_val.to_csv(
    "Phase7_splits/X_val_autoencoder.csv",
    index=False
)

X_test.to_csv(
    "Phase7_splits/X_test_autoencoder.csv",
    index=False
)

y_val.to_csv(
    "Phase7_splits/y_val.csv",
    index=False
)

y_test.to_csv(
    "Phase7_splits/y_test.csv",
    index=False
)

print("\nSaved Successfully")
