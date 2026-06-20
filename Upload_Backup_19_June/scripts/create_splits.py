import pandas as pd

from sklearn.model_selection import train_test_split

print("Loading Dataset...")

df = pd.read_csv(
    "Phase6_final_dataset/scaled_dataset_v2.csv",
    low_memory=False
)

# =====================================
# TRAIN (70%)
# TEST+VAL (30%)
# =====================================

train_df, temp_df = train_test_split(
    df,
    test_size=0.30,
    random_state=42,
    stratify=df["binary_label"]
)

# =====================================
# VALIDATION (15%)
# TEST (15%)
# =====================================

val_df, test_df = train_test_split(
    temp_df,
    test_size=0.50,
    random_state=42,
    stratify=temp_df["binary_label"]
)

print("Train:", train_df.shape)
print("Val  :", val_df.shape)
print("Test :", test_df.shape)

train_df.to_csv(
    "Phase7_splits/train.csv",
    index=False
)

val_df.to_csv(
    "Phase7_splits/val.csv",
    index=False
)

test_df.to_csv(
    "Phase7_splits/test.csv",
    index=False
)

print("Saved Successfully")
