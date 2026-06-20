import pandas as pd
from sklearn.model_selection import train_test_split

print("Loading Dataset...")

df = pd.read_csv(
    "Phase6_final_dataset/scaled_dataset_v2.csv",
    low_memory=False
)

train_df, temp_df = train_test_split(
    df,
    test_size=0.30,
    random_state=42,
    stratify=df["binary_label"]
)

val_df, test_df = train_test_split(
    temp_df,
    test_size=0.50,
    random_state=42,
    stratify=temp_df["binary_label"]
)

test_df.to_csv(
    "ensemble/domain_test_split.csv",
    index=False
)

print(test_df["domain"].value_counts())
print("Saved")
