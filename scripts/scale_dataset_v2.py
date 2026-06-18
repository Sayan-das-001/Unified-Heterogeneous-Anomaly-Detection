import pandas as pd
import joblib

from sklearn.preprocessing import StandardScaler

print("Loading Dataset...")

df = pd.read_csv(
    "Phase6_final_dataset/encoded_dataset.csv",
    low_memory=False
)

exclude_cols = [
    "binary_label",
    "domain",
    "attack_category"
]

feature_cols = [
    col
    for col in df.columns
    if col not in exclude_cols
]

scaler = StandardScaler()

df[feature_cols] = scaler.fit_transform(
    df[feature_cols]
)

joblib.dump(
    scaler,
    "scalers/standard_scaler_v2.pkl"
)

df.to_csv(
    "Phase6_final_dataset/scaled_dataset_v2.csv",
    index=False
)

print("Done")
print(df.shape)

