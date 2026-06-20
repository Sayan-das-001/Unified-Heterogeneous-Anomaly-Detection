import pandas as pd
import joblib

from sklearn.preprocessing import StandardScaler

print("Loading Dataset...")

df = pd.read_csv(
    "Phase6_final_dataset/encoded_dataset.csv",
    low_memory=False
)

# ==========================================
# KEEP LABELS SEPARATE
# ==========================================

labels = df["binary_label"]

# ==========================================
# FEATURES TO SCALE
# ==========================================

exclude_cols = [
    "binary_label"
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
    "scalers/standard_scaler.pkl"
)

df["binary_label"] = labels

print("Saving...")

df.to_csv(
    "Phase6_final_dataset/scaled_dataset.csv",
    index=False
)

print("Done")
print(df.shape)
