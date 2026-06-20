import pandas as pd
import joblib

from sklearn.preprocessing import LabelEncoder

print("Loading Dataset...")

df = pd.read_csv(
    "Phase6_final_dataset/preprocessed_dataset.csv",
    low_memory=False
)

categorical_cols = [
    "proto",
    "state",
    "service",
    "ct_ftp_cmd",
    "domain",
    "attack_category"
]

encoders = {}

for col in categorical_cols:

    print("Encoding:", col)

    le = LabelEncoder()

    df[col] = le.fit_transform(
        df[col].astype(str)
    )

    encoders[col] = le

    joblib.dump(
        le,
        f"encoders/{col}_encoder.pkl"
    )

print("Saving Encoded Dataset...")

df.to_csv(
    "Phase6_final_dataset/encoded_dataset.csv",
    index=False
)

print("Done")
print(df.shape)
