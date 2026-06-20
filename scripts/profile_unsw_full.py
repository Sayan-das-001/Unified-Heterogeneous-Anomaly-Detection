import pandas as pd

cols = pd.read_csv(
    "raw_data/UNSW_FULL/NUSW-NB15_features.csv",
    encoding="latin1"
)

names = cols["Name"].tolist()

files = [
    "UNSW-NB15_1.csv",
    "UNSW-NB15_2.csv",
    "UNSW-NB15_3.csv",
    "UNSW-NB15_4.csv"
]

attack_counts = {}
label_counts = {}

for file in files:

    df = pd.read_csv(
        f"raw_data/UNSW_FULL/{file}",
        header=None,
        names=names,
        low_memory=False
    )

    if "attack_cat" in df.columns:
        attack_counts[file] = (
            df["attack_cat"]
            .fillna("Unknown")
            .value_counts()
            .to_dict()
        )

    if "Label" in df.columns:
        label_counts[file] = (
            df["Label"]
            .value_counts()
            .to_dict()
        )

print("\nATTACK DISTRIBUTION\n")

for k,v in attack_counts.items():
    print(k)
    print(v)

print("\nLABEL DISTRIBUTION\n")

for k,v in label_counts.items():
    print(k)
    print(v)
