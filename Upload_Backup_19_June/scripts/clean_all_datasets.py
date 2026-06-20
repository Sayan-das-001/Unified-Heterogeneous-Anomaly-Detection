import pandas as pd
import os

os.makedirs("Phase2_clean", exist_ok=True)

# =====================================================
# UNSW FULL
# =====================================================

print("Loading UNSW FULL...")

cols = pd.read_csv(
    "raw_data/UNSW_FULL/NUSW-NB15_features.csv",
    encoding="latin1"
)

column_names = cols["Name"].tolist()

files = [
    "UNSW-NB15_1.csv",
    "UNSW-NB15_2.csv",
    "UNSW-NB15_3.csv",
    "UNSW-NB15_4.csv"
]

dfs = []

for f in files:

    df = pd.read_csv(
        f"raw_data/UNSW_FULL/{f}",
        header=None,
        names=column_names,
        low_memory=False
    )

    dfs.append(df)

unsw = pd.concat(
    dfs,
    ignore_index=True
)

print("UNSW Before:", len(unsw))

unsw = unsw.drop_duplicates()

print("UNSW After:", len(unsw))

unsw["domain"] = "UNSW"

unsw["binary_label"] = unsw["Label"]

unsw["attack_category"] = (
    unsw["attack_cat"]
    .fillna("Benign")
    .astype(str)
    .str.strip()
)

unsw.to_csv(
    "Phase2_clean/UNSW_clean.csv",
    index=False
)

print("UNSW Saved")

# =====================================================
# BOT
# =====================================================

print("Loading BoT...")

bot = pd.read_csv(
    "raw_data/BoT_IoT/BoTNeTIoT-L01-v2.csv",
    low_memory=False
)

bot = bot.drop_duplicates()

bot["domain"] = "BoTIoT"

bot["binary_label"] = bot["label"]

bot["attack_category"] = bot["Attack_subType"]

bot.to_csv(
    "Phase2_clean/BoT_clean.csv",
    index=False
)

print("BoT Saved")

# =====================================================
# CIC
# =====================================================

print("Loading CIC...")

train = pd.read_csv(
    "raw_data/CICIoT2023/train.csv",
    low_memory=False
)

test = pd.read_csv(
    "raw_data/CICIoT2023/test.csv",
    low_memory=False
)

val = pd.read_csv(
    "raw_data/CICIoT2023/validation.csv",
    low_memory=False
)

cic = pd.concat(
    [train,test,val],
    ignore_index=True
)

cic = cic.drop_duplicates()

cic["domain"] = "CICIoT2023"

cic["binary_label"] = (
    cic["label"] != "BenignTraffic"
).astype(int)

cic["attack_category"] = cic["label"]

cic.to_csv(
    "Phase2_clean/CIC_clean.csv",
    index=False
)

print("CIC Saved")

# =====================================================
# 5G
# =====================================================

print("Loading 5G...")

g5 = pd.read_csv(
    "raw_data/5G_NIDD/Combined.csv",
    low_memory=False
)

g5 = g5.drop_duplicates()

g5["domain"] = "5G"

g5["binary_label"] = (
    g5["Label"] == "Malicious"
).astype(int)

g5["attack_category"] = g5["Attack Type"]

g5.to_csv(
    "Phase2_clean/5G_clean.csv",
    index=False
)

print("5G Saved")
