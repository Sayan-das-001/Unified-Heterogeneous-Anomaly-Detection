import pandas as pd

print("Loading Harmonized Datasets...")

unsw = pd.read_csv(
    "Phase4_harmonized/UNSW_harmonized.csv",
    low_memory=False
)

bot = pd.read_csv(
    "Phase4_harmonized/BoT_harmonized.csv",
    low_memory=False
)

cic = pd.read_csv(
    "Phase4_harmonized/CIC_harmonized.csv",
    low_memory=False
)

fiveg = pd.read_csv(
    "Phase4_harmonized/5G_harmonized.csv",
    low_memory=False
)

print("Merging...")

heterogeneous = pd.concat(
    [unsw, bot, cic, fiveg],
    ignore_index=True
)

print("Shape:", heterogeneous.shape)

heterogeneous.to_csv(
    "Phase5_heterogeneous/heterogeneous_dataset_1M.csv",
    index=False
)

print("Saved")
