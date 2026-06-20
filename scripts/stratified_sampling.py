import pandas as pd
import os

os.makedirs("Phase3_sampling", exist_ok=True)

SAMPLING_PLAN = {
    "UNSW": {
        "file": "Phase2_clean/UNSW_clean.csv",
        "normal": 285484,
        "attack": 14516
    },
    "BoT": {
        "file": "Phase2_clean/BoT_clean.csv",
        "normal": 184056,
        "attack": 15944
    },
    "CIC": {
        "file": "Phase2_clean/CIC_clean.csv",
        "normal": 4713,
        "attack": 195287
    },
    "5G": {
        "file": "Phase2_clean/5G_clean.csv",
        "normal": 117874,
        "attack": 182126
    }
}

for dataset, cfg in SAMPLING_PLAN.items():

    print(f"\nProcessing {dataset}")

    df = pd.read_csv(cfg["file"], low_memory=False)

    normal_df = df[df["binary_label"] == 0]
    attack_df = df[df["binary_label"] == 1]

    sampled_normal = normal_df.sample(
        n=cfg["normal"],
        random_state=42
    )

    sampled_attack = attack_df.sample(
        n=cfg["attack"],
        random_state=42
    )

    sampled_df = pd.concat(
        [sampled_normal, sampled_attack],
        ignore_index=True
    )

    sampled_df = sampled_df.sample(
        frac=1,
        random_state=42
    ).reset_index(drop=True)

    output_file = f"Phase3_sampling/{dataset}_sampled.csv"

    sampled_df.to_csv(
        output_file,
        index=False
    )

    print(f"Saved -> {output_file}")
    print(f"Rows -> {len(sampled_df)}")

print("\nSampling Complete")
