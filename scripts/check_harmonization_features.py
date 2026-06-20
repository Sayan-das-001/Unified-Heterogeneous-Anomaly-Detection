import pandas as pd

files = {
    "UNSW":"Phase3_sampling/UNSW_sampled.csv",
    "BoT":"Phase3_sampling/BoT_sampled.csv",
    "CIC":"Phase3_sampling/CIC_sampled.csv",
    "5G":"Phase3_sampling/5G_sampled.csv"
}

for name,file in files.items():

    df = pd.read_csv(file,nrows=1)

    print("\n")
    print("="*60)
    print(name)
    print("="*60)

    for col in df.columns:
        print(col)
