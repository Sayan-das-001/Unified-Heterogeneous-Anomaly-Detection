import pandas as pd

files = {
    "UNSW":"Phase2_clean/UNSW_clean.csv",
    "BoT":"Phase2_clean/BoT_clean.csv",
    "CIC":"Phase2_clean/CIC_clean.csv",
    "5G":"Phase2_clean/5G_clean.csv"
}

with open("reports/all_columns.txt","w") as f:

    for name,file in files.items():

        df = pd.read_csv(file,nrows=1)

        f.write("\n")
        f.write("="*60 + "\n")
        f.write(name + "\n")
        f.write("="*60 + "\n")

        for col in df.columns:
            f.write(col + "\n")
