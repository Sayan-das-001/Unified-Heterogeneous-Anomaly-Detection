import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv(
    "paper_assets/feature_analysis/top_10_features.csv"
)

plt.figure(figsize=(8,5))

plt.barh(
    df["Feature"],
    df["Importance"]
)

plt.tight_layout()

plt.savefig(
    "paper_assets/feature_analysis/top10_features.png",
    dpi=300
)

plt.savefig(
    "paper_assets/feature_analysis/top10_features.pdf"
)

print("Saved")
