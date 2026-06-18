import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv(
    "paper_assets/tables/model_comparison.csv"
)

plt.figure(figsize=(10,6))

plt.bar(
    df["Model"],
    df["F1"]
)

plt.xticks(rotation=30)

plt.ylabel("F1 Score")

plt.tight_layout()

plt.savefig(
    "paper_assets/figures/png/f1_comparison.png",
    dpi=300
)

plt.savefig(
    "paper_assets/figures/pdf/f1_comparison.pdf"
)

print("Saved")
