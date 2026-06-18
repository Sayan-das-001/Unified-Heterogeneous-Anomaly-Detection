import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv(
    "paper_assets/ablation/transformer_ablation_detailed.csv"
)

plt.figure(figsize=(8,5))

plt.plot(
    df["Variant"],
    df["F1"],
    marker="o"
)

plt.xticks(rotation=20)

plt.ylabel("F1 Score")

plt.tight_layout()

plt.savefig(
    "paper_assets/ablation/ablation_f1.png",
    dpi=300
)

plt.savefig(
    "paper_assets/ablation/ablation_f1.pdf"
)

print("Saved")
