import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv(
    "paper_assets/final_paper/tables/final_ranking.csv"
)

plt.figure(figsize=(10,6))

plt.bar(
    df["Model"],
    df["F1"]
)

plt.xticks(
    rotation=35,
    ha="right"
)

plt.ylabel("F1 Score")

plt.title(
    "Model Comparison on Heterogeneous Dataset"
)

plt.tight_layout()

plt.savefig(
    "paper_assets/final_paper/plots/final_model_comparison.png",
    dpi=300
)

plt.savefig(
    "paper_assets/final_paper/plots/final_model_comparison.pdf"
)

print("Saved")

