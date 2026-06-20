import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv(
    "paper_assets/proper_ensemble/tables/final_model_ranking.csv"
)

plt.figure(figsize=(10,6))

plt.bar(
    df["Model"],
    df["F1"]
)

plt.xticks(
    rotation=30,
    ha="right"
)

plt.ylabel("F1 Score (%)")
plt.title("Model Comparison on Heterogeneous Dataset")

plt.tight_layout()

plt.savefig(
    "paper_assets/proper_ensemble/tables/model_comparison_f1.png",
    dpi=300
)

plt.savefig(
    "paper_assets/proper_ensemble/tables/model_comparison_f1.pdf"
)

print("Saved")
