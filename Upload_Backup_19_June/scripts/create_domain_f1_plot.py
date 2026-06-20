import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv(
    "paper_assets/final_paper/tables/domain_wise_results.csv"
)

plt.figure(figsize=(8,5))

plt.bar(
    df["Domain"],
    df["F1"]
)

plt.ylabel("F1 Score")
plt.title("Ensemble-4 Domain-wise Performance")

plt.tight_layout()

plt.savefig(
    "paper_assets/final_paper/plots/domain_f1_comparison.png",
    dpi=300
)

plt.savefig(
    "paper_assets/final_paper/plots/domain_f1_comparison.pdf"
)

print("Saved")
