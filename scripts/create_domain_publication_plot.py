import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv(
    "paper_assets/final_paper/tables/domain_results_named.csv"
)

plt.figure(figsize=(8,5))

plt.bar(
    df["Domain"],
    df["F1"]
)

plt.ylabel("F1 Score")
plt.xlabel("Dataset")
plt.title("Domain-wise Evaluation of Proposed Ensemble-4")

plt.tight_layout()

plt.savefig(
    "paper_assets/final_paper/plots/domain_performance.png",
    dpi=300
)

plt.savefig(
    "paper_assets/final_paper/plots/domain_performance.pdf"
)

print("Saved")
