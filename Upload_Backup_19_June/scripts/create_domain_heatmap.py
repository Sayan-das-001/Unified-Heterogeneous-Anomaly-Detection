import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv(
    "paper_assets/final_paper/tables/domain_wise_results.csv"
)

heat = df.set_index(
    "Domain"
)

plt.figure(figsize=(8,4))

sns.heatmap(
    heat,
    annot=True,
    fmt=".3f"
)

plt.title(
    "Domain-wise Performance Metrics"
)

plt.tight_layout()

plt.savefig(
    "paper_assets/final_paper/plots/domain_metrics_heatmap.png",
    dpi=300
)

plt.savefig(
    "paper_assets/final_paper/plots/domain_metrics_heatmap.pdf"
)

print("Saved")
