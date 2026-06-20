import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv(
    "Phase5_heterogeneous/heterogeneous_dataset_1M.csv",
    usecols=["domain"]
)

counts = df["domain"].value_counts()

plt.figure(figsize=(8,5))

counts.plot(
    kind="bar"
)

plt.ylabel("Samples")
plt.title(
    "Dataset Domain Distribution"
)

plt.tight_layout()

plt.savefig(
    "paper_assets/final_paper/plots/domain_distribution.png",
    dpi=300
)

plt.savefig(
    "paper_assets/final_paper/plots/domain_distribution.pdf"
)

print("Saved")
