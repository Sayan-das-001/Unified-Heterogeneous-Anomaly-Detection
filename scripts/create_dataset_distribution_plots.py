import pandas as pd
import matplotlib.pyplot as plt

domain = pd.read_csv(
    "paper_assets/dataset_analysis/domain_distribution.csv"
)

labels = pd.read_csv(
    "paper_assets/dataset_analysis/binary_distribution.csv"
)

plt.figure(figsize=(7,7))

plt.pie(
    domain["count"],
    labels=domain["domain"]
)

plt.tight_layout()

plt.savefig(
    "paper_assets/figures/png/domain_distribution.png",
    dpi=300
)

plt.savefig(
    "paper_assets/figures/pdf/domain_distribution.pdf"
)

plt.close()

plt.figure(figsize=(7,7))

plt.pie(
    labels["count"],
    labels=labels["binary_label"]
)

plt.tight_layout()

plt.savefig(
    "paper_assets/figures/png/binary_distribution.png",
    dpi=300
)

plt.savefig(
    "paper_assets/figures/pdf/binary_distribution.pdf"
)

print("Done")
