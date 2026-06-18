import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv(
    "paper_assets/tables/model_comparison.csv"
)

metrics = [
    "Accuracy",
    "F1",
    "ROC_AUC"
]

for metric in metrics:

    plt.figure(
        figsize=(10,6)
    )

    plt.bar(
        df["Model"],
        df[metric]
    )

    plt.xticks(
        rotation=30
    )

    plt.ylabel(metric)

    plt.tight_layout()

    plt.savefig(
        f"paper_assets/figures/png/{metric}.png",
        dpi=300
    )

    plt.savefig(
        f"paper_assets/figures/pdf/{metric}.pdf"
    )

    plt.close()

print("Done")
