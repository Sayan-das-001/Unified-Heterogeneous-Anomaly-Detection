import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

cm = pd.read_csv(
    "paper_assets/ensemble/tables/ensemble_confusion_matrix.csv"
)

plt.figure(figsize=(6,5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues"
)

plt.title(
    "Stacking Ensemble Confusion Matrix"
)

plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.tight_layout()

plt.savefig(
    "paper_assets/ensemble/confusion_matrices/ensemble_cm.png",
    dpi=300
)

plt.savefig(
    "paper_assets/ensemble/confusion_matrices/ensemble_cm.pdf"
)

print("Saved")
