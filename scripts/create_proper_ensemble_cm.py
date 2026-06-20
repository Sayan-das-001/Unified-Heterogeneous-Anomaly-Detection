import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

cm = pd.read_csv(
    "paper_assets/proper_ensemble/tables/proper_cm.csv",
    header=None
).values

plt.figure(figsize=(6,5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues"
)

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Proper Stacking Ensemble Confusion Matrix")

plt.tight_layout()

plt.savefig(
    "paper_assets/proper_ensemble/confusion_matrices/proper_cm.png",
    dpi=300
)

plt.savefig(
    "paper_assets/proper_ensemble/confusion_matrices/proper_cm.pdf"
)

print("Saved")
