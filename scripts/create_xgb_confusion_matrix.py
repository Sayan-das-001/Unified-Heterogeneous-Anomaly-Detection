import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

cm = pd.read_csv(
    "paper_assets/ensemble/tables/xgb_confusion_matrix.csv"
).values

plt.figure(figsize=(6,5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues"
)

plt.title(
    "XGBoost Confusion Matrix"
)

plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.tight_layout()

plt.savefig(
    "paper_assets/ensemble/confusion_matrices/XGBoost.png",
    dpi=300
)

plt.savefig(
    "paper_assets/ensemble/confusion_matrices/XGBoost.pdf"
)

print("Saved")
