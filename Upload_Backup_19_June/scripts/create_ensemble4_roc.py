import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    roc_curve,
    auc
)

y_true = pd.read_csv(
    "ensemble/predictions/y_true.csv"
).values.flatten()

y_prob = pd.read_csv(
    "ensemble/proper_stacking/ensemble4_probs.csv"
).values.flatten()

fpr, tpr, _ = roc_curve(
    y_true,
    y_prob
)

roc_auc = auc(
    fpr,
    tpr
)

plt.figure(figsize=(7,6))

plt.plot(
    fpr,
    tpr,
    label=f"AUC = {roc_auc:.4f}"
)

plt.plot(
    [0,1],
    [0,1],
    linestyle="--"
)

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")

plt.title(
    "ROC Curve - Proposed Ensemble-4"
)

plt.legend()

plt.tight_layout()

plt.savefig(
    "paper_assets/final_paper/roc/ensemble4_roc.png",
    dpi=300
)

plt.savefig(
    "paper_assets/final_paper/roc/ensemble4_roc.pdf"
)

print("Saved")

