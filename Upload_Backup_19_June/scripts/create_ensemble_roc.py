import pandas as pd

from sklearn.metrics import (
    roc_curve,
    auc
)

import matplotlib.pyplot as plt

y_true = pd.read_csv(
    "paper_assets/roc_curves/data/y_true.csv"
).iloc[:,0]

probs = pd.read_csv(
    "ensemble/predictions/ensemble_probs.csv"
).iloc[:,0]

fpr,tpr,_ = roc_curve(
    y_true,
    probs
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
    "--"
)

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")

plt.title(
    "Stacking Ensemble ROC Curve"
)

plt.legend()

plt.tight_layout()

plt.savefig(
    "paper_assets/ensemble/roc/ensemble_roc.png",
    dpi=300
)

plt.savefig(
    "paper_assets/ensemble/roc/ensemble_roc.pdf"
)

print("Saved")
