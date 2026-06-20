import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    roc_curve,
    roc_auc_score
)

y = pd.read_csv(
    "paper_assets/roc_curves/data/y_true.csv"
).iloc[:,0]

probs = pd.read_csv(
    "ensemble/proper_stacking/proper_probs.csv"
).iloc[:,0]

fpr,tpr,_ = roc_curve(
    y,
    probs
)

auc = roc_auc_score(
    y,
    probs
)

plt.figure(figsize=(7,6))

plt.plot(
    fpr,
    tpr,
    label=f"AUC={auc:.4f}"
)

plt.plot(
    [0,1],
    [0,1],
    linestyle="--"
)

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Proper Stacking Ensemble ROC")
plt.legend()

plt.tight_layout()

plt.savefig(
    "paper_assets/proper_ensemble/roc/proper_ensemble_roc.png",
    dpi=300
)

plt.savefig(
    "paper_assets/proper_ensemble/roc/proper_ensemble_roc.pdf"
)

print("Saved")
