import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.metrics import (
    confusion_matrix
)

y_true = pd.read_csv(
    "ensemble/predictions/y_true.csv"
).values.flatten()

y_prob = pd.read_csv(
    "ensemble/proper_stacking/ensemble4_probs.csv"
).values.flatten()

y_pred = (
    y_prob > 0.5
).astype(int)

cm = confusion_matrix(
    y_true,
    y_pred
)

pd.DataFrame(
    cm
).to_csv(
    "paper_assets/final_paper/confusion_matrices/ensemble4_cm.csv",
    index=False
)

plt.figure(figsize=(6,5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues"
)

plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.title(
    "Ensemble-4 Confusion Matrix"
)

plt.tight_layout()

plt.savefig(
    "paper_assets/final_paper/confusion_matrices/ensemble4_cm.png",
    dpi=300
)

plt.savefig(
    "paper_assets/final_paper/confusion_matrices/ensemble4_cm.pdf"
)

print(cm)
