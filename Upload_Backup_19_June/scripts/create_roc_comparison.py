import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    roc_curve,
    auc
)

y_true = pd.read_csv(
    "paper_assets/roc_curves/data/y_true.csv"
)["binary_label"]

models = {
    "AutoEncoder":
    "paper_assets/roc_curves/data/autoencoder_probs.csv",

    "LSTM V2":
    "paper_assets/roc_curves/data/lstm_v2_probs.csv",

    "Transformer V2":
    "paper_assets/roc_curves/data/transformer_v2_probs.csv"
}

plt.figure(figsize=(8,6))

roc_table = []

for name,file in models.items():

    probs = pd.read_csv(
        file
    )["probability"]

    fpr,tpr,_ = roc_curve(
        y_true,
        probs
    )

    roc_auc = auc(
        fpr,
        tpr
    )

    roc_table.append(
        [name,roc_auc]
    )

    plt.plot(
        fpr,
        tpr,
        label=f"{name} (AUC={roc_auc:.4f})"
    )

plt.plot(
    [0,1],
    [0,1],
    linestyle="--"
)

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")

plt.legend()

plt.tight_layout()

plt.savefig(
    "paper_assets/roc_curves/roc_comparison.png",
    dpi=300
)

plt.savefig(
    "paper_assets/roc_curves/roc_comparison.pdf"
)

pd.DataFrame(
    roc_table,
    columns=[
        "Model",
        "ROC_AUC"
    ]
).to_csv(
    "paper_assets/roc_curves/roc_auc_table.csv",
    index=False
)

print("ROC Figure Saved")
