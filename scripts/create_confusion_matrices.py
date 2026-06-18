import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
import numpy as np

df = pd.read_csv(
    "paper_assets/confusion_matrices/raw/confusion_matrices.csv"
)

for _,row in df.iterrows():

    matrix = np.array([
        [row["TN"],row["FP"]],
        [row["FN"],row["TP"]]
    ])

    disp = ConfusionMatrixDisplay(
        confusion_matrix=matrix
    )

    fig,ax = plt.subplots(
        figsize=(6,6)
    )

    disp.plot(
        ax=ax,
        colorbar=False
    )

    plt.title(
        row["Model"]
    )

    plt.tight_layout()

    png_file = (
        f"paper_assets/confusion_matrices/"
        f"{row['Model']}.png"
    )

    pdf_file = (
        f"paper_assets/confusion_matrices/"
        f"{row['Model']}.pdf"
    )

    plt.savefig(
        png_file,
        dpi=300
    )

    plt.savefig(
        pdf_file
    )

    plt.close()

print("Done")
