import pandas as pd
import matplotlib.pyplot as plt

print("Loading Data...")

df = pd.read_csv(
    "Phase7_splits/X_train_transformer.csv",
    nrows=50000
)

corr = df.corr()

plt.figure(figsize=(14,12))

plt.imshow(
    corr,
    aspect="auto"
)

plt.colorbar()

plt.title(
    "Feature Correlation Heatmap"
)

plt.tight_layout()

plt.savefig(
    "paper_assets/feature_analysis/feature_correlation_heatmap.png",
    dpi=300
)

plt.savefig(
    "paper_assets/feature_analysis/feature_correlation_heatmap.pdf"
)

corr.to_csv(
    "paper_assets/feature_analysis/feature_correlation_matrix.csv"
)

print("Saved")

