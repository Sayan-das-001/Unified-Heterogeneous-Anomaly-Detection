import pandas as pd
import numpy as np

import torch
import torch.nn as nn

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)

# ==================================================
# DEVICE
# ==================================================

device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)

print("Using Device:", device)

# ==================================================
# MODEL
# ==================================================

class AutoEncoder(nn.Module):

    def __init__(self,input_dim):

        super().__init__()

        self.encoder = nn.Sequential(

            nn.Linear(input_dim,64),
            nn.ReLU(),

            nn.Linear(64,32),
            nn.ReLU(),

            nn.Linear(32,16),
            nn.ReLU(),

            nn.Linear(16,8)
        )

        self.decoder = nn.Sequential(

            nn.Linear(8,16),
            nn.ReLU(),

            nn.Linear(16,32),
            nn.ReLU(),

            nn.Linear(32,64),
            nn.ReLU(),

            nn.Linear(64,input_dim)
        )

    def forward(self,x):

        z = self.encoder(x)

        return self.decoder(z)

# ==================================================
# LOAD DATA
# ==================================================

print("Loading Data...")

X_test = pd.read_csv(
    "Phase7_splits/X_test_autoencoder.csv"
)

y_test = pd.read_csv(
    "Phase7_splits/y_test.csv"
)

# ==================================================
# LOAD MODEL
# ==================================================

input_dim = X_test.shape[1]

model = AutoEncoder(
    input_dim
).to(device)

model.load_state_dict(
    torch.load(
        "models/autoencoder/best_autoencoder.pth",
        map_location=device
    )
)

model.eval()

# ==================================================
# RECONSTRUCTION ERRORS
# ==================================================

X_tensor = torch.tensor(
    X_test.values,
    dtype=torch.float32
).to(device)

with torch.no_grad():

    reconstructed = model(X_tensor)

errors = torch.mean(
    (X_tensor - reconstructed) ** 2,
    dim=1
)

errors = errors.cpu().numpy()

# ==================================================
# THRESHOLD
# ==================================================

threshold = np.percentile(
    errors,
    95
)

print("\nThreshold:", threshold)

# ==================================================
# PREDICTIONS
# ==================================================

predictions = (
    errors > threshold
).astype(int)

y_true = y_test["binary_label"].values

# ==================================================
# METRICS
# ==================================================

accuracy = accuracy_score(
    y_true,
    predictions
)

precision = precision_score(
    y_true,
    predictions
)

recall = recall_score(
    y_true,
    predictions
)

f1 = f1_score(
    y_true,
    predictions
)

roc_auc = roc_auc_score(
    y_true,
    errors
)

cm = confusion_matrix(
    y_true,
    predictions
)

# ==================================================
# RESULTS
# ==================================================

print("\n==============================")
print("AUTOENCODER RESULTS")
print("==============================")

print("Accuracy :", accuracy)
print("Precision:", precision)
print("Recall   :", recall)
print("F1 Score :", f1)
print("ROC AUC  :", roc_auc)

print("\nConfusion Matrix")
print(cm)

# ==================================================
# SAVE RESULTS
# ==================================================

with open(
    "results/autoencoder/results.txt",
    "w"
) as f:

    f.write(f"Accuracy: {accuracy}\n")
    f.write(f"Precision: {precision}\n")
    f.write(f"Recall: {recall}\n")
    f.write(f"F1: {f1}\n")
    f.write(f"ROC_AUC: {roc_auc}\n")
    f.write(f"\nConfusion Matrix\n{cm}\n")

print("\nResults Saved")
