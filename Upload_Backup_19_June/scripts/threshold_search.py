import pandas as pd
import numpy as np

import torch
import torch.nn as nn

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ======================
# MODEL
# ======================

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
        return self.decoder(
            self.encoder(x)
        )

# ======================
# LOAD DATA
# ======================

X = pd.read_csv(
    "Phase7_splits/X_test_autoencoder.csv"
)

y = pd.read_csv(
    "Phase7_splits/y_test.csv"
)

# ======================
# LOAD MODEL
# ======================

model = AutoEncoder(
    X.shape[1]
).to(device)

model.load_state_dict(
    torch.load(
        "models/autoencoder/best_autoencoder.pth",
        map_location=device
    )
)

model.eval()

# ======================
# ERRORS
# ======================

X_tensor = torch.tensor(
    X.values,
    dtype=torch.float32
).to(device)

with torch.no_grad():

    recon = model(X_tensor)

errors = torch.mean(
    (X_tensor - recon)**2,
    dim=1
).cpu().numpy()

y_true = y["binary_label"].values

# ======================
# THRESHOLD SEARCH
# ======================

for pct in [70,75,80,85,90,95,97,99]:

    threshold = np.percentile(
        errors,
        pct
    )

    pred = (
        errors > threshold
    ).astype(int)

    acc = accuracy_score(
        y_true,
        pred
    )

    prec = precision_score(
        y_true,
        pred
    )

    rec = recall_score(
        y_true,
        pred
    )

    f1 = f1_score(
        y_true,
        pred
    )

    print(
        f"\nPercentile={pct}"
    )

    print(
        f"ACC={acc:.4f}"
    )

    print(
        f"PREC={prec:.4f}"
    )

    print(
        f"REC={rec:.4f}"
    )

    print(
        f"F1={f1:.4f}"
    )
