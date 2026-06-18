import pandas as pd
import numpy as np
import torch
import torch.nn as nn

device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)

print("Using Device:", device)

# ==========================================
# LOAD DATA
# ==========================================

X_test = pd.read_csv(
    "Phase7_splits/X_test_autoencoder.csv"
)

X_test_tensor = torch.tensor(
    X_test.values,
    dtype=torch.float32
)

print("Test Shape:", X_test_tensor.shape)

# ==========================================
# MODEL
# ==========================================

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

# ==========================================
# LOAD MODEL
# ==========================================

model = AutoEncoder(
    X_test_tensor.shape[1]
).to(device)

model.load_state_dict(
    torch.load(
        "models/autoencoder/best_autoencoder.pth",
        map_location=device
    )
)

model.eval()

print("Model Loaded")

# ==========================================
# RECONSTRUCTION ERRORS
# ==========================================

errors = []

with torch.no_grad():

    for start in range(
        0,
        len(X_test_tensor),
        2048
    ):

        batch = X_test_tensor[
            start:start+2048
        ].to(device)

        recon = model(batch)

        mse = torch.mean(
            (batch - recon) ** 2,
            dim=1
        )

        errors.extend(
            mse.cpu().numpy()
        )

pd.DataFrame(
    errors,
    columns=["probability"]
).to_csv(
    "paper_assets/roc_curves/data/autoencoder_probs.csv",
    index=False
)

print("\nSaved Successfully")
print("Errors:", len(errors))
