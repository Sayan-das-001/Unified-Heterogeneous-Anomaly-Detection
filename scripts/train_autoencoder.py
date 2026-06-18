import pandas as pd
import numpy as np
import torch
import torch.nn as nn

from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset

# ==========================================
# DEVICE
# ==========================================

device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)

print("Using Device:", device)

# ==========================================
# LOAD DATA
# ==========================================

print("Loading Data...")

X_train = pd.read_csv(
    "Phase7_splits/X_train_autoencoder.csv"
)

X_val = pd.read_csv(
    "Phase7_splits/X_val_autoencoder.csv"
)

# ==========================================
# TENSORS
# ==========================================

X_train = torch.tensor(
    X_train.values,
    dtype=torch.float32
)

X_val = torch.tensor(
    X_val.values,
    dtype=torch.float32
)

train_loader = DataLoader(
    TensorDataset(X_train),
    batch_size=2048,
    shuffle=True
)

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
# INIT
# ==========================================

input_dim = X_train.shape[1]

model = AutoEncoder(
    input_dim
).to(device)

criterion = nn.MSELoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)

# ==========================================
# TRAIN
# ==========================================

best_loss = np.inf

epochs = 50

for epoch in range(epochs):

    model.train()

    total_loss = 0

    for batch in train_loader:

        x = batch[0].to(device)

        optimizer.zero_grad()

        output = model(x)

        loss = criterion(
            output,
            x
        )

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

    epoch_loss = (
        total_loss /
        len(train_loader)
    )

    print(
        f"Epoch {epoch+1}/{epochs}"
        f"  Loss={epoch_loss:.6f}"
    )

    if epoch_loss < best_loss:

        best_loss = epoch_loss

        torch.save(
            model.state_dict(),
            "models/autoencoder/best_autoencoder.pth"
        )

print("\nTraining Complete")
print("Best Loss:", best_loss)
