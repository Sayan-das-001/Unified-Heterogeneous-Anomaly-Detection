import pandas as pd
import torch
import torch.nn as nn

from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader

device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)

print("Using Device:", device)

print("Loading Data...")

# =========================
# LOAD DATA
# =========================

X_train = pd.read_csv(
    "Phase7_splits/X_train_transformer.csv"
).values

y_train = pd.read_csv(
    "Phase7_splits/y_train_transformer.csv"
).values.flatten()

X_val = pd.read_csv(
    "Phase7_splits/X_val_transformer.csv"
).values

y_val = pd.read_csv(
    "Phase7_splits/y_val_transformer.csv"
).values.flatten()

# =========================
# TO TENSOR
# =========================

X_train = torch.tensor(
    X_train,
    dtype=torch.float32
)

X_val = torch.tensor(
    X_val,
    dtype=torch.float32
)

y_train = torch.tensor(
    y_train,
    dtype=torch.float32
)

y_val = torch.tensor(
    y_val,
    dtype=torch.float32
)

# Transformer expects:
# (batch, sequence_length, features)

X_train = X_train.unsqueeze(2)
X_val = X_val.unsqueeze(2)

print("Train Shape:", X_train.shape)
print("Val Shape:", X_val.shape)

# =========================
# DATALOADER
# =========================

train_loader = DataLoader(
    TensorDataset(X_train, y_train),
    batch_size=2048,
    shuffle=True
)

val_loader = DataLoader(
    TensorDataset(X_val, y_val),
    batch_size=2048,
    shuffle=False
)

# =========================
# MODEL
# =========================

class TransformerClassifier(nn.Module):

    def __init__(self):

        super().__init__()

        self.embedding = nn.Linear(
            1,
            64
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=64,
            nhead=8,
            batch_first=True
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=2
        )

        self.fc1 = nn.Linear(
            64,
            32
        )

        self.relu = nn.ReLU()

        self.dropout = nn.Dropout(
            0.3
        )

        self.fc2 = nn.Linear(
            32,
            1
        )

    def forward(self, x):

        x = self.embedding(x)

        x = self.transformer(x)

        x = x.mean(dim=1)

        x = self.fc1(x)

        x = self.relu(x)

        x = self.dropout(x)

        x = self.fc2(x)

        return x


model = TransformerClassifier().to(device)

criterion = nn.BCEWithLogitsLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)

best_loss = 999999

print("Model Loaded")

# =========================
# TRAINING
# =========================

for epoch in range(15):

    model.train()

    train_loss = 0

    for X_batch, y_batch in train_loader:

        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        optimizer.zero_grad()

        outputs = model(X_batch)

        loss = criterion(
            outputs.squeeze(),
            y_batch
        )

        loss.backward()

        optimizer.step()

        train_loss += loss.item()

    train_loss /= len(train_loader)

    model.eval()

    val_loss = 0

    with torch.no_grad():

        for X_batch, y_batch in val_loader:

            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            outputs = model(X_batch)

            loss = criterion(
                outputs.squeeze(),
                y_batch
            )

            val_loss += loss.item()

    val_loss /= len(val_loader)

    print(
        f"Epoch {epoch+1}/15 | "
        f"Train Loss={train_loss:.5f} | "
        f"Val Loss={val_loss:.5f}"
    )

    if val_loss < best_loss:

        best_loss = val_loss

        torch.save(
            model.state_dict(),
            "models/transformer/best_transformer.pth"
        )

print("\nTraining Complete")
print("Best Validation Loss:", best_loss)
