import pandas as pd
import torch
import torch.nn as nn

from torch.utils.data import (
    TensorDataset,
    DataLoader
)

device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)

print("Using Device:", device)

print("Loading Data...")

# ====================================
# LOAD DATA
# ====================================

X_train = pd.read_csv(
    "Phase7_splits/X_train_lstm.csv"
).values

y_train = pd.read_csv(
    "Phase7_splits/y_train_lstm.csv"
).values.flatten()

X_val = pd.read_csv(
    "Phase7_splits/X_val_lstm.csv"
).values

y_val = pd.read_csv(
    "Phase7_splits/y_val_lstm.csv"
).values.flatten()

# ====================================
# TO TENSORS
# ====================================

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

X_train = X_train.unsqueeze(2)
X_val = X_val.unsqueeze(2)

print("Train Shape:", X_train.shape)
print("Val Shape:", X_val.shape)

# ====================================
# DATALOADER
# ====================================

train_loader = DataLoader(
    TensorDataset(X_train,y_train),
    batch_size=1024,
    shuffle=True
)

val_loader = DataLoader(
    TensorDataset(X_val,y_val),
    batch_size=1024,
    shuffle=False
)

# ====================================
# MODEL
# ====================================

class LSTMV2(nn.Module):

    def __init__(self):

        super().__init__()

        self.lstm = nn.LSTM(
            input_size=1,
            hidden_size=128,
            num_layers=3,
            batch_first=True,
            bidirectional=True,
            dropout=0.3
        )

        self.bn = nn.BatchNorm1d(
            256
        )

        self.fc1 = nn.Linear(
            256,
            128
        )

        self.fc2 = nn.Linear(
            128,
            64
        )

        self.fc3 = nn.Linear(
            64,
            1
        )

        self.relu = nn.ReLU()

        self.dropout = nn.Dropout(
            0.3
        )

    def forward(self,x):

        x,_ = self.lstm(x)

        x = x[:,-1,:]

        x = self.bn(x)

        x = self.fc1(x)

        x = self.relu(x)

        x = self.dropout(x)

        x = self.fc2(x)

        x = self.relu(x)

        x = self.dropout(x)

        x = self.fc3(x)

        return x

# ====================================
# MODEL SETUP
# ====================================

model = LSTMV2().to(device)

pos_weight = torch.tensor(
    [1.45]
).to(device)

criterion = nn.BCEWithLogitsLoss(
    pos_weight=pos_weight
)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=0.001,
    weight_decay=1e-4
)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="min",
    factor=0.5,
    patience=2
)

best_loss = 999999

patience = 5
counter = 0

print("Model Loaded")

# ====================================
# TRAINING
# ====================================

for epoch in range(30):

    model.train()

    train_loss = 0

    for X_batch,y_batch in train_loader:

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

        for X_batch,y_batch in val_loader:

            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            outputs = model(X_batch)

            loss = criterion(
                outputs.squeeze(),
                y_batch
            )

            val_loss += loss.item()

    val_loss /= len(val_loader)

    scheduler.step(val_loss)

    print(
        f"Epoch {epoch+1}/30 | "
        f"Train Loss={train_loss:.5f} | "
        f"Val Loss={val_loss:.5f}"
    )

    if val_loss < best_loss:

        best_loss = val_loss

        counter = 0

        torch.save(
            model.state_dict(),
            "models/lstm_v2/best_lstm_v2.pth"
        )

    else:

        counter += 1

    if counter >= patience:

        print("\nEarly Stopping Triggered")
        break

print("\nTraining Complete")
print("Best Validation Loss:", best_loss)
