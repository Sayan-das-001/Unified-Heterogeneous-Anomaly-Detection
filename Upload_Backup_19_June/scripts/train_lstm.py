import pandas as pd
import numpy as np
import torch
import torch.nn as nn

from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader

# ======================================
# DEVICE
# ======================================

device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)

print("Using Device:", device)

# ======================================
# LOAD DATA
# ======================================

print("Loading Data...")

X_train = pd.read_csv(
    "Phase7_splits/X_train_lstm.csv"
)

X_val = pd.read_csv(
    "Phase7_splits/X_val_lstm.csv"
)

y_train = pd.read_csv(
    "Phase7_splits/y_train_lstm.csv"
)

y_val = pd.read_csv(
    "Phase7_splits/y_val_lstm.csv"
)

# ======================================
# NUMPY
# ======================================

X_train = X_train.values
X_val = X_val.values

y_train = y_train.values
y_val = y_val.values

# ======================================
# RESHAPE FOR LSTM
# (samples, seq_len, features)
# ======================================

X_train = X_train.reshape(
    X_train.shape[0],
    X_train.shape[1],
    1
)

X_val = X_val.reshape(
    X_val.shape[0],
    X_val.shape[1],
    1
)

# ======================================
# TENSORS
# ======================================

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

# ======================================
# DATALOADERS
# ======================================

train_loader = DataLoader(
    TensorDataset(
        X_train,
        y_train
    ),
    batch_size=2048,
    shuffle=True
)

val_loader = DataLoader(
    TensorDataset(
        X_val,
        y_val
    ),
    batch_size=2048,
    shuffle=False
)

# ======================================
# MODEL
# ======================================

class LSTMClassifier(nn.Module):

    def __init__(self):

        super().__init__()

        self.lstm1 = nn.LSTM(
            input_size=1,
            hidden_size=64,
            batch_first=True
        )

        self.dropout1 = nn.Dropout(0.3)

        self.lstm2 = nn.LSTM(
            input_size=64,
            hidden_size=32,
            batch_first=True
        )

        self.dropout2 = nn.Dropout(0.3)

        self.fc1 = nn.Linear(
            32,
            16
        )

        self.relu = nn.ReLU()

        self.fc2 = nn.Linear(
            16,
            1
        )

    def forward(self,x):

        x,_ = self.lstm1(x)

        x = self.dropout1(x)

        x,_ = self.lstm2(x)

        x = self.dropout2(x)

        x = x[:,-1,:]

        x = self.fc1(x)

        x = self.relu(x)

        x = self.fc2(x)

        return x

# ======================================
# INIT
# ======================================

model = LSTMClassifier().to(device)

criterion = nn.BCEWithLogitsLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)

# ======================================
# TRAINING
# ======================================

best_val_loss = np.inf

epochs = 15

for epoch in range(epochs):

    model.train()

    train_loss = 0

    for X_batch,y_batch in train_loader:

        X_batch = X_batch.to(device)

        y_batch = y_batch.to(device)

        optimizer.zero_grad()

        outputs = model(
            X_batch
        )

        loss = criterion(
            outputs,
            y_batch
        )

        loss.backward()

        optimizer.step()

        train_loss += loss.item()

    # -------------------
    # VALIDATION
    # -------------------

    model.eval()

    val_loss = 0

    with torch.no_grad():

        for X_batch,y_batch in val_loader:

            X_batch = X_batch.to(device)

            y_batch = y_batch.to(device)

            outputs = model(
                X_batch
            )

            loss = criterion(
                outputs,
                y_batch
            )

            val_loss += loss.item()

    train_loss /= len(train_loader)

    val_loss /= len(val_loader)

    print(
        f"Epoch {epoch+1}/{epochs} | "
        f"Train Loss={train_loss:.5f} | "
        f"Val Loss={val_loss:.5f}"
    )

    if val_loss < best_val_loss:

        best_val_loss = val_loss

        torch.save(
            model.state_dict(),
            "models/lstm/best_lstm.pth"
        )

print("\nTraining Complete")
print("Best Validation Loss:", best_val_loss)
