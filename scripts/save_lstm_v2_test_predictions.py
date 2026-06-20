import pandas as pd
import torch
import torch.nn as nn

device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)

print("Using Device:", device)

# ==========================
# LOAD DATA
# ==========================

X_val = pd.read_csv(
    "Phase7_splits/X_test_lstm.csv"
).values

X_val = torch.tensor(
    X_val,
    dtype=torch.float32
)

X_val = X_val.unsqueeze(2)

print("Validation Shape:", X_val.shape)

# ==========================
# MODEL
# ==========================

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

# ==========================
# LOAD MODEL
# ==========================

model = LSTMV2().to(device)

model.load_state_dict(
    torch.load(
        "models/lstm_v2/best_lstm_v2.pth",
        map_location=device
    )
)

model.eval()

print("Model Loaded")

# ==========================
# PREDICTIONS
# ==========================

from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader

val_loader = DataLoader(
    TensorDataset(X_val),
    batch_size=1024,
    shuffle=False
)

all_probs = []

with torch.no_grad():

    for batch in val_loader:

        x = batch[0].to(device)

        probs = torch.sigmoid(
            model(x)
        )

        all_probs.extend(
            probs.cpu().numpy().flatten()
        )

probs = all_probs

pd.DataFrame(
    probs,
    columns=["lstm_prob"]
).to_csv(
    "ensemble/predictions/lstm_probs.csv",
    index=False
)

print("Saved:", len(probs))
