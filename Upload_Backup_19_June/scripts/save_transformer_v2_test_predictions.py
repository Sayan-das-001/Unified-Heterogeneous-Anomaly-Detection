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

# ==========================
# LOAD DATA
# ==========================

X_val = pd.read_csv(
    "Phase7_splits/X_test_transformer.csv"
).values

X_val = torch.tensor(
    X_val,
    dtype=torch.float32
)

X_val = X_val.unsqueeze(2)

print("Validation Shape:", X_val.shape)

# ==========================
# DATALOADER
# ==========================

val_loader = DataLoader(
    TensorDataset(X_val),
    batch_size=1024,
    shuffle=False
)

# ==========================
# MODEL
# ==========================

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
            batch_first=True,
            dropout=0.2
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=2
        )

        self.fc1 = nn.Linear(
            64,
            64
        )

        self.fc2 = nn.Linear(
            64,
            32
        )

        self.fc3 = nn.Linear(
            32,
            1
        )

        self.relu = nn.ReLU()

        self.dropout = nn.Dropout(
            0.2
        )

    def forward(self,x):

        x = self.embedding(x)

        x = self.transformer(x)

        x = x.mean(dim=1)

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

model = TransformerClassifier().to(device)

model.load_state_dict(
    torch.load(
        "models/transformer_v2/best_transformer_v2.pth",
        map_location=device
    )
)

model.eval()

print("Model Loaded")

# ==========================
# PREDICTIONS
# ==========================

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

pd.DataFrame(
    all_probs,
    columns=["transformer_prob"]
).to_csv(
    "ensemble/predictions/transformer_probs.csv",
    index=False
)

print("Saved:", len(all_probs))
