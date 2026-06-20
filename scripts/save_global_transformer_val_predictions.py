import pandas as pd
import numpy as np
import torch
import torch.nn as nn

device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)

print("Using Device:", device)

X_val = pd.read_csv(
    "Phase7_splits/X_val_transformer.csv"
).values

X_val = torch.tensor(
    X_val,
    dtype=torch.float32
).unsqueeze(2)

print("Validation Shape:", X_val.shape)

class GlobalTransformer(nn.Module):

    def __init__(self):

        super().__init__()

        self.embedding = nn.Linear(1,128)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=128,
            nhead=8,
            dropout=0.2,
            batch_first=True
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=4
        )

        self.attention = nn.Linear(
            128,
            1
        )

        self.fc1 = nn.Linear(128,64)
        self.fc2 = nn.Linear(64,32)
        self.fc3 = nn.Linear(32,1)

        self.relu = nn.ReLU()

        self.dropout = nn.Dropout(0.3)

    def forward(self,x):

        x = self.embedding(x)

        x = self.transformer(x)

        weights = torch.softmax(
            self.attention(x),
            dim=1
        )

        x = (weights*x).sum(dim=1)

        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.fc3(x)

        return x

model = GlobalTransformer().to(device)

model.load_state_dict(
    torch.load(
        "models/global_transformer/best_global_transformer.pth",
        map_location=device
    )
)

model.eval()

print("Model Loaded")

probs = []

batch_size = 512

with torch.no_grad():

    for i in range(
        0,
        len(X_val),
        batch_size
    ):

        batch = X_val[
            i:i+batch_size
        ].to(device)

        output = torch.sigmoid(
            model(batch)
        )

        probs.extend(
            output.cpu().numpy().flatten()
        )

pd.DataFrame(
    probs,
    columns=["probability"]
).to_csv(
    "ensemble/validation_predictions/global_transformer_val_probs.csv",
    index=False
)

print("Saved:", len(probs))
