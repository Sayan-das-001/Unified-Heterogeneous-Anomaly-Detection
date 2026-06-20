import pandas as pd
import torch
import torch.nn as nn

device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)

print("Using Device:", device)

# =========================
# LOAD DATA
# =========================

X_test = pd.read_csv(
    "Phase7_splits/X_test_transformer.csv"
).values

X_test = torch.tensor(
    X_test,
    dtype=torch.float32
)

X_test = X_test.unsqueeze(2)

print("Test Shape:", X_test.shape)

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

# =========================
# LOAD MODEL
# =========================

print("Loading Model...")

model = TransformerClassifier().to(device)

model.load_state_dict(
    torch.load(
        "models/transformer_v2/best_transformer_v2.pth",
        map_location=device
    )
)

model.eval()

# =========================
# INFERENCE
# =========================

print("Generating Probabilities...")

probs = []

with torch.no_grad():

    for start in range(
        0,
        len(X_test),
        2048
    ):

        batch = X_test[
            start:start+2048
        ].to(device)

        outputs = model(batch)

        p = torch.sigmoid(outputs)

        probs.extend(
            p.cpu().numpy().flatten()
        )

# =========================
# SAVE
# =========================

pd.DataFrame(
    probs,
    columns=["probability"]
).to_csv(
    "paper_assets/roc_curves/data/transformer_v2_probs.csv",
    index=False
)

print("\nSaved Successfully")
print("Probabilities:", len(probs))
