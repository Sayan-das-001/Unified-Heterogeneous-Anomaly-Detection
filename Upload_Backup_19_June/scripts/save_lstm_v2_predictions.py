import pandas as pd
import torch
import torch.nn as nn

device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)

print("Using Device:", device)

# ====================================
# LOAD TEST DATA
# ====================================

X_test = pd.read_csv(
    "Phase7_splits/X_test_lstm.csv"
).values

y_test = pd.read_csv(
    "Phase7_splits/y_test_lstm.csv"
)

X_test = torch.tensor(
    X_test,
    dtype=torch.float32
)

X_test = X_test.unsqueeze(2)

print("Test Shape:", X_test.shape)

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
# LOAD MODEL
# ====================================

print("Loading Model...")

model = LSTMV2().to(device)

model.load_state_dict(
    torch.load(
        "models/lstm_v2/best_lstm_v2.pth",
        map_location=device
    )
)

model.eval()

# ====================================
# INFERENCE
# ====================================

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

        p = torch.sigmoid(
            outputs
        )

        probs.extend(
            p.cpu().numpy().flatten()
        )

# ====================================
# SAVE
# ====================================

pd.DataFrame(
    probs,
    columns=["probability"]
).to_csv(
    "paper_assets/roc_curves/data/lstm_v2_probs.csv",
    index=False
)

y_test.to_csv(
    "paper_assets/roc_curves/data/y_true.csv",
    index=False
)

print("\nSaved Successfully")
print("Probabilities:", len(probs))
