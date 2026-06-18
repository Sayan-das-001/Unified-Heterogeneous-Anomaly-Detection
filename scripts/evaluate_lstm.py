import pandas as pd
import numpy as np
import torch

from torch import nn

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)

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

X_test = pd.read_csv(
    "Phase7_splits/X_test_lstm.csv"
).values

y_test = pd.read_csv(
    "Phase7_splits/y_test_lstm.csv"
)["binary_label"].values

# Same shape used during training
# (samples, 39) -> (samples, 39, 1)

X_test = torch.tensor(
    X_test,
    dtype=torch.float32
).unsqueeze(-1)

print("Test Shape:", X_test.shape)

# ==========================================
# MODEL
# ==========================================

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

    def forward(self, x):

        x, _ = self.lstm1(x)

        x = self.dropout1(x)

        x, _ = self.lstm2(x)

        x = self.dropout2(x)

        x = x[:, -1, :]

        x = self.fc1(x)

        x = self.relu(x)

        x = self.fc2(x)

        return x

# ==========================================
# LOAD MODEL
# ==========================================

print("Loading Model...")

model = LSTMClassifier().to(device)

model.load_state_dict(
    torch.load(
        "models/lstm/best_lstm.pth",
        map_location=device
    )
)

model.eval()

# ==========================================
# BATCHED INFERENCE
# ==========================================

print("Running Inference...")

batch_size = 2048

probs = []

with torch.no_grad():

    for i in range(
        0,
        len(X_test),
        batch_size
    ):

        batch = X_test[
            i:i + batch_size
        ].to(device)

        logits = model(batch)

        batch_probs = torch.sigmoid(
            logits
        ).cpu().numpy().flatten()

        probs.extend(
            batch_probs
        )

        if i % 20000 == 0:
            print(
                f"Processed {min(i+batch_size,len(X_test))}/{len(X_test)}"
            )

probs = np.array(probs)

preds = (
    probs > 0.5
).astype(int)

# ==========================================
# METRICS
# ==========================================

acc = accuracy_score(
    y_test,
    preds
)

prec = precision_score(
    y_test,
    preds
)

rec = recall_score(
    y_test,
    preds
)

f1 = f1_score(
    y_test,
    preds
)

auc = roc_auc_score(
    y_test,
    probs
)

cm = confusion_matrix(
    y_test,
    preds
)

# ==========================================
# RESULTS
# ==========================================

print("\n==============================")
print("LSTM RESULTS")
print("==============================")

print("Accuracy :", acc)
print("Precision:", prec)
print("Recall   :", rec)
print("F1 Score :", f1)
print("ROC AUC  :", auc)

print("\nConfusion Matrix")
print(cm)

# ==========================================
# SAVE RESULTS
# ==========================================

with open(
    "results/lstm/results.txt",
    "w"
) as f:

    f.write("LSTM RESULTS\n")
    f.write("====================\n")

    f.write(f"Accuracy={acc}\n")
    f.write(f"Precision={prec}\n")
    f.write(f"Recall={rec}\n")
    f.write(f"F1={f1}\n")
    f.write(f"ROC_AUC={auc}\n")

    f.write("\nConfusion Matrix\n")
    f.write(str(cm))

print("\nResults Saved")
