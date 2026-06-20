import pandas as pd
import numpy as np
import torch
import torch.nn as nn

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)

device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)

print("Using Device:", device)

print("Loading Data...")

# =====================================
# LOAD TEST DATA
# =====================================

X_test = pd.read_csv(
    "Phase7_splits/X_test_lstm.csv"
).values

y_test = pd.read_csv(
    "Phase7_splits/y_test_lstm.csv"
).values.flatten()

X_test = torch.tensor(
    X_test,
    dtype=torch.float32
)

X_test = X_test.unsqueeze(2)

print("Test Shape:", X_test.shape)

# =====================================
# MODEL
# =====================================

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

# =====================================
# LOAD MODEL
# =====================================

print("Loading Model...")

model = LSTMV2().to(device)

model.load_state_dict(
    torch.load(
        "models/lstm_v2/best_lstm_v2.pth",
        map_location=device
    )
)

model.eval()

# =====================================
# INFERENCE
# =====================================

print("Running Inference...")

pred_probs = []

batch_size = 2048

with torch.no_grad():

    for start in range(
        0,
        len(X_test),
        batch_size
    ):

        end = start + batch_size

        batch = X_test[start:end].to(device)

        logits = model(batch)

        probs = torch.sigmoid(
            logits
        ).cpu().numpy()

        pred_probs.extend(
            probs.flatten()
        )

pred_probs = np.array(
    pred_probs
)

preds = (
    pred_probs > 0.5
).astype(int)

# =====================================
# METRICS
# =====================================

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

roc = roc_auc_score(
    y_test,
    pred_probs
)

cm = confusion_matrix(
    y_test,
    preds
)

print("\n==============================")
print("LSTM V2 RESULTS")
print("==============================")

print("Accuracy :", acc)
print("Precision:", prec)
print("Recall   :", rec)
print("F1 Score :", f1)
print("ROC AUC  :", roc)

print("\nConfusion Matrix")
print(cm)

with open(
    "results/lstm_v2/results.txt",
    "w"
) as f:

    f.write(f"Accuracy : {acc}\n")
    f.write(f"Precision: {prec}\n")
    f.write(f"Recall   : {rec}\n")
    f.write(f"F1 Score : {f1}\n")
    f.write(f"ROC AUC  : {roc}\n")

print("\nResults Saved")
