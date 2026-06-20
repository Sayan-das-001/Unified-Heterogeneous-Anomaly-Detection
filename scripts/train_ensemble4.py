import pandas as pd
import numpy as np
import joblib

from lightgbm import LGBMClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)

print("Loading Data...")

train_df = pd.read_csv(
    "ensemble/proper_stacking/meta_train_4models.csv"
)

test_df = pd.read_csv(
    "ensemble/proper_stacking/meta_test_4models.csv"
)

X_train = train_df.drop(
    columns=["label"]
)

y_train = train_df["label"]

X_test = test_df.drop(
    columns=["label"]
)

y_test = test_df["label"]

print("Training LightGBM...")

model = LGBMClassifier(
    n_estimators=500,
    learning_rate=0.03,
    max_depth=5,
    num_leaves=31,
    random_state=42
)

model.fit(
    X_train,
    y_train
)

joblib.dump(
    model,
    "models/lightgbm_meta/ensemble4.pkl"
)

probs = model.predict_proba(
    X_test
)[:,1]

preds = (
    probs >= 0.5
).astype(int)

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
    probs
)

cm = confusion_matrix(
    y_test,
    preds
)

print("\n====================")
print("ENSEMBLE 4 RESULTS")
print("====================")
print("Accuracy :", acc)
print("Precision:", prec)
print("Recall   :", rec)
print("F1 Score :", f1)
print("ROC AUC  :", roc)
print(cm)

pd.DataFrame({
    "Accuracy":[acc],
    "Precision":[prec],
    "Recall":[rec],
    "F1":[f1],
    "ROC_AUC":[roc]
}).to_csv(
    "paper_assets/proper_ensemble/tables/ensemble4_metrics.csv",
    index=False
)

pd.DataFrame(
    cm
).to_csv(
    "paper_assets/proper_ensemble/tables/ensemble4_cm.csv",
    index=False
)

pd.DataFrame({
    "probability": probs
}).to_csv(
    "ensemble/proper_stacking/ensemble4_probs.csv",
    index=False
)

print("Saved Everything")
