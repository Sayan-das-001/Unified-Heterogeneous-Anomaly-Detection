import pandas as pd
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

print("Loading Stacking Dataset...")

df = pd.read_csv(
    "ensemble/stacking_dataset.csv"
)

X = df[
    [
        "lstm_prob",
        "transformer_prob",
        "xgb_prob"
    ]
]

y = df["label"]

print("Training Meta Learner...")

model = LGBMClassifier(
    n_estimators=300,
    learning_rate=0.05,
    num_leaves=31,
    random_state=42
)

model.fit(X,y)

probs = model.predict_proba(X)[:,1]

preds = (
    probs > 0.5
).astype(int)

acc = accuracy_score(y,preds)
prec = precision_score(y,preds)
rec = recall_score(y,preds)
f1 = f1_score(y,preds)
roc = roc_auc_score(y,probs)

cm = confusion_matrix(
    y,
    preds
)

print("\n====================")
print("ENSEMBLE RESULTS")
print("====================")

print("Accuracy :",acc)
print("Precision:",prec)
print("Recall   :",rec)
print("F1 Score :",f1)
print("ROC AUC  :",roc)

print(cm)

joblib.dump(
    model,
    "models/lightgbm_meta/lightgbm_meta.pkl"
)

pd.DataFrame(
    probs
).to_csv(
    "ensemble/predictions/ensemble_probs.csv",
    index=False
)

pd.DataFrame(
    cm
).to_csv(
    "paper_assets/ensemble/tables/ensemble_confusion_matrix.csv",
    index=False
)

pd.DataFrame(
    [{
        "Accuracy":acc,
        "Precision":prec,
        "Recall":rec,
        "F1":f1,
        "ROC_AUC":roc
    }]
).to_csv(
    "paper_assets/ensemble/tables/ensemble_results.csv",
    index=False
)

print("\nSaved Everything")
