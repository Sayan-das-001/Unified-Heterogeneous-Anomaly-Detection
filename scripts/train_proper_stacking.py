import pandas as pd
import lightgbm as lgb
import joblib

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
    "ensemble/proper_stacking/meta_train.csv"
)

test_df = pd.read_csv(
    "ensemble/proper_stacking/meta_test.csv"
)

X_train = train_df.drop(
    columns=["label"]
)

y_train = train_df["label"]

X_test = test_df.drop(
    columns=["label"]
)

y_test = test_df["label"]

print("Training LightGBM Meta Learner...")

model = lgb.LGBMClassifier(
    n_estimators=300,
    learning_rate=0.03,
    num_leaves=31,
    random_state=42
)

model.fit(
    X_train,
    y_train
)

joblib.dump(
    model,
    "models/lightgbm_meta/proper_stacking.pkl"
)

probs = model.predict_proba(
    X_test
)[:,1]

preds = (probs > 0.5).astype(int)

acc = accuracy_score(y_test,preds)
prec = precision_score(y_test,preds)
rec = recall_score(y_test,preds)
f1 = f1_score(y_test,preds)
auc = roc_auc_score(y_test,probs)

cm = confusion_matrix(
    y_test,
    preds
)

print("\n====================")
print("PROPER STACKING")
print("====================")

print("Accuracy :",acc)
print("Precision:",prec)
print("Recall   :",rec)
print("F1 Score :",f1)
print("ROC AUC  :",auc)

print(cm)

pd.DataFrame({
    "Metric":[
        "Accuracy",
        "Precision",
        "Recall",
        "F1",
        "ROC_AUC"
    ],
    "Value":[
        acc,
        prec,
        rec,
        f1,
        auc
    ]
}).to_csv(
    "paper_assets/proper_ensemble/tables/proper_stacking_metrics.csv",
    index=False
)

pd.DataFrame(
    probs,
    columns=["ensemble_prob"]
).to_csv(
    "ensemble/proper_stacking/proper_probs.csv",
    index=False
)

pd.DataFrame(
    cm
).to_csv(
    "paper_assets/proper_ensemble/tables/proper_cm.csv",
    index=False
)

print("\nSaved Everything")

