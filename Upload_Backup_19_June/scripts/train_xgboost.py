import pandas as pd
import joblib

from xgboost import XGBClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)

print("Loading Data...")

X_train = pd.read_csv(
    "Phase7_splits/X_train_lstm.csv"
)

X_val = pd.read_csv(
    "Phase7_splits/X_val_lstm.csv"
)

X_test = pd.read_csv(
    "Phase7_splits/X_test_lstm.csv"
)

y_train = pd.read_csv(
    "Phase7_splits/y_train_lstm.csv"
).values.ravel()

y_val = pd.read_csv(
    "Phase7_splits/y_val_lstm.csv"
).values.ravel()

y_test = pd.read_csv(
    "Phase7_splits/y_test_lstm.csv"
).values.ravel()

print("Train Shape:", X_train.shape)
print("Test Shape :", X_test.shape)

print("Training XGBoost...")

model = XGBClassifier(
    n_estimators=500,
    max_depth=8,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="binary:logistic",
    eval_metric="logloss",
    tree_method="hist",
    random_state=42,
    n_jobs=-1
)

model.fit(
    X_train,
    y_train
)

joblib.dump(
    model,
    "models/xgboost/best_xgboost.pkl"
)

print("Model Saved")

probs = model.predict_proba(X_test)[:,1]

preds = (probs > 0.5).astype(int)

acc = accuracy_score(y_test,preds)
prec = precision_score(y_test,preds)
rec = recall_score(y_test,preds)
f1 = f1_score(y_test,preds)
roc = roc_auc_score(y_test,probs)

cm = confusion_matrix(
    y_test,
    preds
)

print("\n====================")
print("XGBOOST RESULTS")
print("====================")
print("Accuracy :",acc)
print("Precision:",prec)
print("Recall   :",rec)
print("F1 Score :",f1)
print("ROC AUC  :",roc)
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
        roc
    ]
}).to_csv(
    "paper_assets/ensemble/tables/xgboost_metrics.csv",
    index=False
)

pd.DataFrame(
    probs,
    columns=["probability"]
).to_csv(
    "ensemble/predictions/xgb_probs.csv",
    index=False
)

pd.DataFrame(
    y_test,
    columns=["label"]
).to_csv(
    "ensemble/predictions/y_true.csv",
    index=False
)

pd.DataFrame(
    cm
).to_csv(
    "paper_assets/ensemble/tables/xgb_confusion_matrix.csv",
    index=False
)

print("\nSaved Everything")
