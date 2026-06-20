import pandas as pd
import joblib

from sklearn.metrics import roc_curve

model = joblib.load(
    "models/xgboost/best_xgboost.pkl"
)

X_test = pd.read_csv(
    "Phase7_splits/X_test_lstm.csv"
)

y_test = pd.read_csv(
    "Phase7_splits/y_test_lstm.csv"
).values.ravel()

probs = model.predict_proba(
    X_test
)[:,1]

fpr,tpr,_ = roc_curve(
    y_test,
    probs
)

pd.DataFrame({
    "fpr":fpr,
    "tpr":tpr
}).to_csv(
    "paper_assets/ensemble/roc/xgb_roc.csv",
    index=False
)

print("Saved")
