import pandas as pd
import joblib

print("Loading Validation Data...")

X_val = pd.read_csv(
    "Phase7_splits/X_val_transformer.csv"
)

print("Loading XGBoost Model...")

model = joblib.load(
    "models/xgboost/best_xgboost.pkl"
)

probs = model.predict_proba(
    X_val
)[:,1]

pd.DataFrame(
    probs,
    columns=["xgb_prob"]
).to_csv(
    "ensemble/validation_predictions/xgb_val_probs.csv",
    index=False
)

print("Saved:", len(probs))
