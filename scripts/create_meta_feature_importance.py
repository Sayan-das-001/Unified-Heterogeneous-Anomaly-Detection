import joblib
import pandas as pd
import matplotlib.pyplot as plt

model = joblib.load(
    "models/lightgbm_meta/ensemble4.pkl"
)

features = [
    "LSTM V2",
    "Transformer V2",
    "Global Transformer",
    "XGBoost"
]

importance = model.feature_importances_

df = pd.DataFrame({
    "Feature": features,
    "Importance": importance
})

df = df.sort_values(
    by="Importance",
    ascending=False
)

print(df)

df.to_csv(
    "paper_assets/final_paper/tables/meta_feature_importance.csv",
    index=False
)

plt.figure(figsize=(8,5))

plt.bar(
    df["Feature"],
    df["Importance"]
)

plt.ylabel("Importance")
plt.title(
    "Meta Learner Feature Importance"
)

plt.tight_layout()

plt.savefig(
    "paper_assets/final_paper/plots/meta_feature_importance.png",
    dpi=300
)

plt.savefig(
    "paper_assets/final_paper/plots/meta_feature_importance.pdf"
)

print("Saved")
