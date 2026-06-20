import pandas as pd

from sklearn.ensemble import RandomForestClassifier

print("Loading Data...")

X = pd.read_csv(
    "Phase7_splits/X_train_transformer.csv"
)

y = pd.read_csv(
    "Phase7_splits/y_train_transformer.csv"
)

print("Training Random Forest...")

rf = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)

rf.fit(
    X,
    y.values.ravel()
)

importance = pd.DataFrame({

    "Feature":X.columns,
    "Importance":rf.feature_importances_

})

importance = importance.sort_values(
    by="Importance",
    ascending=False
)

importance.to_csv(
    "paper_assets/feature_analysis/feature_importance.csv",
    index=False
)

importance.head(10).to_csv(
    "paper_assets/feature_analysis/top_10_features.csv",
    index=False
)

print(
    importance.head(10)
)
