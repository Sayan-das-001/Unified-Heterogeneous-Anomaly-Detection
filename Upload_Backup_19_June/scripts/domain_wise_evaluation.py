import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

test_df = pd.read_csv(
    "ensemble/domain_test_split.csv"
)

probs = pd.read_csv(
    "ensemble/proper_stacking/ensemble4_probs.csv"
)

test_df["prediction"] = (
    probs.iloc[:,0] > 0.5
).astype(int)

results = []

for domain in sorted(
    test_df["domain"].unique()
):

    subset = test_df[
        test_df["domain"] == domain
    ]

    y_true = subset["binary_label"]

    y_pred = subset["prediction"]

    results.append([
        domain,
        accuracy_score(y_true,y_pred),
        precision_score(y_true,y_pred),
        recall_score(y_true,y_pred),
        f1_score(y_true,y_pred)
    ])

results = pd.DataFrame(
    results,
    columns=[
        "Domain",
        "Accuracy",
        "Precision",
        "Recall",
        "F1"
    ]
)

results.to_csv(
    "paper_assets/final_paper/tables/domain_wise_results.csv",
    index=False
)

print(results)
