import pandas as pd

table = pd.read_csv(
    "paper_assets/tables/publication_table.csv"
)

ensemble = pd.DataFrame([
    {
        "Model":"Stacking Ensemble",
        "Accuracy":93.01,
        "Precision":87.97,
        "Recall":95.98,
        "F1":91.81,
        "ROC_AUC":0.9879
    }
])

table = pd.concat(
    [table,ensemble],
    ignore_index=True
)

table = table.sort_values(
    "F1",
    ascending=False
)

table.to_csv(
    "paper_assets/ensemble/tables/model_comparison_with_ensemble.csv",
    index=False
)

print(table)
