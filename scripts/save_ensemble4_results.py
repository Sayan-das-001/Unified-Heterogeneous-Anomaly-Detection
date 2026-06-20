import pandas as pd

results = pd.DataFrame({
    "Metric":[
        "Accuracy",
        "Precision",
        "Recall",
        "F1",
        "ROC_AUC"
    ],
    "Value":[
        0.92542,
        0.85811,
        0.97903,
        0.91459,
        0.98711
    ]
})

results.to_csv(
    "paper_assets/final_paper/tables/ensemble4_metrics.csv",
    index=False
)

print(results)
