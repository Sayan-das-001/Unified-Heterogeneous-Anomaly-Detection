import pandas as pd

df = pd.DataFrame({

    "Comparison":[
        "Ensemble-4 vs XGBoost"
    ],

    "Test":[
        "McNemar"
    ],

    "Statistic":[
        1.9929979730974756
    ],

    "P_Value":[
        0.15802776761173493
    ],

    "Significant":[
        "No"
    ]

})

df.to_csv(
    "paper_assets/final_paper/tables/statistical_validation.csv",
    index=False
)

print(df)
