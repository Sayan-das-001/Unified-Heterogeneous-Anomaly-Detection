import pandas as pd

figs = pd.DataFrame({

    "Figure":[
        "Figure 1",
        "Figure 2",
        "Figure 3",
        "Figure 4"
    ],

    "Description":[
        "Domain Distribution",
        "Label Distribution",
        "ROC Curve Comparison",
        "Model Performance Comparison"
    ]
})

figs.to_csv(
    "paper_assets/tables/figures_index.csv",
    index=False
)

print(figs)
