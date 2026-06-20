import pandas as pd

ablation = pd.DataFrame({

    "Configuration":[

        "LSTM V2",

        "Transformer V2",

        "Global Transformer",

        "XGBoost",

        "LSTM+Transformer+XGB",

        "LSTM+Transformer+Global+XGB"
    ],

    "F1":[

        90.94,

        90.92,

        87.23,

        91.18,

        91.40,

        91.46
    ],

    "ROC_AUC":[

        0.9808,

        0.9795,

        0.9682,

        0.9870,

        0.9871,

        0.98711
    ]
})

ablation.to_csv(
    "paper_assets/final_paper/tables/ablation_study.csv",
    index=False
)

print(ablation)
