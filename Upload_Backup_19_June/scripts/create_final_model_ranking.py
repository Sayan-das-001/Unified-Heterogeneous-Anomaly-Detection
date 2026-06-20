import pandas as pd

ranking = pd.DataFrame({
    "Model":[
        "Proper Stacking Ensemble",
        "XGBoost",
        "LSTM V2",
        "Transformer V2",
        "LSTM",
        "Transformer",
        "Global Transformer",
        "AutoEncoder"
    ],
    "Accuracy":[
        92.53,
        92.47,
        91.92,
        91.98,
        91.92,
        91.43,
        88.21,
        84.26
    ],
    "Precision":[
        86.19,
        87.33,
        83.80,
        84.42,
        84.20,
        83.52,
        78.14,
        91.75
    ],
    "Recall":[
        97.28,
        95.37,
        99.41,
        98.52,
        98.71,
        98.42,
        98.72,
        67.48
    ],
    "F1":[
        91.40,
        91.18,
        90.94,
        90.92,
        90.88,
        90.36,
        87.23,
        77.77
    ],
    "ROC_AUC":[
        0.9871,
        0.9870,
        0.9808,
        0.9795,
        0.9799,
        0.9774,
        0.9682,
        0.7618
    ]
})

ranking.to_csv(
    "paper_assets/proper_ensemble/tables/final_model_ranking.csv",
    index=False
)

print(ranking)
