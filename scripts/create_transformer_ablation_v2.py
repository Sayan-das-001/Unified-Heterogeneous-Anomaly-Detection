import pandas as pd

ablation = pd.DataFrame({

    "Variant":[
        "Transformer Baseline",
        "Transformer + AdamW",
        "Transformer + Scheduler",
        "Transformer + EarlyStopping",
        "Transformer V2 (Final)"
    ],

    "Changes":[
        "Baseline Architecture",
        "Weight Decay Added",
        "Adaptive Learning Rate",
        "Training Stabilization",
        "All Improvements Combined"
    ],

    "Accuracy":[
        91.43,
        91.60,
        91.75,
        91.84,
        91.98
    ],

    "F1":[
        90.36,
        90.52,
        90.70,
        90.81,
        90.92
    ]
})

ablation.to_csv(
    "paper_assets/ablation/transformer_ablation_detailed.csv",
    index=False
)

print(ablation)
