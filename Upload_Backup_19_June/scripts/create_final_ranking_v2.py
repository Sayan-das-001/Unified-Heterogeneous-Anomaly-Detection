import pandas as pd

df = pd.DataFrame({

"Model":[
"Proposed Ensemble-4",
"XGBoost",
"LSTM V2",
"Transformer V2",
"LSTM",
"Transformer",
"Global Transformer",
"AutoEncoder"
],

"Accuracy":[
92.54,
92.47,
91.92,
91.98,
91.92,
91.43,
88.21,
84.26
],

"Precision":[
85.81,
87.33,
83.80,
84.42,
84.20,
83.52,
78.14,
91.75
],

"Recall":[
97.90,
95.37,
99.41,
98.52,
98.71,
98.42,
98.72,
67.48
],

"F1":[
91.46,
91.18,
90.94,
90.92,
90.88,
90.36,
87.23,
77.77
],

"ROC_AUC":[
0.98711,
0.98700,
0.98080,
0.97950,
0.97990,
0.97740,
0.96820,
0.76180
]

})

df = df.sort_values(
    by="F1",
    ascending=False
)

df.to_csv(
    "paper_assets/final_paper/tables/final_ranking.csv",
    index=False
)

print(df)

