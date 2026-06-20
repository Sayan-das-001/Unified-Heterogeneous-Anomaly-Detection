import pandas as pd

df = pd.DataFrame({
    "Dataset":[
        "5G",
        "BoTIoT",
        "CICIoT2023",
        "UNSW"
    ],
    "Attack_Samples":[
        27398,
        2426,
        29243,
        2114
    ],
    "Normal_Samples":[
        17685,
        27507,
        740,
        42887
    ]
})

df.to_csv(
    "paper_assets/final_paper/tables/domain_class_distribution.csv",
    index=False
)

print(df)
