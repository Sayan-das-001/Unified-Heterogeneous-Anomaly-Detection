import pandas as pd
import numpy as np

from statsmodels.stats.contingency_tables import mcnemar

y_true = pd.read_csv(
    "ensemble/predictions/y_true.csv"
).values.flatten()

xgb_probs = pd.read_csv(
    "ensemble/predictions/xgb_probs.csv"
).values.flatten()

ensemble_probs = pd.read_csv(
    "ensemble/proper_stacking/ensemble4_probs.csv"
).values.flatten()

xgb_pred = (
    xgb_probs > 0.5
).astype(int)

ensemble_pred = (
    ensemble_probs > 0.5
).astype(int)

b = np.sum(
    (xgb_pred == y_true)
    &
    (ensemble_pred != y_true)
)

c = np.sum(
    (xgb_pred != y_true)
    &
    (ensemble_pred == y_true)
)

table = [
    [0,b],
    [c,0]
]

result = mcnemar(
    table,
    exact=False,
    correction=True
)

print("\nMcNemar Test")
print("----------------")
print("b =", b)
print("c =", c)
print("Statistic =", result.statistic)
print("P-value =", result.pvalue)

if result.pvalue < 0.05:
    print("\nStatistically Significant")
else:
    print("\nNot Significant")
