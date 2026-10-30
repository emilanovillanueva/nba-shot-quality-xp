import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import log_loss, roc_auc_score

df = pd.read_csv("data/fg_clean.csv", parse_dates=["game_date"])
df = df[(df["dist"] <= 40) & (df["season_type"] == 2)].copy()
df["shot_type"] = df["type_text"].fillna("Unknown").astype("category")

# same split as the baseline: first 80% of shots train, last 20% test
cutoff = df["game_date"].sort_values().iloc[int(len(df) * 0.8)]
train = df[df["game_date"] < cutoff]
test = df[df["game_date"] >= cutoff].copy()

base_features = ["dist", "angle", "is_three"]
gbm_features = base_features + ["shot_type"]

def prep(d, cols):
    X = d[cols].copy()
    X["is_three"] = X["is_three"].astype(float)
    return X

lr = LogisticRegression(max_iter=1000).fit(prep(train, base_features), train["made"])
test["p_lr"] = lr.predict_proba(prep(test, base_features))[:, 1]

gbm = HistGradientBoostingClassifier(
    max_iter=300, learning_rate=0.05, categorical_features="from_dtype", random_state=42
).fit(prep(train, gbm_features), train["made"])
test["p_gbm"] = gbm.predict_proba(prep(test, gbm_features))[:, 1]

print("Test shots:", len(test))
for name, col in [("Logistic regression", "p_lr"), ("Gradient boosting", "p_gbm")]:
    print(f"{name}: log loss {log_loss(test['made'], test[col]):.4f} | AUC {roc_auc_score(test['made'], test[col]):.3f}")

print()
print("Predicted vs actual make rate by distance:")
bins = pd.cut(test["dist"], [0, 4, 10, 16, 22, 26, 40])
print(test.groupby(bins, observed=True)[["p_lr", "p_gbm", "made"]].mean().round(3))

test["shot_value"] = np.where(test["is_three"], 3, 2)
print()
print("Average xP per shot (GBM):", round((test["p_gbm"] * test["shot_value"]).mean(), 3))
print("Actual points per shot:   ", round((test["made"] * test["shot_value"]).mean(), 3))
