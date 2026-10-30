import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score

df = pd.read_csv("data/fg_clean.csv", parse_dates=["game_date"])
df = df[(df["dist"] <= 40) & (df["season_type"] == 2)].copy()  

# split by date: first 80% of game days train, last 20% test
cutoff = df["game_date"].sort_values().iloc[int(len(df) * 0.8)]
train = df[df["game_date"] < cutoff]
test = df[df["game_date"] >= cutoff].copy()

features = ["dist", "angle", "is_three"]
model = LogisticRegression(max_iter=1000)
model.fit(train[features].astype(float), train["made"])

test["p_make"] = model.predict_proba(test[features].astype(float))[:, 1]
test["shot_value"] = np.where(test["is_three"], 3, 2)
test["xp"] = test["p_make"] * test["shot_value"]

# baseline = predict the league-average make rate for every shot
base = np.full(len(test), train["made"].mean())
print("Train shots:", len(train), "| Test shots:", len(test))
print("Log loss, model:   ", round(log_loss(test["made"], test["p_make"]), 4))
print("Log loss, baseline:", round(log_loss(test["made"], base), 4))
print("AUC:", round(roc_auc_score(test["made"], test["p_make"]), 3))

# calibration: do predicted make rates match actual make rates?
test["bin"] = pd.qcut(test["p_make"], 5, duplicates="drop")
print(test.groupby("bin", observed=True)[["p_make", "made"]].mean().round(3))

print("Average xP per shot:", round(test["xp"].mean(), 3))
print("Actual points per shot:", round((test["made"] * test["shot_value"]).mean(), 3))
