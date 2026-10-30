import pandas as pd
import numpy as np

df = pd.read_csv("data/shots_2025.csv")
fg = df[~df["text"].str.contains("free throw", case=False, na=False)].copy()

# flip every shot to the right-hand hoop
flip = fg["coordinate_x"] < 0
fg["x"] = np.where(flip, -fg["coordinate_x"], fg["coordinate_x"])
fg["y"] = np.where(flip, -fg["coordinate_y"], fg["coordinate_y"])

HOOP_X = 41.75
dx = HOOP_X - fg["x"]
fg["dist"] = np.sqrt(dx**2 + fg["y"]**2)
fg["angle"] = np.degrees(np.arctan2(fg["y"].abs(), dx))  # 0 = straight on, 90 = baseline level
fg["is_three"] = fg["text"].str.contains("three point", case=False)
fg["made"] = fg["scoring_play"].astype(int)
fg["blocked"] = fg["text"].str.contains(" blocks ", case=False)

print("Field goal attempts:", len(fg))
print("FG%:", round(fg["made"].mean(), 3))
print("3PA share:", round(fg["is_three"].mean(), 3))
print("3P%:", round(fg.loc[fg["is_three"], "made"].mean(), 3))
print("Closest 3PT (ft):", round(fg.loc[fg["is_three"], "dist"].quantile(0.01), 1))
print()
print(fg["type_text"].value_counts().head(12))
print()
print(fg.groupby(pd.cut(fg["dist"], [0, 4, 10, 16, 22, 26, 40, 100]), observed=True)["made"].agg(["count", "mean"]).round(3))

# who is the shooter on blocked shots: athlete_id_1 or athlete_id_2?
normal = fg[~fg["blocked"]].copy()
normal["shooter"] = normal["text"].str.extract(r"^(.+?) (?:makes|misses)")[0]
id_to_name = normal.dropna(subset=["shooter"]).drop_duplicates("athlete_id_1").set_index("athlete_id_1")["shooter"]
blk = fg[fg["blocked"]].copy()
blk["shooter"] = blk["text"].str.extract(r"blocks (.+?) 's")[0]
print()
print("Blocked shots:", len(blk))
print("shooter = athlete_id_1:", round((blk["athlete_id_1"].map(id_to_name) == blk["shooter"]).mean(), 3))
print("shooter = athlete_id_2:", round((blk["athlete_id_2"].map(id_to_name) == blk["shooter"]).mean(), 3))

cols = ["game_id", "game_date", "period", "team_id", "athlete_id_1", "athlete_id_2", "type_text",
        "text", "x", "y", "dist", "angle", "is_three", "made", "blocked", "score_value"]
fg[[c for c in cols if c in fg.columns]].to_csv("data/fg_clean.csv", index=False)
print("Saved data/fg_clean.csv")
