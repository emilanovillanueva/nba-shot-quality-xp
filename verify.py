import pandas as pd
import numpy as np

df = pd.read_csv("data/shots_2025.csv")
fg = df[~df["text"].str.contains("free throw", case=False, na=False)].copy()
fg["stated_ft"] = fg["text"].str.extract(r"(\d+)-foot").astype(float)

# rotate shots from the left end of the court so every shot targets the same hoop
flip = fg["coordinate_x"] < 0
fg["x"] = np.where(flip, -fg["coordinate_x"], fg["coordinate_x"])
fg["y"] = np.where(flip, -fg["coordinate_y"], fg["coordinate_y"])

HOOP_X = 41.75
fg["dist"] = np.sqrt((fg["x"] - HOOP_X) ** 2 + fg["y"] ** 2)

sub = fg.dropna(subset=["stated_ft"])
err = sub["dist"] - sub["stated_ft"]
print("Shots with a stated distance:", len(sub))
print("Mean absolute error (ft):", round(err.abs().mean(), 2))
print("Median error (ft):", round(err.median(), 2))
print("Share within 2 ft:", round((err.abs() <= 2).mean(), 3))
print(sub[["text", "x", "y", "dist", "stated_ft"]].head(8).to_string())
