import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("data/fg_clean.csv")

# 235,000 dots would be one solid blob, so plot a random sample.
# random_state makes the sample the same every run.
sample = df.sample(n=5000, random_state=42)

made = sample[sample["made"] == 1]
missed = sample[sample["made"] == 0]

fig, ax = plt.subplots(figsize=(7, 8))

# s = dot size, alpha = transparency (so overlapping dots show density)
ax.scatter(missed["y"], missed["x"], s=6, alpha=0.4, color="tab:red", label="Missed")
ax.scatter(made["y"], made["x"], s=6, alpha=0.4, color="tab:green", label="Made")

# the hoop: x = 41.75 ft from center court, y = 0 (middle of the court's width)
ax.scatter([0], [41.75], s=120, facecolors="none", edgecolors="black", linewidths=2, label="Hoop")

ax.set_xlim(-25, 25)   # court width is 50 ft
ax.set_ylim(0, 47)     # half court is 47 ft
ax.set_aspect("equal") # 1 ft horizontally looks the same as 1 ft vertically
ax.set_xlabel("Court width (ft)")
ax.set_ylabel("Distance from half court (ft)")
ax.set_title("2024-25 NBA shot attempts (5,000 random shots, flipped to one hoop)")
ax.legend(loc="lower right")

plt.savefig("shot_map.png", dpi=150, bbox_inches="tight")
print("Saved shot_map.png")
