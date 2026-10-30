import pandas as pd
import matplotlib.pyplot as plt

off = pd.read_csv("data/team_offense.csv", index_col=0)
deff = pd.read_csv("data/team_defense.csv", index_col=0)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

panels = [
    (axes[0], off, "Offense (best = top right)",
     "Shot quality created (expected points per shot)",
     "Shot making (actual minus expected points per shot)"),
    (axes[1], deff, "Defense (best = bottom left)",
     "Shot quality allowed (expected points per shot)",
     "Opponent shot making (actual minus expected)"),
]

for ax, t, title, xlabel, ylabel in panels:
    for team, row in t.iterrows():
        is_okc = team == "OKC"
        ax.scatter(row["xp"], row["shot_making"],
                   color="tab:blue" if is_okc else "lightgray",
                   s=140 if is_okc else 50, zorder=3 if is_okc else 2)
        ax.annotate(team, (row["xp"], row["shot_making"]), xytext=(5, 4),
                    textcoords="offset points",
                    fontsize=9 if is_okc else 7,
                    fontweight="bold" if is_okc else "normal",
                    color="tab:blue" if is_okc else "dimgray")
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")          # shots went in exactly as often as expected
    ax.axvline(t["xp"].mean(), color="black", linewidth=0.8, linestyle=":")  # league-average shot quality
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

fig.suptitle("2024-25 regular season: shot quality vs. shot making, by team")
plt.tight_layout()
plt.savefig("team_chart.png", dpi=150, bbox_inches="tight")
print("Saved team_chart.png")