from sportsdataverse.nba import load_nba_pbp

SEASON = 2025  # hoopR counts a season by the year it ends: 2025 = 2024-25, 2026 = 2025-26

pbp = load_nba_pbp(seasons=[SEASON], return_as_pandas=True)
print(pbp.shape)
print(pbp["game_date"].min(), "to", pbp["game_date"].max())

shots = pbp[pbp["shooting_play"] == True]
print("Shot attempts:", len(shots))
print(shots[["text", "coordinate_x", "coordinate_y", "scoring_play"]].head())

shots.to_csv(f"data/shots_{SEASON}.csv", index=False)