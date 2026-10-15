"""Model inputs computed from the game table (data/pro/games.parquet).

Kept free of torch and lightgbm so that `python -m pro deploy` runs with pandas and sklearn only."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .load import ROLES

SIDES = ["blue", "red"]
NEVER_DAYS = 3000.0  # pc_days_since for a champion the player never played


def side_features(g: pd.DataFrame, sd: str) -> pd.DataFrame:
    """One side's numeric features, named without the side prefix."""
    f = {"team_elo": g[f"{sd}_team_elo"], "roster_elo": g[f"{sd}_roster_elo"],
         "league_offset": g[f"{sd}_league_offset"], "log_team_games": np.log1p(g[f"{sd}_team_games"])}
    if "draft_logit" in g:  # solo queue draft model (python -m draft score), from this side's view
        f["draft_score"] = g.draft_logit * (1 if sd == "blue" else -1)
    for r in ROLES:
        p = f"{sd}_{r}_"
        f[f"{r}_player_elo"] = g[p + "player_elo"]
        f[f"{r}_log_player_games"] = np.log1p(g[p + "player_games"])
        f[f"{r}_log_pc_games"] = np.log1p(g[p + "pc_games"])
        f[f"{r}_log_pc_days"] = np.log1p(g[p + "pc_days_since"].fillna(NEVER_DAYS))
        f[f"{r}_meta_wr"] = g[p + "meta_wr"]
        f[f"{r}_log_meta_games"] = np.log1p(g[p + "meta_games"])
        f[f"{r}_log_patch_games"] = np.log1p(g[p + "patch_games"])
    return pd.DataFrame(f)


def tabular(g: pd.DataFrame, champions=False) -> pd.DataFrame:
    blue, red = side_features(g, "blue"), side_features(g, "red")
    diff = (blue - red).add_prefix("diff_")
    x = pd.concat([blue.add_prefix("blue_"), red.add_prefix("red_"), diff], axis=1)
    x["playoffs"] = g.playoffs
    x["game_in_series"] = g.game_in_series
    x["cross_league"] = g.cross_league.astype(int)
    if champions:
        for sd in SIDES:
            for r in ROLES:
                x[f"{sd}_{r}_champ"] = g[f"{sd}_{r}_champion_idx"].astype("category")
    return x


def logreg_cols(g):
    return [c for c in tabular(g.head()).columns if c.startswith("diff_")]
