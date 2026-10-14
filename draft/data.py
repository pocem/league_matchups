"""Solo queue drafts as champion indices, and the same encoding for pro drafts."""
from __future__ import annotations

import numpy as np
import pandas as pd

SOLO_ROLES = ["top", "jungle", "mid", "bot", "support"]
PRO_ROLES = ["top", "jng", "mid", "bot", "sup"]  # same order
SIDES = ["blue", "red"]

# Riot's internal names (solo queue) -> the display names Oracle's Elixir uses
ALIASES = {"monkeyking": "wukong", "nunu": "nunuwillump", "renata": "renataglasc"}


def norm(name: str) -> str:
    key = "".join(ch for ch in str(name).lower() if ch.isalnum())
    return ALIASES.get(key, key)


def vocab(champions: pd.DataFrame) -> dict[str, int]:
    """Normalized champion name -> index, 0 reserved for unknown."""
    names = sorted({norm(n) for n in champions.champion_name})
    return {n: i + 1 for i, n in enumerate(names)}


def solo_drafts(matches: pd.DataFrame, champions: pd.DataFrame, voc: dict) -> np.ndarray:
    """(n, 2, 5) champion indices from data/matches.parquet (champion ids)."""
    id_to_idx = {cid: voc[norm(name)] for cid, name in zip(champions.champion_id, champions.champion_name)}
    cols = [f"{sd}_{r}" for sd in SIDES for r in SOLO_ROLES]
    return np.vectorize(lambda c: id_to_idx.get(c, 0))(matches[cols].to_numpy()).reshape(-1, 2, 5)


def pro_drafts(games: pd.DataFrame, voc: dict) -> np.ndarray:
    """(n, 2, 5) champion indices from data/pro/games.parquet (champion names)."""
    cols = [f"{sd}_{r}_champion" for sd in SIDES for r in PRO_ROLES]
    return np.vectorize(lambda name: voc.get(norm(name), 0))(games[cols].to_numpy()).reshape(-1, 2, 5)
