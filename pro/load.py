"""Load Oracle's Elixir CSVs into one clean row per player per game, in chronological order."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

ROLES = ["top", "jng", "mid", "bot", "sup"]
SIDES = ["Blue", "Red"]
COLS = ["gameid", "league", "year", "split", "playoffs", "date", "game", "patch", "side", "position",
        "playerid", "playername", "teamid", "teamname", "champion", "result"]


def load(data_dir="pro_play_data") -> pd.DataFrame:
    paths = sorted(Path(data_dir).glob("*_LoL_esports_match_data_from_OraclesElixir.csv"))
    if not paths:
        raise FileNotFoundError(f"no Oracle's Elixir CSVs in {data_dir}")
    df = pd.concat([pd.read_csv(p, usecols=COLS, dtype={"patch": str}) for p in paths], ignore_index=True)
    df = df[df.position.isin(ROLES)].drop_duplicates(["gameid", "side", "position"])
    df["date"] = pd.to_datetime(df.date)
    # Oracle's "year" tags some off-season events with the neighbouring season; use the calendar
    # year so the season boundary is a single point in time
    df["year"] = df.date.dt.year

    # keep games with exactly one player per side and role, and one winner
    complete = df.groupby("gameid").size() == 10
    side_result = df.groupby(["gameid", "side"]).result.agg(["min", "max"]).unstack()
    one_winner = ((side_result["min"] == side_result["max"]).all(axis=1)
                  & (side_result[("min", "Blue")] + side_result[("min", "Red")] == 1))
    ok = complete & one_winner.reindex(complete.index, fill_value=False)
    df = df[df.gameid.isin(ok[ok].index)].copy()

    # stable ids; a few rows lack Oracle's ids, fall back to names
    df["player"] = df.playerid.fillna("name:" + df.playername)
    df["team"] = df.teamid.fillna("name:" + df.teamname)

    df["side"] = pd.Categorical(df.side, SIDES, ordered=True)
    df["position"] = pd.Categorical(df.position, ROLES, ordered=True)
    df = df.sort_values(["date", "gameid", "side", "position"], ignore_index=True)
    df["patch"] = df.patch.ffill().bfill()  # a handful of games have no patch
    return df[["gameid", "date", "year", "league", "split", "playoffs", "game", "patch", "side", "position",
               "player", "playername", "team", "teamname", "champion", "result"]]
