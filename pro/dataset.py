"""Build one row per pro game with point-in-time features for both sides.

Every feature of a game uses only games played before it: the loop records features first,
then updates ratings and stats with the game's result.
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd

from .load import ROLES
from .ratings import Decayed, Elo, EloParams, expected, home_league, shrunk

SIDES = ["blue", "red"]
SLOTS = [(side, role) for side in SIDES for role in ROLES]
META_HALF_LIFE_DAYS = 45.0

# Dropped after testing on 2026: the player's win rate overall and on the champion (raw or
# Elo-adjusted) add nothing once Elo is known. Only how often the player has played the
# champion helps, and time-decaying that count made it worse.
SLOT_FEATURES = ["player_elo", "player_games",
                 "pc_games", "pc_days_since",
                 "meta_games", "meta_wr", "patch_games"]
SIDE_FEATURES = ["team_elo", "team_games", "roster_elo", "league_offset"]


def build(df: pd.DataFrame, params: EloParams | None = None, return_state=False):
    """The game table; with return_state also the ratings and counts after the last game,
    which is what a prediction for an upcoming game needs (see pro/deploy.py)."""
    params = params or EloParams()
    n = len(df) // 10
    first = df.iloc[::10]
    years = first.year.to_numpy()
    leagues = first.league.to_numpy()
    patches = first.patch.to_numpy()
    t_days = ((first.date - first.date.iloc[0]).dt.total_seconds() / 86400).to_numpy()
    players = df.player.to_numpy().reshape(n, 10)
    champs = df.champion.to_numpy().reshape(n, 10)
    teams = df.team.to_numpy().reshape(n, 10)[:, [0, 5]]
    blue_win = df.result.to_numpy()[::10].astype(int)

    player_elo, team_elo = Elo(params), Elo(params)
    league_elo = defaultdict(float)         # home league -> offset added to ratings
    last_home = {}                          # player or team -> last home league
    pc_games = defaultdict(int)             # (player, champion) -> games
    pc_last = {}                            # (player, champion) -> t last played
    meta = Decayed(META_HALF_LIFE_DAYS)     # (champion, role) -> decayed games, wins
    patch_games = defaultdict(int)          # (champion, role, patch) -> games

    slot = {f: np.full((n, 10), np.nan) for f in SLOT_FEATURES}
    side = {f: np.full((n, 2), np.nan) for f in SIDE_FEATURES}
    home = np.empty((n, 2), dtype=object)
    p_team = np.empty(n)
    p_roster = np.empty(n)

    def move_league(elo, key, new_home):
        old = last_home.get(key)
        if old is not None and old != new_home:  # transfer: keep effective strength
            elo.rating[key] += league_elo[old] - league_elo[new_home]
        last_home[key] = new_home

    season = years[0]
    for g in range(n):
        if years[g] != season:
            season = years[g]
            player_elo.new_season()
            team_elo.new_season()
        t = t_days[g]

        homes = [home_league(leagues[g], last_home.get(team)) for team in teams[g]]
        home[g] = homes
        lg = [league_elo[h] for h in homes]
        for s in range(2):
            move_league(team_elo, teams[g, s], homes[s])
            for i in range(5 * s, 5 * s + 5):
                move_league(player_elo, players[g, i], homes[s])

        # --- features, before the game ---
        team_abs = [team_elo.rating[teams[g, s]] + lg[s] for s in range(2)]
        pl_abs = [player_elo.rating[players[g, i]] + lg[i // 5] for i in range(10)]
        roster = [float(np.mean(pl_abs[:5])), float(np.mean(pl_abs[5:]))]
        for s in range(2):
            side["team_elo"][g, s] = team_abs[s]
            side["team_games"][g, s] = team_elo.games[teams[g, s]]
            side["roster_elo"][g, s] = roster[s]
            side["league_offset"][g, s] = lg[s]
        p_team[g] = expected(team_abs[0] - team_abs[1])
        p_roster[g] = expected(roster[0] - roster[1])

        for i in range(10):
            p, c, role = players[g, i], champs[g, i], ROLES[i % 5]
            slot["player_elo"][g, i] = pl_abs[i]
            slot["player_games"][g, i] = player_elo.games[p]
            slot["pc_games"][g, i] = pc_games[(p, c)]
            pct = pc_last.get((p, c))
            slot["pc_days_since"][g, i] = t - pct if pct is not None else np.nan
            mg, mw = meta.get((c, role), t)
            slot["meta_games"][g, i] = mg
            slot["meta_wr"][g, i] = shrunk(mw, mg)
            slot["patch_games"][g, i] = patch_games[(c, role, patches[g])]

        # --- learn from the result ---
        won = [blue_win[g], 1 - blue_win[g]]
        surprise_team = blue_win[g] - p_team[g]
        surprise_roster = blue_win[g] - p_roster[g]
        team_elo.update(teams[g, 0], surprise_team)
        team_elo.update(teams[g, 1], -surprise_team)
        if homes[0] != homes[1]:
            league_elo[homes[0]] += params.k_league * surprise_team
            league_elo[homes[1]] -= params.k_league * surprise_team
        for i in range(10):
            p, c, role = players[g, i], champs[g, i], ROLES[i % 5]
            w = won[i // 5]
            player_elo.update(p, surprise_roster if i < 5 else -surprise_roster)
            pc_games[(p, c)] += 1
            pc_last[(p, c)] = t
            meta.add((c, role), t, w)
            patch_games[(c, role, patches[g])] += 1

    games = _assemble(df, n, players, champs, teams, home, blue_win, slot, side, p_team, p_roster)
    if not return_state:
        return games
    state = {"player_elo": player_elo, "team_elo": team_elo, "league_elo": league_elo,
             "last_home": last_home, "pc_games": pc_games, "pc_last": pc_last, "meta": meta,
             "patch_games": patch_games, "t0": first.date.iloc[0], "t_end": t_days[-1],
             "season": season, "params": params}
    return games, state


def _assemble(df, n, players, champs, teams, home, blue_win, slot, side, p_team, p_roster):
    first = df.iloc[::10].reset_index(drop=True)
    names = df.playername.to_numpy().reshape(n, 10)
    team_names = df.teamname.to_numpy().reshape(n, 10)[:, [0, 5]]
    cols = {
        "gameid": first.gameid, "date": first.date, "year": first.year, "league": first.league,
        "split": first.split, "playoffs": first.playoffs, "game_in_series": first.game, "patch": first.patch,
        "cross_league": home[:, 0] != home[:, 1],
    }
    for s, sd in enumerate(SIDES):
        cols[f"{sd}_team"] = teams[:, s]
        cols[f"{sd}_teamname"] = team_names[:, s]
        cols[f"{sd}_home_league"] = home[:, s]
        for f in SIDE_FEATURES:
            cols[f"{sd}_{f}"] = side[f][:, s]
    for i, (sd, role) in enumerate(SLOTS):
        cols[f"{sd}_{role}_player"] = players[:, i]
        cols[f"{sd}_{role}_playername"] = names[:, i]
        cols[f"{sd}_{role}_champion"] = champs[:, i]
        for f in SLOT_FEATURES:
            cols[f"{sd}_{role}_{f}"] = slot[f][:, i]
    out = pd.DataFrame(cols)
    out["team_elo_diff"] = out.blue_team_elo - out.red_team_elo
    out["roster_elo_diff"] = out.blue_roster_elo - out.red_roster_elo
    out["league_offset_diff"] = out.blue_league_offset - out.red_league_offset
    out["p_blue_team_elo"] = p_team
    out["p_blue_roster_elo"] = p_roster
    out["blue_win"] = blue_win
    return out


def add_indices(games: pd.DataFrame, min_player_games=10, min_team_games=10):
    """Integer ids for embedding layers. 0 = unknown or rare; returns the vocab tables too."""
    vocabs = {}

    def encode(kind, cols, min_count, name_suffix=None):
        counts = pd.concat([games[c] for c in cols]).value_counts()
        keep = counts[counts >= min_count].index if min_count else counts.index
        vocab = pd.DataFrame({kind: sorted(keep)})
        vocab["idx"] = np.arange(1, len(vocab) + 1)
        vocab["games"] = vocab[kind].map(counts)
        if name_suffix:  # readable name: the latest one used with this id (games are in date order)
            names = pd.concat([games[["date", c, c + name_suffix]].set_axis(["date", "id", "name"], axis=1)
                               for c in cols]).sort_values("date", kind="stable")
            vocab["name"] = vocab[kind].map(names.groupby("id").name.last())
        lookup = dict(zip(vocab[kind], vocab.idx))
        for c in cols:
            games[f"{c}_idx"] = games[c].map(lookup).fillna(0).astype("int32")
        vocabs[kind] = vocab

    encode("player", [f"{sd}_{r}_player" for sd, r in SLOTS], min_player_games, name_suffix="name")
    encode("team", [f"{sd}_team" for sd in SIDES], min_team_games, name_suffix="name")
    encode("champion", [f"{sd}_{r}_champion" for sd, r in SLOTS], 0)
    encode("league", [f"{sd}_home_league" for sd in SIDES] + ["league"], 0)
    return games, vocabs


def log_loss(p, y) -> float:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def elo_report(games: pd.DataFrame) -> pd.DataFrame:
    """Log loss of the raw Elo win probabilities per year, vs. always predicting the blue win rate."""
    rows = []
    for year, g in games.groupby("year"):
        y = g.blue_win.to_numpy()
        both = (g.p_blue_team_elo + g.p_blue_roster_elo) / 2
        rows.append({"year": year, "games": len(g), "blue_wr": y.mean(),
                     "base": log_loss(np.full(len(y), y.mean()), y),
                     "team_elo": log_loss(g.p_blue_team_elo.to_numpy(), y),
                     "roster_elo": log_loss(g.p_blue_roster_elo.to_numpy(), y),
                     "average": log_loss(both.to_numpy(), y)})
    return pd.DataFrame(rows)
