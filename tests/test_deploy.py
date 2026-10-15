"""The website's predictions must match the model.

1. Snapshot vs replay: features computed from site/data/snapshot.json for an upcoming game must
   equal the features pro.dataset.build computes when that game is appended to the history.
2. JavaScript vs Python: site/model.js must reproduce pro.deploy.predict on the snapshot's test
   cases (run in V8: pip install mini-racer).

    python -m pytest tests/test_deploy.py      (from the repo root, after python -m pro deploy)
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pro import dataset, deploy, load
from pro import models as M
from pro.load import ROLES

ROOT = Path(__file__).resolve().parents[1]
SNAPSHOT = ROOT / "site" / "data" / "snapshot.json"


def synthetic_game(df, blue, red, champs, date, patch):
    """10 rows in load.load's format for a Worlds game between two (team, players, names) sides."""
    rows = []
    for side, (team, team_name, players), cs in (("Blue", blue, champs[:5]), ("Red", red, champs[5:])):
        for role, pid, c in zip(ROLES, players, cs):
            rows.append({"gameid": "test-game", "date": date, "year": date.year, "league": "WLDs",
                         "split": None, "playoffs": 0, "game": 1, "patch": patch, "side": side,
                         "position": role, "player": pid, "playername": pid, "team": team,
                         "teamname": team_name, "champion": c, "result": int(side == "Blue")})
    out = pd.concat([df, pd.DataFrame(rows)], ignore_index=True)
    out["side"] = pd.Categorical(out.side.astype(str), ["Blue", "Red"], ordered=True)
    out["position"] = pd.Categorical(out.position.astype(str), ROLES, ordered=True)
    return out


@pytest.fixture(scope="module")
def replay():
    df = load.load(ROOT / "pro_play_data")
    games, state = dataset.build(df, return_state=True)
    champions = sorted(set(games[[f"{sd}_{r}_champion" for sd in ("blue", "red") for r in ROLES]]
                           .to_numpy().ravel()))
    strength = {"patches": [], "matches": 0, "values": {}}
    snap = deploy.snapshot(games, state, {"pre": {}, "post": {}, "combiner": {}}, strength, champions)
    return df, games, snap


def test_snapshot_matches_replay(replay):
    df, games, snap = replay
    rng = np.random.default_rng(1)
    recent = games[games.date >= games.date.max() - pd.Timedelta(days=30)]
    g1, g2 = recent.iloc[-1], recent.iloc[len(recent) // 2]
    blue_team, red_team = g1.blue_team, g2.red_team
    blue_players = [g1[f"blue_{r}_player"] for r in ROLES]
    red_players = [g2[f"red_{r}_player"] for r in ROLES]
    blue_players[1] = "test:new-player"                       # never seen before
    other = games[(games.blue_home_league != g1.blue_home_league) & (games.date >= "2026-01-01")].iloc[0]
    blue_players[3] = other.blue_bot_player                   # transfer from another league
    champs = list(rng.choice(snap["champions"], 10, replace=False))
    champs[0] = g1.blue_top_champion                          # a champion the player has history on
    date = games.date.max() + pd.Timedelta(days=2, hours=3)

    big = synthetic_game(df, (blue_team, g1.blue_teamname, blue_players),
                         (red_team, g2.red_teamname, red_players), champs, date, snap["patch"])
    row = dataset.build(big).iloc[[-1]]
    now = deploy.unix_days(date)
    for sd, team, players, cs in (("blue", blue_team, blue_players, champs[:5]),
                                  ("red", red_team, red_players, champs[5:])):
        expected = M.side_features(row, sd).iloc[0]
        got = deploy.side_features(snap, team, players, cs, now)
        for name, value in expected.items():
            assert got[name] == pytest.approx(value, rel=1e-6, abs=1e-4), f"{sd} {name}"


def test_javascript_matches_python():
    mini_racer = pytest.importorskip("py_mini_racer")
    snap = json.loads(SNAPSHOT.read_text())
    ctx = mini_racer.MiniRacer()
    try:
        ctx.eval((ROOT / "site" / "model.js").read_text(encoding="utf-8"))
        ctx.eval(f"var SNAP = {json.dumps(snap)};")
        ctx.eval(f"new Function({json.dumps((ROOT / 'site' / 'app.js').read_text(encoding='utf-8'))})")  # parses
        for case in snap["tests"]:
            (bt, bp, bc), (rt, rp, rc) = case["blue"], case["red"]
            got = json.loads(ctx.eval(
                f"JSON.stringify(LoLModel.predict(SNAP, {json.dumps(bt)}, {json.dumps(rt)}, {json.dumps(bp)}, "
                f"{json.dumps(rp)}, {json.dumps(bc)}, {json.dumps(rc)}, {case['now']}))"))
            for k, v in case["expected"].items():
                assert got[k] == pytest.approx(v, abs=1e-9), k
    finally:
        ctx.close()  # otherwise V8's thread keeps the process alive after the tests
