"""Live predictions: refit on all data and export a snapshot that the website computes from.

    python -m pro deploy      (after downloading the newest Oracle's Elixir CSV)

1. Replay every pro game (pro.dataset.build) and keep the final ratings and counts.
2. Refit two logistic regressions on Jul 2024 - today: pre-draft (no champion features) and
   post-draft (all features).
3. Combining layer for the draft: final logit = a + b * post-draft logit + w * draft logit,
   fitted on recent pro games (patches close to the solo queue data) using predictions of a
   model that never saw them (trained up to Dec 2025), so the pro model isn't overrated.
4. Export site/data/snapshot.json: ratings, rosters, champion stats, model weights, the frozen
   champion strength table (python -m draft freeze) and test cases for the JavaScript.

`features()` below is the reference implementation of what site/model.js computes; the tests
check it against a full replay and against the JavaScript.
"""
from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from . import dataset, load
from . import models as M
from .load import ROLES
from .ratings import BASE, LEAGUE_ALIASES

LIVE_TRAIN_START = "2024-07-01"
OOS_TRAIN_END = "2026-01-01"         # the model whose predictions the combining layer is fitted on
COMBINER_MIN_PATCH = "16.18"         # pro games this close to the solo queue patches
ACTIVE_DAYS = 365                    # teams and players listed on the site
CHAMPION_FEATURES = ("pc_games", "pc_days", "meta_wr", "meta_games", "patch_games")
SHRINK_PRIOR, SHRINK_STRENGTH = 0.5, 10.0
DAY_MS = 86_400_000


def unix_days(ts) -> float:
    return pd.Timestamp(ts).value / 1e6 / DAY_MS


# ---------------------------------------------------------------- reference feature computation

def side_features(snap: dict, team: str, players: list, champs: list | None, now: float) -> dict:
    """One side's features from the snapshot, named like pro.models.side_features."""
    t = snap["teams"][team]
    off = snap["leagues"].get(t["home"], 0.0)
    f = {"team_elo": t["rating"] + off, "league_offset": off, "log_team_games": math.log1p(t["games"])}
    elos = []
    for r, (role, pid) in enumerate(zip(ROLES, players)):
        p = snap["players"].get(pid)
        elo = p["rating"] + snap["leagues"].get(p["home"], 0.0) if p else BASE + off
        elos.append(elo)
        f[f"{role}_player_elo"] = elo
        f[f"{role}_log_player_games"] = math.log1p(p["games"] if p else 0)
        if champs is None:
            continue
        c = champs[r]
        games, last = snap["pc"].get(pid, {}).get(c, (0, None))
        f[f"{role}_log_pc_games"] = math.log1p(games)
        f[f"{role}_log_pc_days"] = math.log1p(now - last if last is not None else M.NEVER_DAYS)
        decay = math.exp(-snap["meta"]["rate"] * (now - snap["meta"]["as_of"]))
        mg, mw = snap["meta"]["values"].get(c, [[0.0, 0.0]] * 5)[r]
        f[f"{role}_meta_wr"] = (mw * decay + SHRINK_PRIOR * SHRINK_STRENGTH) / (mg * decay + SHRINK_STRENGTH)
        f[f"{role}_log_meta_games"] = math.log1p(mg * decay)
        f[f"{role}_log_patch_games"] = math.log1p(snap["patch_games"].get(c, [0] * 5)[r])
    f["roster_elo"] = float(np.mean(elos))
    return f


def linear(model: dict, blue: dict, red: dict) -> float:
    x = np.array([blue[c[5:]] - red[c[5:]] for c in model["features"]])  # "diff_" + side feature
    return float(model["intercept"] + np.dot(model["coef"], (x - model["mean"]) / model["scale"]))


def draft_logit(snap: dict, blue_champs: list, red_champs: list) -> float:
    v = snap["draft"]["values"]
    return float(sum(v.get(b, [0.0] * 5)[r] - v.get(c, [0.0] * 5)[r]
                     for r, (b, c) in enumerate(zip(blue_champs, red_champs))))


def predict(snap, blue_team, red_team, blue_players, red_players, blue_champs=None, red_champs=None,
            now=None) -> dict:
    """P(blue wins): pre-draft, and with champions also post-draft and final (with draft score)."""
    now = snap["data_until"] if now is None else now
    sig = lambda z: 1 / (1 + math.exp(-z))
    m = snap["models"]
    b = side_features(snap, blue_team, blue_players, None, now)
    r = side_features(snap, red_team, red_players, None, now)
    out = {"pre": sig(linear(m["pre"], b, r))}
    if blue_champs and red_champs:
        b = side_features(snap, blue_team, blue_players, blue_champs, now)
        r = side_features(snap, red_team, red_players, red_champs, now)
        z = linear(m["post"], b, r)
        d = draft_logit(snap, blue_champs, red_champs)
        k = m["combiner"]
        out.update(post=sig(z), draft=d, final=sig(k["a"] + k["b"] * z + k["w"] * d))
    return out


# ---------------------------------------------------------------- fitting

def lr_cols(games, champions: bool) -> list:
    cols = M.logreg_cols(games)
    return cols if champions else [c for c in cols if not any(k in c for k in CHAMPION_FEATURES)]


def fit_lr(rows, cols, C) -> dict:
    pipe = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=5000))
    pipe.fit(M.tabular(rows)[cols], rows.blue_win)
    sc, lr = pipe[0], pipe[-1]
    return {"features": cols, "mean": sc.mean_.tolist(), "scale": sc.scale_.tolist(),
            "coef": lr.coef_[0].tolist(), "intercept": float(lr.intercept_[0]), "_pipe": pipe}


def fit_combiner(games, strength: dict, C) -> dict:
    from draft.__main__ import strength_logit  # numpy is already loaded, so torch import order is fine
    from draft.patch_eval import with_draft

    cols = lr_cols(games, True)
    train = games[(games.date >= LIVE_TRAIN_START) & (games.date < OOS_TRAIN_END)]
    recent = games[(games.date >= OOS_TRAIN_END) & (games.patch >= COMBINER_MIN_PATCH)]
    oos = fit_lr(train, cols, C)["_pipe"]
    p = oos.predict_proba(M.tabular(recent)[cols])[:, 1].clip(1e-6, 1 - 1e-6)
    z, d, y = np.log(p / (1 - p)), strength_logit(strength, recent), recent.blue_win.to_numpy()
    k = LogisticRegression(C=1e6).fit(np.c_[z, d], y)
    ll = lambda q: float(-np.mean(y * np.log(q) + (1 - y) * np.log(1 - q)))
    return {"a": float(k.intercept_[0]), "b": float(k.coef_[0][0]), "w": float(k.coef_[0][1]),
            "fitted_on_games": int(len(y)), "patches": sorted(recent.patch.unique().tolist()),
            "cv_log_loss_without_draft": ll(with_draft(p, np.zeros(len(p)), y)),
            "cv_log_loss_with_draft": ll(with_draft(p, d, y))}


# ---------------------------------------------------------------- snapshot

def snapshot(games, state, models: dict, strength: dict, champions: list) -> dict:
    t0 = unix_days(state["t0"])
    end = games.date.max()
    active = games[games.date >= end - pd.Timedelta(days=ACTIVE_DAYS)]
    patch = games.patch.max()  # the newest patch seen (zero-padded, so string order works)
    pe, te, lg = state["player_elo"], state["team_elo"], state["league_elo"]

    teams, worlds = {}, set()
    for _, g in active.iterrows():  # in date order, so the last game wins
        for sd in ("blue", "red"):
            teams[g[f"{sd}_team"]] = {
                "name": g[f"{sd}_teamname"], "league": LEAGUE_ALIASES.get(g.league, g.league),
                "roster": [g[f"{sd}_{r}_player"] for r in ROLES], "last_game": g.date.strftime("%Y-%m-%d")}
            if g.league == "WLDs" and g.year == end.year:
                worlds.add(g[f"{sd}_team"])
    for tid, t in teams.items():
        t.update(rating=te.rating[tid], games=te.games[tid], home=state["last_home"][tid], worlds=tid in worlds)

    player_ids = set(active[[f"{sd}_{r}_player" for sd in ("blue", "red") for r in ROLES]].to_numpy().ravel())
    names, team_of = {}, {}
    for _, g in active.iterrows():  # date order: the latest name and team win
        for sd in ("blue", "red"):
            for r in ROLES:
                names[g[f"{sd}_{r}_player"]] = g[f"{sd}_{r}_playername"]
                team_of[g[f"{sd}_{r}_player"]] = g[f"{sd}_teamname"]
    players = {pid: {"name": names[pid], "team": team_of[pid], "rating": pe.rating[pid],
                     "games": pe.games[pid], "home": state["last_home"][pid]} for pid in player_ids}

    pc = {}
    for (pid, champ), n in state["pc_games"].items():
        if pid in player_ids:
            pc.setdefault(pid, {})[champ] = [n, round(t0 + state["pc_last"][(pid, champ)], 4)]

    t_end = state["t_end"]
    meta = {c: [list(state["meta"].get((c, r), t_end)) for r in ROLES] for c in champions}
    patch_games = {c: [state["patch_games"][(c, r, patch)] for r in ROLES] for c in champions}

    from draft.data import norm
    draft_values = {c: strength["values"][norm(c)] for c in champions if norm(c) in strength["values"]}
    return {
        "generated": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "data_until": round(t0 + t_end, 4), "data_until_date": end.strftime("%Y-%m-%d"), "patch": patch,
        "leagues": dict(lg), "teams": teams, "players": players, "pc": pc,
        "meta": {"rate": state["meta"].rate, "as_of": round(t0 + t_end, 4), "values": meta},
        "patch_games": patch_games, "champions": champions, "roles": ROLES,
        "draft": {"patches": strength["patches"], "matches": strength["matches"], "values": draft_values},
        "constants": {"base": BASE, "never_days": M.NEVER_DAYS, "shrink_prior": SHRINK_PRIOR,
                      "shrink_strength": SHRINK_STRENGTH},
        "models": {k: {kk: vv for kk, vv in v.items() if not kk.startswith("_")} if k != "combiner" else v
                   for k, v in models.items()},
    }


def test_cases(snap: dict, n=30, seed=0) -> list:
    """Random matchups with Python's answers, for checking the JavaScript."""
    rng = np.random.default_rng(seed)
    teams, champs = sorted(snap["teams"]), snap["champions"]
    cases = []
    for i in range(n):
        bt, rt = rng.choice(teams, 2, replace=False)
        bp, rp = list(snap["teams"][bt]["roster"]), list(snap["teams"][rt]["roster"])
        if i % 5 == 0:  # an unknown substitute
            bp[2] = "unknown-player"
        bc, rc = [list(x) for x in rng.choice(champs, (2, 5), replace=False)]
        now = snap["data_until"] + float(rng.uniform(0, 10))
        cases.append({"blue": [str(bt), bp, bc], "red": [str(rt), rp, rc], "now": now,
                      "expected": predict(snap, bt, rt, bp, rp, bc, rc, now)})
    return cases


def deploy(src="pro_play_data", models_dir=Path("models"), site_dir=Path("site")):
    strength_path = models_dir / "draft_strength.json"
    if not strength_path.exists():
        raise SystemExit(f"{strength_path} missing: run `python -m draft freeze` first")
    strength = json.loads(strength_path.read_text())
    C = json.loads((models_dir / "tuned_logistic_regression.json").read_text())["params"]["C"]

    df = load.load(src)
    games, state = dataset.build(df, return_state=True)
    rows = games[games.date >= LIVE_TRAIN_START]
    fitted = {"pre": fit_lr(rows, lr_cols(games, False), C), "post": fit_lr(rows, lr_cols(games, True), C)}
    fitted["combiner"] = fit_combiner(games, strength, C)
    champions = sorted(set(games[[f"{sd}_{r}_champion" for sd in ("blue", "red") for r in ROLES]]
                           .to_numpy().ravel()))

    snap = snapshot(games, state, fitted, strength, champions)
    # the exported weights must reproduce sklearn exactly
    for k in ("pre", "post"):
        x = M.tabular(rows.tail(200))[fitted[k]["features"]]
        z = fitted[k]["_pipe"].decision_function(x)
        mine = fitted[k]["intercept"] + ((x.to_numpy() - fitted[k]["mean"]) / fitted[k]["scale"]) @ fitted[k]["coef"]
        assert np.allclose(z, mine), f"{k}: exported weights don't reproduce sklearn"
    snap["tests"] = test_cases(snap)

    out = site_dir / "data" / "snapshot.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(snap, separators=(",", ":")))
    k = fitted["combiner"]
    print(f"trained on {len(rows):,} games ({LIVE_TRAIN_START} to {snap['data_until_date']}), patch {snap['patch']}")
    print(f"draft combining layer on {k['fitted_on_games']} recent games {k['patches']}: "
          f"a {k['a']:+.3f}  b {k['b']:.3f}  w {k['w']:.3f}; CV log loss "
          f"{k['cv_log_loss_without_draft']:.4f} -> {k['cv_log_loss_with_draft']:.4f}")
    print(f"{len(snap['teams'])} teams ({sum(t['worlds'] for t in snap['teams'].values())} at Worlds), "
          f"{len(snap['players'])} players -> {out} ({out.stat().st_size / 1e6:.1f} MB)")
    return snap
