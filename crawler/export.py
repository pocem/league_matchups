"""Turn the crawl database into one row per match, ready for the draft model."""
from __future__ import annotations

import pandas as pd

from .crawl import APEX_TIERS

ROLES = {"TOP": "top", "JUNGLE": "jungle", "MIDDLE": "mid", "BOTTOM": "bot", "UTILITY": "support"}
SIDES = {100: "blue", 200: "red"}
CHAMP_COLS = [f"{side}_{role}" for side in SIDES.values() for role in ROLES.values()]


def build_table(con, include_remakes=False) -> pd.DataFrame:
    matches = pd.read_sql("SELECT * FROM matches WHERE queue_id = 420", con)
    if not include_remakes:
        matches = matches[matches.remake == 0]
    parts = pd.read_sql("SELECT match_id, puuid, team_id, position, champion_id FROM participants", con)
    tiers = pd.read_sql("SELECT puuid, tier FROM players", con)

    parts = parts[parts.match_id.isin(matches.match_id)].merge(tiers, on="puuid", how="left")
    n_apex = parts.tier.isin(APEX_TIERS).groupby(parts.match_id).sum().rename("n_apex")

    # keep matches where every team has each of the five roles exactly once
    roles = parts[parts.position.isin(ROLES)]
    clean = roles.groupby("match_id").filter(
        lambda g: len(g) == 10 and not g.duplicated(["team_id", "position"]).any())
    wide = clean.pivot(index="match_id", columns=["team_id", "position"], values="champion_id")
    wide.columns = [f"{SIDES[t]}_{ROLES[p]}" for t, p in wide.columns]

    out = (matches.set_index("match_id")
           [["platform", "patch", "game_version", "game_creation", "game_duration", "blue_win"]]
           .join(wide[CHAMP_COLS], how="inner")
           .join(n_apex))
    out["game_creation"] = pd.to_datetime(out.game_creation, unit="ms", utc=True)
    return out.reset_index().sort_values("game_creation", ignore_index=True)


def champions(con) -> pd.DataFrame:
    return pd.read_sql("""SELECT champion_id, MAX(champion_name) AS champion_name
                          FROM participants GROUP BY champion_id ORDER BY champion_name""", con)


def print_stats(con):
    q = lambda sql: con.execute(sql).fetchall()
    print("players on ladder:")
    for platform, tier, n in q("SELECT platform, tier, COUNT(*) FROM players GROUP BY 1, 2 ORDER BY 1, 2"):
        print(f"  {platform:6} {tier:12} {n:>8,}")
    print("\nmatch queue (pending / stored / not found / not solo / failed):")
    for routing, *counts in q("""SELECT routing, SUM(status = 0), SUM(status = 1), SUM(status = 2),
                                       SUM(status = 3), SUM(status = 4)
                                FROM match_queue GROUP BY 1"""):
        print(f"  {routing:9}" + "".join(f"{c:>10,}" for c in counts))
        # one request per match at 100 requests / 2 min on a dev or personal key
        print(f"  {'':9} ~{counts[0] / 72_000:.1f} days left at dev-key rate")
    print("\nstored matches by patch (remakes excluded):")
    for patch, n in q("SELECT patch, COUNT(*) FROM matches WHERE remake = 0 GROUP BY 1 ORDER BY 1"):
        print(f"  {patch:8} {n:>9,}")
