"""Point-in-time strength ratings and running stats for pro games.

Ratings are Elo on the usual 400-point logistic scale. Every entity (player, team) starts at
BASE with a large K that decays towards K_MIN as it plays, so new entities move fast and
established ones are stable. At each new season ratings are pulled part of the way back to
BASE, because rosters and form change over the off-season. Patches do not touch ratings.

Leagues mostly play in closed pools, so a 1600 team in the LCK and a 1600 team in the LFL
are not comparable. A separate league rating fixes this: a side's effective strength is
entity rating + league offset of its home league. In domestic games the league terms cancel;
the league ratings only learn from cross-league games (MSI, Worlds, EWC, EM, KeSPA Cup, ...).
When a player or team changes home league, its rating is shifted by the offset difference so
its effective strength stays the same. Caveat: leagues that never meet (the ERLs, linked only
through EM, vs. the majors) form separate pools whose offsets are not comparable, so the shift
is wrong for e.g. LFL -> LEC moves. Measured, it still beats no shift or re-raising K, because
most moves happen inside a connected pool (LDL -> LPL, LCKC -> LCK, ...).
"""
from __future__ import annotations

import math
from collections import Counter, defaultdict
from dataclasses import dataclass

BASE = 1500.0

# Multi-league events: they don't define a team's home league.
EVENTS = {"MSI", "WLDs", "EWC", "FST", "EM", "KeSPA Cup", "DCup", "AC", "ASI", "WLRQ", "AmCh", "IC"}


def expected(diff: float) -> float:
    return 1.0 / (1.0 + 10.0 ** (-diff / 400.0))


@dataclass
class EloParams:
    # defaults tuned on 2025 log loss of the raw Elo probabilities (2024 = burn-in)
    k_max: float = 120.0         # K for an entity's first game
    k_min: float = 28.0          # K once established
    k_tau: float = 30.0          # games for the K boost to decay by 1/e
    season_keep: float = 0.8     # share of (rating - BASE) kept at a new season
    k_league: float = 24.0


class Elo:
    def __init__(self, params: EloParams):
        self.p = params
        self.rating = defaultdict(lambda: BASE)
        self.games = Counter()

    def k(self, key) -> float:
        p = self.p
        return p.k_min + (p.k_max - p.k_min) * math.exp(-self.games[key] / p.k_tau)

    def update(self, key, delta_score: float):
        """delta_score = actual - expected for this entity's side."""
        self.rating[key] += self.k(key) * delta_score
        self.games[key] += 1

    def new_season(self):
        keep = self.p.season_keep
        for key, r in self.rating.items():
            self.rating[key] = BASE + keep * (r - BASE)


# Renamed leagues keep one rating.
LEAGUE_ALIASES = {"LTA N": "LCS", "LTA S": "CBLOL"}


def home_league(league: str, last_home: str | None) -> str:
    """The league a team represents: the game's league, or its last domestic league at an event."""
    league = LEAGUE_ALIASES.get(league, league)
    if league in EVENTS or league == "LTA":  # LTA = cross-conference games between LTA N and LTA S
        return last_home or league
    return league


class Decayed:
    """Exponentially time-decayed (games, wins) per key. half_life_days=inf means no decay."""

    def __init__(self, half_life_days: float):
        self.rate = math.log(2) / half_life_days if math.isfinite(half_life_days) else 0.0
        self.state = {}  # key -> (games, wins, t_days)

    def get(self, key, t: float) -> tuple[float, float]:
        if key not in self.state:
            return 0.0, 0.0
        g, w, t0 = self.state[key]
        f = math.exp(-self.rate * (t - t0))
        return g * f, w * f

    def add(self, key, t: float, win: float):
        g, w = self.get(key, t)
        self.state[key] = (g + 1, w + win, t)


def shrunk(wins: float, games: float, prior: float = 0.5, strength: float = 10.0) -> float:
    """Win rate pulled towards prior; with no games it is the prior."""
    return (wins + prior * strength) / (games + strength)
