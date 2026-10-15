"""Using a frozen draft score without torch: score pro drafts from the champion strength table
(python -m draft freeze) and add the score to a pro model's prediction."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold

from . import data


def strength_logit(table: dict, games: pd.DataFrame) -> np.ndarray:
    """Draft logit of pro games from a frozen strength table (unknown champions count as 0)."""
    vals = {k: np.array(v) for k, v in table["values"].items()}
    out = np.zeros(len(games))
    for sign, sd in ((1, "blue"), (-1, "red")):
        for r, role in enumerate(data.PRO_ROLES):
            out += sign * games[f"{sd}_{role}_champion"].map(
                lambda c: vals[data.norm(c)][r] if data.norm(c) in vals else 0.0).to_numpy()
    return out


def with_draft(p, draft, y, seed=0) -> np.ndarray:
    """Out-of-fold a + b * logit(p) + w * draft. The pro model's logit is refitted too (a, b), and
    "alone" gets the same treatment with draft = 0, so the comparison only measures w."""
    base = np.log(p / (1 - p))
    out = np.empty_like(p)
    for tr, te in KFold(5, shuffle=True, random_state=seed).split(p):
        lr = LogisticRegression(C=1e6).fit(np.c_[base[tr], draft[tr]], y[tr])
        out[te] = lr.predict_proba(np.c_[base[te], draft[te]])[:, 1]
    return out
