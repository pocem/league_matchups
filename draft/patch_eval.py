"""Does a draft model trained on the pro game's own patch help more than one trained on all patches?

For every patch with enough solo queue games: train draft models on that patch only (the
champion-strength baseline and the network config chosen by `draft train`, picked on a time
holdout of that patch), then score the pro games played on that patch.

On those pro games, three predictions are compared:
    pro model alone                    (data/pro/test_predictions.csv from `pro compare --no-draft`)
    + all-patch draft score            (data/pro/draft_scores.parquet from `draft score`)
    + same-patch draft score
"+ draft" adds w * draft_logit to the pro model's logit, with w fitted out of fold (5-fold
cross-validation over games), so no game is scored with a weight fitted on itself.
"""
from __future__ import annotations

import numpy as np  # before torch, see __main__
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

from . import data, model
from .combine import with_draft

PRO_MODELS = ["logistic_regression", "lightgbm", "neural_net"]


def per_patch_scores(matches, champions, games, net_cfg, min_games) -> pd.DataFrame:
    voc = data.vocab(champions)
    rows = []
    for patch, solo in matches.groupby("patch"):
        pro = games[games.patch == patch]
        if len(solo) < min_games or pro.empty:
            continue
        solo = solo.sort_values("game_creation")
        ch = torch.tensor(data.solo_drafts(solo, champions, voc), dtype=torch.long)
        y = torch.tensor(solo.blue_win.to_numpy(), dtype=torch.float32)
        cut = int(len(y) * 0.9)
        candidates = [{"model": {"kind": "strength", "n_champ": len(voc) + 1}, "lr": 3e-3,
                       "weight_decay": 0.0, "batch_size": 512},
                      {**net_cfg, "model": {**net_cfg["model"], "n_champ": len(voc) + 1}}]
        fits = [(model.fit(ch[:cut], y[:cut], c, val=(ch[cut:], y[cut:])), c) for c in candidates]
        (m, _, val), cfg = min(fits, key=lambda f: f[0][2])
        m.eval()
        with torch.no_grad():
            logit = m.draft_logit(torch.tensor(data.pro_drafts(pro, voc), dtype=torch.long)).numpy()
        print(f"  patch {patch}: {len(solo):,} solo games -> {cfg['model']['kind']} (holdout {val:.4f}), "
              f"{len(pro)} pro games")
        rows.append(pd.DataFrame({"gameid": pro.gameid, "patch_draft_logit": logit}))
    return pd.concat(rows) if rows else pd.DataFrame(columns=["gameid", "patch_draft_logit"])


def evaluate(scored: pd.DataFrame, rng_seed=0) -> pd.DataFrame:
    y = scored.blue_win.to_numpy()
    ll = lambda p: -(y * np.log(p) + (1 - y) * np.log(1 - p))
    rng = np.random.default_rng(rng_seed)
    boot = rng.integers(0, len(y), (2000, len(y)))
    rows = []
    for m in PRO_MODELS:
        p = scored[m].clip(1e-6, 1 - 1e-6).to_numpy()
        alone = with_draft(p, np.zeros(len(p)), y)  # same recalibration, no draft: fair baseline
        variants = {"alone": alone,
                    "+ all-patch draft": with_draft(p, scored.draft_logit.to_numpy(), y),
                    "+ same-patch draft": with_draft(p, scored.patch_draft_logit.to_numpy(), y)}
        for name, q in variants.items():
            d = ll(q) - ll(alone)
            rows.append({"model": m, "variant": name, "log_loss": ll(q).mean(), "vs_alone": d.mean(),
                         "ci_low": np.percentile(d[boot].mean(1), 2.5),
                         "ci_high": np.percentile(d[boot].mean(1), 97.5)})
    return pd.DataFrame(rows)


def run(data_dir, model_dir, min_games):
    matches = pd.read_parquet(data_dir / "matches.parquet")
    champions = pd.read_csv(data_dir / "champions.csv")
    games = pd.read_parquet(data_dir / "pro" / "games.parquet")
    net_cfg = torch.load(model_dir / "draft_net.pt", weights_only=False)["cfg"]
    print("solo games per patch:", matches.patch.value_counts().sort_index().to_dict())

    scores = per_patch_scores(matches, champions, games, net_cfg, min_games)
    if scores.empty:
        print(f"no patch has both >= {min_games:,} solo games and pro games yet")
        return
    preds = pd.read_csv(data_dir / "pro" / "test_predictions.csv")
    scored = (preds.merge(scores, on="gameid")
              .merge(pd.read_parquet(data_dir / "pro" / "draft_scores.parquet"), on="gameid"))
    print(f"\n{len(scored)} pro test games on patches with same-patch draft data")
    print("AUC of the draft score alone:  all-patch "
          f"{roc_auc_score(scored.blue_win, scored.draft_logit):.4f}   same-patch "
          f"{roc_auc_score(scored.blue_win, scored.patch_draft_logit):.4f}")
    with pd.option_context("display.float_format", "{:+.4f}".format, "display.width", 200):
        print(evaluate(scored).to_string(index=False))
