"""Compare models on the pro game table, tested on 2026.

Models with a stopping point (trees, epochs) pick it on train = Jul 2024 - Jun 2025 vs.
valid = Jul - Dec 2025, then refit on all of Jul 2024 - Dec 2025 with that setting. The first
half of 2024 is left out as Elo burn-in."""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import sklearn
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch import nn

from .load import ROLES

SIDES = ["blue", "red"]
TRAIN_START, VALID_START, TEST_START = "2024-07-01", "2025-07-01", "2026-01-01"
MAJOR = {"LCK", "LPL", "LEC", "LCS", "LTA N", "LCP", "CBLOL", "MSI", "WLDs", "EWC", "FST"}
NEVER_DAYS = 3000.0  # pc_days_since for a champion the player never played


# ---------------------------------------------------------------- features

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


def split(g: pd.DataFrame):
    train = g[(g.date >= TRAIN_START) & (g.date < VALID_START)]
    valid = g[(g.date >= VALID_START) & (g.date < TEST_START)]
    test = g[g.date >= TEST_START]
    return train, valid, test


# ---------------------------------------------------------------- classic models

# Each fit_* returns (test predictions, fitted model), fitted on train + valid. Hyperparameters
# default to the hand-picked values below; `python -m pro tune` searches for better ones.

LOGREG_DEFAULTS = {"C": 0.1}
LGBM_DEFAULTS = dict(learning_rate=0.02, num_leaves=15, min_child_samples=60, feature_fraction=0.6,
                     bagging_fraction=0.8, bagging_freq=1, lambda_l2=5.0)
LGBM_FIXED = dict(objective="binary", cat_smooth=50, min_data_per_group=100, verbose=-1, seed=0)
NN_DEFAULTS = dict(hidden=64, n_layers=2, dropout=0.3, weight_decay=1e-3, lr=1e-3, batch_size=256)
NN_ARCH = ("hidden", "n_layers", "dropout")  # the rest of NN_DEFAULTS are training settings


def logreg_cols(g):
    return [c for c in tabular(g.head()).columns if c.startswith("diff_")]


def fit_logreg(tr, va, te, params=None):
    p = {**LOGREG_DEFAULTS, **(params or {})}
    full = pd.concat([tr, va])
    cols = logreg_cols(tr)
    m = make_pipeline(StandardScaler(), LogisticRegression(C=p["C"], max_iter=5000))
    m.fit(tabular(full)[cols], full.blue_win)
    return m.predict_proba(tabular(te)[cols])[:, 1], m


def lgbm_early_stop(params, x_tr, y_tr, x_va, y_va):
    """Train with early stopping on the validation set; returns the booster (best_iteration set)."""
    dtr = lgb.Dataset(x_tr, y_tr)
    dva = lgb.Dataset(x_va, y_va, reference=dtr)
    return lgb.train({**LGBM_FIXED, **params}, dtr, 10000, valid_sets=[dva],
                     callbacks=[lgb.early_stopping(200, verbose=False)])


def fit_lgbm(tr, va, te, champions=False, params=None):
    p = {**LGBM_DEFAULTS, **(params or {})}
    m = lgbm_early_stop(p, tabular(tr, champions), tr.blue_win, tabular(va, champions), va.blue_win)
    full = pd.concat([tr, va])
    m = lgb.train({**LGBM_FIXED, **p}, lgb.Dataset(tabular(full, champions), full.blue_win), m.best_iteration)
    return m.predict(tabular(te, champions)), m


# ---------------------------------------------------------------- neural net

class SideScorer(nn.Module):
    """Scores one side against the other: own and opponent stats, champions and players."""

    def __init__(self, n_num, n_champ, n_player, d_champ, d_player, hidden, n_layers, dropout):
        super().__init__()
        self.champ = nn.Embedding(n_champ, d_champ) if d_champ else None
        self.player = nn.Embedding(n_player, d_player) if d_player else None
        self.emb_drop = nn.Dropout(dropout / 2)
        width = 2 * n_num + 15 * d_champ + 10 * d_player  # champs: own, opp, lane products
        layers = []
        for _ in range(n_layers):
            layers += [nn.Linear(width, hidden), nn.ReLU(), nn.Dropout(dropout)]
            width = hidden
        self.mlp = nn.Sequential(*layers, nn.Linear(width, 1))

    def forward(self, num_own, num_opp, ch_own, ch_opp, pl_own, pl_opp):
        parts = [num_own, num_opp]
        if self.champ is not None:
            own, opp = self.emb_drop(self.champ(ch_own)), self.emb_drop(self.champ(ch_opp))  # (B, 5, d)
            parts += [own.flatten(1), opp.flatten(1), (own * opp).flatten(1)]                # lane matchups
        if self.player is not None:
            parts += [self.emb_drop(self.player(pl_own)).flatten(1), self.emb_drop(self.player(pl_opp)).flatten(1)]
        return self.mlp(torch.cat(parts, 1)).squeeze(1)


class DraftNet(nn.Module):
    """logit(blue wins) = s(blue, red) - s(red, blue) + blue bias: swapping sides flips the
    prediction exactly, so the net doesn't have to learn that from data."""

    def __init__(self, **kw):
        super().__init__()
        self.scorer = SideScorer(**kw)
        self.blue_bias = nn.Parameter(torch.zeros(1))

    def forward(self, num, ch, pl):
        a = self.scorer(num[:, 0], num[:, 1], ch[:, 0], ch[:, 1], pl[:, 0], pl[:, 1])
        b = self.scorer(num[:, 1], num[:, 0], ch[:, 1], ch[:, 0], pl[:, 1], pl[:, 0])
        return a - b + self.blue_bias


def nn_arrays(g, mean=None, std=None):
    num = np.stack([side_features(g, sd).to_numpy(np.float32) for sd in SIDES], 1)  # (n, 2, F)
    if mean is None:
        flat = num.reshape(-1, num.shape[2])
        mean, std = flat.mean(0), flat.std(0) + 1e-6
    num = (num - mean) / std
    idx = lambda kind: np.stack([g[[f"{sd}_{r}_{kind}_idx" for r in ROLES]].to_numpy() for sd in SIDES], 1)
    tensors = (torch.tensor(num), torch.tensor(idx("champion"), dtype=torch.long),
               torch.tensor(idx("player"), dtype=torch.long), torch.tensor(g.blue_win.to_numpy(), dtype=torch.float32))
    return tensors, mean, std


def train_nn(X, arch: dict, p: dict, seed: int, n_epochs: int, val=None, patience=10):
    """Train one DraftNet. With val, stops early and returns (model, best epoch, best val loss)."""
    loss_fn = nn.BCEWithLogitsLoss()
    torch.manual_seed(seed)
    model = DraftNet(n_num=X[0].shape[2], **arch)
    opt = torch.optim.AdamW(model.parameters(), lr=p["lr"], weight_decay=p["weight_decay"])
    best, best_epoch = np.inf, 0
    for epoch in range(1, n_epochs + 1):
        model.train()
        for b in torch.randperm(len(X[3])).split(p["batch_size"]):
            opt.zero_grad()
            loss_fn(model(X[0][b], X[1][b], X[2][b]), X[3][b]).backward()
            opt.step()
        if val is not None:
            model.eval()
            with torch.no_grad():
                v = loss_fn(model(*val[:3]), val[3]).item()
            if v < best - 1e-5:
                best, best_epoch = v, epoch
            elif epoch - best_epoch >= patience:
                break
    return model, best_epoch, best


def fit_nn(tr, va, te, n_champ, n_player, d_champ=0, d_player=0, seeds=(0, 1, 2), params=None,
           epochs=150):
    """Per seed: find the best epoch count on validation, refit on train + valid for that many
    epochs. Predicts the mean over seeds; the model is the seeds' weights + input scaling."""
    p = {**NN_DEFAULTS, **(params or {})}
    arch = dict(n_champ=n_champ, n_player=n_player, d_champ=d_champ, d_player=d_player,
                **{k: p[k] for k in NN_ARCH})
    trX, mean, std = nn_arrays(tr)
    vaX, _, _ = nn_arrays(va, mean, std)
    full = pd.concat([tr, va])
    fullX, mean, std = nn_arrays(full)
    teX, _, _ = nn_arrays(te, mean, std)

    preds, states, n_epochs = [], [], []
    for seed in seeds:
        _, n_best, _ = train_nn(trX, arch, p, seed, epochs, val=vaX)
        model, _, _ = train_nn(fullX, arch, p, seed, max(n_best, 1))
        model.eval()
        with torch.no_grad():
            preds.append(torch.sigmoid(model(*teX[:3])).numpy())
        states.append(model.state_dict())
        n_epochs.append(n_best)
    bundle = {"config": {"n_num": fullX[0].shape[2], **arch}, "params": p, "state_dicts": states, "epochs": n_epochs,
              "mean": mean, "std": std, "features": list(side_features(tr.head(), "blue").columns)}
    return np.mean(preds, 0), bundle


# ---------------------------------------------------------------- comparison

def scores(y, p, major) -> dict:
    return {"log_loss": log_loss(y, p), "brier": brier_score_loss(y, p), "accuracy": accuracy_score(y, p > 0.5),
            "auc": roc_auc_score(y, p), "log_loss_major": log_loss(y[major], p[major])}


def compare(games: pd.DataFrame, vocab_sizes: dict, params: dict | None = None):
    """params: {model name: hyperparameters}, e.g. from tuning; missing models use defaults."""
    params = params or {}
    tr, va, te = split(games)
    n_champ, n_player = vocab_sizes["champion"] + 1, vocab_sizes["player"] + 1
    blue_rate = pd.concat([tr, va]).blue_win.mean()
    # Champion and player embeddings (fit_nn d_champ / d_player) wait for the solo queue
    # pretraining; on pro data alone they didn't beat the stats-only net.
    models = {
        "always blue rate": lambda: (np.full(len(te), blue_rate), None),
        "raw roster Elo": lambda: (te.p_blue_roster_elo.to_numpy(), None),
        "logistic_regression": lambda: fit_logreg(tr, va, te, params=params.get("logistic_regression")),
        "lightgbm": lambda: fit_lgbm(tr, va, te, params=params.get("lightgbm")),
        "neural_net": lambda: fit_nn(tr, va, te, n_champ, n_player, params=params.get("neural_net")),
    }
    y = te.blue_win.to_numpy()
    major = te.league.isin(MAJOR).to_numpy()
    preds, rows, fitted = {}, [], {}
    for name, fit in models.items():
        print(f"  fitting {name} ...", flush=True)
        p, model = fit()
        preds[name] = p = np.clip(p, 1e-6, 1 - 1e-6)
        if model is not None:
            fitted[name] = model
        rows.append({"model": name, **scores(y, p, major)})
    table = pd.DataFrame(rows).set_index("model")
    pred_df = te[["gameid", "date", "league", "blue_teamname", "red_teamname", "blue_win"]].copy()
    for name, p in preds.items():
        pred_df[name] = p
    print(f"train {len(tr):,}  valid {len(va):,}  test {len(te):,} games ({major.sum():,} in major leagues)")
    return table, pred_df, fitted


def save(fitted: dict, table: pd.DataFrame, out_dir: Path, params: dict | None = None):
    """Write the fitted models plus a metadata.json saying how they were trained and scored."""
    out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(fitted["logistic_regression"], out_dir / "logistic_regression.joblib")
    fitted["lightgbm"].save_model(str(out_dir / "lightgbm.txt"))
    torch.save(fitted["neural_net"], out_dir / "neural_net.pt")
    meta = {
        "trained_on": f"{TRAIN_START} up to {TEST_START}",
        "stopping_point_chosen_on": f"train {TRAIN_START} - {VALID_START} vs valid {VALID_START} - {TEST_START}",
        "tested_on": f"from {TEST_START}",
        "input": "rows of data/pro/games.parquet, features via pro.models.tabular and nn_arrays",
        "hyperparameters": {
            "logistic_regression": {**LOGREG_DEFAULTS, **(params or {}).get("logistic_regression", {})},
            "lightgbm": {**LGBM_DEFAULTS, **(params or {}).get("lightgbm", {}),
                         "n_trees": fitted["lightgbm"].num_trees()},
            "neural_net": {**fitted["neural_net"]["params"], "epochs": fitted["neural_net"]["epochs"]},
        },
        "test_scores": table.loc[list(fitted)].round(4).to_dict("index"),
        "versions": {"sklearn": sklearn.__version__, "lightgbm": lgb.__version__, "torch": torch.__version__},
    }
    (out_dir / "metadata.json").write_text(json.dumps(meta, indent=2))
