"""Draft-only win model trained on high-elo solo queue: P(blue wins | 10 champions by role).

Same antisymmetric form as the pro net: logit = s(blue, red) - s(red, blue) + blue bias. The
scorer sees its own 5 champion embeddings, the opponent's 5, and the per-lane products
(own_r * opp_r), so it can learn champion strength, team synergy and lane counters.
`draft_logit` leaves out the blue bias, because side balance in solo queue isn't pro side balance.
"""
from __future__ import annotations

import copy
import itertools

import numpy as np
import torch
from torch import nn


class Scorer(nn.Module):
    def __init__(self, n_champ, dim, hidden, dropout):
        super().__init__()
        self.emb = nn.Embedding(n_champ, dim)
        self.drop = nn.Dropout(dropout)
        width, layers = 15 * dim, []
        for h in hidden:
            layers += [nn.Linear(width, h), nn.ReLU(), nn.Dropout(dropout)]
            width = h
        self.mlp = nn.Sequential(*layers, nn.Linear(width, 1))

    def forward(self, own, opp):
        a, b = self.drop(self.emb(own)), self.drop(self.emb(opp))  # (B, 5, dim)
        return self.mlp(torch.cat([a.flatten(1), b.flatten(1), (a * b).flatten(1)], 1)).squeeze(1)


class ChampionStrength(nn.Module):
    """Baseline without synergy or counters: one learned value per champion and role."""

    def __init__(self, n_champ, **_):
        super().__init__()
        self.w = nn.Embedding(n_champ, 5)
        nn.init.zeros_(self.w.weight)

    def forward(self, own, opp):
        return self.w(own).gather(2, torch.arange(5).expand(len(own), 5).unsqueeze(2)).sum((1, 2))


class DraftModel(nn.Module):
    def __init__(self, kind="net", **kw):
        super().__init__()
        self.scorer = Scorer(**kw) if kind == "net" else ChampionStrength(**kw)
        self.blue_bias = nn.Parameter(torch.zeros(1))

    def draft_logit(self, ch):  # ch: (B, 2, 5)
        return self.scorer(ch[:, 0], ch[:, 1]) - self.scorer(ch[:, 1], ch[:, 0])

    def forward(self, ch):
        return self.draft_logit(ch) + self.blue_bias


def fit(ch, y, cfg, val=None, epochs=None, seed=0, max_epochs=60, patience=5):
    """Train one model. With val: early stopping, returns (model, best epoch, best val loss)."""
    torch.manual_seed(seed)
    model = DraftModel(**cfg["model"])
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    loss_fn = nn.BCEWithLogitsLoss()
    best, best_state, best_epoch = np.inf, None, 0
    for epoch in range(1, (epochs or max_epochs) + 1):
        model.train()
        for b in torch.randperm(len(y)).split(cfg["batch_size"]):
            opt.zero_grad()
            loss_fn(model(ch[b]), y[b]).backward()
            opt.step()
        if val is not None:
            v = evaluate(model, *val)
            if v < best - 1e-5:
                best, best_state, best_epoch = v, copy.deepcopy(model.state_dict()), epoch
            elif epoch - best_epoch >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_epoch, best


def evaluate(model, ch, y) -> float:
    model.eval()
    with torch.no_grad():
        return nn.functional.binary_cross_entropy_with_logits(model(ch), y).item()


def grid(n_champ):
    """Configurations to try: the strength-only baseline, then nets of growing size."""
    yield {"model": {"kind": "strength", "n_champ": n_champ}, "lr": 3e-3, "weight_decay": 0.0, "batch_size": 512}
    # first tries at lr 1e-3, dropout 0.1 overfit after 2 epochs, hence slow and heavily regularized
    for dim, hidden, dropout, wd in itertools.product(
            [8, 16, 32], [(64,), (256, 128, 64), (160, 128, 64, 32, 16)], [0.3, 0.5], [1e-2, 1e-1]):
        yield {"model": {"kind": "net", "n_champ": n_champ, "dim": dim, "hidden": hidden, "dropout": dropout},
               "lr": 3e-4, "weight_decay": wd, "batch_size": 512}
