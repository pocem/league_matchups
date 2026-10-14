"""Hyperparameter search for the pro models. 2026 (the test year) is never touched here.

Each candidate is scored on two time folds and the validation log losses are averaged:
    fold 1: train Jul - Dec 2024, validate Jan - Jun 2025
    fold 2: train Jul 2024 - Jun 2025, validate Jul - Dec 2025
Two folds (~8.8k validation games) make it harder to overfit a lucky validation set than one.
Searches use Optuna (TPE) and are stored in SQLite, so they can be resumed or inspected.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from . import models as M

FOLDS = [("2024-07-01", "2025-01-01", "2025-07-01"),
         ("2024-07-01", "2025-07-01", "2026-01-01")]
NN_SEEDS = (0, 1)  # per fold; averaging two seeds keeps lucky initialisations from winning


def folds(games):
    for start, cut, end in FOLDS:
        yield games[(games.date >= start) & (games.date < cut)], games[(games.date >= cut) & (games.date < end)]


# ---------------------------------------------------------------- logistic regression

def tune_logreg(games) -> tuple[dict, pd.DataFrame]:
    data = [(M.tabular(tr)[M.logreg_cols(tr)], tr.blue_win, M.tabular(va)[M.logreg_cols(va)], va.blue_win)
            for tr, va in folds(games)]
    rows = []
    for C in np.logspace(-4, 2, 25):
        losses = []
        for xtr, ytr, xva, yva in data:
            m = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=5000)).fit(xtr, ytr)
            losses.append(log_loss(yva, m.predict_proba(xva)[:, 1]))
        rows.append({"C": C, "fold1": losses[0], "fold2": losses[1], "mean": np.mean(losses)})
    table = pd.DataFrame(rows)
    return {"C": float(table.loc[table["mean"].idxmin(), "C"])}, table


# ---------------------------------------------------------------- lightgbm

def lgbm_objective(games):
    data = [(M.tabular(tr), tr.blue_win, M.tabular(va), va.blue_win) for tr, va in folds(games)]

    def objective(trial: optuna.Trial) -> float:
        p = {
            "learning_rate": trial.suggest_float("learning_rate", 0.003, 0.1, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 2, 64, log=True),
            "max_depth": trial.suggest_categorical("max_depth", [-1, 2, 3, 4, 6, 8]),
            "min_child_samples": trial.suggest_int("min_child_samples", 10, 500, log=True),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.1, 1.0),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.4, 1.0),
            "bagging_freq": 1,
            "lambda_l1": trial.suggest_float("lambda_l1", 1e-3, 30, log=True),
            "lambda_l2": trial.suggest_float("lambda_l2", 1e-3, 100, log=True),
            "min_gain_to_split": trial.suggest_float("min_gain_to_split", 0.0, 1.0),
            "num_threads": 4,
        }
        losses, iters = [], []
        for k, (xtr, ytr, xva, yva) in enumerate(data):
            m = M.lgbm_early_stop(p, xtr, ytr, xva, yva)
            losses.append(m.best_score["valid_0"]["binary_logloss"])
            iters.append(m.best_iteration)
            trial.report(float(np.mean(losses)), k)
            if trial.should_prune():
                raise optuna.TrialPruned()
        trial.set_user_attr("best_iterations", iters)
        return float(np.mean(losses))

    return objective


# ---------------------------------------------------------------- neural net

def nn_objective(games, n_champ, n_player):
    data = []
    for tr, va in folds(games):
        trX, mean, std = M.nn_arrays(tr)
        vaX, _, _ = M.nn_arrays(va, mean, std)
        data.append((trX, vaX))

    def objective(trial: optuna.Trial) -> float:
        p = {
            "hidden": trial.suggest_categorical("hidden", [16, 32, 64, 128, 256]),
            "n_layers": trial.suggest_int("n_layers", 1, 3),
            "dropout": trial.suggest_float("dropout", 0.0, 0.6),
            "weight_decay": trial.suggest_float("weight_decay", 1e-5, 0.3, log=True),
            "lr": trial.suggest_float("lr", 1e-4, 1e-2, log=True),
            "batch_size": trial.suggest_categorical("batch_size", [64, 128, 256, 512, 1024]),
        }
        arch = dict(n_champ=n_champ, n_player=n_player, d_champ=0, d_player=0,
                    **{k: p[k] for k in M.NN_ARCH})
        losses, epochs = [], []
        for k, (trX, vaX) in enumerate(data):
            for seed in NN_SEEDS:
                _, best_epoch, best = M.train_nn(trX, arch, p, seed, 150, val=vaX)
                losses.append(best)
                epochs.append(best_epoch)
            trial.report(float(np.mean(losses)), k)
            if trial.should_prune():
                raise optuna.TrialPruned()
        trial.set_user_attr("best_epochs", epochs)
        return float(np.mean(losses))

    return objective


# ---------------------------------------------------------------- driver

def run_study(name, objective, minutes, storage_dir: Path, defaults: dict):
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        study_name=name, direction="minimize", load_if_exists=True,
        storage=f"sqlite:///{(storage_dir / f'tune_{name}.sqlite').as_posix()}",
        sampler=optuna.samplers.TPESampler(seed=0, n_startup_trials=20),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=10))
    if not study.trials:
        study.enqueue_trial(defaults, skip_if_exists=True)  # the current hand-picked settings first
    start = time.time()

    def progress(study, trial):
        if trial.number % 10 == 0 or trial.value == study.best_value:
            print(f"  [{name}] trial {trial.number:4d}  {(time.time() - start) / 60:5.1f} min  "
                  f"best {study.best_value:.5f}", flush=True)

    study.optimize(objective, timeout=minutes * 60, callbacks=[progress], gc_after_trial=True)
    return study


def tune(model: str, minutes: float, data_dir: Path, out_dir: Path):
    games = pd.read_parquet(data_dir / "games.parquet")
    out_dir.mkdir(parents=True, exist_ok=True)
    if model == "logistic_regression":
        best, table = tune_logreg(games)
        print(table.to_string(index=False, float_format="%.5f"))
        result = {"params": best, "cv_log_loss": float(table["mean"].min())}
    else:
        torch.set_num_threads(4)
        if model == "lightgbm":
            defaults = {**{k: v for k, v in M.LGBM_DEFAULTS.items() if k != "bagging_freq"},
                        "max_depth": -1, "lambda_l1": 1e-3, "min_gain_to_split": 0.0}
            study = run_study(model, lgbm_objective(games), minutes, data_dir, defaults)
        else:
            sizes = {k: len(pd.read_csv(data_dir / f"vocab_{k}.csv")) + 1 for k in ("champion", "player")}
            study = run_study(model, nn_objective(games, sizes["champion"], sizes["player"]),
                              minutes, data_dir, M.NN_DEFAULTS)
        done = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        default = done[0].value if done else None
        result = {"params": study.best_params, "cv_log_loss": study.best_value,
                  "cv_log_loss_defaults": default, "trials": len(study.trials),
                  "completed": len(done), "best_trial_attrs": study.best_trial.user_attrs}
    (out_dir / f"tuned_{model}.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


def load_tuned(out_dir: Path) -> dict:
    params = {}
    for name in ("logistic_regression", "lightgbm", "neural_net"):
        path = out_dir / f"tuned_{name}.json"
        if path.exists():
            params[name] = json.loads(path.read_text())["params"]
    return params
