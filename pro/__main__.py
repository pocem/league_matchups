"""python -m pro build     -> data/pro/games.parquet (+ games.csv to read) + vocab CSVs
python -m pro tune M    -> hyperparameter search for model M, writes models/tuned_M.json
python -m pro compare   -> model comparison on 2026 (data/pro/model_comparison.csv), fitted models in models/
                           (uses models/tuned_*.json when present)
python -m pro fetch     -> download the newest Oracle's Elixir CSVs into pro_play_data/
python -m pro deploy    -> refit on all data, export site/data/snapshot.json for the website"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from . import dataset, load


def main():
    ap = argparse.ArgumentParser(prog="pro", description="Pro play dataset.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("build", help="build the point-in-time game table")
    p.add_argument("--src", default="pro_play_data")
    p.add_argument("--out", default="data/pro")
    p.add_argument("--min-player-games", type=int, default=10)
    c = sub.add_parser("compare", help="fit and compare models, test on 2026")
    c.add_argument("--data", default="data/pro")
    c.add_argument("--save", default="models", help="directory for the fitted models")
    c.add_argument("--defaults", action="store_true", help="ignore tuned hyperparameters in --save")
    c.add_argument("--no-draft", action="store_true", help="leave out the solo queue draft score")
    t = sub.add_parser("tune", help="hyperparameter search on 2024-2025 (never touches 2026)")
    t.add_argument("model", choices=["logistic_regression", "lightgbm", "neural_net"])
    t.add_argument("--minutes", type=float, default=15)
    t.add_argument("--data", default="data/pro")
    t.add_argument("--save", default="models", help="writes tuned_<model>.json here")
    f = sub.add_parser("fetch", help="download the newest Oracle's Elixir CSVs")
    f.add_argument("--dest", default="pro_play_data")
    a = sub.add_parser("auth", help="one-time Google sign-in for fetch's copy route")
    a.add_argument("client_secrets", help="OAuth client JSON (Desktop app) from Google Cloud Console")
    d = sub.add_parser("deploy", help="refit on all data and export site/data/snapshot.json")
    d.add_argument("--src", default="pro_play_data")
    d.add_argument("--site", default="site")
    args = ap.parse_args()
    if args.cmd == "fetch":
        from . import fetch
        fetch.fetch(Path(args.dest))
        return
    if args.cmd == "auth":
        from . import fetch
        fetch.authorize(Path(args.client_secrets))
        return
    if args.cmd == "deploy":
        from . import deploy
        return deploy.deploy(args.src, Path("models"), Path(args.site))
    if args.cmd == "compare":
        return compare(Path(args.data), Path(args.save), args.defaults, not args.no_draft)
    if args.cmd == "tune":
        from . import tune
        return tune.tune(args.model, args.minutes, Path(args.data), Path(args.save))

    df = load.load(args.src)
    games = dataset.build(df)
    games, vocabs = dataset.add_indices(games, min_player_games=args.min_player_games)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    games.to_parquet(out / "games.parquet", index=False)
    games.to_csv(out / "games.csv", index=False, float_format="%.4f")
    for kind, vocab in vocabs.items():
        vocab.to_csv(out / f"vocab_{kind}.csv", index=False)
    print(f"wrote {len(games):,} games x {games.shape[1]} columns to {out / 'games.parquet'} and games.csv")
    with pd.option_context("display.float_format", "{:.4f}".format):
        print(dataset.elo_report(games).to_string(index=False))


def compare(data: Path, save_dir: Path, defaults=False, draft=True):
    from . import models, tune  # heavy imports (torch, lightgbm) only when needed

    games = pd.read_parquet(data / "games.parquet")
    if draft and (data / "draft_scores.parquet").exists():
        games = games.merge(pd.read_parquet(data / "draft_scores.parquet"), on="gameid", how="left")
        print("using the solo queue draft score")
    sizes = {k: len(pd.read_csv(data / f"vocab_{k}.csv")) for k in ("champion", "player")}
    params = {} if defaults else tune.load_tuned(save_dir)
    if params:
        print(f"using tuned hyperparameters for: {', '.join(params)}")
    table, preds, fitted = models.compare(games, sizes, params)
    models.save(fitted, table, save_dir, params)
    table.to_csv(data / "model_comparison.csv", float_format="%.4f")
    preds.to_csv(data / "test_predictions.csv", index=False, float_format="%.4f")
    with pd.option_context("display.float_format", "{:.4f}".format, "display.width", 200):
        print(table.to_string())
    print(f"saved models to {save_dir}/")


if __name__ == "__main__":
    main()
