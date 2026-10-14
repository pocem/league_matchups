"""python -m pro build   -> data/pro/games.parquet (+ games.csv to read) + vocab CSVs"""
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
    args = ap.parse_args()

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


if __name__ == "__main__":
    main()
