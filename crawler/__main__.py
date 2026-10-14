"""python -m crawler {players,run,stats,export} --help"""
from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

from . import crawl, export, store
from .riot import PLATFORM_ROUTING, AuthError, RiotClient, parse_limits


def api_key() -> str:
    key = os.environ.get("RIOT_API_KEY")
    if not key and Path(".env").exists():
        for line in Path(".env").read_text().splitlines():
            name, _, value = line.partition("=")
            if name.strip() == "RIOT_API_KEY":
                key = value.strip().strip("'\"")
    if not key:
        sys.exit("Set RIOT_API_KEY in the environment or in a .env file")
    return key


def platforms(value: str) -> list[str]:
    out = [p.strip().lower() for p in value.split(",") if p.strip()]
    if bad := [p for p in out if p not in PLATFORM_ROUTING]:
        raise argparse.ArgumentTypeError(f"unknown platform(s) {bad}; choose from {sorted(PLATFORM_ROUTING)}")
    return out


def unix_date(value: str) -> int:
    return int(datetime.fromisoformat(value).replace(tzinfo=timezone.utc).timestamp())


def main():
    ap = argparse.ArgumentParser(prog="crawler", description="Crawl high-elo ranked solo queue matches.")
    ap.add_argument("--db", default="data/crawl.sqlite")
    sub = ap.add_subparsers(dest="cmd", required=True)

    def ladder_args(p):
        p.add_argument("--platforms", type=platforms, default=platforms("kr,euw1,na1"))
        p.add_argument("--min-tier", default="MASTER", choices=crawl.TIERS,
                       help="lowest ladder tier to crawl (DIAMOND adds many players and requests)")
        p.add_argument("--app-limits", type=parse_limits, default=None,
                       help="starting limits like '20:1,100:120'; corrected from Riot's headers anyway")

    p_players = sub.add_parser("players", help="only refresh the ladder snapshot")
    ladder_args(p_players)

    p_run = sub.add_parser("run", help="refresh ladder, scan match ids, download matches")
    ladder_args(p_run)
    p_run.add_argument("--since", type=unix_date, default=unix_date("2026-01-01"),
                       help="earliest game date, YYYY-MM-DD (set to the season start)")
    p_run.add_argument("--skip-players", action="store_true", help="reuse the stored ladder snapshot")
    p_run.add_argument("--no-ids", action="store_true", help="only download already queued matches")
    p_run.add_argument("--no-matches", action="store_true", help="only scan match ids (fast count)")
    p_run.add_argument("--refresh-hours", type=int, default=24,
                       help="re-scan a player's match ids if the last scan is older than this")
    p_run.add_argument("--id-workers", type=int, default=2)
    p_run.add_argument("--match-workers", type=int, default=6)
    p_run.add_argument("--keep-raw", action="store_true",
                       help="also store gzipped full match JSON (~5-8 KB each) for later features")

    sub.add_parser("stats", help="show crawl progress")

    p_exp = sub.add_parser("export", help="write one row per match to Parquet")
    p_exp.add_argument("--out", default="data/matches.parquet")
    p_exp.add_argument("--min-apex", type=int, default=0,
                       help="keep matches with at least this many current Master+ players")
    p_exp.add_argument("--patch-prefix", default="", help="e.g. '16.' for this season's patches")
    p_exp.add_argument("--include-remakes", action="store_true")

    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        datefmt="%H:%M:%S")
    Path(args.db).parent.mkdir(parents=True, exist_ok=True)
    con = store.connect(args.db)

    if args.cmd == "stats":
        export.print_stats(con)
    elif args.cmd == "export":
        table = export.build_table(con, args.include_remakes)
        table = table[(table.n_apex >= args.min_apex) & table.patch.str.startswith(args.patch_prefix)]
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        table.to_parquet(args.out, index=False)
        champs_path = Path(args.out).with_name("champions.csv")
        export.champions(con).to_csv(champs_path, index=False)
        print(f"wrote {len(table):,} matches to {args.out} and champion names to {champs_path}")
    else:
        asyncio.run(_crawl(con, args))


async def _crawl(con, args):
    kwargs = {"app_limits": args.app_limits} if args.app_limits else {}
    try:
        async with RiotClient(api_key(), **kwargs) as client:
            if args.cmd == "players":
                await crawl.fetch_players(client, con, args.platforms, args.min_tier)
            else:
                await crawl.run(client, con, args)
    except* AuthError as eg:
        sys.exit(f"Stopped: {eg.exceptions[0]}. Dev keys expire every 24 h; progress is saved, "
                 "rerun with a fresh key.")


if __name__ == "__main__":
    main()
