"""Crawl pipeline: ladder -> players' match ids -> match details.

Per routing region (americas, europe, asia, sea) id workers and match workers run
concurrently and share that region's rate limit, so matches start flowing right away.
"""
from __future__ import annotations

import asyncio
import logging
import time
from collections import defaultdict

from . import store
from .riot import PLATFORM_ROUTING, RiotClient, RiotError

log = logging.getLogger(__name__)

QUEUE = "RANKED_SOLO_5x5"
TIERS = ["CHALLENGER", "GRANDMASTER", "MASTER", "DIAMOND"]
APEX_TIERS = {"CHALLENGER", "GRANDMASTER", "MASTER"}
DIVISIONS = ["I", "II", "III", "IV"]

# highest elo first, so the best games are downloaded first
TIER_SORT = "CASE tier WHEN 'CHALLENGER' THEN 0 WHEN 'GRANDMASTER' THEN 1 WHEN 'MASTER' THEN 2 ELSE 3 END"


async def fetch_players(client: RiotClient, con, platforms, min_tier="MASTER"):
    tiers = TIERS[:TIERS.index(min_tier) + 1]
    for platform in platforms:
        now = int(time.time())
        for tier in tiers:
            rows = []
            if tier in APEX_TIERS:
                data = await client.get(platform, f"/lol/league/v4/{tier.lower()}leagues/by-queue/{QUEUE}",
                                        method=f"league-{tier}")
                entries = data["entries"] if data else []
                rows += [(e["puuid"], platform, tier, e["leaguePoints"], e["wins"], e["losses"], now)
                         for e in entries]
            else:
                for div in DIVISIONS:
                    page = 1
                    while entries := await client.get(
                            platform, f"/lol/league-exp/v4/entries/{QUEUE}/{tier}/{div}",
                            params={"page": page}, method="league-exp"):
                        rows += [(e["puuid"], platform, tier, e["leaguePoints"], e["wins"], e["losses"], now)
                                 for e in entries]
                        page += 1
            store.upsert_players(con, rows)
            log.info("%s %s: %d players", platform, tier, len(rows))


def players_to_crawl(con, platforms, tiers, refresh_hours):
    """Players currently on the ladder whose match ids are missing or older than refresh_hours."""
    cutoff = int(time.time()) - refresh_hours * 3600
    out = []
    for platform in platforms:
        latest = con.execute("SELECT MAX(seen_at) FROM players WHERE platform = ?", (platform,)).fetchone()[0]
        if latest is None:
            log.warning("%s: no players stored, run without --skip-players first", platform)
            continue
        marks = ",".join("?" * len(tiers))
        out += con.execute(f"""
            SELECT puuid, ids_until FROM players
            WHERE platform = ? AND tier IN ({marks}) AND seen_at >= ?
              AND (ids_until IS NULL OR ids_until < ?)
            ORDER BY {TIER_SORT}, lp DESC
        """, (platform, *tiers, latest - 6 * 3600, cutoff)).fetchall()
    return out


async def fetch_ids(client, con, routing, puuid, start_time, end_time) -> int:
    start, new = 0, 0
    while True:
        ids = await client.get(routing, f"/lol/match/v5/matches/by-puuid/{puuid}/ids", params={
            "queue": store.SOLO_QUEUE_ID, "startTime": start_time, "endTime": end_time,
            "start": start, "count": 100,
        }, method="match-ids")
        if not ids:
            break
        new += store.queue_match_ids(con, routing, ids)
        if len(ids) < 100:
            break
        start += 100
    con.execute("UPDATE players SET ids_until = ? WHERE puuid = ?", (end_time, puuid))
    con.commit()
    return new


class Progress:
    def __init__(self):
        self.t0 = time.monotonic()
        self.players = defaultdict(int)
        self.new_ids = defaultdict(int)
        self.matches = defaultdict(int)

    async def report(self, con, client, every=60):
        while True:
            await asyncio.sleep(every)
            hours = (time.monotonic() - self.t0) / 3600
            for routing in sorted(set(self.players) | set(self.matches)):
                pending = con.execute("SELECT COUNT(*) FROM match_queue WHERE routing = ? AND status = 0",
                                      (routing,)).fetchone()[0]
                log.info("%s: %d players scanned, %d new match ids, %d matches stored (%.0f/h), %d pending",
                         routing, self.players[routing], self.new_ids[routing], self.matches[routing],
                         self.matches[routing] / hours, pending)
            log.info("total requests: %d", client.requests)


async def crawl_routing(client, con, routing, platforms, args, progress: Progress):
    tiers = TIERS[:TIERS.index(args.min_tier) + 1]
    end_time = int(time.time())
    players = asyncio.Queue()
    if not args.no_ids:
        for puuid, ids_until in players_to_crawl(con, platforms, tiers, args.refresh_hours):
            # 10 min overlap on refresh; INSERT OR IGNORE drops the repeats
            players.put_nowait((puuid, max(args.since, (ids_until or 0) - 600)))
        log.info("%s: %d players to scan for match ids", routing, players.qsize())
    ids_done = asyncio.Event()

    async def id_worker():
        while not players.empty():
            puuid, start_time = players.get_nowait()
            try:
                progress.new_ids[routing] += await fetch_ids(client, con, routing, puuid, start_time, end_time)
            except RiotError as e:
                # ids_until stays unset, so the next run scans this player again
                log.warning("%s: skipping player: %s", routing, e)
                continue
            progress.players[routing] += 1

    async def ids_phase():
        async with asyncio.TaskGroup() as tg:
            for _ in range(args.id_workers):
                tg.create_task(id_worker())
        ids_done.set()
        log.info("%s: match id scan finished", routing)

    matches = asyncio.Queue(maxsize=args.match_workers * 4)
    inflight = set()

    async def feeder():
        while True:
            rows = con.execute("""SELECT match_id FROM match_queue WHERE routing = ? AND status = 0
                                  ORDER BY match_id DESC LIMIT 1000""", (routing,)).fetchall()
            batch = [m for (m,) in rows if m not in inflight]
            if not batch:
                if ids_done.is_set():
                    break
                await asyncio.sleep(5)
                continue
            for m in batch:
                inflight.add(m)
                await matches.put(m)
        for _ in range(args.match_workers):
            await matches.put(None)

    async def match_worker():
        while (match_id := await matches.get()) is not None:
            try:
                data = await client.get(routing, f"/lol/match/v5/matches/{match_id}", method="match")
                if store.store_match(con, match_id, data, args.keep_raw) == store.STORED:
                    progress.matches[routing] += 1
            except RiotError as e:
                log.warning("%s: %s", match_id, e)
                store.set_status(con, match_id, store.FAILED)
            finally:
                inflight.discard(match_id)

    async with asyncio.TaskGroup() as tg:
        tg.create_task(ids_phase())
        if not args.no_matches:
            tg.create_task(feeder())
            for _ in range(args.match_workers):
                tg.create_task(match_worker())
    log.info("%s: done", routing)


async def run(client: RiotClient, con, args):
    if not args.skip_players:
        await fetch_players(client, con, args.platforms, args.min_tier)
    by_routing = defaultdict(list)
    for p in args.platforms:
        by_routing[PLATFORM_ROUTING[p]].append(p)

    progress = Progress()
    reporter = asyncio.create_task(progress.report(con, client))
    try:
        async with asyncio.TaskGroup() as tg:
            for routing, platforms in by_routing.items():
                tg.create_task(crawl_routing(client, con, routing, platforms, args, progress))
    finally:
        reporter.cancel()
