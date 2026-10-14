"""SQLite storage. Every step is resumable: the crawl state lives entirely in this file."""
from __future__ import annotations

import gzip
import json
import sqlite3

SCHEMA = """
CREATE TABLE IF NOT EXISTS players (
    puuid      TEXT PRIMARY KEY,
    platform   TEXT NOT NULL,
    tier       TEXT NOT NULL,
    lp         INTEGER,
    wins       INTEGER,
    losses     INTEGER,
    seen_at    INTEGER NOT NULL,   -- unix time this player was last seen on the ladder
    ids_until  INTEGER             -- match ids fetched up to this unix time
);

-- status: 0 pending, 1 stored, 2 not found, 3 not solo queue, 4 failed
CREATE TABLE IF NOT EXISTS match_queue (
    match_id  TEXT PRIMARY KEY,
    routing   TEXT NOT NULL,
    status    INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS match_queue_pending ON match_queue (routing, status, match_id);

CREATE TABLE IF NOT EXISTS matches (
    match_id       TEXT PRIMARY KEY,
    platform       TEXT,
    queue_id       INTEGER,
    game_version   TEXT,
    patch          TEXT,
    game_creation  INTEGER,        -- unix ms
    game_duration  INTEGER,        -- seconds
    remake         INTEGER,
    blue_win       INTEGER
);

CREATE TABLE IF NOT EXISTS participants (
    match_id       TEXT NOT NULL,
    puuid          TEXT NOT NULL,
    team_id        INTEGER,        -- 100 blue, 200 red
    position       TEXT,           -- TOP JUNGLE MIDDLE BOTTOM UTILITY, sometimes ''
    champion_id    INTEGER,
    champion_name  TEXT,
    PRIMARY KEY (match_id, puuid)
);

CREATE TABLE IF NOT EXISTS raw (
    match_id  TEXT PRIMARY KEY,
    data      BLOB                 -- gzipped match JSON
);
"""

PENDING, STORED, NOT_FOUND, NOT_SOLO, FAILED = range(5)
SOLO_QUEUE_ID = 420


def connect(path) -> sqlite3.Connection:
    con = sqlite3.connect(path)
    con.execute("PRAGMA journal_mode=WAL")
    con.execute("PRAGMA synchronous=NORMAL")
    con.executescript(SCHEMA)
    return con


def upsert_players(con, rows):
    """rows: (puuid, platform, tier, lp, wins, losses, seen_at)"""
    con.executemany("""
        INSERT INTO players (puuid, platform, tier, lp, wins, losses, seen_at)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT (puuid) DO UPDATE SET
            platform = excluded.platform, tier = excluded.tier, lp = excluded.lp,
            wins = excluded.wins, losses = excluded.losses, seen_at = excluded.seen_at
    """, rows)
    con.commit()


def queue_match_ids(con, routing, match_ids) -> int:
    cur = con.executemany("INSERT OR IGNORE INTO match_queue (match_id, routing) VALUES (?, ?)",
                          [(m, routing) for m in match_ids])
    return cur.rowcount


def set_status(con, match_id, status):
    con.execute("UPDATE match_queue SET status = ? WHERE match_id = ?", (status, match_id))
    con.commit()


def store_match(con, match_id, data, keep_raw=False) -> int:
    """Parse a match-v5 response into compact rows. Returns the queue status it ended with."""
    if data is None:
        set_status(con, match_id, NOT_FOUND)
        return NOT_FOUND
    info = data["info"]
    if info.get("queueId") != SOLO_QUEUE_ID:
        set_status(con, match_id, NOT_SOLO)
        return NOT_SOLO

    parts = info["participants"]
    duration = info.get("gameDuration", 0)
    remake = duration < 300 or any(p.get("gameEndedInEarlySurrender") for p in parts)
    blue_win = next((t["win"] for t in info.get("teams", []) if t["teamId"] == 100), None)
    version = info.get("gameVersion", "")

    con.execute("INSERT OR REPLACE INTO matches VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)", (
        match_id, info.get("platformId"), info["queueId"], version,
        ".".join(version.split(".")[:2]), info.get("gameCreation"), duration,
        int(remake), None if blue_win is None else int(blue_win),
    ))
    con.executemany("INSERT OR REPLACE INTO participants VALUES (?, ?, ?, ?, ?, ?)", [
        (match_id, p["puuid"], p["teamId"], p.get("teamPosition", ""),
         p["championId"], p.get("championName"))
        for p in parts
    ])
    if keep_raw:
        con.execute("INSERT OR REPLACE INTO raw VALUES (?, ?)",
                    (match_id, gzip.compress(json.dumps(data).encode())))
    con.execute("UPDATE match_queue SET status = ? WHERE match_id = ?", (STORED, match_id))
    con.commit()
    return STORED
