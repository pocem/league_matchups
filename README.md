# league_matchups

Predict ranked solo queue win probability from the draft (champion + role per side).

## Data: high-elo match crawler

Uses the official Riot API: ladder (`league-v4`) -> each player's ranked solo match ids
(`match-v5`) -> match details, deduplicated. State lives in `data/crawl.sqlite`, so the
crawl can be stopped and resumed at any time.

```bash
pip install -r requirements.txt
echo RIOT_API_KEY=RGAPI-... > .env

python -m crawler run --platforms kr,euw1,na1 --since 2026-01-08   # season start
python -m crawler stats
python -m crawler export --patch-prefix 16. --min-apex 3
```

- `run --no-matches` only scans match ids, which gives you an exact count of available
  matches before you spend days downloading them.
- Rerunning `run` resumes. Players are rescanned for new games after `--refresh-hours`
  (default 24), so a daily run keeps the dataset current.
- Each routing region (americas, europe, asia, sea) has its own rate limit, so crawling
  one server per region in parallel multiplies throughput.
- A dev or personal key allows about 72k requests/day per region. Dev keys expire every
  24 h: rerun with a fresh key and it continues.

### Output

`data/matches.parquet`, one row per match:

| column | meaning |
|---|---|
| `blue_top` ... `red_support` | champion id per side and role |
| `blue_win` | 1 if blue side won |
| `patch`, `game_creation` | for patch holdout and time decay |
| `n_apex` | how many of the 10 players are on the current Master+ ladder |

`data/champions.csv` maps champion ids to names. Remakes and games with missing or
duplicate roles are dropped.

`n_apex` uses the ladder at crawl time. Riot doesn't expose a player's rank at the time
of a game, and early-season games happen right after the rank reset, so filter on
`n_apex` and `patch` to keep lobbies that really are high elo.
