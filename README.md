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

## Pro play dataset

`python -m pro build` turns the Oracle's Elixir CSVs in `pro_play_data/` into
`data/pro/games.parquet`: one row per game (about 30k, 2024-2026), with features for both
sides. Every feature uses only games played earlier, so any time-based split is leak-free.
The `vocab_*.csv` files map player, team, champion and league ids to the `*_idx` columns
used by embedding layers. Index 0 means unknown or rare (fewer than 10 games).

| columns | meaning |
|---|---|
| `{side}_{role}_player`, `_champion` (+ `_idx`) | who played what. Roles: top jng mid bot sup |
| `{side}_{role}_player_elo`, `_player_games` | player rating and experience |
| `{side}_{role}_pc_games`, `_pc_days_since` | how often and how recently the player has played this champion |
| `{side}_{role}_meta_games`, `_meta_wr`, `_patch_games` | champion+role pro meta: 45-day decayed and on this patch |
| `{side}_team_elo`, `_team_games`, `_roster_elo` | team-id rating and the mean of the 5 players' ratings |
| `{side}_home_league`, `_league_offset`, `cross_league` | league strength, only non-zero across leagues |
| `p_blue_team_elo`, `p_blue_roster_elo` | raw Elo win probabilities (baselines) |
| `blue_win` | label |

Ratings (`pro/ratings.py`): Elo starts at 1500 with a K that decays from 120 to 28 over the
first games. Each new calendar year keeps 80% of a rating's distance from 1500. Effective
strength = rating + home league offset. League offsets learn only from cross-league games
(MSI, Worlds, EWC, EM, KeSPA Cup...). Player Elo updates from the roster-mean expectation,
so it tracks rosters through subs and transfers. Raw Elo log loss in 2026: 0.616 for roster
Elo, 0.632 for team Elo, 0.691 for the base rate.

What the extra features add, measured with a logistic regression (train Jul 2024-2025, test 2026):
Elo only 0.6153 -> 0.6074 with the player and champion features. Nearly all of that comes from
`pc_games` (comfort on the champion) and `meta_wr`. The player's win rate overall and on the
champion (raw, or adjusted for opponents via Elo) added nothing on top of Elo, so they were
dropped. Time-decaying `pc_games` (45-365 day half-life) made it worse, so it is an all-time count.

## Pro play models

`python -m pro compare` fits logistic regression, LightGBM and a neural net, then tests them on
all 2026 games. The number of trees or epochs is chosen on Jul-Dec 2025. Each model is then
refit on Jul 2024 - Dec 2025 (the first half of 2024 is Elo burn-in). The fitted models go to
`models/`, with `metadata.json` holding the training window and test scores. Each model's 2026
predictions go to `data/pro/test_predictions.csv`.

The neural net scores the game from each side's point of view:
logit = s(blue, red) - s(red, blue) + blue bias. Swapping the sides flips the prediction exactly.

| 2026 test | log loss | accuracy | log loss, major leagues |
|---|---|---|---|
| always blue rate | 0.691 | 53.5% | 0.688 |
| raw roster Elo | 0.617 | 65.2% | 0.645 |
| logistic regression | 0.600 | 67.1% | 0.637 |
| LightGBM | 0.601 | 66.9% | 0.635 |
| neural net | 0.600 | 67.1% | 0.635 |

The three models are statistically tied (bootstrap). Random forest and LightGBM with champion
ids were significantly worse and were dropped. Champion and player embeddings learned on pro data
alone didn't help. They are still available in `fit_nn` for the solo queue pretraining later.

### Hyperparameter tuning

```bash
python -m pro tune logistic_regression          # grid over C
python -m pro tune lightgbm --minutes 25        # Optuna TPE search
python -m pro tune neural_net --minutes 25
python -m pro compare                           # refit with models/tuned_*.json, test on 2026
```

Tuning never sees 2026. Each candidate is scored by its average validation log loss over two
time folds: train Jul-Dec 2024 → validate Jan-Jun 2025, and train Jul 2024-Jun 2025 → validate
Jul-Dec 2025. The neural net averages 2 seeds per fold. The searches are stored in
`data/pro/tune_*.sqlite`, so rerunning a search continues it. The winners are written to
`models/tuned_*.json`. `compare --defaults` ignores them.
