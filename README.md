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
- `run --until 2026-09-23` only downloads queued matches played before that date. It
  finds the cut with a binary search over the queue, and the newer matches stay queued.
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

## Solo queue draft model (Stage 1)

```bash
python -m crawler export --patch-prefix 16.   # data/matches.parquet from the crawl
python -m draft train                         # grid search, models/draft_net.pt + draft_metadata.json
python -m draft score                         # data/pro/draft_scores.parquet, one draft logit per pro game
python -m pro compare                         # uses the draft score when present (--no-draft to leave it out)
```

The model predicts P(blue wins) from the 10 champions alone. It uses the same antisymmetric form
as the pro net: champion embeddings by role, plus own × opponent lane products. Pro drafts are
scored without the solo queue blue-side bias.

First run (99k matches, almost all on patches 16.19-16.20, test = newest 10%): log loss 0.6893,
vs 0.6921 for always predicting the blue rate. No network beat a plain per-champion-per-role
strength baseline (also 0.6893), so 100k games aren't enough to learn synergies or counters.
On pro games the score is weak but real (AUC 0.513 overall, 0.533 on patches 16.15+). It
improves 2026 log loss by 0.0005 for LR, and by 0.001-0.0017 on patches 16.15+. Pro play runs a
few patches behind solo queue, so draft data from the same patch as the pro game is the
biggest lever.

### Same-patch draft test

Pro play runs a few patches behind solo queue. `python -m draft patch-eval` trains a draft model
on each patch's solo queue games alone. It then checks whether that score helps the pro models
more than the all-patch score, on the pro games played on that same patch:

```bash
python -m crawler export --patch-prefix 16.   # refresh solo games
python -m pro compare --no-draft             # baseline predictions in data/pro/test_predictions.csv
python -m draft patch-eval                   # patches with >= 20k solo games (--min-games)
```

The crawl downloads newest matches first, and its queue holds patches 16.16-16.19, which pro play
was on in Aug-Oct 2026 (about 2,000 pro games). Each of those patches becomes testable as it
fills up.

## Website (GitHub Pages)

`site/` is a static page that runs the logistic regression in the browser. The page has
searchable team, player and champion inputs, pre-filled lineups and a side swap. It shows the
probability before and after the draft, fair odds, and a breakdown of what moves the number.
`.github/workflows/pages.yml` publishes `site/` on every push to `main` that changes it. One-time
setup: repo Settings → Pages → Source: GitHub Actions.

**Daily update, automatic.** `.github/workflows/daily.yml` runs on GitHub every 6 hours. It
runs `python -m pro fetch` (newest Oracle's Elixir CSV from its public Google Drive folder),
then `pro deploy` and the tests. If there are new games it commits the snapshot and publishes
the site. Once that day's snapshot exists, the scheduled runs skip. "Run workflow" in the
Actions tab forces an update. Google Drive sometimes blocks downloads of popular files for a
few hours, and the later runs retry. A repo secret `GOOGLE_API_KEY` (a free Google Cloud API
key with the Drive API enabled) adds a second download route. The daily job only needs
`requirements-deploy.txt` (no torch or lightgbm). The champion strength table
(`models/draft_strength.json`) stays as committed until you rerun `python -m draft freeze`
locally.

```bash
python -m draft freeze --patches 16.19 16.20   # once per patch: champion strength table
python -m pro fetch                             # newest Oracle's Elixir CSVs (the daily job does this)
python -m pro deploy                            # refit + site/data/snapshot.json (the daily job does this)
python -m pytest tests/test_deploy.py           # the snapshot and the JavaScript match the model
cd site && python -m http.server 8000           # preview at http://localhost:8000
```

`pro deploy` refits two LRs on Jul 2024 to today: pre-draft (without champion features) and
post-draft. It fits the draft combining layer (`a + b · post-draft logit + w · draft logit`) on
recent pro games using predictions of a model that hadn't seen them. It exports ratings,
rosters, champion stats, weights and 30 test cases. The page checks itself against those test
cases and shows the result in the footer.
