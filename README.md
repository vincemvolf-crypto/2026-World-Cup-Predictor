# World Cup Player Prop Predictor

Predicts 12 per-player stats for every starting player in a match
(passes attempted, shots, shots on target, saves, dribbles attempted,
clearances, crosses, shots assisted, tackles, goals, assists, fantasy
score), grouped by position so irrelevant stats (e.g. GK shots, CB saves)
are correctly zeroed out.

## How it works

```
data_collection.py      -> pulls raw stats from FBref + national team Elo + WC fixtures
feature_engineering.py  -> builds rolling "recent form" features per player
model.py                 -> trains one LightGBM model per (position group, stat)
predict.py               -> generates predictions for a specific matchup
```

## Setup

```bash
python -m venv venv
source venv/bin/activate          # or venv\Scripts\activate on Windows
pip install -r requirements.txt
```

## First run (build everything from scratch)

```bash
python data_collection.py        # ~10-30 min depending on rate limits
python feature_engineering.py
python model.py
```

## Generate predictions for a specific match

1. Create `data/processed/lineups.csv` with the predicted/confirmed starting XI:

```csv
team,player,position,is_starter
Brazil,Alisson,GK,1
Brazil,Marquinhos,DF,1
...
France,Mike Maignan,GK,1
...
```

   `position` must be one of: GK, DF, MF, FW (use config.POSITION_GROUPS
   to map FBref's raw position codes if needed).

2. Run:

```bash
python predict.py --home "Brazil" --away "France" --neutral
```

This writes:
- `predictions/Brazil_vs_France.csv` — every starter's 12-stat prediction
- `predictions/Brazil_vs_France_summary.csv` — team totals per stat

## Keeping it updated through the tournament ("automated pipeline")

There's no way to run a 24/7 background service from this chat, but you
can automate the refresh cycle on your own machine:

- **Daily**: `python data_collection.py && python feature_engineering.py`
  to pull newly-played World Cup matches and update Elo/form.
- **Weekly or after enough WC matches accumulate**: `python model.py`
  to retrain with the new World Cup data folded in.
- **Per matchday**: update `lineups.csv` with confirmed XIs (released
  ~75 min before kickoff) and rerun `predict.py` for each fixture.

On Linux/Mac, wire this up with `cron`; on Windows, Task Scheduler. If
you use Claude Code locally, you can also have it run this sequence on a
schedule and message you the output.

## Get the website live (easiest path — no command line needed)

1. **Create a GitHub account** (free) at github.com if you don't have one.
2. **Create a new repository**: click the "+" top-right → "New repository".
   Name it something like `worldcup-predictor`. Set it to **Public**
   (required for free GitHub Pages). Click "Create repository".
3. **Upload the project**: on the new repo page, click "uploading an
   existing file", then drag the entire contents of this
   `worldcup_predictor` folder into the browser window (including the
   hidden `.github` folder — if your file manager hides it, show hidden
   files first, or use the "Add file → Upload files" button which usually
   shows everything). Commit the upload.
4. **Turn on GitHub Pages**: repo → Settings tab → Pages (left sidebar) →
   under "Build and deployment", set Source to "Deploy from a branch",
   branch = `main`, folder = `/site`. Save. GitHub gives you a URL like
   `https://yourusername.github.io/worldcup-predictor/` — that's your
   website, live in a minute or two (showing sample data at first).
5. **Run the pipeline once**: repo → Actions tab → you'll see "Update
   World Cup Predictions" → click it → "Run workflow" button → Run. This
   takes several minutes (it's downloading and processing real data).
   When it's done (green checkmark), refresh your website — it now shows
   real predictions.
6. **After that**: it runs automatically every day on its own (the cron
   schedule in `.github/workflows/update.yml`). You don't have to do
   anything unless you want to update `lineups.csv` or `set_piece_takers.csv`
   — for those, edit the file directly on GitHub (click the file, pencil
   icon to edit, commit), and either wait for the next scheduled run or
   trigger it manually from the Actions tab as in step 5.

That's the whole setup — everything after step 6 is automatic.

## Known gaps / things to refine

- **`shots_assisted`** is approximated via FBref's shot-creating-actions
  (live passes) column — FBref doesn't expose a literal "secondary assist"
  stat under that name. Check `config.TARGET_STATS` and adjust if you find
  a better-matching column.
- **Elo scraping** (`eloratings.net`) doesn't have an official API — the
  TSV export endpoint may change. If `fetch_national_team_elo()` fails,
  create `data/raw/national_elo.csv` manually (columns: `team,elo`) from
  https://www.eloratings.net/.
- **International form blending** (`fetch_international_matches`) depends
  on `soccerdata` recognizing the competition names used — these may need
  adjusting. If it fails entirely, the pipeline still works fine on club
  form alone (see `config.INTERNATIONAL_FORM_WEIGHT`).
- **`set_piece_takers.csv`** and **`venue_factors.csv`** are manual inputs
  (see `data/processed/` and `data/raw/`) — fill these in for the teams/
  venues you care about most; defaults are neutral (no adjustment).
- **Expected minutes** (`minutes_model.py`) is most useful once real WC
  data starts coming in — pre-tournament it's based purely on club rotation
  patterns, which may not match international rotation behavior.
- Model accuracy should be backtested (`backtest.py`) against earlier
  2025-26 club season matches before trusting outputs.


## Disclaimer

This is a statistical model, not a guarantee. Sportsbooks price player
props using similar (and often more data-rich) models, so treat outputs
as one input into your own judgment, not a standalone betting signal.
