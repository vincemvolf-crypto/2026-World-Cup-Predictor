"""
Central configuration for the World Cup player-prop prediction pipeline.

Edit this file first - almost everything downstream (data collection,
features, models, predictions) reads from here.
"""

# -----------------------------------------------------------------------
# THE 12 TARGET STATS
# Keys = internal names used everywhere in the pipeline.
# Values = (FBref source table, FBref column name) - used by data_collection.py
# -----------------------------------------------------------------------
TARGET_STATS = {
    "passes_attempted":   ("passing",    "Att"),        # Passes attempted
    "shots":              ("shooting",   "Sh"),         # Shots total
    "shots_on_target":    ("shooting",   "SoT"),        # Shots on target
    "saves":              ("keeper",     "Saves"),      # GK saves
    "dribbles_attempted": ("possession", "Att_take_on"),# Take-ons attempted
    "clearances":         ("defense",    "Clr"),        # Clearances
    "crosses":            ("passing",    "Crs"),        # Crosses
    "shots_assisted":     ("gca",        "SCA_PassLive"),# Shot-creating passes (proxy)
    "tackles":            ("defense",    "Tkl"),        # Tackles
    "goals":              ("standard",   "Gls"),        # Goals
    "assists":            ("standard",   "Ast"),        # Assists
    # fantasy_score is NOT scraped - it's computed from the other 11
}

# -----------------------------------------------------------------------
# POSITION GROUPS
# Used to (a) train separate models per group (different stat profiles)
# and (b) zero-out / mask irrelevant stats in the output.
# -----------------------------------------------------------------------
POSITION_GROUPS = {
    "GK": ["GK"],
    "DF": ["DF", "DF,MF"],
    "MF": ["MF", "MF,DF", "MF,FW"],
    "FW": ["FW", "FW,MF"],
}

# Which of the 12 stats are realistically non-zero for each group.
# Anything not listed is forced to 0 in the final output for that group.
RELEVANT_STATS = {
    "GK": ["passes_attempted", "saves", "clearances"],
    "DF": ["passes_attempted", "clearances", "tackles", "crosses",
           "shots", "shots_on_target", "dribbles_attempted",
           "shots_assisted", "goals", "assists"],
    "MF": ["passes_attempted", "tackles", "shots", "shots_on_target",
           "dribbles_attempted", "shots_assisted", "crosses",
           "clearances", "goals", "assists"],
    "FW": ["passes_attempted", "shots", "shots_on_target",
           "dribbles_attempted", "shots_assisted", "crosses",
           "tackles", "goals", "assists"],
}

# -----------------------------------------------------------------------
# FANTASY SCORE FORMULA (outfield players)
# Tune these weights to match whatever fantasy platform you're targeting
# (these defaults are a generic FPL-style soccer scoring scheme).
# -----------------------------------------------------------------------
FANTASY_WEIGHTS = {
    "goals": 5.0,            # per goal (varies by position in real FPL; simplified here)
    "assists": 3.0,
    "shots_on_target": 0.5,
    "tackles": 0.3,
    "clearances": 0.1,
    "dribbles_attempted": 0.2,
    "passes_attempted": 0.01,
    "crosses": 0.1,
    "shots_assisted": 0.5,
}

GK_FANTASY_WEIGHTS = {
    "saves": 0.5,
    "passes_attempted": 0.01,
    "clearances": 0.1,
}

# -----------------------------------------------------------------------
# RECENT FORM WINDOW
# -----------------------------------------------------------------------
FORM_WINDOW_MATCHES = 8          # rolling average over last N matches
MIN_MINUTES_FOR_SAMPLE = 20      # ignore cameo appearances under this

# -----------------------------------------------------------------------
# DATA SOURCES
# -----------------------------------------------------------------------
# FBref competition IDs / season strings used by the `soccerdata` library
# for "recent form" (club football, most recently completed seasons)
CLUB_LEAGUES = [
    "ENG-Premier League",
    "ESP-La Liga",
    "GER-Bundesliga",
    "ITA-Serie A",
    "FRA-Ligue 1",
]
CLUB_SEASON = "2526"  # 2025-26 season, FBref season string format

# FBref competition string for the World Cup itself
WC_LEAGUE = "WC"
WC_SEASON = "2026"

# National team Elo ratings (free, no key) - used as opponent-strength feature
ELO_RATINGS_URL = "http://api.clubelo.com"  # club elo - see team_strength.py
WORLD_FOOTBALL_ELO_URL = "https://www.eloratings.net"  # national team elo (scrape)

# -----------------------------------------------------------------------
# PATHS
# -----------------------------------------------------------------------
DATA_DIR = "data"
RAW_DIR = f"{DATA_DIR}/raw"
PROCESSED_DIR = f"{DATA_DIR}/processed"
MODEL_DIR = "models"
PREDICTIONS_DIR = "predictions"

# -----------------------------------------------------------------------
# ACCURACY-BOOSTING FEATURES
# -----------------------------------------------------------------------

# How much weight to give international-match form vs club form when both
# are available (0 = ignore international data, 1 = ignore club data).
# International sample sizes are small, so this stays low.
INTERNATIONAL_FORM_WEIGHT = 0.35

# Rolling window for the expected-minutes model (separate from the stat
# form window - minutes patterns change faster, e.g. coming back from injury)
MINUTES_FORM_WINDOW = 6

# Stats that scale roughly linearly with minutes played - these get
# multiplied by (expected_minutes / 90) at prediction time.
MINUTES_SCALED_STATS = [
    "passes_attempted", "shots", "shots_on_target", "saves",
    "dribbles_attempted", "clearances", "crosses", "shots_assisted",
    "tackles",
]
# goals/assists are left unscaled by minutes alone - they're rare events
# better driven by the model's own features, but expected_minutes is still
# fed in as an input feature to those models too.

# Work-rate stats affected by altitude/heat (see venue_factors.csv)
VENUE_SENSITIVE_STATS = [
    "passes_attempted", "tackles", "dribbles_attempted", "clearances", "crosses",
]

# Set-piece taker bonus multiplier applied to these stats
SET_PIECE_STATS = ["crosses", "shots_assisted"]
SET_PIECE_BONUS = 1.3

# Manual input file locations (you create/edit these)
SET_PIECE_TAKERS_FILE = f"{PROCESSED_DIR}/set_piece_takers.csv"   # team,player
VENUE_FACTORS_FILE = f"{RAW_DIR}/venue_factors.csv"               # venue,altitude_factor,heat_factor

# International match data (national team appearances)
INTERNATIONAL_TABLE = f"{PROCESSED_DIR}/international_form_table.parquet"

