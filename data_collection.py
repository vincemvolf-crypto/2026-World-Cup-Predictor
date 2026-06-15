"""
data_collection.py

Pulls the raw data the pipeline needs, all from free sources:

1. Player match-level stats from FBref (via the `soccerdata` library)
   - "Recent form" source: last completed club season(s) for the Big 5 leagues
   - "Tournament" source: World Cup 2026 matches as they're played
2. National team Elo ratings from eloratings.net (opponent-strength proxy)
3. World Cup 2026 fixtures from openfootball (free JSON, no key)

Run this on a schedule (e.g. nightly during the tournament) to refresh data.

NOTE: FBref's robots.txt asks for a polite crawl rate. `soccerdata` already
caches responses to disk and rate-limits requests - don't disable that.

Install deps first:
    pip install soccerdata pandas requests beautifulsoup4 lxml
"""

import os
import time
import requests
import pandas as pd

import config


def _ensure_dirs():
    os.makedirs(config.RAW_DIR, exist_ok=True)
    os.makedirs(config.PROCESSED_DIR, exist_ok=True)


def fetch_player_match_stats(leagues, season, label):
    """
    Pull all the FBref stat tables we need (standard, shooting, passing,
    defense, possession, gca, keeper) for the given league(s)/season and
    merge them into one long-format dataframe of per-player-per-match rows.
    """
    import soccerdata as sd

    fbref = sd.FBref(leagues=leagues, seasons=[season])

    stat_types = ["standard", "shooting", "passing", "defense",
                   "possession", "gca", "keeper", "misc"]

    frames = []
    for stat_type in stat_types:
        print(f"  fetching '{stat_type}' for {label}...")
        try:
            df = fbref.read_player_match_stats(stat_type=stat_type)
        except Exception as e:
            print(f"    !! failed: {e}")
            continue
        df = df.reset_index()
        df["stat_type"] = stat_type
        frames.append(df)
        time.sleep(1)  # be polite even though soccerdata already throttles

    if not frames:
        raise RuntimeError(f"No data fetched for {label}")

    combined = pd.concat(frames, ignore_index=True)
    out_path = f"{config.RAW_DIR}/fbref_{label}_raw.parquet"
    combined.to_parquet(out_path)
    print(f"  saved {len(combined):,} rows -> {out_path}")
    return combined


def fetch_recent_club_form():
    """Most recently completed club season for the Big 5 leagues."""
    print("Fetching recent club-season form (Big 5 leagues)...")
    return fetch_player_match_stats(
        leagues=config.CLUB_LEAGUES,
        season=config.CLUB_SEASON,
        label="club_form",
    )


def fetch_world_cup_matches():
    """World Cup 2026 player match stats (updates as the tournament progresses)."""
    print("Fetching World Cup 2026 match data...")
    return fetch_player_match_stats(
        leagues=[config.WC_LEAGUE],
        season=config.WC_SEASON,
        label="world_cup",
    )


def fetch_national_team_elo():
    """
    National-team Elo ratings from eloratings.net.
    Used as the 'opponent quality' feature.

    eloratings.net doesn't have a documented JSON API, so this scrapes the
    ratings table. If the site structure changes, update the parsing here -
    everything downstream just expects columns: team, elo.
    """
    print("Fetching national team Elo ratings...")
    url = "https://www.eloratings.net/World.tsv"  # bulk TSV export
    try:
        resp = requests.get(url, timeout=20)
        resp.raise_for_status()
        from io import StringIO
        df = pd.read_csv(StringIO(resp.text), sep="\t", header=None)
        # Columns vary by export version - inspect df.head() and adjust.
        # Typically: rank, team, elo, ...
        df.columns = [f"col_{i}" for i in range(df.shape[1])]
        df = df.rename(columns={"col_1": "team", "col_2": "elo"})
        df = df[["team", "elo"]]
    except Exception as e:
        print(f"  !! Elo fetch failed ({e}); falling back to manual CSV.")
        print("  -> Create data/raw/national_elo.csv manually with columns: team,elo")
        return None

    out_path = f"{config.RAW_DIR}/national_elo.csv"
    df.to_csv(out_path, index=False)
    print(f"  saved -> {out_path}")
    return df


def fetch_wc_fixtures():
    """World Cup 2026 fixtures from openfootball (free, public domain JSON)."""
    print("Fetching World Cup 2026 fixtures...")
    url = "https://raw.githubusercontent.com/openfootball/worldcup.json/master/2026/wc2026.json"
    try:
        resp = requests.get(url, timeout=20)
        resp.raise_for_status()
        data = resp.json()
        matches = pd.json_normalize(data.get("matches", []))
        out_path = f"{config.RAW_DIR}/wc2026_fixtures.csv"
        matches.to_csv(out_path, index=False)
        print(f"  saved {len(matches)} fixtures -> {out_path}")
        return matches
    except Exception as e:
        print(f"  !! Fixture fetch failed: {e}")
        return None


def fetch_international_matches():
    """
    Pull recent international match stats (World Cup qualifiers, Nations
    League, friendlies, continental championships) so the model can see
    how a player performs for their national team specifically - their
    role often differs from their club role.

    `soccerdata`'s FBref wrapper supports several international
    competitions by name (e.g. "Copa America", "UEFA Euro Qualifying").
    There's no single "all internationals" league code, so this pulls a
    few of the bigger ones - extend the `competitions` list for more
    coverage of the teams you care about.

    If a competition name isn't recognized by soccerdata (these change
    over time), it's skipped with a warning rather than failing the whole
    run.
    """
    print("Fetching international match data...")
    competitions = [
        "Copa America",
        "UEFA Euro Qualifying",
        "UEFA Nations League",
    ]

    frames = []
    for comp in competitions:
        try:
            df = fetch_player_match_stats(
                leagues=[comp], season=config.WC_SEASON,
                label=f"intl_{comp.replace(' ', '_')}"
            )
            frames.append(df)
        except Exception as e:
            print(f"  skipping '{comp}': {e}")

    if not frames:
        print("  No international data fetched. The model will run on club "
              "form only (still works fine - see README on this tradeoff). "
              "Alternative: manually export player match logs from each "
              "national team's FBref page into "
              "data/raw/fbref_international_raw.parquet using the same "
              "column layout as the club data, then it'll be picked up "
              "automatically.")
        return None

    combined = pd.concat(frames, ignore_index=True)
    out_path = f"{config.RAW_DIR}/fbref_international_raw.parquet"
    combined.to_parquet(out_path)
    print(f"  saved {len(combined):,} rows -> {out_path}")
    return combined


def main():
    _ensure_dirs()
    fetch_recent_club_form()
    fetch_world_cup_matches()
    fetch_international_matches()
    fetch_national_team_elo()
    fetch_wc_fixtures()
    print("Done. Raw data is in data/raw/")


if __name__ == "__main__":
    main()
