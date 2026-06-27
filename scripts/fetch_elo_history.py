import csv
import io
import json
import os
import re
import time
from pathlib import Path
import unicodedata

import polars as pl
import requests

DATA_DIR = Path("data")
OUTPUT_ELO_HISTORY = DATA_DIR / "elo_history.parquet"
OUTPUT_TEAM_CODES = DATA_DIR / "team_code_map.json"

ELO_BASE = "https://eloratings.net"
TEAMS_TSV = f"{ELO_BASE}/en.teams.tsv"
TOURNAMENTS_TSV = f"{ELO_BASE}/en.tournaments.tsv"
RESULTS_URL = "https://raw.githubusercontent.com/martj42/international_results/refs/heads/master/results.csv"

HEADERS = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36"}
SESSION = requests.Session()
SESSION.headers.update(HEADERS)


def get_with_retry(url: str, max_retries=3, timeout=30) -> requests.Response:
    for attempt in range(max_retries):
        try:
            r = SESSION.get(url, timeout=timeout)
            r.raise_for_status()
            return r
        except (requests.ConnectionError, requests.Timeout) as e:
            if attempt == max_retries - 1:
                raise
            print(f"   Retry {attempt + 1}/{max_retries} for {url}")
            time.sleep(2 ** attempt)
    raise RuntimeError(f"Failed to download {url}")


def fix_double_encoding(name: str) -> str:
    try:
        return name.encode("latin-1").decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError):
        return name


def strip_accents(s: str) -> str:
    nfkd = unicodedata.normalize("NFKD", s)
    return "".join(c for c in nfkd if not unicodedata.category(c).startswith("M"))


def name_to_url(name: str) -> str:
    clean = strip_accents(name).strip()
    clean = re.sub(r"[^a-zA-Z0-9\s-]", "", clean)
    clean = clean.replace(" ", "_")
    return clean


def download_team_names() -> dict[str, tuple[str, str]]:
    r = get_with_retry(TEAMS_TSV)

    code_to_name: dict[str, str] = {}
    name_to_code: dict[str, str] = {}
    name_to_urlslug: dict[str, str] = {}

    reader = csv.reader(io.StringIO(r.text), delimiter="\t")
    for row in reader:
        if not row or len(row) < 2:
            continue
        code = row[0].strip()
        primary_name = unicodedata.normalize("NFC", fix_double_encoding(row[1].strip()))
        url_slug = name_to_url(primary_name)

        code_to_name[code] = primary_name
        name_to_code[primary_name] = code
        name_to_urlslug[primary_name] = url_slug

        for alt in row[2:]:
            alt = unicodedata.normalize("NFC", fix_double_encoding(alt.strip()))
            if alt and alt not in name_to_code:
                name_to_code[alt] = code

    return code_to_name, name_to_code, name_to_urlslug


def download_tournament_names() -> dict[str, str]:
    r = get_with_retry(TOURNAMENTS_TSV)
    mapping: dict[str, str] = {}
    reader = csv.reader(io.StringIO(r.text), delimiter="\t")
    for row in reader:
        if not row or len(row) < 2:
            continue
        mapping[row[0].strip()] = row[1].strip()
    return mapping


def get_teams_from_results() -> set[str]:
    r = get_with_retry(RESULTS_URL, timeout=60)
    teams: set[str] = set()
    reader = csv.DictReader(io.StringIO(r.text))
    for row in reader:
        teams.add(row["home_team"].strip())
        teams.add(row["away_team"].strip())
    return teams


def normalize_for_elo(name: str) -> str:
    name = unicodedata.normalize("NFC", name)
    m = {
        "Czech Republic": "Czechia",
        "Bosnia-Herzegovina": "Bosnia and Herzegovina",
        "Congo DR": "DR Congo",
        "United States of America": "United States",
        "Korea Republic": "South Korea",
        "Côte d'Ivoire": "Ivory Coast",
        "Curacao": "Curaçao",
        "Türkiye": "Turkey",
        "Republic of Ireland": "Ireland",
        "German DR": "Germany DR",
        "Vietnam Republic": "Vietnam",
        "United States Virgin Islands": "US Virgin Islands",
        "Timor-Leste": "East Timor",
    }
    return m.get(name, name)


def download_team_tsv(url_slug: str) -> str | None:
    url = f"{ELO_BASE}/{url_slug}.tsv"
    for attempt in range(2):
        try:
            r = SESSION.get(url, timeout=30)
            if r.status_code == 404:
                return None
            r.raise_for_status()
            return r.text
        except (requests.ConnectionError, requests.Timeout) as e:
            if attempt == 1:
                return None
            time.sleep(2)
    return None


def parse_team_tsv(
    text: str,
    team_name: str,
    team_code: str,
    code_to_name: dict[str, str],
    tourn_map: dict[str, str],
) -> list[dict]:
    rows = []
    reader = csv.reader(io.StringIO(text), delimiter="\t")
    for row in reader:
        if len(row) < 16:
            continue
        year, month, day = row[0], row[1], row[2]
        home_code = row[3].strip()
        away_code = row[4].strip()
        home_score = int(row[5])
        away_score = int(row[6])
        tourn_code = row[7].strip()
        neutral = row[8].strip() if len(row) > 8 else ""
        elo_change = int(row[9]) if row[9] else 0
        home_elo = int(row[10]) if row[10] else 1500
        away_elo = int(row[11]) if row[11] else 1500
        home_rank_change = row[12].strip() if len(row) > 12 else ""
        away_rank_change = row[13].strip() if len(row) > 13 else ""
        home_rank_str = row[14].strip() if len(row) > 14 else ""
        away_rank_str = row[15].strip() if len(row) > 15 else ""

        try:
            home_rank = int(home_rank_str)
        except ValueError:
            home_rank = -1
        try:
            away_rank = int(away_rank_str)
        except ValueError:
            away_rank = -1

        date_str = f"{year}-{month.zfill(2)}-{day.zfill(2)}"
        home_name = code_to_name.get(home_code, home_code)
        away_name = code_to_name.get(away_code, away_code)
        tourn_name = tourn_map.get(tourn_code, tourn_code)

        if team_code == home_code:
            team_elo = home_elo
            team_rank = home_rank
            team_goals_for = home_score
            team_goals_against = away_score
            opp_name = away_name
        else:
            team_elo = away_elo
            team_rank = away_rank
            team_goals_for = away_score
            team_goals_against = home_score
            opp_name = home_name

        rows.append({
            "team": team_name,
            "date": date_str,
            "year": int(year),
            "month": int(month),
            "day": int(day),
            "opponent": opp_name,
            "home_team": home_name,
            "away_team": away_name,
            "home_score": home_score,
            "away_score": away_score,
            "goals_for": team_goals_for,
            "goals_against": team_goals_against,
            "tournament_code": tourn_code,
            "tournament": tourn_name,
            "neutral": neutral,
            "elo_before": team_elo,
            "rank_before": team_rank,
            "elo_change": elo_change,
            "elo_opponent": away_elo if team_code == home_code else home_elo,
        })
    return rows


def main():
    print("=" * 60)
    print("  ELO HISTORY EXTRACTION")
    print("=" * 60)
    DATA_DIR.mkdir(exist_ok=True)

    print("\n[1] Downloading team name mappings...")
    code_to_name, name_to_code, name_to_urlslug = download_team_names()
    print(f"   {len(name_to_code)} names, {len(code_to_name)} codes in eloratings")

    print("\n[2] Downloading tournament name mappings...")
    tourn_map = download_tournament_names()
    print(f"   {len(tourn_map)} tournament codes mapped")

    print("\n[3] Identifying teams from training data...")
    results_teams = get_teams_from_results()
    print(f"   {len(results_teams)} unique teams in results.csv")

    print("\n[4] Mapping team names to eloratings entries...")
    mapped_codes: set[str] = set()
    mapped_names: set[str] = set()
    unmapped: list[str] = []

    for team_name in sorted(results_teams):
        mapped_name = normalize_for_elo(team_name)
        code = name_to_code.get(mapped_name)
        if code:
            mapped_codes.add(code)
            mapped_names.add(code_to_name[code])
        else:
            unmapped.append(team_name)

    if unmapped:
        print(f"   WARNING: {len(unmapped)} teams unmapped:")
        for t in unmapped:
            print(f"     - {t}")

    print(f"   {len(mapped_codes)} unique team codes mapped")
    code_list = sorted(mapped_codes)
    name_list = sorted(mapped_names)

    code_map_out = {}
    for code in code_list:
        name = code_to_name.get(code, code)
        slug = name_to_urlslug.get(name, name_to_url(name))
        code_map_out[code] = {"name": name, "slug": slug}

    with open(OUTPUT_TEAM_CODES, "w", encoding="utf-8") as f:
        json.dump(code_map_out, f, indent=2, ensure_ascii=False)
    print(f"   Code map saved to {OUTPUT_TEAM_CODES}")

    print(f"\n[5] Downloading TSVs for {len(code_list)} teams...")
    all_matches: list[dict] = []
    errors = 0

    for i, code in enumerate(code_list):
        name = code_to_name.get(code, code)
        slug = name_to_urlslug.get(name, name_to_url(name))
        if i > 0 and i % 25 == 0:
            print(f"   Progress: {i}/{len(code_list)}...")
        tsv = download_team_tsv(slug)
        if tsv is None:
            print(f"   WARNING: No TSV for {name} (slug: {slug})")
            errors += 1
            continue
        matches = parse_team_tsv(tsv, name, code, code_to_name, tourn_map)
        all_matches.extend(matches)

    print(f"   Downloaded {len(code_list) - errors}/{len(code_list)} team files")
    print(f"   Total match records: {len(all_matches)}")

    print(f"\n[6] Saving to {OUTPUT_ELO_HISTORY}...")
    df = pl.DataFrame(all_matches)
    df = df.sort("date")
    df.write_parquet(OUTPUT_ELO_HISTORY)
    print(f"   Rows: {df.height}")
    print(f"   Columns: {df.width}")
    print(f"   Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"   Unique teams: {df['team'].n_unique()}")
    print(f"   Memory: {df.estimated_size('mb'):.1f} MB")

    team_stats = df.group_by("team").agg([
        pl.col("date").count().alias("matches"),
        pl.col("elo_before").last().alias("current_elo"),
    ]).sort("current_elo", descending=True)
    print("\n   Top 10 by current ELO:")
    for row in team_stats.head(10).iter_rows(named=True):
        print(f"     {row['team']:30s}  ELO: {row['current_elo']:5.0f}  ({row['matches']} matches)")

    print("\nDone!")


if __name__ == "__main__":
    main()
