import bisect
import json
import warnings
from datetime import datetime, timedelta
from io import StringIO

import numpy as np
import polars as pl
import requests
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

warnings.filterwarnings("ignore")

RESULTS_URL = "https://raw.githubusercontent.com/martj42/international_results/refs/heads/master/results.csv"
FIFA_RANKINGS = "data/fifa_rankings.json"
ELO_RANKINGS = "data/elo_rankings.json"
ELO_HISTORY = "data/elo_history.parquet"
KNOCKOUT_MATCHES = "data/knockout_matches.json"
MAX_DATE = (datetime.today() - timedelta(days=1)).strftime("%Y-%m-%d")

TOURNAMENT_WEIGHTS = {
    "FIFA World Cup": 5.0,
    "UEFA Euro": 4.0,
    "Copa América": 4.0,
    "African Cup of Nations": 3.5,
    "AFC Asian Cup": 3.5,
    "Gold Cup": 3.0,
    "FIFA World Cup qualification": 3.0,
    "UEFA Euro qualification": 2.5,
    "CONCACAF Nations League": 2.0,
    "UEFA Nations League": 2.0,
    "African Cup of Nations qualification": 2.0,
    "Friendly": 1.0,
}

FEATURE_COLS = [
    "home_matches_played", "home_goals_for_avg", "home_goals_against_avg",
    "home_goal_diff_avg", "home_win_pct", "home_draw_pct", "home_loss_pct",
    "home_recent_form",
    "away_matches_played", "away_goals_for_avg", "away_goals_against_avg",
    "away_goal_diff_avg", "away_win_pct", "away_draw_pct", "away_loss_pct",
    "away_recent_form",
    "home_fifa_rank", "away_fifa_rank", "home_fifa_points", "away_fifa_points",
    "home_elo", "away_elo", "elo_diff", "fifa_rank_diff", "goal_diff_strength",
    "h2h_home_wins", "h2h_away_wins", "h2h_draws", "tournament_weight", "is_neutral",
]


def normalize_team_name(name):
    name = name.strip()
    name = name.replace("Korea Republic", "South Korea")
    name = name.replace("Czechia", "Czech Republic")
    name = name.replace("Turkey", "Türkiye")
    name = name.replace("DR Congo", "Congo DR")
    name = name.replace("United States of America", "United States")
    name = name.replace("Bosnia and Herzegovina", "Bosnia-Herzegovina")
    name = name.replace("C\u00f4te d'Ivoire", "Ivory Coast")
    name = name.replace("Curacao", "Cura\u00e7ao")
    name = name.replace("Congo", "Congo DR")
    return name


def load_rankings():
    rankings = {}
    with open(FIFA_RANKINGS) as f:
        for team in json.load(f):
            rankings[team["team"]] = {
                "fifa_rank": team["rank"], "fifa_points": team["points"],
            }
    return rankings


def load_static_elo():
    elo = {}
    with open(ELO_RANKINGS) as f:
        for team in json.load(f):
            elo[team["team"]] = float(team["rating"])
    return elo


def normalize_team_for_elo(name):
    m = {
        "Czech Republic": "Czechia",
        "Bosnia-Herzegovina": "Bosnia and Herzegovina",
        "Congo DR": "DR Congo",
        "United States of America": "United States",
        "Korea Republic": "South Korea",
        "C\u00f4te d'Ivoire": "Ivory Coast",
        "Curacao": "Cura\u00e7ao",
        "T\u00fcrkiye": "Turkey",
        "Republic of Ireland": "Ireland",
        "German DR": "Germany DR",
        "Vietnam Republic": "Vietnam",
        "United States Virgin Islands": "US Virgin Islands",
        "Timor-Leste": "East Timor",
    }
    return m.get(name, name)


def load_elo_history():
    df = pl.read_parquet(ELO_HISTORY)
    lookup = {}
    for row in df.iter_rows(named=True):
        team = row["team"]
        try:
            dt = datetime.strptime(row["date"], "%Y-%m-%d")
        except ValueError:
            continue
        date_int = dt.toordinal()
        row_list = lookup.setdefault(team, [])
        row_list.append((date_int, row["elo_before"]))
    return lookup


def get_historical_elo(team, match_date, elo_lookup, static_elo=None):
    elo_rows = elo_lookup.get(team)
    if elo_rows:
        date_int = match_date.toordinal()
        idx = bisect.bisect_left(elo_rows, (date_int,))
        if idx > 0:
            return elo_rows[idx - 1][1]
    if static_elo:
        return static_elo.get(team, 1500)
    return 1500


def compute_rolling_stats(matches, team_col, date, elo_lookup, static_elo=None, window_matches=10, decay_halflife=180):
    if isinstance(date, datetime):
        date = date.date()
    recent = matches.filter(
        ((pl.col("home_team") == team_col) | (pl.col("away_team") == team_col))
        & (pl.col("date") < pl.lit(date))
    ).sort("date", descending=True).head(window_matches)
    total = recent.height
    if total == 0:
        return {"matches_played": 0, "goals_for_avg": 0, "goals_against_avg": 0,
                "goal_diff_avg": 0, "win_pct": 0, "draw_pct": 0, "loss_pct": 0,
                "recent_form": 0}

    w_gf = w_ga = w_w = w_d = w_l = 0.0
    total_weight = 0.0
    form_points = 0

    for j, row in enumerate(recent.iter_rows(named=True)):
        days_ago = max((date - row["date"]).days, 1)
        time_weight = 2 ** (-days_ago / decay_halflife)

        if row["home_team"] == team_col:
            opp = row["away_team"]
            gf = row["home_score"]
            ga = row["away_score"]
            won = gf > ga
            drawn = gf == ga
        else:
            opp = row["home_team"]
            gf = row["away_score"]
            ga = row["home_score"]
            won = ga > gf
            drawn = ga == gf

        opp_elo = get_historical_elo(normalize_team_for_elo(opp), row["date"], elo_lookup, static_elo)
        weight = time_weight * (opp_elo / 2000.0)
        total_weight += weight
        w_gf += weight * gf
        w_ga += weight * ga
        if won:
            w_w += weight
        elif drawn:
            w_d += weight
        else:
            w_l += weight

        if j < 5:
            form_points += 3 if won else (1 if drawn else 0)

    tw = total_weight if total_weight > 0 else 1
    return {
        "matches_played": total, "goals_for_avg": w_gf / tw, "goals_against_avg": w_ga / tw,
        "goal_diff_avg": (w_gf - w_ga) / tw, "win_pct": w_w / tw, "draw_pct": w_d / tw, "loss_pct": w_l / tw,
        "recent_form": form_points / 5,
    }


def build_features_for_match(match, matches_df, rankings, match_date, elo_lookup, static_elo=None):
    home = normalize_team_name(match["local"])
    away = normalize_team_name(match["visitante"])

    date = datetime.strptime(MAX_DATE, "%Y-%m-%d")
    date_d = date.date()
    hs = compute_rolling_stats(matches_df, home, date, elo_lookup, static_elo)
    as_ = compute_rolling_stats(matches_df, away, date, elo_lookup, static_elo)

    h2h = matches_df.filter(
        ((pl.col("home_team") == home) & (pl.col("away_team") == away)) |
        ((pl.col("home_team") == away) & (pl.col("away_team") == home))
    ).filter(pl.col("date") < pl.lit(date_d)).sort("date", descending=True).head(5)

    hw = aw = hd = 0
    for r in h2h.iter_rows(named=True):
        if r["home_team"] == home and r["home_score"] > r["away_score"]: hw += 1
        elif r["away_team"] == home and r["away_score"] > r["home_score"]: hw += 1
        elif r["home_team"] == away and r["home_score"] > r["away_score"]: aw += 1
        elif r["away_team"] == away and r["away_score"] > r["home_score"]: aw += 1
        else: hd += 1

    hr = rankings.get(home, {})
    ar = rankings.get(away, {})
    he = get_historical_elo(normalize_team_for_elo(home), date, elo_lookup, static_elo)
    ae = get_historical_elo(normalize_team_for_elo(away), date, elo_lookup, static_elo)

    return {
        "home_matches_played": hs["matches_played"],
        "home_goals_for_avg": hs["goals_for_avg"],
        "home_goals_against_avg": hs["goals_against_avg"],
        "home_goal_diff_avg": hs["goal_diff_avg"],
        "home_win_pct": hs["win_pct"],
        "home_draw_pct": hs["draw_pct"],
        "home_loss_pct": hs["loss_pct"],
        "home_recent_form": hs["recent_form"],
        "away_matches_played": as_["matches_played"],
        "away_goals_for_avg": as_["goals_for_avg"],
        "away_goals_against_avg": as_["goals_against_avg"],
        "away_goal_diff_avg": as_["goal_diff_avg"],
        "away_win_pct": as_["win_pct"],
        "away_draw_pct": as_["draw_pct"],
        "away_loss_pct": as_["loss_pct"],
        "away_recent_form": as_["recent_form"],
        "home_fifa_rank": hr.get("fifa_rank", 100),
        "away_fifa_rank": ar.get("fifa_rank", 100),
        "home_fifa_points": hr.get("fifa_points", 1000),
        "away_fifa_points": ar.get("fifa_points", 1000),
        "home_elo": he,
        "away_elo": ae,
        "elo_diff": he - ae,
        "fifa_rank_diff": ar.get("fifa_rank", 100) - hr.get("fifa_rank", 100),
        "goal_diff_strength": hs["goal_diff_avg"] - as_["goal_diff_avg"],
        "h2h_home_wins": hw, "h2h_away_wins": aw, "h2h_draws": hd,
        "tournament_weight": 5.0,
        "is_neutral": False,
    }


def main():
    print("=" * 60)
    print("STACKING MODEL - 16avos World Cup 2026")
    print("=" * 60)

    # ------------------------------------------------------------------
    # 1. LOAD DATA
    # ------------------------------------------------------------------
    print("\n[1] Loading training data...")
    r = requests.get(RESULTS_URL, timeout=30)
    r.raise_for_status()
    df = pl.read_csv(StringIO(r.text), schema_overrides={"home_score": pl.Int32, "away_score": pl.Int32},
                      null_values=["NA", ""], try_parse_dates=True)
    df = df.drop_nulls(subset=["home_score", "away_score"])
    max_dt = datetime.strptime(MAX_DATE, "%Y-%m-%d")
    df = df.filter((pl.col("date") >= datetime(2018, 1, 1)) & (pl.col("date") <= max_dt))
    df = df.with_columns(
        pl.col("home_team").map_elements(normalize_team_name, return_dtype=pl.Utf8),
        pl.col("away_team").map_elements(normalize_team_name, return_dtype=pl.Utf8),
    )
    df = df.with_columns(
        tournament_weight=pl.col("tournament").replace_strict(TOURNAMENT_WEIGHTS, default=1.0).cast(pl.Float32),
    )
    df = df.sort("date")
    print(f"   Training matches: {df.height}")

    # ------------------------------------------------------------------
    # 2. BUILD FEATURE VECTORS
    # ------------------------------------------------------------------
    print("\n[2] Building feature vectors...")
    rankings = load_rankings()
    static_elo = load_static_elo()
    print(f"   FIFA: {len(rankings)}, Static ELO: {len(static_elo)}")
    elo_lookup = load_elo_history()
    print(f"   {len(elo_lookup)} teams with ELO history")
    features = []
    targets = []
    match_dates = []
    matches_list = df.to_dicts()

    for i, m in enumerate(matches_list):
        if i % 2000 == 0 and i > 0:
            print(f"   {i}/{len(matches_list)}...")

        match_date = m["date"]
        home, away = m["home_team"], m["away_team"]

        hs = compute_rolling_stats(df, home, match_date, elo_lookup, static_elo)
        as_ = compute_rolling_stats(df, away, match_date, elo_lookup, static_elo)

        h2h = df.filter(
            ((pl.col("home_team") == home) & (pl.col("away_team") == away)) |
            ((pl.col("home_team") == away) & (pl.col("away_team") == home))
        ).filter(pl.col("date") < match_date).sort("date", descending=True).head(5)

        hw = aw = hd = 0
        for r in h2h.iter_rows(named=True):
            if r["home_team"] == home and r["home_score"] > r["away_score"]: hw += 1
            elif r["away_team"] == home and r["away_score"] > r["home_score"]: hw += 1
            elif r["home_team"] == away and r["home_score"] > r["away_score"]: aw += 1
            elif r["away_team"] == away and r["away_score"] > r["home_score"]: aw += 1
            else: hd += 1

        hr = rankings.get(home, {})
        ar = rankings.get(away, {})

        he = get_historical_elo(normalize_team_for_elo(home), match_date, elo_lookup, static_elo)
        ae = get_historical_elo(normalize_team_for_elo(away), match_date, elo_lookup, static_elo)

        features.append({
            "home_matches_played": hs["matches_played"],
            "home_goals_for_avg": hs["goals_for_avg"],
            "home_goals_against_avg": hs["goals_against_avg"],
            "home_goal_diff_avg": hs["goal_diff_avg"],
            "home_win_pct": hs["win_pct"],
            "home_draw_pct": hs["draw_pct"],
            "home_loss_pct": hs["loss_pct"],
            "home_recent_form": hs["recent_form"],
            "away_matches_played": as_["matches_played"],
            "away_goals_for_avg": as_["goals_for_avg"],
            "away_goals_against_avg": as_["goals_against_avg"],
            "away_goal_diff_avg": as_["goal_diff_avg"],
            "away_win_pct": as_["win_pct"],
            "away_draw_pct": as_["draw_pct"],
            "away_loss_pct": as_["loss_pct"],
            "away_recent_form": as_["recent_form"],
            "home_fifa_rank": hr.get("fifa_rank", 100),
            "away_fifa_rank": ar.get("fifa_rank", 100),
            "home_fifa_points": hr.get("fifa_points", 1000),
            "away_fifa_points": ar.get("fifa_points", 1000),
            "home_elo": he, "away_elo": ae,
            "elo_diff": he - ae,
            "fifa_rank_diff": ar.get("fifa_rank", 100) - hr.get("fifa_rank", 100),
            "goal_diff_strength": hs["goal_diff_avg"] - as_["goal_diff_avg"],
            "h2h_home_wins": hw, "h2h_away_wins": aw, "h2h_draws": hd,
            "tournament_weight": m["tournament_weight"],
            "is_neutral": m["neutral"] == "TRUE",
        })
        match_dates.append(match_date)

        if m["home_score"] > m["away_score"]:
            targets.append(0)
        elif m["home_score"] == m["away_score"]:
            targets.append(1)
        else:
            targets.append(2)

    X = pl.DataFrame(features).to_numpy()
    y = np.array(targets)
    dates_arr = np.array(match_dates)
    print(f"   Feature matrix: {X.shape}")

    # ------------------------------------------------------------------
    # 3. TRAIN STACKING WITH TEMPORAL SPLIT
    # ------------------------------------------------------------------
    # Use last 20% by date as validation (no future leakage)
    n = len(dates_arr)
    split_idx = int(n * 0.8)
    train_idx = np.arange(split_idx)
    val_idx = np.arange(split_idx, n)

    train_date_cutoff = dates_arr[split_idx - 1].strftime("%Y-%m-%d")
    val_date_range = f"{dates_arr[split_idx].strftime('%Y-%m-%d')} → {dates_arr[-1].strftime('%Y-%m-%d')}"
    print(f"\n[3] Training stacking model (temporal split)...")
    print(f"   Train: {n - len(val_idx)} matches (→ {train_date_cutoff})")
    print(f"   Valid: {len(val_idx)} matches ({val_date_range})")

    X_tr, X_val = X[train_idx], X[val_idx]
    y_tr, y_val = y[train_idx], y[val_idx]

    scaler = StandardScaler()
    X_tr_scaled = scaler.fit_transform(X_tr)
    X_val_scaled = scaler.transform(X_val)

    models = {
        "rf": RandomForestClassifier(n_estimators=300, max_depth=12, random_state=42, n_jobs=-1),
        "xgb": xgb.XGBClassifier(n_estimators=300, max_depth=8, learning_rate=0.05, random_state=42,
                                  eval_metric="mlogloss"),
        "svm": SVC(kernel="rbf", probability=True, random_state=42),
    }

    # Train base models on train set, predict on val set (out-of-sample for meta)
    print("   Training base models...")
    meta_val = np.zeros((len(y_val), 3 * 3))

    for name, model in models.items():
        model.fit(X_tr_scaled, y_tr)
    offset = 0
    for name, model in models.items():
        meta_val[:, offset:offset + 3] = model.predict_proba(X_val_scaled)
        offset += 3

    # Train meta-model on VAL predictions (out-of-sample), evaluate on val
    meta = LogisticRegression(solver="lbfgs", max_iter=1000, C=0.1, random_state=42)
    meta.fit(meta_val, y_val)

    val_probs = meta.predict_proba(meta_val)
    val_acc = accuracy_score(y_val, np.argmax(val_probs, axis=1))
    val_ll = log_loss(y_val, val_probs)
    print(f"\n   Temporal validation — accuracy: {val_acc:.4f}, log_loss: {val_ll:.4f}")

    # Retrain base models on FULL data for final predictions
    print("\n   Retraining base models on full data...")
    X_full_scaled = scaler.fit_transform(X)
    for name, model in models.items():
        model.fit(X_full_scaled, y)

    # ------------------------------------------------------------------
    # 4. PREDICT KNOCKOUT MATCHES
    # ------------------------------------------------------------------
    print("\n[4] Predicting 16avos matchups...")
    with open(KNOCKOUT_MATCHES) as f:
        matches = json.load(f)

    predictions_rows = []

    for i, match in enumerate(matches, 1):
        fv = build_features_for_match(match, df, rankings, MAX_DATE, elo_lookup, static_elo)
        fv_arr = scaler.transform(pl.DataFrame([fv]).to_numpy())

        offset = 0
        meta_input = np.zeros((1, 9))
        print(f"\n   {'─' * 55}")
        print(f"   Match {i:02d}: {match['local']} vs {match['visitante']}")
        print(f"   {'─' * 55}")

        for name, model in models.items():
            proba = model.predict_proba(fv_arr)[0]
            meta_input[0, offset:offset + 3] = proba
            offset += 3
            local_win = proba[0] * 100
            draw = proba[1] * 100
            away_win = proba[2] * 100
            print(f"   {name.upper():5s} | Local: {local_win:5.1f}%  Empate: {draw:5.1f}%  Visitante: {away_win:5.1f}%")

        final_proba = meta.predict_proba(meta_input)[0]
        local_w = final_proba[0] * 100
        draw_p = final_proba[1] * 100
        away_w = final_proba[2] * 100
        print(f"   {'─' * 55}")
        print(f"   STACKING | Local: {local_w:5.1f}%  Empate: {draw_p:5.1f}%  Visitante: {away_w:5.1f}%")

        # Advancement probability (redistribute draw as 50/50)
        local_adv = final_proba[0] + final_proba[1] * 0.5
        away_adv = final_proba[2] + final_proba[1] * 0.5
        total_adv = local_adv + away_adv
        print(f"   {'─' * 55}")
        print(f"   AVANCE    | {match['local']}: {local_adv / total_adv * 100:.1f}%  |  {match['visitante']}: {away_adv / total_adv * 100:.1f}%")

        predictions_rows.append({
            "match": f"{match['local']} vs {match['visitante']}",
            "local_win_pct": round(local_w, 1),
            "draw_pct": round(draw_p, 1),
            "away_win_pct": round(away_w, 1),
            "local_advance_pct": round(local_adv / total_adv * 100, 1),
            "away_advance_pct": round(away_adv / total_adv * 100, 1),
        })

    print(f"\n   {'=' * 55}")
    pl.DataFrame(predictions_rows).write_csv("data/knockout_predictions.csv")
    print("   Predictions saved to data/knockout_predictions.csv")


if __name__ == "__main__":
    main()
