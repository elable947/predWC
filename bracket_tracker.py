import json
import sys
from datetime import datetime, timedelta
from io import StringIO

import numpy as np
import polars as pl
import requests

RESULTS_URL = "https://raw.githubusercontent.com/martj42/international_results/refs/heads/master/results.csv"
KNOCKOUT_MATCHES = "data/knockout_matches.json"
PREDICTIONS = "data/knockout_predictions.csv"
PREDICTIONS_NLP = "data/knockout_predictions_nlp.csv"
WC_YEAR = 2026


def normalize_team_name(name):
    name = name.strip()
    name = name.replace("Korea Republic", "South Korea")
    name = name.replace("Czechia", "Czech Republic")
    name = name.replace("Turkey", "T\u00fcrkiye")
    name = name.replace("DR Congo", "Congo DR")
    name = name.replace("United States of America", "United States")
    name = name.replace("Bosnia and Herzegovina", "Bosnia-Herzegovina")
    name = name.replace("C\u00f4te d'Ivoire", "Ivory Coast")
    name = name.replace("Curacao", "Cura\u00e7ao")
    if name == "Congo":
        name = "Congo DR"
    return name


def fetch_actual_results():
    r = requests.get(RESULTS_URL, timeout=30)
    r.raise_for_status()
    df = pl.read_csv(
        StringIO(r.text),
        schema_overrides={"home_score": pl.Int32, "away_score": pl.Int32},
        null_values=["NA", ""],
        try_parse_dates=True,
    )
    df = df.drop_nulls(subset=["home_score", "away_score"])
    df = df.with_columns(
        pl.col("home_team").map_elements(normalize_team_name, return_dtype=pl.Utf8),
        pl.col("away_team").map_elements(normalize_team_name, return_dtype=pl.Utf8),
    )
    df = df.filter(pl.col("date") >= datetime(WC_YEAR, 1, 1))
    return df


def find_actual_match(df, local, away):
    matches = df.filter(
        (
            ((pl.col("home_team") == local) & (pl.col("away_team") == away))
            | ((pl.col("home_team") == away) & (pl.col("away_team") == local))
        )
        & (pl.col("tournament") == "FIFA World Cup")
    ).sort("date", descending=True)
    if matches.height == 0:
        return None
    row = matches.row(0, named=True)
    if row["home_team"] == local:
        return {"date": row["date"], "local_score": row["home_score"], "away_score": row["away_score"]}
    else:
        return {"date": row["date"], "local_score": row["away_score"], "away_score": row["home_score"]}


def result_label(local, away, ls, aws):
    if ls > aws:
        return local, "Local"
    if ls < aws:
        return away, "Visitante"
    return "Empate", "Empate"


def winner_from_probs(local_pct, draw_pct, away_pct):
    if local_pct > away_pct and local_pct > draw_pct:
        return "Local"
    if away_pct > local_pct and away_pct > draw_pct:
        return "Visitante"
    return "Empate"


def format_score(goal_str):
    parts = goal_str.split("-")
    if len(parts) == 2:
        try:
            return int(parts[0]), int(parts[1])
        except ValueError:
            return None, None
    return None, None


def main():
    use_nlp = "--nlp" in sys.argv

    csv_path = PREDICTIONS_NLP if use_nlp else PREDICTIONS
    label = "NLP" if use_nlp else "BASE"

    print("=" * 68)
    print(f"  BRACKET TRACKER — 16avos World Cup 2026  [{label}]")
    print("=" * 68)

    with open(KNOCKOUT_MATCHES) as f:
        bracket = json.load(f)

    try:
        pred_df = pl.read_csv(csv_path)
    except FileNotFoundError:
        print(f"\n  ERROR: {csv_path} not found.")
        print(f"  Run: uv run python stacking_model.py{' --nlp' if use_nlp else ''}")
        sys.exit(1)

    print("\n  Fetching actual results...")
    actual_df = fetch_actual_results()
    print(f"  {actual_df.height} matches in {WC_YEAR}")

    rows = []
    correct = 0
    played = 0

    for match in bracket:
        local = match["local"]
        away = match["visitante"]

        pred_row = pred_df.filter(
            pl.col("match") == f"{local} vs {away}"
        )
        if pred_row.height == 0:
            continue
        pred = pred_row.row(0, named=True)

        actual = find_actual_match(actual_df, local, away)
        is_played = actual is not None

        pred_winner = winner_from_probs(pred["local_win_pct"], pred["draw_pct"], pred["away_win_pct"])

        if is_played:
            played += 1
            actual_winner_name, actual_winner_label = result_label(
                local, away, actual["local_score"], actual["away_score"]
            )
            correct_flag = pred_winner == actual_winner_label
            if correct_flag:
                correct += 1
        else:
            actual_winner_name = ""
            actual_winner_label = ""
            correct_flag = None

        rows.append({
            "local": local,
            "away": away,
            "played": is_played,
            "actual_local_goals": actual["local_score"] if is_played else None,
            "actual_away_goals": actual["away_score"] if is_played else None,
            "actual_winner": actual_winner_label if is_played else "",
            "pred_winner": pred_winner,
            "correct": correct_flag,
            "local_win_pct": pred["local_win_pct"],
            "draw_pct": pred["draw_pct"],
            "away_win_pct": pred["away_win_pct"],
            "pred_score": pred.get("most_likely_score", ""),
            "pred_score_pct": pred.get("most_likely_score_pct", 0),
            "exp_goals_local": pred.get("expected_goals_local", 0),
            "exp_goals_away": pred.get("expected_goals_away", 0),
        })

    print()
    print(f"  {'=' * 68}")
    print(f"  {'PARTIDO':^42s}  {'PRED':^12s}  {'REAL':^12s}")
    print(f"  {'=' * 68}")

    for r in rows:
        match_str = f"{r['local']:20s} vs {r['away']:20s}"

        if r["played"]:
            actual_str = f"{r['actual_local_goals']}-{r['actual_away_goals']}  {r['actual_winner'][:6]}"
            pred_str = f"{r['pred_score']:>5s}  {r['pred_winner'][:6]}"

            if r["correct"]:
                icon = "+"
                status_color = "OK"
            else:
                icon = "x"
                status_color = "ERROR"
        else:
            actual_str = f"{'—':^12s}"
            pred_str = f"{r['pred_score']:>5s}  {r['pred_winner'][:6]}"
            icon = " "
            status_color = "PEND"

        print(f"  {icon}  {match_str}  {pred_str:12s}  {actual_str:12s}")

    print(f"  {'=' * 68}")

    if played > 0:
        pct = correct / played * 100
        print(f"\n  Aciertos: {correct}/{played} ({pct:.1f}%)")
    else:
        print(f"\n  Aun no se ha jugado ningun partido de 16avos.")

    print(f"\n  Pendientes: {len(rows) - played}/{len(rows)}")

    print()
    print("  --- DETALLE POR PARTIDO ---")
    print()

    for r in rows:
        match_str = f"{r['local']} vs {r['away']}"

        if r["played"]:
            status = "JUGADO"
        else:
            status = "PENDIENTE"

        print(f"  [{status}]  {match_str}")
        print(f"         Prediccion:  {r['pred_score']:>5s} ({r['pred_score_pct']:.0f}%)  "
              f"L={r['local_win_pct']:.0f}%  E={r['draw_pct']:.0f}%  V={r['away_win_pct']:.0f}%  "
              f"Gexp={r['exp_goals_local']:.2f}-{r['exp_goals_away']:.2f}")

        if r["played"]:
            actual_winner_name, _ = result_label(r["local"], r["away"], r["actual_local_goals"], r["actual_away_goals"])
            correct_str = "ACIERTO" if r["correct"] else "FALLO"
            print(f"         Real:        {r['actual_local_goals']}-{r['actual_away_goals']}  "
                  f"Ganador: {actual_winner_name:20s}  [{correct_str}]")
        print()

    print(f"  {'=' * 68}")
    print(f"  Resumen: {correct}/{played} aciertos ({correct / played * 100:.1f}%)" if played > 0 else "  Aun sin partidos jugados.")
    print(f"  {'=' * 68}")


if __name__ == "__main__":
    main()
