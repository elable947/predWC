import json
import sys
from datetime import datetime
from io import StringIO

import numpy as np
import polars as pl
import requests

# Must come before other matplotlib imports
if "--graphs" in sys.argv:
    if "--save" in sys.argv:
        import matplotlib
        matplotlib.use("Agg")

    import matplotlib.pyplot as plt
    import seaborn as sns

RESULTS_URL = "https://raw.githubusercontent.com/martj42/international_results/refs/heads/master/results.csv"
KNOCKOUT_MATCHES = "data/knockout_matches.json"
PREDICTIONS = "data/knockout_predictions.csv"
PREDICTIONS_NLP = "data/knockout_predictions_nlp.csv"
WC_YEAR = 2026

LOCAL_COLOR = "#3498db"
AWAY_COLOR = "#e67e22"
DRAW_COLOR = "#f39c12"
OK_COLOR = "#2ecc71"
ERROR_COLOR = "#e74c3c"
PENDING_COLOR = "#bdc3c7"


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


# ────────────────────────────────────────────────────────────────
#  PLOTTING
# ────────────────────────────────────────────────────────────────


def plot_bracket_overview(rows):
    fig, ax = plt.subplots(figsize=(12, 9))
    fig.canvas.manager.set_window_title("Bracket Tracker — Overview")

    y = np.arange(len(rows))
    height = 0.35

    for i, r in enumerate(rows):
        status = r["status"]
        if status == "correct":
            color = OK_COLOR
        elif status == "incorrect":
            color = ERROR_COLOR
        else:
            color = PENDING_COLOR

        ax.barh(i, 1, height, color=color, alpha=0.3)

        label = f"{r['local']:20s} vs {r['away']:20s}"
        ax.text(0, i, label, va="center", fontsize=7, fontfamily="monospace")

        if status != "pending":
            pred = r["pred_winner"]
            actual = r["actual_winner"]
            pred_str = f"{r['pred_score']} ({pred[:4]})"
            actual_str = f"{r['actual_local_goals']}-{r['actual_away_goals']} ({actual[:4]})"
            ax.text(0.5, i + height / 2, pred_str, va="center", fontsize=7, ha="center", color="#555")
            ax.text(0.5, i - height / 2, actual_str, va="center", fontsize=7, ha="center",
                    color=OK_COLOR if status == "correct" else ERROR_COLOR)

    ax.set_yticks(y)
    ax.set_yticklabels([""] * len(rows))
    ax.set_xlim(0, 1)
    ax.set_ylim(-1, len(rows))
    ax.axis("off")
    ax.set_title("Bracket — 16avos World Cup 2026  (  + Acierto    x Fallo    \u2014 Pendiente  )",
                 fontsize=11, fontweight="bold")

    # Legend as colored blocks
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=OK_COLOR, alpha=0.3, label="Acierto"),
        Patch(facecolor=ERROR_COLOR, alpha=0.3, label="Fallo"),
        Patch(facecolor=PENDING_COLOR, alpha=0.3, label="Pendiente"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=9)

    fig.tight_layout()
    return fig


def plot_confidence_vs_outcome(rows):
    played = [r for r in rows if r["status"] != "pending"]
    if not played:
        return None

    fig, axes = plt.subplots(1, len(played), figsize=(4 * len(played), 4))
    fig.canvas.manager.set_window_title("Confianza vs Resultado Real")
    if len(played) == 1:
        axes = [axes]

    colors = [LOCAL_COLOR, DRAW_COLOR, AWAY_COLOR]

    for idx, r in enumerate(played):
        ax = axes[idx]
        labels = ["Local", "Empate", "Visitante"]
        vals = [r["local_win_pct"], r["draw_pct"], r["away_win_pct"]]
        bars = ax.bar(labels, vals, color=colors, width=0.6)

        actual_idx = 0 if r["actual_winner"] == "Local" else (2 if r["actual_winner"] == "Visitante" else 1)
        bars[actual_idx].set_edgecolor("black")
        bars[actual_idx].set_linewidth(3)

        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                    f"{v:.0f}%", ha="center", fontsize=9)

        short = r["local"].split()[0][:8] + " vs " + r["away"].split()[0][:8]
        ax.set_title(f"{short}\n{r['actual_local_goals']}-{r['actual_away_goals']}",
                     fontsize=9, fontweight="bold")
        ax.set_ylim(0, 105)
        ax.set_ylabel("%")

    fig.suptitle("Confianza del modelo vs resultado real (borde negro = resultado real)",
                 fontsize=12, y=1.02)
    fig.tight_layout()
    return fig


def plot_accuracy_summary(rows):
    n_correct = sum(1 for r in rows if r["status"] == "correct")
    n_incorrect = sum(1 for r in rows if r["status"] == "incorrect")
    n_pending = sum(1 for r in rows if r["status"] == "pending")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    fig.canvas.manager.set_window_title("Accuracy Summary")

    if n_correct + n_incorrect > 0:
        played = n_correct + n_incorrect
        pct = n_correct / played * 100
        labels_donut = [f"Aciertos\n{n_correct}", f"Fallos\n{n_incorrect}"]
        sizes = [n_correct, n_incorrect]
        colors_donut = [OK_COLOR, ERROR_COLOR]
        ax1.pie(sizes, labels=labels_donut, colors=colors_donut, autopct="",
                startangle=90, textprops={"fontsize": 11, "fontweight": "bold"})
        ax1.set_title(f"Partidos jugados: {played}/{len(rows)}  ({pct:.0f}%)", fontsize=11)
    else:
        ax1.text(0.5, 0.5, "Aun sin partidos jugados", ha="center", va="center", fontsize=11)
        ax1.set_title("Accuracy", fontsize=11)

    # Bar: total breakdown
    categories = ["Aciertos", "Fallos", "Pendientes"]
    counts = [n_correct, n_incorrect, n_pending]
    bar_colors = [OK_COLOR, ERROR_COLOR, PENDING_COLOR]
    bars = ax2.bar(categories, counts, color=bar_colors, width=0.5)
    for bar, v in zip(bars, counts):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                 str(v), ha="center", fontsize=12, fontweight="bold")
    ax2.set_ylim(0, max(counts) + 3)
    ax2.set_ylabel("Partidos")
    ax2.set_title(f"Total: {len(rows)} partidos", fontsize=11)

    fig.tight_layout()
    return fig


def plot_score_comparison(rows):
    played = [r for r in rows if r["status"] != "pending"]
    if not played:
        return None

    fig, ax = plt.subplots(figsize=(8, 4))
    fig.canvas.manager.set_window_title("Score Comparison")

    x = np.arange(len(played))
    width = 0.3

    pred_scores = []
    actual_scores = []
    labels = []
    for r in played:
        pg, _ = format_score(r["pred_score"])
        pred_scores.append(pg if pg is not None else 0)
        actual_scores.append(r["actual_local_goals"] + r["actual_away_goals"])
        labels.append(f"{r['local'].split()[0][:6]}-{r['away'].split()[0][:6]}")

    ax.bar(x - width / 2, pred_scores, width, label="Goles predichos (score mas probable)", color="#555")
    ax.bar(x + width / 2, actual_scores, width, label="Goles reales", color="#e67e22")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("Total goles")
    ax.set_title("Goles totales por partido: prediccion vs real")
    ax.legend(fontsize=9)

    for i, (p, a) in enumerate(zip(pred_scores, actual_scores)):
        ax.text(i - width / 2, p + 0.1, str(p), ha="center", fontsize=8, color="#555")
        ax.text(i + width / 2, a + 0.1, str(a), ha="center", fontsize=8, color="#e67e22")

    fig.tight_layout()
    return fig


def format_score(goal_str):
    parts = goal_str.split("-")
    if len(parts) == 2:
        try:
            return int(parts[0]), int(parts[1])
        except ValueError:
            return None, None
    return None, None


# ────────────────────────────────────────────────────────────────
#  MAIN
# ────────────────────────────────────────────────────────────────


def main():
    use_graphs = "--graphs" in sys.argv
    use_nlp = "--nlp" in sys.argv
    save_mode = "--save" in sys.argv

    csv_path = PREDICTIONS_NLP if use_nlp else PREDICTIONS
    label = "NLP" if use_nlp else "BASE"

    print("=" * 68)
    print(f"  BRACKET TRACKER \u2014 16avos World Cup 2026  [{label}]")
    print("=" * 68)

    if use_graphs:
        import os
        if "--save" not in sys.argv:
            if os.environ.get("DISPLAY") is None and sys.platform != "darwin":
                print("\n  ERROR: No display available. Use --save to generate PNGs.")
                print("    uv run python bracket_tracker.py --graphs --save")
                sys.exit(1)
        sns.set_style("whitegrid")
        plt.rcParams.update({"font.size": 11, "figure.dpi": 120})

    with open(KNOCKOUT_MATCHES) as f:
        bracket = json.load(f)

    try:
        pred_df = pl.read_csv(csv_path)
    except FileNotFoundError:
        fl = " --nlp" if use_nlp else ""
        print(f"\n  ERROR: {csv_path} not found.")
        print(f"  Run: uv run python stacking_model.py{fl}")
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

        pred_row = pred_df.filter(pl.col("match") == f"{local} vs {away}")
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
            status = "correct" if correct_flag else "incorrect"
        else:
            actual_winner_label = ""
            status = "pending"
            correct_flag = None

        rows.append({
            "local": local,
            "away": away,
            "status": status,
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

    # ── Text output ──
    print()
    print(f"  {'=' * 68}")
    print(f"  {'PARTIDO':^42s}  {'PRED':^12s}  {'REAL':^12s}")
    print(f"  {'=' * 68}")

    for r in rows:
        match_str = f"{r['local']:20s} vs {r['away']:20s}"

        if r["played"]:
            actual_str = f"{r['actual_local_goals']}-{r['actual_away_goals']}  {r['actual_winner'][:6]}"
            pred_str = f"{r['pred_score']:>5s}  {r['pred_winner'][:6]}"
            icon = "+" if r["correct"] else "x"
        else:
            actual_str = f"{'\u2014':^12s}"
            pred_str = f"{r['pred_score']:>5s}  {r['pred_winner'][:6]}"
            icon = " "

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
        status_txt = "JUGADO" if r["played"] else "PENDIENTE"

        print(f"  [{status_txt}]  {match_str}")
        print(f"         Prediccion:  {r['pred_score']:>5s} ({r['pred_score_pct']:.0f}%)  "
              f"L={r['local_win_pct']:.0f}%  E={r['draw_pct']:.0f}%  V={r['away_win_pct']:.0f}%  "
              f"Gexp={r['exp_goals_local']:.2f}-{r['exp_goals_away']:.2f}")

        if r["played"]:
            actual_winner_name, _ = result_label(
                r["local"], r["away"], r["actual_local_goals"], r["actual_away_goals"]
            )
            cstr = "ACIERTO" if r["correct"] else "FALLO"
            print(f"         Real:        {r['actual_local_goals']}-{r['actual_away_goals']}  "
                  f"Ganador: {actual_winner_name:20s}  [{cstr}]")
        print()

    print(f"  {'=' * 68}")
    if played > 0:
        print(f"  Resumen: {correct}/{played} aciertos ({correct / played * 100:.1f}%)")
    else:
        print("  Aun sin partidos jugados.")
    print(f"  {'=' * 68}")

    # ── Graphs ──
    if use_graphs:
        figs = [
            ("bracket_overview.png", plot_bracket_overview(rows)),
            ("accuracy_summary.png", plot_accuracy_summary(rows)),
            ("confidence_vs_outcome.png", plot_confidence_vs_outcome(rows)),
            ("score_comparison.png", plot_score_comparison(rows)),
        ]

        if save_mode:
            print("\nSaving charts...")
            for name, fig in figs:
                if fig is None:
                    continue
                fig.savefig(f"data/{name}", dpi=120, bbox_inches="tight")
                print(f"  Saved data/{name}")
                plt.close(fig)
            print("\nDone.")
        else:
            print("\nOpening interactive windows...")
            for _, fig in figs:
                if fig is None:
                    continue
            plt.show()


if __name__ == "__main__":
    main()
